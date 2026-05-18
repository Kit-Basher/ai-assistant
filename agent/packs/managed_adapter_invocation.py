from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent.packs.lifecycle import PackLifecycleResult, PackLifecycleService, render_lifecycle_response
from agent.packs.managed_adapters import (
    ADAPTER_LOCAL_FILE_IMPORT,
    GRANT_GRANTED,
    KNOWN_ADAPTER_KINDS,
    ManagedAdapterSpec,
    redact_private_history_path,
    validate_managed_adapter_spec,
)


OP_VALIDATE_GRANT = "validate_grant"
OP_DESCRIBE_CAPABILITY = "describe_capability"
OP_DRY_RUN = "dry_run"


@dataclass(frozen=True)
class ManagedAdapterOperation:
    adapter_kind: str
    name: str
    summary: str
    reads_content: bool = False
    writes_content: bool = False
    uses_network: bool = False
    executes_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_kind": self.adapter_kind,
            "name": self.name,
            "summary": self.summary,
            "reads_content": bool(self.reads_content),
            "writes_content": bool(self.writes_content),
            "uses_network": bool(self.uses_network),
            "executes_code": bool(self.executes_code),
        }


MANAGED_ADAPTER_OPERATION_REGISTRY: dict[str, dict[str, ManagedAdapterOperation]] = {
    ADAPTER_LOCAL_FILE_IMPORT: {
        OP_VALIDATE_GRANT: ManagedAdapterOperation(
            adapter_kind=ADAPTER_LOCAL_FILE_IMPORT,
            name=OP_VALIDATE_GRANT,
            summary="Validate that a metadata-only grant matches the pack's adapter declaration.",
        ),
        OP_DESCRIBE_CAPABILITY: ManagedAdapterOperation(
            adapter_kind=ADAPTER_LOCAL_FILE_IMPORT,
            name=OP_DESCRIBE_CAPABILITY,
            summary="Describe the granted adapter capability with redacted metadata and privacy notes.",
        ),
        OP_DRY_RUN: ManagedAdapterOperation(
            adapter_kind=ADAPTER_LOCAL_FILE_IMPORT,
            name=OP_DRY_RUN,
            summary="Re-check safe metadata for the user-selected local file without reading contents.",
        ),
    }
}

OPERATION_ALIASES = {
    "describe_grant": OP_DESCRIBE_CAPABILITY,
    "dry_run_access": OP_DRY_RUN,
}


@dataclass(frozen=True)
class ManagedAdapterInvocationRequest:
    pack_id: str | None
    canonical_id: str | None
    pack_name: str
    adapter_kind: str
    operation: str
    user_id: str | None = None
    thread_id: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    permission_grant_id: str | None = None
    grant_evidence: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "canonical_id": self.canonical_id,
            "pack_name": self.pack_name,
            "adapter_kind": self.adapter_kind,
            "operation": self.operation,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "parameters": dict(self.parameters),
            "permission_grant_id": self.permission_grant_id,
            "grant_evidence": _redact_grant(dict(self.grant_evidence)),
            "dry_run": bool(self.dry_run),
        }


@dataclass(frozen=True)
class ManagedAdapterInvocationError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ManagedAdapterInvocationResult:
    ok: bool
    adapter_kind: str
    operation: str
    did_work: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    errors: tuple[ManagedAdapterInvocationError, ...] = ()
    privacy_notes: tuple[str, ...] = ()
    redactions_applied: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "adapter_kind": self.adapter_kind,
            "operation": self.operation,
            "did_work": self.did_work,
            "summary": self.summary,
            "data": dict(self.data),
            "errors": [error.to_dict() for error in self.errors],
            "privacy_notes": list(self.privacy_notes),
            "redactions_applied": list(self.redactions_applied),
        }


class ManagedAdapterInvoker:
    def __init__(self, *, lifecycle_service: PackLifecycleService | None = None) -> None:
        self._lifecycle_service = lifecycle_service or PackLifecycleService()

    def invoke(
        self,
        request: ManagedAdapterInvocationRequest,
        *,
        lifecycle: PackLifecycleResult | dict[str, Any] | None = None,
        pack: dict[str, Any] | None = None,
        adapter_declarations: list[dict[str, Any]] | None = None,
        permission_grants: list[dict[str, Any]] | None = None,
    ) -> ManagedAdapterInvocationResult:
        lifecycle_row = _lifecycle_dict(lifecycle)
        if not lifecycle_row and isinstance(pack, dict):
            lifecycle_row = self._lifecycle_service.evaluate(
                imported_pack=pack,
                permission_grants=permission_grants or [],
            ).to_dict()
        if lifecycle_row and not bool(lifecycle_row.get("usable")):
            summary = (
                "I cannot invoke that external pack yet. "
                f"{render_lifecycle_response(lifecycle_row)}"
            )
            return _error_result(
                request,
                "lifecycle_not_usable",
                summary,
                details={
                    "state": lifecycle_row.get("state"),
                    "missing_gate": lifecycle_row.get("missing_gate"),
                    "next_step": lifecycle_row.get("next_step"),
                },
            )

        adapter_kind = _clean(request.adapter_kind)
        adapter_operations = MANAGED_ADAPTER_OPERATION_REGISTRY.get(adapter_kind)
        if adapter_operations is None:
            code = "adapter_kind_unsupported" if adapter_kind in KNOWN_ADAPTER_KINDS else "adapter_kind_unknown"
            return _error_result(
                request,
                code,
                f"Managed adapter {adapter_kind or 'unknown'} is not supported by this runtime.",
            )
        operation = _canonical_operation(request.operation)
        adapter_operation = adapter_operations.get(operation)
        if adapter_operation is None:
            return _error_result(
                request,
                "operation_unsupported",
                f"The managed adapter invocation layer is available, but operation {operation or 'unknown'} is not implemented for {adapter_kind} yet.",
            )
        request = replace(request, adapter_kind=adapter_kind, operation=adapter_operation.name)
        adapter = _find_adapter(adapter_declarations or [], adapter_kind)
        if adapter is None:
            return _error_result(request, "adapter_declaration_missing", "The pack does not declare the requested managed adapter.")
        ok, errors, spec = validate_managed_adapter_spec(adapter)
        if not ok:
            return _error_result(
                request,
                "adapter_declaration_invalid",
                "The pack's managed adapter declaration is invalid.",
                details={"errors": errors},
            )
        grant = _find_grant(request, permission_grants or [])
        if grant is None:
            return _error_result(request, "permission_grant_missing", "No matching metadata grant exists for this pack and adapter.")
        metadata_ok, metadata_errors = _validate_grant_metadata(grant, spec)
        if not metadata_ok:
            return _error_result(
                request,
                "permission_grant_invalid",
                "The metadata grant no longer matches the adapter policy.",
                details={"errors": metadata_errors, "grant": _redact_grant(grant)},
            )

        if adapter_operation.name == OP_VALIDATE_GRANT:
            return _ok_result(
                request,
                summary=f"{request.pack_name} has a matching metadata grant for {adapter_kind}. No file contents were read.",
                data={"operation": adapter_operation.to_dict(), "grant": _redact_grant(grant), "adapter": spec.to_dict()},
                did_work=False,
            )
        if adapter_operation.name == OP_DESCRIBE_CAPABILITY:
            return _ok_result(
                request,
                summary=f"{request.pack_name} can request {adapter_kind} through a core-owned managed adapter. No file contents were read.",
                data={"operation": adapter_operation.to_dict(), "grant": _redact_grant(grant), "adapter": spec.to_dict()},
                did_work=False,
            )
        if adapter_operation.name == OP_DRY_RUN:
            return self._dry_run(request, operation=adapter_operation, grant=grant, spec=spec)
        return _error_result(
            request,
            "operation_unsupported",
            f"Operation {adapter_operation.name} is registered but has no implementation.",
        )

    def _dry_run(
        self,
        request: ManagedAdapterInvocationRequest,
        *,
        operation: ManagedAdapterOperation,
        grant: dict[str, Any],
        spec: ManagedAdapterSpec,
    ) -> ManagedAdapterInvocationResult:
        path = _clean(grant.get("granted_path"))
        metadata = dict(grant.get("path_metadata") if isinstance(grant.get("path_metadata"), dict) else {})
        errors: list[str] = []
        resolved = Path(path).expanduser() if path else None
        exists = False
        is_file = False
        size_bytes: int | None = None
        extension = _clean(metadata.get("extension")).lower() or (resolved.suffix.lower() if resolved else "")
        if resolved is None:
            errors.append("granted_path_missing")
        else:
            try:
                exists = resolved.exists()
                is_file = resolved.is_file()
                if is_file:
                    size_bytes = resolved.stat().st_size
            except OSError:
                errors.append("path_metadata_unavailable")
        if not exists:
            errors.append("path_not_found")
        if exists and not is_file:
            errors.append("path_is_not_file")
        if extension not in set(spec.allowed_extensions):
            errors.append("extension_not_allowed")
        if size_bytes is not None and size_bytes > int(spec.max_file_size_mb) * 1024 * 1024:
            errors.append("file_too_large")
        data = {
            "operation": operation.to_dict(),
            "path_redacted": redact_private_history_path(path),
            "exists": exists,
            "is_file": is_file,
            "extension": extension,
            "size_bytes": size_bytes,
            "max_file_size_mb": spec.max_file_size_mb,
            "read_contents": False,
            "indexed_contents": False,
        }
        if errors:
            return _error_result(
                request,
                "dry_run_failed",
                f"The {request.adapter_kind} dry run failed metadata checks. No file contents were read.",
                details={"errors": sorted(dict.fromkeys(errors)), "data": data},
            )
        return _ok_result(
            request,
            summary=f"{request.pack_name} dry-run access passed for the selected local file. No file contents were read or indexed.",
            data=data,
            did_work=True,
        )


def _find_adapter(rows: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for row in rows:
        if isinstance(row, dict) and _clean(row.get("kind")) == kind:
            return dict(row)
    return None


def _canonical_operation(value: Any) -> str:
    operation = _clean(value)
    return OPERATION_ALIASES.get(operation, operation)


def _find_grant(request: ManagedAdapterInvocationRequest, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    pack_ids = {_clean(request.pack_id), _clean(request.canonical_id)}
    pack_ids.discard("")
    for row in rows:
        if not isinstance(row, dict):
            continue
        if request.permission_grant_id and _clean(row.get("grant_id")) != _clean(request.permission_grant_id):
            continue
        if _clean(row.get("adapter_kind")) != _clean(request.adapter_kind):
            continue
        if _clean(row.get("state")) != GRANT_GRANTED:
            continue
        if pack_ids and _clean(row.get("pack_id")) not in pack_ids:
            continue
        return dict(row)
    evidence = request.grant_evidence if isinstance(request.grant_evidence, dict) else {}
    if evidence and _clean(evidence.get("adapter_kind")) == _clean(request.adapter_kind) and _clean(evidence.get("state")) == GRANT_GRANTED:
        if not pack_ids or _clean(evidence.get("pack_id")) in pack_ids:
            return dict(evidence)
    return None


def _validate_grant_metadata(grant: dict[str, Any], spec: ManagedAdapterSpec) -> tuple[bool, list[str]]:
    errors: list[str] = []
    metadata = grant.get("path_metadata") if isinstance(grant.get("path_metadata"), dict) else {}
    extension = _clean(metadata.get("extension")).lower() or Path(_clean(grant.get("granted_path"))).suffix.lower()
    if extension not in set(spec.allowed_extensions):
        errors.append("extension_not_allowed")
    size = metadata.get("size_bytes")
    try:
        size_int = int(size) if size is not None else None
    except (TypeError, ValueError):
        size_int = None
        errors.append("size_metadata_invalid")
    if size_int is not None and size_int > int(spec.max_file_size_mb) * 1024 * 1024:
        errors.append("file_too_large")
    if metadata.get("is_file") is False:
        errors.append("path_is_not_file")
    return (not errors, sorted(dict.fromkeys(errors)))


def _ok_result(
    request: ManagedAdapterInvocationRequest,
    *,
    summary: str,
    data: dict[str, Any],
    did_work: bool,
) -> ManagedAdapterInvocationResult:
    return ManagedAdapterInvocationResult(
        ok=True,
        adapter_kind=request.adapter_kind,
        operation=request.operation,
        did_work=did_work,
        summary=summary,
        data=_redact_nested(data),
        privacy_notes=(
            "Raw local paths are redacted in normal response payloads.",
            "File contents were not read, parsed, uploaded, or indexed.",
            "No network, subprocess, dependency install, or generated handler execution is used.",
        ),
        redactions_applied=("local_path",),
    )


def _error_result(
    request: ManagedAdapterInvocationRequest,
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> ManagedAdapterInvocationResult:
    return ManagedAdapterInvocationResult(
        ok=False,
        adapter_kind=request.adapter_kind,
        operation=request.operation,
        did_work=False,
        summary=message,
        data={},
        errors=(ManagedAdapterInvocationError(code=code, message=message, details=_redact_nested(details or {})),),
        privacy_notes=(
            "No file contents were read.",
            "No network, subprocess, dependency install, or generated handler execution is used.",
        ),
        redactions_applied=("local_path",),
    )


def _redact_grant(grant: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(grant)
    if _clean(redacted.get("granted_path")):
        redacted["granted_path"] = redact_private_history_path(_clean(redacted.get("granted_path")))
    if _clean(redacted.get("requested_path")):
        redacted["requested_path"] = redact_private_history_path(_clean(redacted.get("requested_path")))
    metadata = redacted.get("path_metadata") if isinstance(redacted.get("path_metadata"), dict) else {}
    if metadata:
        redacted["path_metadata"] = _redact_nested(metadata)
    return redacted


def _redact_nested(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"granted_path", "requested_path", "path"}:
                out[key_text] = redact_private_history_path(_clean(item))
            elif key_text.endswith("_path") and isinstance(item, str):
                out[key_text] = redact_private_history_path(item)
            else:
                out[key_text] = _redact_nested(item)
        return out
    if isinstance(value, list):
        return [_redact_nested(item) for item in value]
    return value


def _lifecycle_dict(value: PackLifecycleResult | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, PackLifecycleResult):
        return value.to_dict()
    return dict(value) if isinstance(value, dict) else {}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
