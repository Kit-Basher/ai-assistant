from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


ADAPTER_LOCAL_FILE_IMPORT = "local_file_import"
ADAPTER_LOCAL_DIRECTORY_IMPORT = "local_directory_import"
ADAPTER_GOOGLE_TAKEOUT_YOUTUBE_HISTORY = "google_takeout_youtube_history"
ADAPTER_TRANSCRIPT_LOOKUP = "transcript_lookup"
ADAPTER_NETWORK_FETCH = "network_fetch"

ENABLED_ADAPTER_KINDS = {ADAPTER_LOCAL_FILE_IMPORT}
KNOWN_ADAPTER_KINDS = {
    ADAPTER_LOCAL_FILE_IMPORT,
    ADAPTER_LOCAL_DIRECTORY_IMPORT,
    ADAPTER_GOOGLE_TAKEOUT_YOUTUBE_HISTORY,
    ADAPTER_TRANSCRIPT_LOOKUP,
    ADAPTER_NETWORK_FETCH,
}
USER_SELECTED_FILE_ONLY = "user_selected_file_only"
GRANT_REQUESTED = "requested"
GRANT_GRANTED = "granted"
GRANT_DENIED = "denied"

EXECUTABLE_OR_SCRIPT_EXTENSIONS = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".exe",
    ".js",
    ".mjs",
    ".ps1",
    ".py",
    ".rb",
    ".sh",
}


@dataclass(frozen=True)
class ManagedAdapterSpec:
    kind: str
    purpose: str
    allowed_extensions: tuple[str, ...]
    max_file_size_mb: int
    path_policy: str
    stores_local_index: bool = False
    network_allowed: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ManagedAdapterSpec":
        allowed = value.get("allowed_extensions") if isinstance(value.get("allowed_extensions"), list) else []
        return cls(
            kind=str(value.get("kind") or "").strip().lower(),
            purpose=str(value.get("purpose") or "").strip(),
            allowed_extensions=tuple(_normalize_extension(item) for item in allowed if str(item).strip()),
            max_file_size_mb=max(1, int(value.get("max_file_size_mb") or 1)),
            path_policy=str(value.get("path_policy") or "").strip().lower(),
            stores_local_index=bool(value.get("stores_local_index")),
            network_allowed=bool(value.get("network_allowed")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "purpose": self.purpose,
            "allowed_extensions": list(self.allowed_extensions),
            "max_file_size_mb": int(self.max_file_size_mb),
            "path_policy": self.path_policy,
            "stores_local_index": bool(self.stores_local_index),
            "network_allowed": bool(self.network_allowed),
        }


@dataclass(frozen=True)
class ManagedAdapterPermissionRequest:
    request_id: str
    pack_id: str
    pack_name: str
    adapter: ManagedAdapterSpec
    requested_path: str | None = None
    requested_path_redacted: str | None = None
    state: str = GRANT_REQUESTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "pack_id": self.pack_id,
            "pack_name": self.pack_name,
            "adapter": self.adapter.to_dict(),
            "requested_path": self.requested_path,
            "requested_path_redacted": self.requested_path_redacted,
            "state": self.state,
        }


@dataclass(frozen=True)
class ManagedAdapterGrant:
    grant_id: str
    request_id: str
    pack_id: str
    pack_name: str
    adapter_kind: str
    state: str
    granted_path: str | None
    granted_path_redacted: str | None
    path_metadata: dict[str, Any]
    granted_at: int
    permissions_granted: tuple[str, ...] = ()
    executes_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "request_id": self.request_id,
            "pack_id": self.pack_id,
            "pack_name": self.pack_name,
            "adapter_kind": self.adapter_kind,
            "state": self.state,
            "granted_path": self.granted_path,
            "granted_path_redacted": self.granted_path_redacted,
            "path_metadata": dict(self.path_metadata),
            "permissions_granted": list(self.permissions_granted),
            "executes_code": bool(self.executes_code),
            "granted_at": int(self.granted_at),
        }


def _normalize_extension(value: Any) -> str:
    ext = str(value or "").strip().lower()
    if ext and not ext.startswith("."):
        ext = "." + ext
    return ext


def validate_managed_adapter_spec(spec: ManagedAdapterSpec | dict[str, Any]) -> tuple[bool, list[str], ManagedAdapterSpec]:
    adapter = spec if isinstance(spec, ManagedAdapterSpec) else ManagedAdapterSpec.from_mapping(spec if isinstance(spec, dict) else {})
    errors: list[str] = []
    if not adapter.kind:
        errors.append("adapter_kind_missing")
    elif adapter.kind not in KNOWN_ADAPTER_KINDS:
        errors.append("adapter_kind_unknown")
    elif adapter.kind not in ENABLED_ADAPTER_KINDS:
        errors.append("adapter_kind_disabled")
    if adapter.kind == ADAPTER_LOCAL_FILE_IMPORT and adapter.network_allowed:
        errors.append("local_file_import_network_not_allowed")
    if adapter.path_policy != USER_SELECTED_FILE_ONLY:
        errors.append("path_policy_must_be_user_selected_file_only")
    if not adapter.allowed_extensions:
        errors.append("allowed_extensions_required")
    for ext in adapter.allowed_extensions:
        if ext in {"*", ".*", "*.*"} or "*" in ext:
            errors.append("wildcard_extensions_not_allowed")
        if ext in EXECUTABLE_OR_SCRIPT_EXTENSIONS:
            errors.append("executable_extensions_not_allowed")
    if adapter.kind == ADAPTER_LOCAL_FILE_IMPORT:
        if any(str(flag).lower() in {"true", "1", "yes"} for flag in (getattr(adapter, "directory_scanning", None),)):
            errors.append("directory_scanning_not_allowed")
    return (not errors, sorted(dict.fromkeys(errors)), adapter)


def validate_managed_adapter_declarations(rows: Any) -> tuple[bool, list[str], list[dict[str, Any]]]:
    if not isinstance(rows, list):
        return True, [], []
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        ok, row_errors, adapter = validate_managed_adapter_spec(row if isinstance(row, dict) else {})
        normalized.append(adapter.to_dict())
        for error in row_errors:
            errors.append(f"managed_adapters[{index}].{error}")
        raw = row if isinstance(row, dict) else {}
        for blocked_key in ("dependencies", "dependency_installs", "handler", "handler_py", "recursive", "directory_scanning"):
            if raw.get(blocked_key):
                errors.append(f"managed_adapters[{index}].{blocked_key}_not_allowed")
    return (not errors, sorted(dict.fromkeys(errors)), normalized)


def redact_private_history_path(path: str | None) -> str | None:
    cleaned = str(path or "").strip()
    if not cleaned:
        return None
    name = Path(cleaned).name or "selected-file"
    return f"<redacted-local-history-path>/{name}"


_PATH_RE = re.compile(r"(?P<path>(?:~|/)[^\s\"']+)")


def extract_local_path(text: str | None) -> str | None:
    match = _PATH_RE.search(str(text or ""))
    if match is None:
        return None
    return str(match.group("path") or "").strip().rstrip(".,;)")


def build_permission_request(
    *,
    pack_id: str,
    pack_name: str,
    adapter: ManagedAdapterSpec | dict[str, Any],
    requested_path: str | None = None,
) -> ManagedAdapterPermissionRequest:
    spec = adapter if isinstance(adapter, ManagedAdapterSpec) else ManagedAdapterSpec.from_mapping(adapter if isinstance(adapter, dict) else {})
    payload = json.dumps(
        {
            "pack_id": str(pack_id or "").strip(),
            "pack_name": str(pack_name or "").strip(),
            "adapter": spec.to_dict(),
            "requested_path": redact_private_history_path(requested_path),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    request_id = "adapter-request-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return ManagedAdapterPermissionRequest(
        request_id=request_id,
        pack_id=str(pack_id or "").strip(),
        pack_name=str(pack_name or "Generated pack").strip(),
        adapter=spec,
        requested_path=str(requested_path or "").strip() or None,
        requested_path_redacted=redact_private_history_path(requested_path),
    )


def render_permission_preview(request: ManagedAdapterPermissionRequest) -> str:
    adapter = request.adapter
    extensions = ", ".join(adapter.allowed_extensions) or "none"
    selected = request.requested_path_redacted or "not provided yet"
    stored = "a derived local search index may be stored later" if adapter.stores_local_index else "no derived index is requested"
    network = "no network access is used" if not adapter.network_allowed else "network access was requested and is not allowed for this adapter"
    return (
        f"{request.pack_name} needs permission for managed adapter {adapter.kind}. "
        f"Data accessed: one user-selected local file only, with allowed extensions {extensions}, up to {adapter.max_file_size_mb} MB. "
        f"Selected path: {selected}. "
        f"Purpose: {adapter.purpose or 'local import'}. "
        f"Stored data: {stored}; raw file contents are not logged or added to support context. "
        f"Blocked: arbitrary path scanning, directory imports, browser profile scraping, OAuth, network fetches, dependency installs, and executable pack code. "
        f"Network: {network}. "
        "Say yes to record this grant metadata only; I will not read or parse the file in this phase."
    ).strip()


def validate_local_file_path_metadata(path: str, adapter: ManagedAdapterSpec) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    resolved = Path(path).expanduser()
    metadata = {
        "path_redacted": redact_private_history_path(str(path)),
        "extension": resolved.suffix.lower(),
        "exists": resolved.exists(),
        "is_file": False,
        "size_bytes": None,
        "max_file_size_mb": int(adapter.max_file_size_mb),
    }
    if not resolved.exists():
        errors.append("path_not_found")
        return False, errors, metadata
    if not resolved.is_file():
        errors.append("path_is_not_file")
        metadata["is_file"] = False
        return False, errors, metadata
    metadata["is_file"] = True
    suffix = resolved.suffix.lower()
    if suffix not in set(adapter.allowed_extensions):
        errors.append("extension_not_allowed")
    size_bytes = resolved.stat().st_size
    metadata["size_bytes"] = size_bytes
    if size_bytes > int(adapter.max_file_size_mb) * 1024 * 1024:
        errors.append("file_too_large")
    return (not errors, errors, metadata)


def grants_store_path(storage_root: str | Path) -> Path:
    return Path(storage_root).expanduser().resolve() / "managed_adapter_grants.json"


def list_adapter_grants(storage_root: str | Path) -> list[dict[str, Any]]:
    path = grants_store_path(storage_root)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(row) for row in parsed if isinstance(row, dict)] if isinstance(parsed, list) else []


def record_adapter_grant(storage_root: str | Path, grant: ManagedAdapterGrant) -> dict[str, Any]:
    path = grants_store_path(storage_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list_adapter_grants(storage_root)
    payload = grant.to_dict()
    rows = [row for row in rows if str(row.get("grant_id") or "") != grant.grant_id]
    rows.append(payload)
    path.write_text(json.dumps(rows, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def create_metadata_only_grant(
    *,
    request: ManagedAdapterPermissionRequest,
    state: str = GRANT_GRANTED,
    path_metadata: dict[str, Any] | None = None,
) -> ManagedAdapterGrant:
    payload = json.dumps(
        {
            "request_id": request.request_id,
            "pack_id": request.pack_id,
            "adapter_kind": request.adapter.kind,
            "path": request.requested_path,
            "state": state,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return ManagedAdapterGrant(
        grant_id="adapter-grant-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        request_id=request.request_id,
        pack_id=request.pack_id,
        pack_name=request.pack_name,
        adapter_kind=request.adapter.kind,
        state=state,
        granted_path=request.requested_path,
        granted_path_redacted=request.requested_path_redacted,
        path_metadata=dict(path_metadata or {}),
        permissions_granted=(),
        executes_code=False,
        granted_at=int(time.time()),
    )
