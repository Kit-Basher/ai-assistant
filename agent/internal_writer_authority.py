from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import secrets
import sqlite3
import sys
import time
import threading
from typing import Any, Callable, TypeVar

from .capability_policy import stable_fingerprint


INTERNAL_WRITER_REGISTRY_SCHEMA = "personal-agent.internal-writer-registry.v1"
FORBIDDEN_PUBLIC_AUTHORITY_KEYS = frozenset(
    {
        "internal_authority",
        "internal_writer_authority",
        "internal_writer_context",
        "internal_writer_id",
        "service_identity",
        "writer_identity",
    }
)
FORBIDDEN_INTERNAL_ARGUMENT_KEYS = frozenset(
    {
        "capability_id",
        "executor",
        "executor_id",
        "shell",
        "command",
        "url",
        "path",
        "target_path",
        "trusted_invocation_context",
    }
)
_FACTORY_KEY = object()
_ISSUED: dict[str, tuple[str, str, str]] = {}
_ISSUED_LOCK = threading.RLock()
T = TypeVar("T")


def default_internal_writer_registry_path() -> Path:
    checkout = Path(__file__).resolve().parents[1] / "docs" / "operator" / "INTERNAL_WRITER_REGISTRY_V1.json"
    if checkout.is_file():
        return checkout
    installed = Path(sys.prefix) / "share" / "personal-agent" / "INTERNAL_WRITER_REGISTRY_V1.json"
    if installed.is_file():
        return installed
    raise FileNotFoundError("internal_writer_registry_not_installed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: Any, *, key_hint: str = "") -> Any:
    key = str(key_hint or "").lower()
    if any(marker in key for marker in ("token", "secret", "password", "authorization", "cookie", "content")):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, key_hint=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, key_hint=key_hint) for item in list(value)[:100]]
    if isinstance(value, str) and len(value) > 256:
        return value[:256] + "...[truncated]"
    return value


def _contains_forbidden_key(value: Any, forbidden: frozenset[str]) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key or "").strip().lower()
            if normalized in forbidden:
                return normalized
            nested = _contains_forbidden_key(item, forbidden)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _contains_forbidden_key(item, forbidden)
            if nested:
                return nested
    return None


def _validate_argument_limits(value: Any, *, max_string_bytes: int, max_collection_items: int) -> None:
    if isinstance(value, str) and len(value.encode("utf-8")) > max_string_bytes:
        raise ValueError("internal_writer_argument_string_too_large")
    if isinstance(value, dict):
        if len(value) > max_collection_items:
            raise ValueError("internal_writer_argument_collection_too_large")
        for item in value.values():
            _validate_argument_limits(item, max_string_bytes=max_string_bytes, max_collection_items=max_collection_items)
    elif isinstance(value, (list, tuple)):
        if len(value) > max_collection_items:
            raise ValueError("internal_writer_argument_collection_too_large")
        for item in value:
            _validate_argument_limits(item, max_string_bytes=max_string_bytes, max_collection_items=max_collection_items)


def reject_public_internal_authority_claim(payload: Any) -> str | None:
    """Return the forbidden key when serialized/public input claims authority."""
    return _contains_forbidden_key(payload, FORBIDDEN_PUBLIC_AUTHORITY_KEYS)


@dataclass(frozen=True)
class InternalWriterAuthority:
    writer_id: str
    capability_id: str
    trigger: str
    operation_id: str
    runtime_mode: str
    nonce: str

    @classmethod
    def _create(
        cls,
        factory_key: object,
        *,
        writer_id: str,
        capability_id: str,
        trigger: str,
        operation_id: str,
        runtime_mode: str,
    ) -> "InternalWriterAuthority":
        if factory_key is not _FACTORY_KEY:
            raise ValueError("internal_writer_factory_required")
        nonce = secrets.token_hex(24)
        authority = cls(writer_id, capability_id, trigger, operation_id, runtime_mode, nonce)
        with _ISSUED_LOCK:
            _ISSUED[nonce] = (writer_id, capability_id, operation_id)
        return authority


class InternalWriterRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or default_internal_writer_registry_path()).expanduser().resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != INTERNAL_WRITER_REGISTRY_SCHEMA:
            raise ValueError("internal_writer_registry_schema_invalid")
        rows = payload.get("writers") if isinstance(payload.get("writers"), list) else []
        self._writers: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("writer_id") or "").strip():
                raise ValueError("internal_writer_registry_entry_invalid")
            writer_id = str(row["writer_id"])
            if writer_id in self._writers:
                raise ValueError("internal_writer_registry_duplicate")
            self._writers[writer_id] = row

    def get(self, writer_id: str) -> dict[str, Any] | None:
        row = self._writers.get(str(writer_id or ""))
        return dict(row) if row else None

    def rows(self) -> list[dict[str, Any]]:
        return [dict(self._writers[key]) for key in sorted(self._writers)]


class InternalWriterFactory:
    """Trusted runtime/scheduler factory; serialized payloads cannot create it."""

    def __init__(self, *, trigger: str, registry: InternalWriterRegistry | None = None) -> None:
        normalized = str(trigger or "").strip().lower()
        if normalized not in {"runtime", "scheduler", "system"}:
            raise ValueError("internal_writer_trigger_invalid")
        self.trigger = normalized
        self.registry = registry or InternalWriterRegistry()

    def issue(self, writer_id: str, *, operation_id: str, runtime_mode: str = "safe") -> InternalWriterAuthority:
        definition = self.registry.get(writer_id)
        if definition is None:
            raise ValueError("internal_writer_unregistered")
        if str(definition.get("disposition") or "") not in {
            "trusted_bookkeeping",
            "scheduled_maintenance",
        }:
            raise ValueError("internal_writer_not_internal")
        if self.trigger not in set(definition.get("allowed_triggers") or []):
            raise ValueError("internal_writer_trigger_denied")
        mode = str(runtime_mode or "").strip().lower()
        if mode not in set(definition.get("modes") or []):
            raise ValueError("internal_writer_mode_denied")
        operation = str(operation_id or "").strip()
        if not operation:
            raise ValueError("internal_writer_operation_id_required")
        return InternalWriterAuthority._create(
            _FACTORY_KEY,
            writer_id=writer_id,
            capability_id=str(definition.get("capability_id") or ""),
            trigger=self.trigger,
            operation_id=operation,
            runtime_mode=mode,
        )


class InternalWriterJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(50):
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS internal_writer_operations (
                    operation_key TEXT PRIMARY KEY,
                    writer_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    target_scope TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    state TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    request_json TEXT NOT NULL,
                    receipt_json TEXT
                        )
                        """
                    )
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 49:
                    raise
                time.sleep(0.02)
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        self._restrict_files()
        return connection

    def _restrict_files(self) -> None:
        for candidate in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            if candidate.exists() and not candidate.is_symlink():
                os.chmod(candidate, 0o600)

    def reserve(self, operation_key: str, row: dict[str, Any]) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT INTO internal_writer_operations (
                        operation_key, writer_id, operation_id, capability_id,
                        operation, resource_type, target_scope, trigger, state,
                        started_at, request_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'executing', ?, ?)""",
                    (
                        operation_key,
                        row["writer_id"],
                        row["operation_id"],
                        row["capability_id"],
                        row["operation"],
                        row["resource_type"],
                        row["target_scope"],
                        row["trigger"],
                        _utc_now(),
                        json.dumps(_redact(row), sort_keys=True, ensure_ascii=True),
                    ),
                )
                connection.execute("COMMIT")
                return True
            except sqlite3.IntegrityError:
                connection.execute("ROLLBACK")
                return False
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def finish(self, operation_key: str, *, state: str, receipt: dict[str, Any]) -> None:
        if state not in {"succeeded", "failed", "indeterminate"}:
            raise ValueError("internal_writer_terminal_state_invalid")
        with self._connect() as connection:
            connection.execute(
                "UPDATE internal_writer_operations SET state = ?, finished_at = ?, receipt_json = ? WHERE operation_key = ? AND state = 'executing'",
                (state, _utc_now(), json.dumps(_redact(receipt), sort_keys=True, ensure_ascii=True), operation_key),
            )

    def reconcile_stale(self, *, before_iso: str) -> int:
        """Mark unknown executing outcomes indeterminate; never make them retryable."""
        with self._connect() as connection:
            return int(connection.execute(
                "UPDATE internal_writer_operations SET state = 'indeterminate', finished_at = ?, receipt_json = ? WHERE state = 'executing' AND started_at <= ?",
                (_utc_now(), json.dumps({"ok": False, "reason": "stale_execution_outcome_unknown"}), str(before_iso)),
            ).rowcount)


def execute_internal_write(
    *,
    authority: InternalWriterAuthority,
    operation: str,
    resource_type: str,
    target_scope: str,
    arguments: dict[str, Any],
    callback: Callable[[], T],
    journal: InternalWriterJournal,
    registry: InternalWriterRegistry | None = None,
) -> T:
    definitions = registry or InternalWriterRegistry()
    definition = definitions.get(authority.writer_id)
    if definition is None:
        raise ValueError("internal_writer_unregistered")
    with _ISSUED_LOCK:
        issued = _ISSUED.pop(authority.nonce, None)
    if issued != (authority.writer_id, authority.capability_id, authority.operation_id):
        raise ValueError("internal_writer_authority_invalid_or_replayed")
    if authority.capability_id != str(definition.get("capability_id") or ""):
        raise ValueError("internal_writer_capability_mismatch")
    if authority.trigger not in set(definition.get("allowed_triggers") or []):
        raise ValueError("internal_writer_trigger_denied")
    if authority.runtime_mode not in set(definition.get("modes") or []):
        raise ValueError("internal_writer_mode_denied")
    if operation not in set(definition.get("allowed_operations") or []):
        raise ValueError("internal_writer_operation_denied")
    if resource_type not in set(definition.get("resource_types") or []):
        raise ValueError("internal_writer_resource_denied")
    if target_scope not in set(definition.get("target_scopes") or []):
        raise ValueError("internal_writer_target_denied")
    forbidden = _contains_forbidden_key(arguments, FORBIDDEN_INTERNAL_ARGUMENT_KEYS | FORBIDDEN_PUBLIC_AUTHORITY_KEYS)
    if forbidden:
        raise ValueError(f"internal_writer_argument_denied:{forbidden}")
    schema = definition.get("argument_schema") if isinstance(definition.get("argument_schema"), dict) else {}
    if len(arguments) > int(schema.get("max_properties") or 0):
        raise ValueError("internal_writer_arguments_too_many")
    if len(json.dumps(arguments, ensure_ascii=True).encode("utf-8")) > int(schema.get("max_total_bytes") or 8192):
        raise ValueError("internal_writer_arguments_too_large")
    _validate_argument_limits(
        arguments,
        max_string_bytes=max(1, int(schema.get("max_string_bytes") or 1024)),
        max_collection_items=max(1, int(schema.get("max_collection_items") or 100)),
    )
    inspected_module = inspect.getmodule(callback)
    callback_module = str(getattr(callback, "__module__", "") or (inspected_module.__name__ if inspected_module else ""))
    if callback_module != str(definition.get("module") or ""):
        raise ValueError("internal_writer_callback_module_mismatch")
    operation_key = stable_fingerprint(
        {
            "writer_id": authority.writer_id,
            "operation_id": authority.operation_id,
            "capability_id": authority.capability_id,
            "operation": operation,
            "resource_type": resource_type,
            "target_scope": target_scope,
        }
    )
    request = {
        "writer_id": authority.writer_id,
        "operation_id": authority.operation_id,
        "capability_id": authority.capability_id,
        "operation": operation,
        "resource_type": resource_type,
        "target_scope": target_scope,
        "trigger": authority.trigger,
        "arguments": arguments,
    }
    if not journal.reserve(operation_key, request):
        raise ValueError("internal_writer_duplicate_operation")
    try:
        result = callback()
    except Exception as exc:
        journal.finish(operation_key, state="failed", receipt={"ok": False, "error": exc.__class__.__name__})
        raise
    journal.finish(operation_key, state="succeeded", receipt={"ok": True, "result_type": type(result).__name__})
    return result


def perform_registered_internal_write(
    *,
    writer_id: str,
    operation: str,
    resource_type: str,
    target_scope: str,
    arguments: dict[str, Any],
    callback: Callable[[], T],
    journal_path: str | Path,
    trigger: str = "runtime",
    runtime_mode: str = "safe",
    operation_id: str | None = None,
) -> T | None:
    """Convenience boundary for trusted module-owned bookkeeping writers."""
    durable_id = str(operation_id or "").strip() or stable_fingerprint(
        {
            "writer_id": writer_id,
            "operation": operation,
            "resource_type": resource_type,
            "target_scope": target_scope,
            "arguments": arguments,
        }
    )
    authority = InternalWriterFactory(trigger=trigger).issue(
        writer_id,
        operation_id=durable_id,
        runtime_mode=runtime_mode,
    )
    try:
        return execute_internal_write(
            authority=authority,
            operation=operation,
            resource_type=resource_type,
            target_scope=target_scope,
            arguments=arguments,
            callback=callback,
            journal=InternalWriterJournal(journal_path),
        )
    except ValueError as exc:
        if str(exc) == "internal_writer_duplicate_operation":
            return None
        raise
