from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from agent.config import canonical_state_dir


MANAGED_ACTION_STATUSES = {
    "planned",
    "running",
    "verified",
    "rolled_back",
    "failed",
    "recovery_needed",
}

_SENSITIVE_KEY_TOKENS = {
    "api_key",
    "authorization",
    "body",
    "content",
    "credential",
    "deleted_keys",
    "identifier",
    "memory",
    "password",
    "private",
    "prompt",
    "raw",
    "resource",
    "secret",
    "source_ref",
    "target",
    "text",
    "token",
    "value",
    "path",
}
_TELEGRAM_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_LONG_STRING_LIMIT = 256


def default_managed_action_journal_db_path() -> Path:
    return canonical_state_dir() / "managed_actions.db"


def redact_journal_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            lowered = normalized_key.lower()
            if lowered.endswith("_hash") or lowered.endswith("_hashes"):
                redacted[normalized_key] = redact_journal_value(item)
            elif any(token in lowered for token in _SENSITIVE_KEY_TOKENS):
                redacted[normalized_key] = "***redacted***"
            else:
                redacted[normalized_key] = redact_journal_value(item)
        return redacted
    if isinstance(value, list):
        return [redact_journal_value(item) for item in value[:100]]
    if isinstance(value, tuple):
        return [redact_journal_value(item) for item in value[:100]]
    if isinstance(value, str):
        scrubbed = _OPENAI_KEY_RE.sub("***redacted***", _TELEGRAM_TOKEN_RE.sub("***redacted***", value))
        if len(scrubbed) > _LONG_STRING_LIMIT:
            return f"{scrubbed[:_LONG_STRING_LIMIT]}...[truncated]"
        return scrubbed
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


class PersistentManagedActionJournalStore:
    """SQLite-backed current-state store for redacted managed-action journals."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or default_managed_action_journal_db_path()).expanduser().resolve()

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS managed_action_journals (
                    action_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    target_redacted TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    recovery_needed INTEGER NOT NULL DEFAULT 0,
                    recovery_hint TEXT,
                    journal_json TEXT NOT NULL,
                    owned_resources_json TEXT NOT NULL,
                    verification_json TEXT NOT NULL,
                    rollback_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_managed_action_journals_status_updated
                ON managed_action_journals(status, updated_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_managed_action_journals_recovery_updated
                ON managed_action_journals(recovery_needed, updated_at)
                """
            )

    def upsert(
        self,
        journal: Any,
        *,
        status: str,
        recovery_hint: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        if status not in MANAGED_ACTION_STATUSES:
            raise ValueError(f"unknown managed-action status: {status}")
        self.init_schema()
        payload = journal.to_dict() if hasattr(journal, "to_dict") else dict(journal)
        redacted = redact_journal_value(payload)
        if not isinstance(redacted, dict):
            raise ValueError("journal payload must be a mapping")

        action_id = str(redacted.get("action_id") or "").strip()
        action_type = str(redacted.get("action_type") or "").strip()
        if not action_id or not action_type:
            raise ValueError("journal payload must include action_id and action_type")

        now = updated_at or datetime.now(timezone.utc).isoformat()
        created_at = str(redacted.get("created_at") or now)
        completed_at = now if status in {"verified", "rolled_back", "failed"} else None
        recovery_needed = 1 if status == "recovery_needed" else 0
        owned_resources = {
            "created_resources": redacted.get("created_resources") if isinstance(redacted.get("created_resources"), list) else [],
            "changed_resources": redacted.get("changed_resources") if isinstance(redacted.get("changed_resources"), list) else [],
        }
        verification = redacted.get("verification_result") if isinstance(redacted.get("verification_result"), dict) else {}
        rollback = redacted.get("rollback_result") if isinstance(redacted.get("rollback_result"), dict) else {}

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO managed_action_journals (
                    action_id,
                    action_type,
                    target_redacted,
                    status,
                    created_at,
                    updated_at,
                    completed_at,
                    recovery_needed,
                    recovery_hint,
                    journal_json,
                    owned_resources_json,
                    verification_json,
                    rollback_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_id) DO UPDATE SET
                    action_type=excluded.action_type,
                    target_redacted=excluded.target_redacted,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    completed_at=excluded.completed_at,
                    recovery_needed=excluded.recovery_needed,
                    recovery_hint=excluded.recovery_hint,
                    journal_json=excluded.journal_json,
                    owned_resources_json=excluded.owned_resources_json,
                    verification_json=excluded.verification_json,
                    rollback_json=excluded.rollback_json
                """,
                (
                    action_id,
                    action_type,
                    str(redacted.get("target") or ""),
                    status,
                    created_at,
                    now,
                    completed_at,
                    recovery_needed,
                    redact_journal_value(recovery_hint) if recovery_hint else None,
                    _json_dumps(redacted),
                    _json_dumps(owned_resources),
                    _json_dumps(verification),
                    _json_dumps(rollback),
                ),
            )
        return self.get(action_id) or {}

    def get(self, action_id: str) -> dict[str, Any] | None:
        self.init_schema()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM managed_action_journals WHERE action_id = ?",
                (str(action_id or "").strip(),),
            ).fetchone()
        return _row_to_dict(row) if row else None

    def recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        self.init_schema()
        max_rows = max(1, min(int(limit or 20), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM managed_action_journals
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max_rows,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def incomplete(self, *, limit: int = 20) -> list[dict[str, Any]]:
        self.init_schema()
        max_rows = max(1, min(int(limit or 20), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM managed_action_journals
                WHERE status IN ('planned', 'running', 'recovery_needed') OR recovery_needed = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max_rows,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "action_id": row["action_id"],
        "action_type": row["action_type"],
        "target_redacted": row["target_redacted"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "recovery_needed": bool(row["recovery_needed"]),
        "recovery_hint": row["recovery_hint"],
        "journal": _json_loads(row["journal_json"], {}),
        "owned_resources": _json_loads(row["owned_resources_json"], {}),
        "verification_result": _json_loads(row["verification_json"], {}),
        "rollback_result": _json_loads(row["rollback_json"], {}),
    }


__all__ = [
    "MANAGED_ACTION_STATUSES",
    "PersistentManagedActionJournalStore",
    "default_managed_action_journal_db_path",
    "redact_journal_value",
]
