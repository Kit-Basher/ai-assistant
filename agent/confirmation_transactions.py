from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from .capability_policy import stable_fingerprint
from .mutation_plan import validate_mutation_confirmation


CONFIRMATION_TRANSACTION_SCHEMA_VERSION = 1
TERMINAL_STATES = frozenset({"succeeded", "failed", "indeterminate"})
VALID_STATES = frozenset({"reserved", "executing", *TERMINAL_STATES})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expiry_epoch(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def operation_key_for_plan(plan: dict[str, Any]) -> str:
    return stable_fingerprint(
        {
            "plan_id": str(plan.get("plan_id") or ""),
            "plan_fingerprint": str(plan.get("plan_fingerprint") or ""),
            "capability_id": str(plan.get("capability_id") or ""),
            "executor_id": str(plan.get("executor_id") or ""),
        }
    )


def confirmation_key_for_scope(plan: dict[str, Any], confirmation: dict[str, Any]) -> str:
    return stable_fingerprint(
        {
            "confirmation_id": str(confirmation.get("confirmation_id") or ""),
            "plan_id": str(plan.get("plan_id") or ""),
            "plan_fingerprint": str(plan.get("plan_fingerprint") or ""),
            "actor_id": str(confirmation.get("actor_id") or ""),
            "thread_id": str(confirmation.get("thread_id") or ""),
            "session_id": str(confirmation.get("session_id") or ""),
        }
    )


@dataclass(frozen=True)
class ReservationResult:
    allowed: bool
    reason_code: str
    operation_key: str
    state: str
    owner_id: str = ""


class ConfirmationTransactionStore:
    """Durable exactly-once reservation ledger for confirmed mutations.

    SQLite's ``BEGIN IMMEDIATE`` serializes the uniqueness check and insert
    across threads and processes.  A row is never deleted or reset to pending.
    This deliberately favors duplicate prevention over automatic recovery.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stale_after_seconds: int = 300,
        legacy_paths: list[str | Path] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stale_after_seconds = max(30, int(stale_after_seconds))
        self._initialize()
        self._migrate_legacy_paths(legacy_paths or [])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        self._restrict_files()
        return connection

    def _restrict_files(self) -> None:
        for candidate in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            if candidate.exists() and not candidate.is_symlink():
                os.chmod(candidate, 0o600)

    def _initialize(self) -> None:
        for attempt in range(50):
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS confirmation_transaction_meta (
                            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                            schema_version INTEGER NOT NULL
                        )
                        """
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO confirmation_transaction_meta (singleton, schema_version) VALUES (1, ?)",
                        (CONFIRMATION_TRANSACTION_SCHEMA_VERSION,),
                    )
                    version = connection.execute(
                        "SELECT schema_version FROM confirmation_transaction_meta WHERE singleton = 1"
                    ).fetchone()
                    if version is None or int(version[0]) != CONFIRMATION_TRANSACTION_SCHEMA_VERSION:
                        raise ValueError("confirmation_transaction_schema_unsupported")
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS confirmation_transactions (
                    operation_key TEXT PRIMARY KEY,
                    confirmation_key TEXT NOT NULL UNIQUE,
                    plan_id TEXT NOT NULL,
                    plan_fingerprint TEXT NOT NULL,
                    target_fingerprint TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    executor_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    state TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    reserved_at TEXT NOT NULL,
                    executing_at TEXT,
                    finished_at TEXT,
                    updated_epoch INTEGER NOT NULL,
                    result_json TEXT,
                    reconciliation_reason TEXT,
                    schema_version INTEGER NOT NULL
                        )
                        """
                    )
                    self._restrict_files()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 49:
                    raise
                time.sleep(0.02)

    def _migrate_legacy_paths(self, paths: list[str | Path]) -> None:
        for raw in paths:
            source = Path(raw).expanduser().resolve()
            if source == self.path or not source.is_file() or source.is_symlink():
                continue
            with sqlite3.connect(str(source), timeout=30.0) as old:
                exists = old.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'confirmation_transactions'"
                ).fetchone()
                if not exists:
                    continue
                columns = [str(row[1]) for row in old.execute("PRAGMA table_info(confirmation_transactions)")]
                canonical_columns = (
                    "operation_key", "confirmation_key", "plan_id", "plan_fingerprint",
                    "target_fingerprint", "capability_id", "executor_id", "actor_id",
                    "thread_id", "session_id", "expires_at", "state", "owner_id",
                    "reserved_at", "executing_at", "finished_at", "updated_epoch",
                    "result_json", "reconciliation_reason", "schema_version",
                )
                if not set(canonical_columns).issubset(set(columns)):
                    raise ValueError("legacy_confirmation_transaction_schema_unsupported")
                old.row_factory = sqlite3.Row
                rows = old.execute(
                    f"SELECT {','.join(canonical_columns)} FROM confirmation_transactions"
                ).fetchall()
            with self._connect() as current:
                current.execute("BEGIN IMMEDIATE")
                try:
                    for row in rows:
                        values = dict(row)
                        current.execute(
                            f"INSERT OR IGNORE INTO confirmation_transactions ({','.join(canonical_columns)}) VALUES ({','.join('?' for _ in canonical_columns)})",
                            tuple(values[name] for name in canonical_columns),
                        )
                    current.execute("COMMIT")
                except Exception:
                    current.execute("ROLLBACK")
                    raise

    def reserve(self, *, plan: dict[str, Any], confirmation: dict[str, Any]) -> ReservationResult:
        operation_key = operation_key_for_plan(plan)
        confirmation_key = confirmation_key_for_scope(plan, confirmation)
        owner_id = f"reservation-{uuid.uuid4().hex}"
        now_epoch = int(time.time())
        now_iso = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                try:
                    validate_mutation_confirmation(plan, confirmation, now=now_epoch)
                except ValueError as exc:
                    connection.execute("ROLLBACK")
                    return ReservationResult(False, str(exc), operation_key, "rejected")
                existing = connection.execute(
                    "SELECT operation_key, state, owner_id, updated_epoch FROM confirmation_transactions WHERE operation_key = ? OR confirmation_key = ?",
                    (operation_key, confirmation_key),
                ).fetchone()
                if existing is not None:
                    state = str(existing["state"] or "")
                    age = max(0, now_epoch - int(existing["updated_epoch"] or 0))
                    if state in {"reserved", "executing"} and age >= self.stale_after_seconds:
                        recovered_state = "failed" if state == "reserved" else "indeterminate"
                        reason = "stale_before_execution" if state == "reserved" else "stale_execution_outcome_unknown"
                        connection.execute(
                            "UPDATE confirmation_transactions SET state = ?, finished_at = ?, updated_epoch = ?, reconciliation_reason = ? WHERE operation_key = ?",
                            (recovered_state, now_iso, now_epoch, reason, str(existing["operation_key"])),
                        )
                        state = recovered_state
                    connection.execute("COMMIT")
                    reason_code = (
                        "mutation_confirmation_in_progress"
                        if state in {"reserved", "executing"}
                        else "mutation_confirmation_reconciliation_required"
                        if state == "indeterminate"
                        else "mutation_confirmation_replayed"
                    )
                    return ReservationResult(False, reason_code, operation_key, state, str(existing["owner_id"] or ""))
                if _expiry_epoch(plan.get("expires_at")) <= time.time():
                    connection.execute("COMMIT")
                    return ReservationResult(False, "mutation_confirmation_expired", operation_key, "expired")
                connection.execute(
                    """
                    INSERT INTO confirmation_transactions (
                        operation_key, confirmation_key, plan_id, plan_fingerprint,
                        target_fingerprint, capability_id, executor_id, actor_id,
                        thread_id, session_id, expires_at, state, owner_id,
                        reserved_at, updated_epoch, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?)
                    """,
                    (
                        operation_key,
                        confirmation_key,
                        str(plan.get("plan_id") or ""),
                        str(plan.get("plan_fingerprint") or ""),
                        str(plan.get("target_fingerprint") or ""),
                        str(plan.get("capability_id") or ""),
                        str(plan.get("executor_id") or ""),
                        str(confirmation.get("actor_id") or ""),
                        str(confirmation.get("thread_id") or ""),
                        str(confirmation.get("session_id") or ""),
                        str(plan.get("expires_at") or ""),
                        owner_id,
                        now_iso,
                        now_epoch,
                        CONFIRMATION_TRANSACTION_SCHEMA_VERSION,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return ReservationResult(True, "reserved", operation_key, "reserved", owner_id)

    def mark_executing(self, operation_key: str, owner_id: str) -> bool:
        return self._transition(operation_key, owner_id, from_state="reserved", to_state="executing")

    def finish(self, operation_key: str, owner_id: str, *, state: str, result: dict[str, Any]) -> bool:
        if state not in TERMINAL_STATES:
            raise ValueError("confirmation_transaction_terminal_state_invalid")
        now_epoch = int(time.time())
        result_json = json.dumps(result, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                changed = connection.execute(
                    """
                    UPDATE confirmation_transactions
                    SET state = ?, finished_at = ?, updated_epoch = ?, result_json = ?
                    WHERE operation_key = ? AND owner_id = ? AND state = 'executing'
                    """,
                    (state, _utc_now(), now_epoch, result_json, operation_key, owner_id),
                ).rowcount
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return changed == 1

    def _transition(self, operation_key: str, owner_id: str, *, from_state: str, to_state: str) -> bool:
        if from_state not in VALID_STATES or to_state not in VALID_STATES:
            raise ValueError("confirmation_transaction_state_invalid")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                changed = connection.execute(
                    "UPDATE confirmation_transactions SET state = ?, executing_at = ?, updated_epoch = ? WHERE operation_key = ? AND owner_id = ? AND state = ?",
                    (to_state, _utc_now(), int(time.time()), operation_key, owner_id, from_state),
                ).rowcount
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return changed == 1

    def get(self, operation_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM confirmation_transactions WHERE operation_key = ?",
                (str(operation_key or ""),),
            ).fetchone()
        return dict(row) if row is not None else None

    def reconcile_stale(self, *, now_epoch: int | None = None) -> dict[str, int]:
        current = int(time.time() if now_epoch is None else now_epoch)
        cutoff = current - self.stale_after_seconds
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                reserved = connection.execute(
                    "UPDATE confirmation_transactions SET state = 'failed', finished_at = ?, updated_epoch = ?, reconciliation_reason = 'stale_before_execution' WHERE state = 'reserved' AND updated_epoch <= ?",
                    (_utc_now(), current, cutoff),
                ).rowcount
                executing = connection.execute(
                    "UPDATE confirmation_transactions SET state = 'indeterminate', finished_at = ?, updated_epoch = ?, reconciliation_reason = 'stale_execution_outcome_unknown' WHERE state = 'executing' AND updated_epoch <= ?",
                    (_utc_now(), current, cutoff),
                ).rowcount
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {"failed_before_execution": int(reserved), "indeterminate": int(executing)}
