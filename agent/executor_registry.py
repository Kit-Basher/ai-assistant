from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
import uuid


SUPPORT_BUNDLE_SCHEMA_VERSION = "support_bundle.v2"
BACKUP_SCHEMA_VERSION = "backup.v1"
BACKUP_MAX_TOTAL_BYTES = 2 * 1024 * 1024
BACKUP_MAX_FILE_BYTES = 256 * 1024
BACKUP_MAX_JOURNAL_ENTRIES = 8
EXECUTOR_JOURNAL_MAX_RECORD_BYTES = 64 * 1024
EXECUTOR_JOURNAL_MAX_STRING_BYTES = 1024
SUPPORT_BUNDLE_MAX_TOTAL_BYTES = 2 * 1024 * 1024
SUPPORT_BUNDLE_MAX_FILE_BYTES = 256 * 1024
CLEANUP_MAX_CANDIDATES = 50
CLEANUP_MAX_SCAN_ENTRIES = 10000

SECRET_KEY_HINTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "confirmation_token",
    "cookie",
    "password",
    "private_key",
    "secret",
    "server.secret_key",
    "sudo",
    "token",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_executor_value(value: Any, *, key_hint: str = "") -> Any:
    normalized_key = str(key_hint or "").lower()
    if any(hint in normalized_key for hint in SECRET_KEY_HINTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            redacted[str(key)] = redact_executor_value(item, key_hint=str(key))
        return redacted
    if isinstance(value, list):
        return [redact_executor_value(item, key_hint=key_hint) for item in value[:100]]
    if isinstance(value, tuple):
        return [redact_executor_value(item, key_hint=key_hint) for item in value[:100]]
    if isinstance(value, str):
        lowered = value.lower()
        if any(
            hint in lowered
            for hint in (
                "bot token",
                "bearer ",
                "api_key=",
                "apikey=",
                "authorization:",
                "password=",
                "secret=",
                "token=",
                "x-api-key",
            )
        ):
            return "[REDACTED]"
        if len(value) > 512:
            return value[:512] + "...[truncated]"
    return value


def safe_path_label(path: Any) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    home = str(Path.home())
    if raw == home:
        return "~"
    if raw.startswith(home + "/"):
        raw = "~/" + raw[len(home) + 1 :]
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if len(parts) <= 4:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{'/'.join(parts[:2])}/.../{parts[-1]}#sha256:{digest}"


def support_bundle_redact(value: Any, *, key_hint: str = "") -> Any:
    redacted = redact_executor_value(value, key_hint=key_hint)
    normalized_key = str(key_hint or "").lower()
    if isinstance(redacted, dict):
        safe: dict[str, Any] = {}
        for key, item in redacted.items():
            key_text = str(key)
            safe[key_text] = support_bundle_redact(item, key_hint=key_text)
        return safe
    if isinstance(redacted, list):
        return [support_bundle_redact(item, key_hint=key_hint) for item in redacted[:80]]
    if isinstance(redacted, str):
        if any(token in normalized_key for token in ("path", "file", "dir", "root", "cwd")):
            return safe_path_label(redacted)
    return redacted


@dataclass(frozen=True)
class ExecutorResult:
    ok: bool
    mutated: bool
    executor_id: str
    plan_id: str
    action_type: str
    target: str
    started_at: str
    finished_at: str
    resources_touched: list[str] = field(default_factory=list)
    journal_id: str = ""
    rollback_available: bool = False
    rollback_hint: str = ""
    error_code: str | None = None
    user_message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "ok": bool(self.ok),
            "mutated": bool(self.mutated),
            "executor_id": self.executor_id,
            "plan_id": self.plan_id,
            "action_type": self.action_type,
            "target": self.target,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "resources_touched": list(self.resources_touched),
            "journal_id": self.journal_id,
            "rollback_available": bool(self.rollback_available),
            "rollback_hint": self.rollback_hint,
            "error_code": self.error_code,
            "user_message": self.user_message,
            "details": dict(self.details),
        }
        return redact_executor_value(payload)


ExecutorFn = Callable[[dict[str, Any], dict[str, Any]], ExecutorResult | dict[str, Any]]


class ExecutorPartialFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        resources_touched: list[str] | None = None,
        rollback_hint: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.resources_touched = list(resources_touched or [])
        self.rollback_hint = rollback_hint
        self.details = dict(details or {})


@dataclass(frozen=True)
class ExecutorSpec:
    executor_id: str
    action_type: str
    status: str
    run: ExecutorFn | None = None
    rollback_available: bool = False
    rollback_hint: str = ""


class MutationJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> str:
        journal_id = str(record.get("journal_id") or f"executor-{uuid.uuid4().hex[:12]}")
        payload = _bounded_journal_record(redact_executor_value({**record, "journal_id": journal_id}))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        return journal_id

    def recent(self, limit: int = 20, *, max_tail_bytes: int = 512 * 1024, max_line_bytes: int = 64 * 1024) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - max_tail_bytes))
                data = handle.read(max_tail_bytes)
        except OSError:
            return []
        if size > max_tail_bytes:
            _, _, data = data.partition(b"\n")
        lines = data.decode("utf-8", errors="replace").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-max(0, int(limit)) :]:
            if len(line.encode("utf-8", errors="replace")) > max_line_bytes:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
        return out


class ExecutorRegistry:
    def __init__(self, journal_path: str | Path) -> None:
        self.journal = MutationJournal(journal_path)
        self._executors: dict[str, ExecutorSpec] = {}

    def register(self, spec: ExecutorSpec) -> None:
        action_type = str(spec.action_type or "").strip().lower()
        if not action_type:
            raise ValueError("executor action_type is required")
        self._executors[action_type] = spec

    def lookup(self, action_type: str) -> ExecutorSpec | None:
        return self._executors.get(str(action_type or "").strip().lower())

    def execute_confirmed_plan(
        self,
        *,
        plan: dict[str, Any],
        action: dict[str, Any],
        high_risk_confirmed: bool = True,
    ) -> ExecutorResult:
        started_at = utc_now_iso()
        plan_id = str(plan.get("plan_id") or action.get("pending_id") or "").strip()
        action_type = str(plan.get("action_type") or "").strip().lower()
        target = str(plan.get("target") or "unspecified").strip() or "unspecified"
        executor_status = str(plan.get("executor_status") or "unavailable").strip().lower() or "unavailable"
        risk_level = str(plan.get("risk_level") or "").strip().lower()
        journal_id = f"executor-{uuid.uuid4().hex[:12]}"

        def _result(
            *,
            ok: bool,
            mutated: bool,
            executor_id: str,
            error_code: str | None,
            user_message: str,
            resources_touched: list[str] | None = None,
            rollback_available: bool = False,
            rollback_hint: str = "",
            details: dict[str, Any] | None = None,
        ) -> ExecutorResult:
            finished_at = utc_now_iso()
            result = ExecutorResult(
                ok=ok,
                mutated=mutated,
                executor_id=executor_id,
                plan_id=plan_id,
                action_type=action_type,
                target=target,
                started_at=started_at,
                finished_at=finished_at,
                resources_touched=list(resources_touched or []),
                journal_id=journal_id,
                rollback_available=rollback_available,
                rollback_hint=rollback_hint,
                error_code=error_code,
                user_message=user_message,
                details=dict(details or {}),
            )
            self.journal.append(
                {
                    "journal_id": journal_id,
                    "event": "executor_result",
                    "plan": plan,
                    "action": action,
                    "result": result.to_dict(),
                }
            )
            return result

        if not plan_id:
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="plan_id_missing",
                user_message="I blocked execution because the confirmed plan had no plan_id.",
            )
        if str(action.get("pending_id") or "").strip() and plan_id != str(action.get("pending_id") or "").strip():
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="plan_id_mismatch",
                user_message="I blocked execution because the confirmed plan_id did not match the pending action.",
            )
        if risk_level == "high" and not high_risk_confirmed:
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="high_risk_confirmation_required",
                user_message="I blocked this high-risk action because it did not have explicit confirmation.",
            )
        if executor_status == "preview_only":
            return _result(
                ok=False,
                mutated=False,
                executor_id="preview_only",
                error_code="executor_not_enabled",
                user_message="This plan was confirmed, but its executor is preview-only. I did not mutate state.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing was changed.",
            )
        if executor_status == "unavailable":
            return _result(
                ok=False,
                mutated=False,
                executor_id="unavailable",
                error_code="executor_unavailable",
                user_message="This plan was confirmed, but no executor is available. I did not mutate state.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing was changed.",
            )
        if executor_status != "enabled":
            return _result(
                ok=False,
                mutated=False,
                executor_id="executor_registry",
                error_code="executor_status_unknown",
                user_message="This plan used an unknown executor status, so I blocked it.",
            )
        spec = self.lookup(action_type)
        if spec is None or str(spec.status or "").strip().lower() != "enabled" or not callable(spec.run):
            return _result(
                ok=False,
                mutated=False,
                executor_id=(spec.executor_id if spec else "unregistered"),
                error_code="executor_unavailable",
                user_message="This action has no enabled executor in the registry. I did not mutate state.",
                rollback_available=False,
                rollback_hint="No rollback needed because nothing was changed.",
            )
        try:
            raw = spec.run(plan, action)
            if isinstance(raw, ExecutorResult):
                result = raw
            else:
                finished_at = utc_now_iso()
                result = ExecutorResult(
                    ok=bool(raw.get("ok")),
                    mutated=bool(raw.get("mutated")),
                    executor_id=str(raw.get("executor_id") or spec.executor_id),
                    plan_id=plan_id,
                    action_type=action_type,
                    target=target,
                    started_at=started_at,
                    finished_at=finished_at,
                    resources_touched=[str(item) for item in raw.get("resources_touched", []) if str(item).strip()]
                    if isinstance(raw.get("resources_touched"), list)
                    else [],
                    journal_id=journal_id,
                    rollback_available=bool(raw.get("rollback_available", spec.rollback_available)),
                    rollback_hint=str(raw.get("rollback_hint") or spec.rollback_hint),
                    error_code=str(raw.get("error_code") or "") or None,
                    user_message=str(raw.get("user_message") or ""),
                    details=dict(raw.get("details") if isinstance(raw.get("details"), dict) else {}),
                )
        except ExecutorPartialFailure as exc:
            return _result(
                ok=False,
                mutated=False,
                executor_id=spec.executor_id,
                error_code="executor_partial_failure",
                user_message=f"{target} did not finish. I recorded partial artifacts and did not verify a completed mutation.",
                resources_touched=exc.resources_touched,
                rollback_available=True,
                rollback_hint=exc.rollback_hint or spec.rollback_hint,
                details={"exception": exc.__class__.__name__, **exc.details},
            )
        except Exception as exc:  # noqa: BLE001 - executor boundary must return safe failure.
            return _result(
                ok=False,
                mutated=False,
                executor_id=spec.executor_id,
                error_code="executor_exception_before_verified_mutation",
                user_message=f"{target} did not finish. I did not verify any mutation.",
                rollback_available=spec.rollback_available,
                rollback_hint=spec.rollback_hint,
                details={"exception": exc.__class__.__name__},
            )
        result = ExecutorResult(
            **{
                **result.to_dict(),
                "journal_id": journal_id,
                "started_at": result.started_at or started_at,
                "finished_at": result.finished_at or utc_now_iso(),
            }
        )
        self.journal.append(
            {
                "journal_id": journal_id,
                "event": "executor_result",
                "plan": plan,
                "action": action,
                "result": result.to_dict(),
            }
        )
        return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(support_bundle_redact(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _bounded_journal_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated:max_depth]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                out["truncated_keys"] = True
                break
            out[str(key)] = _bounded_journal_value(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        out = [_bounded_journal_value(item, depth=depth + 1) for item in value[:40]]
        if len(value) > 40:
            out.append("[truncated:list]")
        return out
    if isinstance(value, tuple):
        return _bounded_journal_value(list(value), depth=depth)
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="replace")
        if len(encoded) > EXECUTOR_JOURNAL_MAX_STRING_BYTES:
            return encoded[:EXECUTOR_JOURNAL_MAX_STRING_BYTES].decode("utf-8", errors="replace") + "...[truncated]"
    return value


def _compact_journal_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    resources = result.get("resources_touched") if isinstance(result.get("resources_touched"), list) else []
    return redact_executor_value(
        {
            "journal_id": record.get("journal_id") or result.get("journal_id"),
            "event": record.get("event"),
            "compacted": True,
            "compaction_reason": "journal_record_size_cap",
            "plan_id": result.get("plan_id") or plan.get("plan_id") or action.get("pending_id"),
            "action_type": result.get("action_type") or plan.get("action_type"),
            "target": result.get("target") or plan.get("target"),
            "executor_id": result.get("executor_id"),
            "ok": result.get("ok"),
            "mutated": result.get("mutated"),
            "error_code": result.get("error_code"),
            "resources_touched_count": len(resources),
            "rollback_available": result.get("rollback_available"),
        }
    )


def _bounded_journal_record(record: dict[str, Any]) -> dict[str, Any]:
    bounded = _bounded_journal_value(record)
    encoded = json.dumps(bounded, sort_keys=True, ensure_ascii=True).encode("utf-8")
    if len(encoded) <= EXECUTOR_JOURNAL_MAX_RECORD_BYTES:
        return bounded if isinstance(bounded, dict) else {"record": bounded}
    return _compact_journal_record(record)


def _status_summary(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


def _bounded_backup_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[truncated:max_depth]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                out["truncated_keys"] = True
                break
            out[str(key)] = _bounded_backup_value(item, depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_bounded_backup_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, tuple):
        return [_bounded_backup_value(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        if len(value) > 512:
            return value[:512] + "...[truncated]"
    return value


def _write_backup_json(path: Path, payload: dict[str, Any]) -> int:
    text = json.dumps(support_bundle_redact(_bounded_backup_value(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > BACKUP_MAX_FILE_BYTES:
        raise ValueError(f"backup_file_size_cap_exceeded:{path.name}:{len(encoded)}")
    path.write_bytes(encoded)
    return len(encoded)


def _write_support_json(path: Path, payload: dict[str, Any]) -> int:
    text = json.dumps(support_bundle_redact(_bounded_backup_value(payload)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > SUPPORT_BUNDLE_MAX_FILE_BYTES:
        raise ValueError(f"support_bundle_file_size_cap_exceeded:{path.name}:{len(encoded)}")
    path.write_bytes(encoded)
    return len(encoded)


def _summarize_executor_journal_entries(entries: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for entry in entries[-BACKUP_MAX_JOURNAL_ENTRIES:]:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else {}
        resources = result.get("resources_touched") if isinstance(result.get("resources_touched"), list) else []
        summaries.append(
            support_bundle_redact(
                {
                    "journal_id": entry.get("journal_id") or result.get("journal_id"),
                    "event": entry.get("event"),
                    "plan_id": result.get("plan_id") or plan.get("plan_id"),
                    "action_type": result.get("action_type") or plan.get("action_type"),
                    "target": result.get("target") or plan.get("target"),
                    "executor_id": result.get("executor_id"),
                    "ok": result.get("ok"),
                    "mutated": result.get("mutated"),
                    "error_code": result.get("error_code"),
                    "resources_touched_count": len(resources),
                    "rollback_available": result.get("rollback_available"),
                    "started_at": result.get("started_at"),
                    "finished_at": result.get("finished_at"),
                }
            )
        )
    return summaries


def build_support_bundle_manifest(
    *,
    root: Path,
    diagnostics: dict[str, Any],
    included_files: list[str],
) -> dict[str, Any]:
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    return support_bundle_redact(
        {
            "bundle_schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "runtime_commit": version.get("git_commit"),
            "checkout_commit": diagnostics.get("checkout_commit"),
            "runtime_instance": version.get("runtime_instance"),
            "included_files": included_files,
            "bundle_path": str(root),
            "redaction_policy": (
                "Token, API key, password, bearer, secret, confirmation-token, raw secret-file, "
                "raw log, and broad private-path values are redacted or summarized."
            ),
        }
    )


def _approved_backup_root(action: dict[str, Any]) -> Path:
    raw = str(action.get("backup_root") or "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    else:
        root = (Path.home() / ".local/share/personal-agent/backups").resolve()
    home = Path.home().resolve()
    state_root = (home / ".local/share/personal-agent").resolve()
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if root != state_root and state_root not in root.parents and root != tmp_root and tmp_root not in root.parents:
        raise ValueError("backup_root_not_approved")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_dir(root: Path, *, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def _tree_size_and_count(path: Path, *, max_entries: int = CLEANUP_MAX_SCAN_ENTRIES) -> tuple[int, int, bool]:
    total = 0
    count = 0
    truncated = False
    try:
        if path.is_file() or path.is_symlink():
            return int(path.lstat().st_size), 1, False
        for child in path.rglob("*"):
            count += 1
            if count > max_entries:
                truncated = True
                break
            try:
                stat = child.lstat()
            except OSError:
                continue
            total += int(stat.st_size)
    except OSError:
        return 0, count, True
    return total, count, truncated


def _cleanup_allowed_roots() -> dict[str, Path]:
    state_root = (Path.home() / ".local/share/personal-agent").resolve()
    return {
        "backup": (state_root / "backups").resolve(),
        "runtime_release": (state_root / "runtime/releases").resolve(),
        "support_tmp": Path(tempfile.gettempdir()).resolve(),
    }


def _is_contained(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def _cleanup_candidate_kind(classification: str) -> str | None:
    normalized = str(classification or "").strip().lower()
    if normalized in {"oversized backup artifact", "old backup artifact"}:
        return "backup"
    if normalized == "old support bundle artifact":
        return "support_tmp"
    if normalized == "old runtime release":
        return "runtime_release"
    return None


def _cleanup_path_allowed(path: Path, *, kind: str) -> tuple[bool, str]:
    roots = _cleanup_allowed_roots()
    root = roots.get(kind)
    if root is None:
        return False, "cleanup_candidate_kind_not_allowed"
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False, "cleanup_path_unresolvable"
    if not _is_contained(resolved, root):
        return False, "cleanup_path_outside_owned_root"
    if kind == "backup" and not resolved.name.startswith("personal-agent-backup-"):
        return False, "cleanup_backup_name_not_owned"
    if kind == "support_tmp" and not (
        resolved.name.startswith("personal-agent-support-") or resolved.name.startswith("agent-support-")
    ):
        return False, "cleanup_support_name_not_owned"
    if kind == "runtime_release" and resolved == root:
        return False, "cleanup_runtime_release_root_protected"
    return True, ""


def _cleanup_has_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if path.is_dir():
            for index, child in enumerate(path.rglob("*")):
                if index > CLEANUP_MAX_SCAN_ENTRIES:
                    return True
                if child.is_symlink():
                    return True
    except OSError:
        return True
    return False


def execute_cleanup(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    preview = action.get("cleanup_preview") if isinstance(action.get("cleanup_preview"), dict) else {}
    candidates = preview.get("candidates") if isinstance(preview.get("candidates"), list) else []
    protected = preview.get("protected") if isinstance(preview.get("protected"), list) else []
    protected_paths = {
        str(item.get("canonical_path") or "").strip()
        for item in protected
        if isinstance(item, dict) and str(item.get("canonical_path") or "").strip()
    }
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    bytes_recovered = 0

    for raw in candidates[:CLEANUP_MAX_CANDIDATES]:
        if not isinstance(raw, dict):
            skipped.append({"reason": "cleanup_candidate_malformed"})
            continue
        label = str(raw.get("path") or raw.get("canonical_path") or "").strip()
        canonical = str(raw.get("canonical_path") or "").strip()
        classification = str(raw.get("classification") or "").strip()
        if not canonical:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_canonical_path_missing"})
            continue
        if not bool(raw.get("safe_to_delete_later")):
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_candidate_not_marked_safe"})
            continue
        if canonical in protected_paths:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_candidate_became_protected"})
            continue
        kind = _cleanup_candidate_kind(classification)
        if kind is None:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_classification_not_deletable"})
            continue
        path = Path(canonical).expanduser()
        allowed, reason = _cleanup_path_allowed(path, kind=kind)
        if not allowed:
            skipped.append({"path": label, "classification": classification, "reason": reason})
            continue
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_candidate_missing"})
            continue
        if os.path.ismount(str(resolved)):
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_mount_point_protected"})
            continue
        if _cleanup_has_symlink(resolved):
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_symlink_protected"})
            continue
        current_size, current_count, truncated = _tree_size_and_count(resolved)
        preview_size = raw.get("size_bytes")
        preview_count = raw.get("file_count")
        if isinstance(preview_size, int) and current_size != preview_size:
            skipped.append(
                {
                    "path": label,
                    "classification": classification,
                    "reason": "cleanup_candidate_changed_after_preview",
                    "preview_size_bytes": preview_size,
                    "current_size_bytes": current_size,
                }
            )
            continue
        if isinstance(preview_count, int) and current_count != preview_count:
            skipped.append(
                {
                    "path": label,
                    "classification": classification,
                    "reason": "cleanup_candidate_changed_after_preview",
                    "preview_file_count": preview_count,
                    "current_file_count": current_count,
                }
            )
            continue
        if truncated:
            skipped.append({"path": label, "classification": classification, "reason": "cleanup_scan_truncated"})
            continue
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
        except Exception as exc:  # noqa: BLE001 - independent cleanup failures are reported, not thrown.
            failures.append({"path": label, "classification": classification, "reason": exc.__class__.__name__})
            continue
        bytes_recovered += current_size
        deleted.append({"path": label, "classification": classification, "size_bytes": current_size})

    ok = not failures
    mutated = bool(deleted)
    if deleted and failures:
        status = "partial_failure"
        message = (
            f"Cleanup partially finished. Removed {len(deleted)} Personal Agent artifact(s) and recovered "
            f"{bytes_recovered} bytes, but {len(failures)} candidate(s) failed."
        )
        error_code = "cleanup_partial_failure"
    elif deleted:
        status = "completed"
        message = f"Cleanup finished. Removed {len(deleted)} old Personal Agent artifact(s) and recovered {bytes_recovered} bytes."
        error_code = None
    elif failures:
        status = "failed"
        message = "Cleanup did not remove anything because every attempted candidate failed."
        error_code = "cleanup_failed"
    else:
        status = "no_op"
        message = "Cleanup found no eligible candidates to delete after revalidation. I did not remove anything."
        error_code = None
    return {
        "ok": ok,
        "mutated": mutated,
        "executor_id": "operator.cleanup.v1",
        "resources_touched": [str(item.get("path") or "") for item in deleted if str(item.get("path") or "").strip()],
        "rollback_available": False,
        "rollback_hint": (
            "Cleanup deletion is not automatically reversible. The latest valid backup, current runtime, secret store, "
            "and active service files were protected by policy."
        ),
        "error_code": error_code,
        "user_message": message,
        "details": {
            "status": status,
            "deleted": deleted,
            "skipped": skipped[:50],
            "protected_count": len(protected),
            "failures": failures[:50],
            "bytes_recovered": bytes_recovered,
        },
    }


def build_backup_manifest(
    *,
    root: Path,
    diagnostics: dict[str, Any],
    included_files: list[str],
    excluded_files: list[str],
    file_sizes: dict[str, int] | None = None,
    total_size_bytes: int | None = None,
) -> dict[str, Any]:
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    policy = (
        "Backup v1 stores redacted JSON summaries only. Raw secret-store files, "
        "tokens, API keys, passwords, raw logs, model caches, arbitrary home data, "
        "and unreviewed pack/source contents are excluded. No encryption is applied "
        "because raw secret material is not included; treat the backup as local-sensitive."
    )
    return support_bundle_redact(
        {
            "backup_schema_version": BACKUP_SCHEMA_VERSION,
            "created_at": utc_now_iso(),
            "runtime_commit": version.get("git_commit"),
            "runtime_instance": version.get("runtime_instance"),
            "included_files": included_files,
            "excluded_files": excluded_files,
            "file_sizes": file_sizes or {},
            "total_size_bytes": total_size_bytes,
            "size_caps": {
                "max_total_bytes": BACKUP_MAX_TOTAL_BYTES,
                "max_file_bytes": BACKUP_MAX_FILE_BYTES,
                "max_journal_entries": BACKUP_MAX_JOURNAL_ENTRIES,
            },
            "redaction/encryption policy": policy,
            "redaction_encryption_policy": policy,
            "restore_status": "dry_run_only",
            "live_restore": "restore_not_enabled",
            "backup_path": str(root),
        }
    )


def create_additive_backup(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    plan_id = str(plan.get("plan_id") or action.get("pending_id") or "unknown").strip() or "unknown"
    backup_root = _approved_backup_root(action)
    root = _artifact_dir(backup_root, prefix="personal-agent-backup")
    root.mkdir(mode=0o700, parents=False, exist_ok=False)

    diagnostics = action.get("diagnostics") if isinstance(action.get("diagnostics"), dict) else {}
    executor_recent = action.get("executor_journal_recent") if isinstance(action.get("executor_journal_recent"), list) else []
    backup_sources = action.get("backup_sources") if isinstance(action.get("backup_sources"), dict) else {}
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    ready = diagnostics.get("ready") if isinstance(diagnostics.get("ready"), dict) else {}
    state = diagnostics.get("state") if isinstance(diagnostics.get("state"), dict) else {}
    search = diagnostics.get("search_status") if isinstance(diagnostics.get("search_status"), dict) else {}
    telegram = diagnostics.get("telegram_status") if isinstance(diagnostics.get("telegram_status"), dict) else {}
    packs = diagnostics.get("packs_state") if isinstance(diagnostics.get("packs_state"), dict) else {}
    doctor = diagnostics.get("doctor") if isinstance(diagnostics.get("doctor"), dict) else {}
    git = diagnostics.get("git") if isinstance(diagnostics.get("git"), dict) else {}

    excluded_files = [
        "raw secret-store files",
        "raw credentials such as Telegram/provider/API authentication values and passwords",
        "raw logs and full support bundles",
        "arbitrary home directory files",
        "model caches/downloads and GGUF/model artifacts",
        "raw external pack archives, SKILL.md, AGENTS.md, and untrusted source text",
        "browser caches, downloaded pages, and search result page contents",
        "live restore output; restore is dry-run/preview-only in Backup v1",
    ]
    files: dict[str, dict[str, Any]] = {}
    files["state_database_summary.json"] = {
        "source": "runtime state database summary",
        "mode": "summary_only_raw_database_excluded",
        "state_database": backup_sources.get("state_database"),
    }
    files["preferences_summary.json"] = {
        "source": "preferences summary",
        "mode": "summary_only",
        "preferences": backup_sources.get("preferences", {"status": "not_exported_in_backup_v1"}),
    }
    files["memory_anchors_summary.json"] = {
        "source": "memory/anchors summary",
        "mode": "summary_only_raw_memory_text_excluded",
        "memory": backup_sources.get("memory", {"status": "summary_only"}),
    }
    pack_counts = packs.get("counts") if isinstance(packs.get("counts"), dict) else {}
    files["pack_metadata_summary.json"] = {
        "source": "pack metadata summary",
        "ok": packs.get("ok", True),
        "counts": pack_counts,
        "state": packs.get("state"),
        "warnings": packs.get("warnings") if isinstance(packs.get("warnings"), list) else [],
        "raw_pack_text": "excluded",
    }
    files["runtime_config_summary.json"] = {
        "version": version,
        "ready": _status_summary(
            ready,
            ("ready", "phase", "startup_phase", "runtime_mode", "failure_code", "next_action", "state_label", "reason", "blocker", "next_step"),
        ),
        "state": _status_summary(state, ("ok", "ready", "runtime_mode", "state_label", "reason", "next_action", "search", "telegram", "packs", "memory")),
        "search_status": _status_summary(
            search,
            ("ok", "enabled", "provider", "endpoint_configured", "available", "reason", "next_action", "search_state", "base_url", "managed_service"),
        ),
        "telegram_status": _status_summary(
            telegram,
            (
                "ok",
                "enabled",
                "configured",
                "token_source",
                "state",
                "effective_state",
                "service_installed",
                "service_active",
                "service_enabled",
                "lock_present",
                "lock_live",
                "lock_stale",
                "next_action",
            ),
        ),
    }
    journal_entries = _summarize_executor_journal_entries(executor_recent)
    files["executor_registry_journal_summary.json"] = {
        "entries": journal_entries,
        "entry_count": len(executor_recent),
        "included_entry_count": len(journal_entries),
        "source": "executor_registry_journal_recent_summary_only",
    }
    files["diagnostics_summary.json"] = {
        "doctor": doctor or _status_summary(ready, ("ready", "runtime_mode", "state_label", "reason", "next_action")),
        "git_runtime_freshness": {
            "runtime_commit": version.get("git_commit"),
            "checkout_commit": diagnostics.get("checkout_commit"),
            "runtime_instance": version.get("runtime_instance"),
            "git": git,
        },
        "readiness_proof": diagnostics.get("readiness_proof"),
        "docs_truth": diagnostics.get("docs_truth"),
    }
    files["support_bundle_style_summary.json"] = {
        "source": "support-bundle-style redacted summaries",
        "included": sorted(files.keys()),
        "redaction": "same redaction helper as Support Bundle v2",
    }
    files["backup_summary.json"] = {
        "created_at": utc_now_iso(),
        "plan_id": plan_id,
        "action_type": str(plan.get("action_type") or "operator.backup"),
        "target": str(plan.get("target") or "backup assistant"),
        "backup_schema_version": BACKUP_SCHEMA_VERSION,
        "restore_status": "dry_run_only",
        "live_restore": "restore_not_enabled",
        "contents": sorted(files.keys()),
        "size_caps": {
            "max_total_bytes": BACKUP_MAX_TOTAL_BYTES,
            "max_file_bytes": BACKUP_MAX_FILE_BYTES,
            "max_journal_entries": BACKUP_MAX_JOURNAL_ENTRIES,
        },
    }

    included_files = sorted([*files.keys(), "manifest.json"])
    file_sizes: dict[str, int] = {}
    try:
        total_size = 0
        for name, payload in files.items():
            size = _write_backup_json(root / name, payload)
            file_sizes[name] = size
            total_size += size
            if total_size > BACKUP_MAX_TOTAL_BYTES:
                raise ValueError(f"backup_total_size_cap_exceeded:{total_size}")
        manifest = build_backup_manifest(
            root=root,
            diagnostics=diagnostics,
            included_files=included_files,
            excluded_files=excluded_files,
            file_sizes=file_sizes,
            total_size_bytes=total_size,
        )
        manifest_text = json.dumps(support_bundle_redact(_bounded_backup_value(manifest)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        manifest_size = len(manifest_text.encode("utf-8"))
        if manifest_size > BACKUP_MAX_FILE_BYTES:
            raise ValueError(f"backup_file_size_cap_exceeded:manifest.json:{manifest_size}")
        if total_size + manifest_size > BACKUP_MAX_TOTAL_BYTES:
            raise ValueError(f"backup_total_size_cap_exceeded:{total_size + manifest_size}")
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        file_sizes["manifest.json"] = manifest_size
        total_size += manifest_size
    except Exception as exc:
        resources = [str(path) for path in sorted(root.glob("*.json"))]
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.backup.v1",
            "resources_touched": resources,
            "rollback_available": True,
            "rollback_hint": f"Remove only the partial backup directory created for this failed action: {root}",
            "error_code": "backup_v1_failed_before_final_manifest",
            "user_message": "Backup v1 did not finish. I did not write a final manifest or verify a usable backup.",
            "details": {"artifact_path": str(root), "partial": True, "error": exc.__class__.__name__},
        }
    resources = [str(root / name) for name in included_files]
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.backup.v1",
        "resources_touched": resources,
        "rollback_available": True,
        "rollback_hint": f"Remove only the newly created backup directory: {root}",
        "user_message": f"Backup v1 created at {root}. It contains redacted summaries only; live restore is not enabled.",
        "details": {"artifact_path": str(root), "files": included_files, "manifest_path": str(root / "manifest.json")},
    }


def create_redacted_support_bundle(plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    plan_id = str(plan.get("plan_id") or action.get("pending_id") or "unknown").strip() or "unknown"
    root = Path(tempfile.mkdtemp(prefix="personal-agent-support-"))
    diagnostics = action.get("diagnostics") if isinstance(action.get("diagnostics"), dict) else {}
    executor_recent = action.get("executor_journal_recent") if isinstance(action.get("executor_journal_recent"), list) else []
    files: dict[str, dict[str, Any]] = {}
    version = diagnostics.get("version") if isinstance(diagnostics.get("version"), dict) else {}
    ready = diagnostics.get("ready") if isinstance(diagnostics.get("ready"), dict) else {}
    state = diagnostics.get("state") if isinstance(diagnostics.get("state"), dict) else {}
    search = diagnostics.get("search_status") if isinstance(diagnostics.get("search_status"), dict) else {}
    telegram = diagnostics.get("telegram_status") if isinstance(diagnostics.get("telegram_status"), dict) else {}
    packs = diagnostics.get("packs_state") if isinstance(diagnostics.get("packs_state"), dict) else {}
    doctor = diagnostics.get("doctor") if isinstance(diagnostics.get("doctor"), dict) else {}
    proof = diagnostics.get("readiness_proof") if isinstance(diagnostics.get("readiness_proof"), dict) else {}
    docs_truth = diagnostics.get("docs_truth") if isinstance(diagnostics.get("docs_truth"), dict) else {}
    git = diagnostics.get("git") if isinstance(diagnostics.get("git"), dict) else {}

    files["doctor_summary.json"] = {
        "source": "doctor/runtime summary",
        "summary": doctor or _status_summary(ready, ("ready", "runtime_mode", "state_label", "reason", "next_action")),
    }
    files["version.json"] = version
    files["ready.json"] = _status_summary(
        ready,
        ("ready", "phase", "startup_phase", "runtime_mode", "failure_code", "next_action", "state_label", "reason", "blocker", "next_step", "message"),
    )
    files["state_summary.json"] = _status_summary(
        state,
        ("ok", "ready", "runtime_mode", "state_label", "reason", "next_action", "search", "telegram", "packs", "memory"),
    )
    files["search_status.json"] = _status_summary(
        search,
        ("ok", "enabled", "provider", "endpoint_configured", "available", "reason", "next_action", "search_state", "base_url", "managed_service"),
    )
    files["telegram_status.json"] = _status_summary(
        telegram,
        (
            "ok",
            "enabled",
            "configured",
            "token_source",
            "state",
            "effective_state",
            "service_installed",
            "service_active",
            "service_enabled",
            "lock_present",
            "lock_live",
            "lock_stale",
            "next_action",
        ),
    )
    pack_counts = packs.get("counts") if isinstance(packs.get("counts"), dict) else {}
    files["packs_state_summary.json"] = {
        "ok": packs.get("ok", True),
        "counts": pack_counts,
        "state": packs.get("state"),
        "warnings": packs.get("warnings") if isinstance(packs.get("warnings"), list) else [],
    }
    files["executor_registry_journal_summary.json"] = {
        "entries": _summarize_executor_journal_entries(executor_recent),
        "entry_count": len(executor_recent),
        "included_entry_count": min(len(executor_recent), BACKUP_MAX_JOURNAL_ENTRIES),
        "source": "executor_registry_journal_recent_summary_only",
    }
    files["readiness_proof_summary.json"] = {
        "prove_ready": proof,
        "docs_truth": docs_truth,
    }
    files["git_runtime_freshness.json"] = {
        "runtime_commit": version.get("git_commit"),
        "checkout_commit": diagnostics.get("checkout_commit"),
        "runtime_instance": version.get("runtime_instance"),
        "git": git,
    }
    files["support_summary.json"] = {
        "created_at": utc_now_iso(),
        "plan_id": plan_id,
        "action_type": str(plan.get("action_type") or "operator.support_bundle"),
        "target": str(plan.get("target") or "support bundle"),
        "bundle_schema_version": SUPPORT_BUNDLE_SCHEMA_VERSION,
        "redaction": "secrets, tokens, API keys, raw private values, raw logs, and broad private paths are redacted or summarized",
        "contents": sorted(files.keys()),
    }

    included_files = sorted([*files.keys(), "manifest.json"])
    file_sizes: dict[str, int] = {}
    try:
        total_size = 0
        for name, payload in files.items():
            size = _write_support_json(root / name, payload)
            file_sizes[name] = size
            total_size += size
            if total_size > SUPPORT_BUNDLE_MAX_TOTAL_BYTES:
                raise ValueError(f"support_bundle_total_size_cap_exceeded:{total_size}")
        manifest = build_support_bundle_manifest(root=root, diagnostics=diagnostics, included_files=included_files)
        manifest["file_sizes"] = file_sizes
        manifest["total_size_bytes"] = total_size
        manifest["size_caps"] = {
            "max_total_bytes": SUPPORT_BUNDLE_MAX_TOTAL_BYTES,
            "max_file_bytes": SUPPORT_BUNDLE_MAX_FILE_BYTES,
            "max_journal_entries": BACKUP_MAX_JOURNAL_ENTRIES,
        }
        manifest_text = json.dumps(support_bundle_redact(_bounded_backup_value(manifest)), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        manifest_size = len(manifest_text.encode("utf-8"))
        if manifest_size > SUPPORT_BUNDLE_MAX_FILE_BYTES:
            raise ValueError(f"support_bundle_file_size_cap_exceeded:manifest.json:{manifest_size}")
        if total_size + manifest_size > SUPPORT_BUNDLE_MAX_TOTAL_BYTES:
            raise ValueError(f"support_bundle_total_size_cap_exceeded:{total_size + manifest_size}")
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        file_sizes["manifest.json"] = manifest_size
        total_size += manifest_size
    except Exception as exc:
        resources = [str(path) for path in sorted(root.glob("*.json"))]
        return {
            "ok": False,
            "mutated": False,
            "executor_id": "operator.support_bundle.v1",
            "resources_touched": resources,
            "rollback_available": True,
            "rollback_hint": f"Remove only the partial support bundle directory created for this failed action: {root}",
            "error_code": "support_bundle_v2_failed_before_final_manifest",
            "user_message": "Support bundle creation did not finish. I did not write a final manifest or verify a usable bundle.",
            "details": {"artifact_path": str(root), "partial": True, "error": exc.__class__.__name__},
        }
    resources = [str(root / name) for name in included_files]
    return {
        "ok": True,
        "mutated": True,
        "executor_id": "operator.support_bundle.v1",
        "resources_touched": resources,
        "rollback_available": True,
        "rollback_hint": f"Remove only the newly created support bundle directory: {root}",
        "user_message": f"Support bundle created at {root}. It contains redacted diagnostics only.",
        "details": {"artifact_path": str(root), "files": included_files, "manifest_path": str(root / "manifest.json")},
    }
