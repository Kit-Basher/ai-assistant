from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable
import uuid


SUPPORT_BUNDLE_SCHEMA_VERSION = "support_bundle.v2"

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
        payload = redact_executor_value({**record, "journal_id": journal_id})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        return journal_id

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-max(0, int(limit)) :]:
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


def _status_summary(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if key in payload}


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
        "entries": executor_recent[:20],
        "entry_count": len(executor_recent),
        "source": "executor_registry_journal_recent",
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
    manifest = build_support_bundle_manifest(root=root, diagnostics=diagnostics, included_files=included_files)
    _write_json(root / "manifest.json", manifest)
    for name, payload in files.items():
        _write_json(root / name, payload)
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
