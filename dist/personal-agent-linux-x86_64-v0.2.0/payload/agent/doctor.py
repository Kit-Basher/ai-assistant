from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from memory.db import MemoryDB
from agent.cards import validate_cards_payload
from agent.nl_router import build_cards_payload
from skills.observe_now.handler import observe_now
from agent.audit_log import redact
from agent.diagnostics import CommandResult, redact_secrets, run_command
from agent.golden_path import next_step_for_failure
from agent.logging_bootstrap import configure_logging_if_needed
from agent.secret_store import SecretStore
from agent.startup_checks import run_startup_checks
from agent.skills.system_health import collect_system_health
from agent.telegram_runtime_state import get_telegram_runtime_state, telegram_control_env
from agent.config import (
    canonical_config_dir,
    canonical_log_path,
    canonical_state_dir,
    load_config,
    resolved_default_db_path,
    runtime_port,
    runtime_instance,
    runtime_root_path,
    runtime_service_name,
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


# Legacy checks retained for backwards compatibility with existing tests.
def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_systemd_units() -> bool:
    return _env_truthy("AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS", default=False)


def _missing_unit_result(check_name: str, unit_name: str, *, strict: bool) -> CheckResult:
    if strict:
        return CheckResult(check_name, False, f"{unit_name} missing")
    hint = "set AGENT_DOCTOR_REQUIRE_SYSTEMD_UNITS=1 to enforce"
    return CheckResult(check_name, True, f"{unit_name} missing (skipped; {hint})")


def _check_llm_router() -> CheckResult:
    try:
        from agent.config import load_config
        from agent.llm.router import LLMRouter

        config = load_config(require_telegram_token=False)
        router = LLMRouter(config)
        snapshot = router.doctor_snapshot()
    except Exception as exc:
        return CheckResult("llm", False, f"llm diagnostics unavailable ({exc})")

    providers = snapshot.get("providers") or []
    models = snapshot.get("models") or []
    env = snapshot.get("env") or {}
    circuits = snapshot.get("circuits") or {}

    provider_names = ",".join(sorted(item.get("name", "") for item in providers if item.get("name"))) or "none"
    model_ids = ",".join(sorted(item.get("id", "") for item in models if item.get("id"))) or "none"
    env_present = ",".join(sorted(env.get("present") or [])) or "none"
    env_missing = ",".join(sorted(env.get("missing") or [])) or "none"
    open_circuits = sorted(model_id for model_id, state in circuits.items() if state.get("open"))
    circuit_state = "open=" + ",".join(open_circuits) if open_circuits else "open=none"

    message = (
        f"providers={provider_names}; models={model_ids}; "
        f"env_present={env_present}; env_missing={env_missing}; circuits={circuit_state}"
    )
    return CheckResult("llm", True, message)


def expected_schema_from_version(version_text: str) -> int | None:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)\s*$", version_text or "")
    if not match:
        return None
    return int(match.group(2))


def _check_db(db_path: str) -> CheckResult:
    if not db_path or not os.path.isfile(db_path):
        return CheckResult("db", False, "db path missing")
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        conn.close()
        if not row:
            return CheckResult("db", False, "schema_version missing")
        int(row["value"])
        return CheckResult("db", True, "schema_version readable")
    except Exception:
        return CheckResult("db", False, "schema_version unreadable")


def _systemd_available(run: Callable[..., subprocess.CompletedProcess[str]]) -> bool:
    try:
        proc = run(["systemctl", "--version"], check=False, capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def _unit_exists(unit_name: str, run: Callable[..., subprocess.CompletedProcess[str]]) -> bool:
    try:
        proc = run(["systemctl", "cat", unit_name], check=False, capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def _unit_state(unit_name: str, run: Callable[..., subprocess.CompletedProcess[str]]) -> tuple[str, str]:
    enabled = "unknown"
    active = "unknown"
    try:
        proc_enabled = run(
            ["systemctl", "is-enabled", unit_name],
            check=False,
            capture_output=True,
            text=True,
        )
        enabled = (proc_enabled.stdout or "").strip() or "unknown"
    except Exception:
        enabled = "unknown"
    try:
        proc_active = run(
            ["systemctl", "is-active", unit_name],
            check=False,
            capture_output=True,
            text=True,
        )
        active = (proc_active.stdout or "").strip() or "unknown"
    except Exception:
        active = "unknown"
    return enabled, active


def _systemd_show_value(
    unit_name: str,
    prop: str,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    try:
        proc = run(
            ["systemctl", "show", unit_name, f"--property={prop}", "--value"],
            check=False,
            capture_output=True,
            text=True,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def _has_recent_observe_audit_success(db_path: str, window_hours: int = 24) -> bool:
    if not db_path or not os.path.isfile(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT created_at
            FROM audit_log
            WHERE action_type = 'observe_now_scheduled' AND status = 'executed'
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return False
        created = datetime.fromisoformat(str(row["created_at"]))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(window_hours))
        return created >= cutoff
    except Exception:
        return False


def _check_systemd_units(run: Callable[..., subprocess.CompletedProcess[str]], db_path: str) -> CheckResult:
    svc = "personal-agent-observe.service"
    timer = "personal-agent-observe.timer"
    strict = _require_systemd_units()
    if not _systemd_available(run):
        return CheckResult("systemd", True, "systemd unavailable (skipped)")
    if not _unit_exists(svc, run):
        return _missing_unit_result("systemd", svc, strict=strict)
    if not _unit_exists(timer, run):
        return _missing_unit_result("systemd", timer, strict=strict)
    svc_enabled, svc_active = _unit_state(svc, run)
    timer_enabled, timer_active = _unit_state(timer, run)
    if timer_enabled != "enabled" or timer_active != "active":
        return CheckResult(
            "systemd",
            False,
            f"timer enabled={timer_enabled} active={timer_active}",
        )
    if svc_active == "failed":
        return CheckResult("systemd", False, f"service active={svc_active}")
    exec_main_status = _systemd_show_value(svc, "ExecMainStatus", run)
    proof_ok = exec_main_status == "0" or _has_recent_observe_audit_success(db_path, window_hours=24)
    if not proof_ok:
        return CheckResult(
            "systemd",
            False,
            f"timer enabled={timer_enabled} active={timer_active}; service active={svc_active}; proof missing",
        )
    msg = (
        f"service active={svc_active}; timer enabled={timer_enabled} active={timer_active}; "
        f"proof={'execmainstatus=0' if exec_main_status == '0' else 'recent_audit'}"
    )
    return CheckResult("systemd", True, msg)


def _check_daily_brief_timer(run: Callable[..., subprocess.CompletedProcess[str]]) -> CheckResult:
    svc = "personal-agent-daily-brief.service"
    timer = "personal-agent-daily-brief.timer"
    strict = _require_systemd_units()
    if not _systemd_available(run):
        return CheckResult("daily_brief_timer", True, "systemd unavailable (skipped)")
    if not _unit_exists(svc, run):
        return _missing_unit_result("daily_brief_timer", svc, strict=strict)
    if not _unit_exists(timer, run):
        return _missing_unit_result("daily_brief_timer", timer, strict=strict)
    _svc_enabled, svc_active = _unit_state(svc, run)
    timer_enabled, timer_active = _unit_state(timer, run)
    if timer_enabled != "enabled" or timer_active != "active":
        return CheckResult(
            "daily_brief_timer",
            False,
            f"timer enabled={timer_enabled} active={timer_active}",
        )
    if svc_active == "failed":
        return CheckResult("daily_brief_timer", False, f"service active={svc_active}")
    exec_main_status = _systemd_show_value(svc, "ExecMainStatus", run) or "unknown"
    msg = (
        f"service active={svc_active}; timer enabled={timer_enabled} "
        f"active={timer_active}; last_exit={exec_main_status}"
    )
    return CheckResult("daily_brief_timer", True, msg)


def _check_version(version_path: str, schema_version: int | None) -> CheckResult:
    if not os.path.isfile(version_path):
        return CheckResult("version", False, "VERSION missing")
    try:
        text = Path(version_path).read_text(encoding="utf-8").strip()
    except Exception:
        return CheckResult("version", False, "VERSION unreadable")
    expected = expected_schema_from_version(text)
    if expected is None:
        return CheckResult("version", False, "VERSION format invalid")
    if schema_version is None:
        return CheckResult("version", False, "schema unavailable")
    if expected != schema_version:
        return CheckResult("version", False, f"schema mismatch version={text} schema={schema_version}")
    return CheckResult("version", True, f"version={text} schema={schema_version}")


def _read_schema_version(db_path: str) -> int | None:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return int(row["value"])
    except Exception:
        return None


def _check_observe_now_dry_run(repo_root: str) -> CheckResult:
    schema_path = os.path.join(repo_root, "memory", "schema.sql")
    if not os.path.isfile(schema_path):
        return CheckResult("observe_now", False, "schema.sql missing")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "doctor.db")
            db = MemoryDB(db_path)
            db.init_schema(schema_path)
            observe_now({"db": db, "timezone": "UTC", "user_id": "doctor"}, user_id="doctor")
            db.close()
        return CheckResult("observe_now", True, "dry-run execution ok")
    except Exception:
        return CheckResult("observe_now", False, "dry-run execution failed")


def run_doctor(repo_root: str, db_path: str, version_path: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(_check_db(db_path))
    results.append(_check_llm_router())
    schema_version = _read_schema_version(db_path)
    results.append(_check_systemd_units(subprocess.run, db_path))
    results.append(_check_daily_brief_timer(subprocess.run))
    results.append(_check_version(version_path, schema_version))
    results.append(_check_observe_now_dry_run(repo_root))
    return results


def default_db_path(repo_root: str) -> str:
    env_path = os.getenv("AGENT_DB_PATH", "").strip()
    if env_path:
        return env_path
    _ = repo_root
    return resolved_default_db_path()


# New deterministic doctor report engine.
_DOCTOR_STATUS_ORDER = {"OK": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class DoctorCheck:
    check_id: str
    status: str
    detail_short: str
    detail_long: str | None = None
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "detail_short": self.detail_short,
            "detail_long": self.detail_long,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class DoctorReport:
    trace_id: str
    generated_at: str
    summary_status: str
    checks: list[DoctorCheck]
    next_action: str | None
    fixes_applied: list[str]
    support_bundle_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "generated_at": self.generated_at,
            "summary_status": self.summary_status,
            "checks": [item.to_dict() for item in self.checks],
            "next_action": self.next_action,
            "fixes_applied": list(self.fixes_applied),
            "support_bundle_path": self.support_bundle_path,
        }


def _doctor_logger() -> logging.Logger:
    configure_logging_if_needed()
    return logging.getLogger("agent.doctor")


def _doctor_trace_id(now_epoch: int | None = None) -> str:
    epoch = int(now_epoch if now_epoch is not None else time.time())
    return f"doctor-{epoch}-{os.getpid()}"


def _status_from_checks(checks: list[DoctorCheck]) -> str:
    worst = "OK"
    for item in checks:
        if _DOCTOR_STATUS_ORDER.get(item.status, 0) > _DOCTOR_STATUS_ORDER.get(worst, 0):
            worst = item.status
    return worst


def _first_next_action(checks: list[DoctorCheck]) -> str | None:
    for wanted in ("FAIL", "WARN"):
        for item in checks:
            if item.status == wanted and item.next_action:
                return item.next_action
    return None


def _log_check(trace_id: str, check: DoctorCheck) -> None:
    logger = _doctor_logger()
    payload = {
        "trace_id": trace_id,
        "check_id": check.check_id,
        "status": check.status,
        "detail_short": check.detail_short,
    }
    if check.detail_long:
        payload["detail_long"] = check.detail_long
    if check.next_action:
        payload["next_action"] = check.next_action
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _safe_now_iso(now_epoch: int | None = None) -> str:
    return datetime.fromtimestamp(int(now_epoch if now_epoch is not None else time.time()), tz=timezone.utc).isoformat()


def _redact_bundle_value(value: Any) -> Any:
    redacted = redact(value)
    if isinstance(redacted, dict):
        return {str(key): _redact_bundle_value(item) for key, item in redacted.items()}
    if isinstance(redacted, list):
        return [_redact_bundle_value(item) for item in redacted]
    if isinstance(redacted, str):
        return redact_secrets(redacted)
    return redacted


def _effective_secret_store_path() -> str:
    configured = os.getenv("AGENT_SECRET_STORE_PATH", "").strip()
    if configured:
        return configured
    return str(canonical_state_dir() / "secrets.enc.json")


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _telegram_enabled_for_doctor() -> bool:
    try:
        return bool(get_telegram_runtime_state(env=telegram_control_env()).get("enabled", False))
    except Exception:
        return False


def _telegram_optional_check(check_id: str) -> DoctorCheck:
    return DoctorCheck(
        check_id=check_id,
        status="OK",
        detail_short="telegram adapter disabled (optional)",
    )


def _token_preview(token: str | None) -> str:
    value = str(token or "").strip()
    if not value:
        return "missing"
    if len(value) <= 8:
        return "***redacted***"
    return f"{value[:4]}...{value[-4:]}"


def _check_python_runtime() -> DoctorCheck:
    detail = f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} exec={sys.executable}"
    return DoctorCheck(check_id="env.python", status="OK", detail_short=detail)


def _check_repo_readable(repo_root: Path) -> DoctorCheck:
    ok = repo_root.is_dir() and os.access(repo_root, os.R_OK)
    if ok:
        return DoctorCheck(check_id="env.repo", status="OK", detail_short=f"repo={repo_root}")
    return DoctorCheck(
        check_id="env.repo",
        status="FAIL",
        detail_short=f"repo not readable: {repo_root}",
        next_action="Ensure the repository path exists and is readable.",
    )


def _check_secret_store_path() -> DoctorCheck:
    path = Path(_effective_secret_store_path()).expanduser()
    if not path.is_file():
        return DoctorCheck(
            check_id="env.secret_store",
            status="WARN",
            detail_short=f"secret_store missing: {path}",
            next_action=f"Create readable secret store file at {path}.",
        )
    if not os.access(path, os.R_OK):
        return DoctorCheck(
            check_id="env.secret_store",
            status="FAIL",
            detail_short=f"secret_store unreadable: {path}",
            next_action=f"Fix read permissions for {path}.",
        )
    try:
        store = SecretStore(path=str(path))
        store.validate()
    except Exception as exc:  # pragma: no cover - defensive
        return DoctorCheck(
            check_id="env.secret_store",
            status="FAIL",
            detail_short=f"secret_store decrypt failed: {exc.__class__.__name__}",
            next_action=f"Repair or replace secret store at {path}.",
        )
    return DoctorCheck(
        check_id="env.secret_store",
        status="OK",
        detail_short=f"secret_store readable: {path}",
    )


def _check_required_dirs() -> DoctorCheck:
    required = [
        canonical_state_dir(),
        canonical_config_dir(),
        Path.home() / ".config" / "systemd" / "user",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if not missing:
        return DoctorCheck(check_id="env.required_dirs", status="OK", detail_short="required dirs present")
    return DoctorCheck(
        check_id="env.required_dirs",
        status="WARN",
        detail_short="missing dirs: " + ", ".join(missing),
        next_action="Run: python -m agent doctor --fix",
    )


def _check_telegram_dropin() -> DoctorCheck:
    dropin = Path.home() / ".config" / "systemd" / "user" / "personal-agent-telegram.service.d" / "override.conf"
    if not dropin.is_file():
        return DoctorCheck(
            check_id="telegram.dropin",
            status="WARN",
            detail_short=f"missing drop-in: {dropin}",
            next_action="Run: python -m agent doctor --fix",
        )
    try:
        content = dropin.read_text(encoding="utf-8")
    except Exception:
        return DoctorCheck(
            check_id="telegram.dropin",
            status="WARN",
            detail_short=f"unreadable drop-in: {dropin}",
            next_action=f"Fix read permissions for {dropin}",
        )
    if "AGENT_SECRET_STORE_PATH=" not in content:
        return DoctorCheck(
            check_id="telegram.dropin",
            status="WARN",
            detail_short="AGENT_SECRET_STORE_PATH missing in telegram drop-in",
            next_action="Run: python -m agent doctor --fix",
        )
    return DoctorCheck(check_id="telegram.dropin", status="OK", detail_short=f"drop-in configured: {dropin}")


def _check_write_mode_safe() -> DoctorCheck:
    writes_enabled = _env_truthy("ENABLE_WRITES", default=False)
    if writes_enabled:
        return DoctorCheck(
            check_id="runtime.safe_mode",
            status="WARN",
            detail_short="ENABLE_WRITES is on",
            next_action="Set ENABLE_WRITES=0 for read-only safe mode.",
        )
    return DoctorCheck(check_id="runtime.safe_mode", status="OK", detail_short="read-only safe mode is active")


def _run_systemctl_user(unit: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.6,
        )
    except Exception:
        return False, "unknown"
    state = (proc.stdout or "").strip() or "unknown"
    return proc.returncode == 0 and state == "active", state


def _check_systemd_service(unit: str, check_id: str) -> DoctorCheck:
    active, state = _run_systemctl_user(unit)
    if active:
        return DoctorCheck(check_id=check_id, status="OK", detail_short=f"{unit} active")
    return DoctorCheck(
        check_id=check_id,
        status="WARN",
        detail_short=f"{unit} state={state}",
        next_action=f"Run: systemctl --user restart {unit}",
    )


def _check_telegram_poller_singleton() -> DoctorCheck:
    try:
        state = get_telegram_runtime_state(env=telegram_control_env())
        if bool(state.get("lock_present", False)) and not bool(state.get("service_active", False)):
            if bool(state.get("lock_stale", False)):
                return DoctorCheck(
                    check_id="process.telegram_pollers",
                    status="WARN",
                    detail_short="stale telegram lock detected",
                    next_action="Run: python -m agent telegram_enable",
                )
            if bool(state.get("lock_live", False)):
                return DoctorCheck(
                    check_id="process.telegram_pollers",
                    status="FAIL",
                    detail_short="telegram lock held by another process",
                    next_action="Stop duplicate Telegram pollers and keep only one running instance.",
                )
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,args"],
            check=False,
            capture_output=True,
            text=True,
            timeout=0.6,
        )
    except Exception:
        return DoctorCheck(check_id="process.telegram_pollers", status="WARN", detail_short="unable to inspect process list")
    lines = []
    for row in (proc.stdout or "").splitlines():
        low = row.lower()
        if "telegram_adapter" in low and "python" in low:
            lines.append(row.strip())
    if len(lines) <= 1:
        return DoctorCheck(check_id="process.telegram_pollers", status="OK", detail_short=f"telegram pollers={len(lines)}")
    return DoctorCheck(
        check_id="process.telegram_pollers",
        status="FAIL",
        detail_short=f"multiple telegram pollers detected ({len(lines)})",
        detail_long=" | ".join(lines[:4]),
        next_action="Stop duplicate Telegram pollers and keep only one running instance.",
    )


def _api_get_json(url: str, timeout_seconds: float = 0.8) -> tuple[bool, dict[str, Any] | str]:
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        return False, f"{exc.__class__.__name__}:{exc}"
    try:
        parsed = json.loads(raw or "{}")
    except Exception as exc:
        return False, f"json_error:{exc.__class__.__name__}"
    if isinstance(parsed, dict):
        return True, parsed
    return False, "non_object_json"


def _check_llm_availability(api_base: str) -> DoctorCheck:
    ready_ok, ready_payload_or_error = _api_get_json(f"{api_base.rstrip('/')}/ready", timeout_seconds=0.8)
    ready_payload = ready_payload_or_error if ready_ok and isinstance(ready_payload_or_error, dict) else {}
    payload = ready_payload.get("llm") if isinstance(ready_payload.get("llm"), dict) else {}
    if not payload:
        status_ok, payload_or_error = _api_get_json(f"{api_base.rstrip('/')}/llm/status", timeout_seconds=0.8)
        if not status_ok:
            return DoctorCheck(
                check_id="llm.availability",
                status="WARN",
                detail_short=f"/llm/status unavailable: {payload_or_error}",
                next_action=f"Run: systemctl --user restart {runtime_service_name()}",
            )
        payload = payload_or_error if isinstance(payload_or_error, dict) else {}
    provider = str(payload.get("default_provider") or "unknown").strip() or "unknown"
    model = str(payload.get("resolved_default_model") or payload.get("default_model") or "unknown").strip() or "unknown"
    provider_health = payload.get("active_provider_health") if isinstance(payload.get("active_provider_health"), dict) else {}
    model_health = payload.get("active_model_health") if isinstance(payload.get("active_model_health"), dict) else {}
    allow_remote = bool(payload.get("allow_remote_fallback", True))
    provider_state = str(provider_health.get("status") or "unknown").strip().lower()
    model_state = str(model_health.get("status") or "unknown").strip().lower()
    detail = (
        f"provider={provider} model={model} provider_status={provider_state} "
        f"model_status={model_state} allow_remote_fallback={str(allow_remote).lower()}"
    )
    if provider_state == "ok" and model_state == "ok":
        return DoctorCheck(check_id="llm.availability", status="OK", detail_short=detail)
    return DoctorCheck(
        check_id="llm.availability",
        status="FAIL",
        detail_short=detail,
        next_action="Run: python -m agent setup",
    )


def _check_telegram_token(online: bool) -> DoctorCheck:
    store = SecretStore(path=_effective_secret_store_path())
    token = (store.get_secret("telegram:bot_token") or "").strip() or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return DoctorCheck(
            check_id="telegram.token",
            status="WARN",
            detail_short="telegram token missing",
            next_action="Run: python -m agent.secrets set telegram:bot_token",
        )
    detail = f"token={_token_preview(token)} source={'secret_store' if store.get_secret('telegram:bot_token') else 'env'}"
    if not online:
        return DoctorCheck(check_id="telegram.token", status="OK", detail_short=detail)
    ok, payload_or_error = _api_get_json(
        f"https://api.telegram.org/bot{token}/getMe",
        timeout_seconds=1.0,
    )
    if not ok:
        return DoctorCheck(
            check_id="telegram.token",
            status="WARN",
            detail_short=f"token present, online check failed: {payload_or_error}",
            next_action="Verify network access and Telegram token validity.",
        )
    payload = payload_or_error if isinstance(payload_or_error, dict) else {}
    if bool(payload.get("ok", False)):
        return DoctorCheck(check_id="telegram.token", status="OK", detail_short=detail + " online=ok")
    return DoctorCheck(
        check_id="telegram.token",
        status="WARN",
        detail_short=detail + " online=failed",
        next_action="Rotate Telegram token and restart telegram service.",
    )


def _check_logging_to_stdout() -> DoctorCheck:
    root = logging.getLogger()
    handlers = root.handlers
    if not handlers:
        return DoctorCheck(
            check_id="logging.stdout",
            status="WARN",
            detail_short="no logging handlers configured",
            next_action="Configure logging to stdout for journald visibility.",
        )
    return DoctorCheck(check_id="logging.stdout", status="OK", detail_short=f"stdout/journald handlers={len(handlers)}")


def _doctor_checks(
    *,
    repo_root: Path,
    online: bool,
    api_base_url: str,
) -> list[DoctorCheck]:
    telegram_enabled = _telegram_enabled_for_doctor()
    checks = [
        _check_python_runtime(),
        _check_repo_readable(repo_root),
        _check_secret_store_path(),
        _check_required_dirs(),
        _check_write_mode_safe(),
        _check_systemd_service(runtime_service_name(), "systemd.api_service"),
        _check_llm_availability(api_base_url),
        _check_logging_to_stdout(),
    ]
    if telegram_enabled:
        checks.extend(
            [
                _check_telegram_dropin(),
                _check_systemd_service("personal-agent-telegram.service", "systemd.telegram_service"),
                _check_telegram_poller_singleton(),
                _check_telegram_token(online=online),
            ]
        )
    else:
        checks.extend(
            [
                _telegram_optional_check("telegram.dropin"),
                _telegram_optional_check("systemd.telegram_service"),
                _telegram_optional_check("process.telegram_pollers"),
                _telegram_optional_check("telegram.token"),
            ]
        )
    return checks


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _safe_fix_dirs() -> list[str]:
    changed: list[str] = []
    paths = [
        canonical_state_dir(),
        canonical_config_dir(),
        Path.home() / ".config" / "systemd" / "user",
        Path.home() / ".config" / "systemd" / "user" / "personal-agent-telegram.service.d",
    ]
    for path in paths:
        if path.is_dir():
            continue
        path.mkdir(parents=True, exist_ok=True)
        changed.append(f"created_dir:{path}")
    return changed


def _safe_fix_telegram_dropin() -> list[str]:
    changed: list[str] = []
    secret_path = Path(_effective_secret_store_path()).expanduser().resolve()
    if not secret_path.is_file() or not os.access(secret_path, os.R_OK):
        return changed
    dropin_dir = Path.home() / ".config" / "systemd" / "user" / "personal-agent-telegram.service.d"
    dropin_dir.mkdir(parents=True, exist_ok=True)
    dropin_path = dropin_dir / "override.conf"
    target_line = f"Environment=AGENT_SECRET_STORE_PATH={secret_path}"
    existing = ""
    if dropin_path.is_file():
        try:
            existing = dropin_path.read_text(encoding="utf-8")
        except Exception:
            existing = ""
    if target_line in existing:
        return changed
    content = "[Service]\n" + target_line + "\n"
    dropin_path.write_text(content, encoding="utf-8")
    changed.append(f"wrote_dropin:{dropin_path}")
    return changed


def _safe_fix_stale_locks() -> list[str]:
    changed: list[str] = []
    root = canonical_state_dir()
    if not root.is_dir():
        return changed
    patterns = ["telegram_poll*.lock", "telegram_poll.lock"]
    for pattern in patterns:
        for lock_path in sorted(root.glob(pattern)):
            try:
                raw = lock_path.read_text(encoding="utf-8").strip().splitlines()
            except Exception:
                raw = []
            if not raw:
                continue
            first = str(raw[0]).strip()
            if not first.isdigit():
                continue
            pid = int(first)
            if _is_pid_running(pid):
                continue
            try:
                lock_path.unlink()
                changed.append(f"removed_stale_lock:{lock_path}")
            except OSError:
                continue
    return changed


def _safe_fix_runtime_storage_upgrade(repo_root: Path) -> list[str]:
    changed: list[str] = []
    migrations = [
        (repo_root / "memory" / "agent.db", canonical_state_dir() / "agent.db", "copied_legacy_db"),
        (repo_root / "logs" / "agent.jsonl", canonical_log_path(), "copied_legacy_log"),
    ]
    for source, target, label in migrations:
        if target.exists() or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        changed.append(f"{label}:{source}->{target}")
    return changed


def _backup_targets_manifest() -> list[dict[str, str]]:
    return [
        {
            "label": "operator_config",
            "path": str(canonical_config_dir()),
        },
        {
            "label": "state_dir",
            "path": str(canonical_state_dir()),
        },
        {
            "label": "log_file",
            "path": str(canonical_log_path()),
        },
        {
            "label": "systemd_user_dir",
            "path": str(Path.home() / ".config" / "systemd" / "user"),
        },
    ]


def _recovery_manifest() -> dict[str, Any]:
    return {
        "canonical_commands": {
            "doctor": "python -m agent doctor",
            "collect_diagnostics": "python -m agent doctor --collect-diagnostics",
            "safe_fix": "python -m agent doctor --fix",
            "setup": "python -m agent setup",
        },
        "backup_targets": _backup_targets_manifest(),
        "backup_notes": [
            "Stop user services before copying state for backup or restore.",
            "Configured discovery sources and policies live under the canonical config/state directories and are included by copying those paths.",
        ],
        "restore_steps": [
            f"Stop {runtime_service_name()} and personal-agent-telegram.service if they are running.",
            "Restore the copied operator config and state files to their canonical paths.",
            "Run: python -m agent doctor --fix",
            "Restart the user services and confirm /ready and /health are healthy.",
        ],
        "corrupt_state_recovery": [
            "If a state file is corrupt, move the broken file aside instead of editing it in place.",
            "Restore from backup when available, then run: python -m agent doctor --fix",
        ],
        "failed_upgrade_recovery": [
            "Reinstall the supported build or repo checkout.",
            "Run: python -m agent doctor --fix to recreate missing local directories, drop-ins, and migrated state.",
        ],
    }


def _diagnostics_paths_manifest(repo_root: Path) -> dict[str, str]:
    instance = runtime_instance()
    service_name = "personal-agent-api-dev.service" if instance == "dev" else "personal-agent-api.service"
    launcher_target = "personal-agent-webui-dev" if instance == "dev" else "personal-agent-webui"
    return {
        "runtime_instance": instance,
        "repo_root": str(repo_root),
        "runtime_root": str(runtime_root_path()),
        "install_mode": "editable_checkout" if instance == "dev" else "stable_installed",
        "service_name": service_name,
        "launcher_target": launcher_target,
        "config_dir": str(canonical_config_dir()),
        "state_dir": str(canonical_state_dir()),
        "db_path": str(resolved_default_db_path()),
        "log_path": str(canonical_log_path()),
        "secret_store_path": _effective_secret_store_path(),
    }


def _diagnostics_fetch(url: str) -> dict[str, Any]:
    ok, payload_or_error = _api_get_json(url, timeout_seconds=0.8)
    if not ok:
        return {
            "ok": False,
            "error": str(payload_or_error),
        }
    return {
        "ok": True,
        "payload": payload_or_error if isinstance(payload_or_error, dict) else {},
    }


def _diagnostics_runtime_payloads(api_base_url: str) -> dict[str, Any]:
    base = str(api_base_url or f"http://127.0.0.1:{runtime_port()}").rstrip("/")
    return {
        "health": _diagnostics_fetch(f"{base}/health"),
        "ready": _diagnostics_fetch(f"{base}/ready"),
        "runtime": _diagnostics_fetch(f"{base}/runtime"),
        "llm_status": _diagnostics_fetch(f"{base}/llm/status"),
        "memory": _diagnostics_fetch(f"{base}/memory"),
    }


def _diagnostics_command_ok(result: CommandResult) -> bool:
    return not result.permission_denied and not result.not_available and result.error is None and int(result.returncode or 0) == 0


def _diagnostics_nmc_rows(result: CommandResult) -> list[dict[str, str]]:
    if not _diagnostics_command_ok(result):
        return []
    rows: list[dict[str, str]] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(":")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "device": parts[0],
                "type": parts[1],
                "state": parts[2],
                "connection": parts[3],
            }
        )
    return rows


def _diagnostics_suspend_resume_lines(result: CommandResult) -> list[str]:
    if not _diagnostics_command_ok(result):
        return []
    keywords = ("suspend", "resume", "sleep", "wakeup", "wake up", "suspended", "resumed")
    matches: list[str] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            matches.append(line)
    return matches[-8:]


_BLUETOOTH_SERVICE_ACTIVE_RE = re.compile(r"^\s*Active:\s*(.+)$", re.IGNORECASE)
_BLUETOOTH_SERVICE_LOADED_RE = re.compile(r"^\s*Loaded:\s*(.+)$", re.IGNORECASE)
_BLUETOOTH_DEVICE_RE = re.compile(r"^Device\s+([0-9A-F:]{17})\s+(.+)$", re.IGNORECASE)
_BLUETOOTH_INFO_BOOL_RE = re.compile(r"^\s*(Connected|Paired|Trusted|Blocked):\s*(yes|no)\s*$", re.IGNORECASE)
_PRINTER_CUPS_SERVICE_ACTIVE_RE = re.compile(r"^\s*Active:\s*(.+)$", re.IGNORECASE)
_PRINTER_CUPS_SERVICE_LOADED_RE = re.compile(r"^\s*Loaded:\s*(.+)$", re.IGNORECASE)
_PRINTER_CUPS_PRINTER_RE = re.compile(r"^\s*printer\s+(\S+)\s+(.*)$", re.IGNORECASE)
_PRINTER_CUPS_DEFAULT_RE = re.compile(r"^\s*system default destination:\s*(.+)$", re.IGNORECASE)
_GENERIC_DEVICE_LOG_KEYWORDS = (
    "error",
    "fail",
    "failed",
    "warning",
    "warn",
    "timeout",
    "not found",
    "not detected",
    "not recognized",
    "firmware",
    "reset",
    "disconnect",
    "offline",
    "blocked",
    "denied",
    "usb",
    "pci",
    "device",
    "probe",
    "hotplug",
    "i/o error",
)
_STORAGE_DF_HEADER_RE = re.compile(r"^Filesystem\s+Type\s+Size\s+Used\s+Avail\s+Use%\s+Mounted on$", re.IGNORECASE)
_STORAGE_PSEUDO_FS_TYPES = {
    "autofs",
    "bpf",
    "cgroup",
    "cgroup2",
    "configfs",
    "debugfs",
    "devpts",
    "devtmpfs",
    "efivarfs",
    "fusectl",
    "hugetlbfs",
    "mqueue",
    "proc",
    "pstore",
    "securityfs",
    "squashfs",
    "sysfs",
    "tmpfs",
    "tracefs",
}
_STORAGE_LOG_KEYWORDS = (
    "no space left on device",
    "disk full",
    "out of space",
    "read-only file system",
    "write failed",
    "cannot write",
    "save failed",
    "enospc",
    "i/o error",
)


def _bluetooth_bool_from_value(value: str | None) -> bool | None:
    lowered = str(value or "").strip().lower()
    if lowered in {"yes", "true", "1", "on"}:
        return True
    if lowered in {"no", "false", "0", "off"}:
        return False
    return None


def _bluetooth_service_snapshot(result: CommandResult) -> dict[str, Any]:
    text = redact_secrets(result.stdout).strip()
    active_state = None
    active_detail = None
    loaded_state = None
    loaded_detail = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        active_match = _BLUETOOTH_SERVICE_ACTIVE_RE.match(line)
        if active_match:
            active_detail = active_match.group(1).strip()
            active_state = active_detail.split(" ", 1)[0].strip().lower() or None
            continue
        loaded_match = _BLUETOOTH_SERVICE_LOADED_RE.match(line)
        if loaded_match:
            loaded_detail = loaded_match.group(1).strip()
            loaded_state = loaded_detail.split(";", 1)[0].strip().split(" ", 1)[0].strip().lower() or None
            continue
    return {
        "available": _diagnostics_command_ok(result),
        "source": "systemctl status bluetooth",
        "active_state": active_state,
        "active_detail": active_detail,
        "loaded_state": loaded_state,
        "loaded_detail": loaded_detail,
        "error_kind": None
        if _diagnostics_command_ok(result)
        else (
            "permission_denied"
            if result.permission_denied
            else "not_available"
            if result.not_available
            else "command_failed"
        ),
    }


def _bluetooth_controller_snapshot(result: CommandResult) -> dict[str, Any]:
    text = redact_secrets(result.stdout).strip()
    controller = None
    powered = None
    discoverable = None
    pairable = None
    discovering = None
    alias = None
    address = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("controller "):
            parts = line.split()
            if len(parts) >= 2:
                address = parts[1]
            if len(parts) >= 3:
                controller = " ".join(parts[2:]).strip() or None
            continue
        key, _, value = line.partition(":")
        if not _:
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key == "alias":
            alias = normalized_value or None
        elif normalized_key == "powered":
            powered = _bluetooth_bool_from_value(normalized_value)
        elif normalized_key == "discoverable":
            discoverable = _bluetooth_bool_from_value(normalized_value)
        elif normalized_key == "pairable":
            pairable = _bluetooth_bool_from_value(normalized_value)
        elif normalized_key == "discovering":
            discovering = _bluetooth_bool_from_value(normalized_value)
    return {
        "available": _diagnostics_command_ok(result),
        "source": "bluetoothctl show",
        "address": address,
        "controller": controller,
        "alias": alias,
        "powered": powered,
        "discoverable": discoverable,
        "pairable": pairable,
        "discovering": discovering,
        "error_kind": None
        if _diagnostics_command_ok(result)
        else (
            "permission_denied"
            if result.permission_denied
            else "not_available"
            if result.not_available
            else "command_failed"
        ),
    }


def _bluetooth_devices_from_command(result: CommandResult) -> list[dict[str, Any]]:
    if not _diagnostics_command_ok(result):
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _BLUETOOTH_DEVICE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "address": match.group(1).strip(),
                "name": match.group(2).strip() or None,
            }
        )
    return rows


def _bluetooth_device_info_snapshot(result: CommandResult) -> dict[str, Any]:
    text = redact_secrets(result.stdout).strip()
    connected = None
    paired = None
    trusted = None
    blocked = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _BLUETOOTH_INFO_BOOL_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = _bluetooth_bool_from_value(match.group(2))
        if key == "connected":
            connected = value
        elif key == "paired":
            paired = value
        elif key == "trusted":
            trusted = value
        elif key == "blocked":
            blocked = value
    return {
        "available": _diagnostics_command_ok(result),
        "connected": connected,
        "paired": paired,
        "trusted": trusted,
        "blocked": blocked,
        "error_kind": None
        if _diagnostics_command_ok(result)
        else (
            "permission_denied"
            if result.permission_denied
            else "not_available"
            if result.not_available
            else "command_failed"
        ),
    }


def _bluetooth_device_rows(
    paired_result: CommandResult,
    *,
    run_command_fn: Callable[..., CommandResult],
) -> list[dict[str, Any]]:
    rows = _bluetooth_devices_from_command(paired_result)
    if not rows:
        fallback_result = run_command_fn(["bluetoothctl", "devices"], timeout_s=1.0)
        rows = _bluetooth_devices_from_command(fallback_result)
    for row in rows[:3]:
        info_result = run_command_fn(["bluetoothctl", "info", str(row.get("address") or "")], timeout_s=1.0)
        row.update(_bluetooth_device_info_snapshot(info_result))
    return rows


def _bluetooth_log_lines(result: CommandResult) -> list[str]:
    if not _diagnostics_command_ok(result):
        return []
    keywords = (
        "bluetooth",
        "bluez",
        "hci",
        "a2dp",
        "avrcp",
        "headset",
        "headphone",
        "audio",
        "codec",
        "pair",
        "connect",
        "disconnect",
        "pipewire",
        "pulseaudio",
        "sink",
        "source",
    )
    matches: list[str] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            matches.append(line)
    return matches[-8:]


def _printer_cups_service_snapshot(result: CommandResult) -> dict[str, Any]:
    text = redact_secrets(result.stdout).strip()
    active_detail = ""
    loaded_detail = ""
    description = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        active_match = _PRINTER_CUPS_SERVICE_ACTIVE_RE.match(line)
        if active_match:
            active_detail = active_match.group(1).strip()
            continue
        loaded_match = _PRINTER_CUPS_SERVICE_LOADED_RE.match(line)
        if loaded_match:
            loaded_detail = loaded_match.group(1).strip()
            continue
        if line.lower().startswith("description:"):
            description = line.split(":", 1)[1].strip()
    active_state = active_detail.split(" ", 1)[0].strip().lower() if active_detail else "unknown"
    loaded_state = loaded_detail.split(" ", 1)[0].strip().lower() if loaded_detail else "unknown"
    return {
        "available": _diagnostics_command_ok(result),
        "source": "systemctl status cups",
        "active_state": active_state,
        "active_detail": active_detail or None,
        "loaded_state": loaded_state,
        "loaded_detail": loaded_detail or None,
        "description": description or None,
        "error_kind": None
        if _diagnostics_command_ok(result)
        else (
            "permission_denied"
            if result.permission_denied
            else "not_available"
            if result.not_available
            else "command_failed"
        ),
    }


def _printer_cups_printer_rows(result: CommandResult) -> dict[str, Any]:
    text = redact_secrets(result.stdout).strip()
    default_printer = None
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        default_match = _PRINTER_CUPS_DEFAULT_RE.match(line)
        if default_match:
            default_printer = default_match.group(1).strip() or None
            continue
        match = _PRINTER_CUPS_PRINTER_RE.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        status_text = match.group(2).strip()
        lowered = status_text.lower()
        state = "unknown"
        enabled = None
        accepting = None
        if "disabled" in lowered:
            state = "disabled"
            enabled = False
        elif "offline" in lowered:
            state = "offline"
        elif "printing" in lowered:
            state = "printing"
        elif "idle" in lowered:
            state = "idle"
        if "enabled" in lowered:
            enabled = True
        if "not accepting" in lowered:
            accepting = False
        elif "accepting" in lowered:
            accepting = True
        rows.append(
            {
                "name": name,
                "state": state,
                "enabled": enabled,
                "accepting": accepting,
                "details": status_text[:180] or None,
            }
        )
    return {
        "available": _diagnostics_command_ok(result),
        "source": "lpstat -p -d",
        "default_printer": default_printer,
        "printer_count": len(rows),
        "rows": rows,
        "error_kind": None
        if _diagnostics_command_ok(result)
        else (
            "permission_denied"
            if result.permission_denied
            else "not_available"
            if result.not_available
            else "command_failed"
        ),
    }


def _printer_cups_job_rows(result: CommandResult) -> dict[str, Any]:
    text = redact_secrets(result.stdout).strip()
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        queue_name = None
        job_id = None
        owner = None
        if parts:
            first = parts[0]
            if "-" in first:
                queue_name, job_tail = first.rsplit("-", 1)
                if job_tail.isdigit():
                    job_id = job_tail
            if len(parts) > 1:
                owner = parts[1].strip() or None
        rows.append(
            {
                "queue": queue_name,
                "job_id": job_id,
                "owner": owner,
                "text": line[:180],
            }
        )
    return {
        "available": _diagnostics_command_ok(result),
        "source": "lpstat -o",
        "match_count": len(rows),
        "rows": rows,
        "error_kind": None
        if _diagnostics_command_ok(result)
        else (
            "permission_denied"
            if result.permission_denied
            else "not_available"
            if result.not_available
            else "command_failed"
        ),
    }


_PRINTER_CUPS_LOG_KEYWORDS = (
    "cups",
    "printer",
    "print",
    "queue",
    "job",
    "spool",
    "scheduler",
    "backend",
    "filter",
    "paused",
    "stopped",
    "disabled",
    "offline",
    "error",
    "fail",
    "denied",
    "timeout",
    "blocked",
)


def _printer_cups_log_lines(result: CommandResult) -> list[str]:
    if not _diagnostics_command_ok(result):
        return []
    matches: list[str] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in _PRINTER_CUPS_LOG_KEYWORDS):
            matches.append(line[:220])
    return matches[-8:]


def _generic_device_snapshot_lines(result: CommandResult, *, limit: int = 8) -> list[str]:
    if not _diagnostics_command_ok(result):
        return []
    lines: list[str] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lines.append(line[:180])
    return lines[-limit:]


def _generic_device_log_lines(result: CommandResult) -> list[str]:
    if not _diagnostics_command_ok(result):
        return []
    matches: list[str] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in _GENERIC_DEVICE_LOG_KEYWORDS):
            matches.append(line[:220])
    return matches[-8:]


def _storage_format_bytes(value: int | float | None) -> str:
    amount = float(value or 0)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 0
    while amount >= 1024.0 and unit_index < len(units) - 1:
        amount /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(amount)} {units[unit_index]}"
    return f"{amount:.1f} {units[unit_index]}"


def _storage_disk_rows(result: CommandResult) -> list[dict[str, Any]]:
    if not result.stdout.strip():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line or _STORAGE_DF_HEADER_RE.match(line):
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        source, fstype, size, used, avail, used_pct, mountpoint = parts[:7]
        try:
            used_pct_value = float(str(used_pct).strip().rstrip("%"))
        except ValueError:
            used_pct_value = 0.0
        rows.append(
            {
                "device": source.strip() or None,
                "mountpoint": mountpoint.strip() or None,
                "fstype": fstype.strip() or None,
                "size": size.strip() or None,
                "used": used.strip() or None,
                "avail": avail.strip() or None,
                "used_pct": round(used_pct_value, 2),
            }
        )
    rows.sort(key=lambda row: float(row.get("used_pct") or 0.0), reverse=True)
    return [row for row in rows if str(row.get("fstype") or "").strip().lower() not in _STORAGE_PSEUDO_FS_TYPES]


def _storage_consumers_from_du(result: CommandResult, *, base_path: str) -> list[dict[str, Any]]:
    if not result.stdout.strip():
        return []
    rows: list[dict[str, Any]] = []
    base_abs = str(Path(base_path).expanduser().resolve())
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            size_bytes = int(parts[0].strip())
        except ValueError:
            continue
        path = parts[1].strip()
        try:
            if str(Path(path).expanduser().resolve()) == base_abs:
                continue
        except OSError:
            pass
        rows.append({"path": path, "size_bytes": size_bytes, "base_path": base_abs})
    rows.sort(key=lambda row: int(row.get("size_bytes") or 0), reverse=True)
    return rows


def _storage_recent_log_lines(result: CommandResult) -> list[str]:
    if not result.stdout.strip():
        return []
    matches: list[str] = []
    for raw_line in redact_secrets(result.stdout).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in _STORAGE_LOG_KEYWORDS):
            matches.append(line)
    return matches[-8:]


def collect_storage_disk_diagnostics_snapshot(
    *,
    run_command_fn: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    trace_id = _doctor_trace_id()
    generated_at = _safe_now_iso()

    df_result = run_command_fn(
        ["df", "-hT", "--local", "--output=source,fstype,size,used,avail,pcent,target"],
        timeout_s=1.5,
    )
    journal_result = run_command_fn(["journalctl", "-b", "--no-pager", "-n", "120"], timeout_s=2.0)

    disk_rows = _storage_disk_rows(df_result)
    consumer_rows: list[dict[str, Any]] = []
    for candidate in [str(Path.home()), "/var", "/tmp"]:
        if not candidate or not os.path.exists(candidate):
            continue
        du_result = run_command_fn(["du", "-x", "-B1", "-d", "1", candidate], timeout_s=1.5)
        consumer_rows.extend(_storage_consumers_from_du(du_result, base_path=candidate))
    consumer_rows.sort(key=lambda row: int(row.get("size_bytes") or 0), reverse=True)
    consumer_rows = consumer_rows[:5]
    log_lines = _storage_recent_log_lines(journal_result)

    mount_preview = [
        row
        for row in disk_rows[:4]
        if isinstance(row, dict) and str(row.get("mountpoint") or "").strip()
    ]
    if disk_rows:
        max_used_pct = max(float(row.get("used_pct") or 0.0) for row in disk_rows)
    else:
        max_used_pct = 0.0
    high_mounts = [
        row for row in disk_rows if float(row.get("used_pct") or 0.0) >= 85.0
    ]

    notes: list[str] = []
    if high_mounts:
        notes.append(
            "High filesystem usage on "
            + ", ".join(
                f"{str(row.get('mountpoint') or '?')} ({float(row.get('used_pct') or 0.0):.1f}%)"
                for row in high_mounts[:3]
            )
            + "."
        )
    elif disk_rows:
        notes.append("Filesystem usage is visible in the compact snapshot.")
    if consumer_rows:
        notes.append(f"Largest obvious consumer: {str(consumer_rows[0].get('path') or '')}.")
    if log_lines:
        notes.append("Recent logs contain disk-full or write-failure markers.")
    else:
        notes.append("No recent disk-full or write-failure markers were visible.")

    status = "ok"
    if max_used_pct >= 95.0:
        status = "critical"
    elif high_mounts or log_lines:
        status = "warn"

    assessment = "; ".join(notes) if notes else "No obvious storage failure was visible in this compact snapshot."
    if max_used_pct >= 95.0:
        next_action = "Free space on the fullest filesystem and retry the save."
    elif high_mounts:
        next_action = "Free space on the affected filesystem and retry the write."
    elif log_lines:
        next_action = "Retry the save after clearing some space and check the target mount."
    else:
        next_action = "If the problem persists, capture a full doctor bundle or inspect the target path."

    return {
        "trace_id": trace_id,
        "generated_at": generated_at,
        "preset": "storage_disk",
        "storage": {
            "filesystems": {
                "source": "df -hT --local --output=source,fstype,size,used,avail,pcent,target",
                "available": bool(df_result.stdout.strip()),
                "rows": disk_rows,
            },
            "consumers": {
                "source": "du -x -B1 -d 1 <candidate paths>",
                "available": bool(consumer_rows),
                "match_count": len(consumer_rows),
                "entries": consumer_rows,
            },
            "logs": {
                "source": "journalctl -b",
                "available": bool(journal_result.stdout.strip()),
                "match_count": len(log_lines),
                "matches": log_lines,
            },
        },
        "summary": {
            "status": status,
            "assessment": assessment,
            "notes": notes,
            "next_action": next_action,
        },
    }


def collect_printer_cups_diagnostics_snapshot(
    *,
    run_command_fn: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    trace_id = _doctor_trace_id()
    generated_at = _safe_now_iso()

    service_result = run_command_fn(["systemctl", "status", "cups", "--no-pager", "--full"], timeout_s=1.5)
    printer_result = run_command_fn(["lpstat", "-p", "-d"], timeout_s=1.0)
    jobs_result = run_command_fn(["lpstat", "-o"], timeout_s=1.0)
    journal_result = run_command_fn(["journalctl", "-u", "cups", "--no-pager", "-n", "120"], timeout_s=2.0)

    service_payload = _printer_cups_service_snapshot(service_result)
    printer_payload = _printer_cups_printer_rows(printer_result)
    job_payload = _printer_cups_job_rows(jobs_result)
    log_lines = _printer_cups_log_lines(journal_result)
    printer_rows = printer_payload.get("rows") if isinstance(printer_payload.get("rows"), list) else []
    job_rows = job_payload.get("rows") if isinstance(job_payload.get("rows"), list) else []

    enabled_printers = [row for row in printer_rows if isinstance(row, dict) and row.get("enabled") is not False]
    offline_printers = [row for row in printer_rows if isinstance(row, dict) and str(row.get("state") or "").strip().lower() in {"offline", "disabled"}]
    notes: list[str] = []
    if service_payload.get("active_state") != "active":
        notes.append("CUPS service is not active in the snapshot.")
    if printer_payload.get("default_printer"):
        notes.append(f"Default printer is set to {printer_payload['default_printer']}.")
    else:
        notes.append("No default printer was visible in the compact snapshot.")
    if job_rows:
        notes.append(f"Print queue shows {len(job_rows)} job(s).")
    if offline_printers:
        preview = ", ".join(str(row.get("name") or "?") for row in offline_printers[:3])
        notes.append(f"At least one printer appears offline or disabled: {preview}.")
    elif printer_rows:
        notes.append("Printer status is visible in the compact snapshot.")
    if log_lines:
        lowered_logs = " ".join(line.lower() for line in log_lines)
        if any(token in lowered_logs for token in ("fail", "error", "stuck", "offline", "disabled", "denied", "stopp")):
            notes.append("Recent CUPS logs contain failure markers.")
        else:
            notes.append("Recent CUPS activity is visible in the journal window.")
    else:
        notes.append("No recent CUPS journal matches were found in the compact window.")

    status = "ok"
    if service_payload.get("active_state") != "active":
        status = "critical"
    elif job_rows or offline_printers or not printer_rows or log_lines:
        status = "warn"

    assessment = "; ".join(notes) if notes else "No obvious printer/CUPS failure was visible in this compact snapshot."
    if service_payload.get("active_state") != "active":
        next_action = "Restart CUPS and retry printing."
    elif job_rows and log_lines:
        next_action = "Clear or re-submit the print queue, then retry the job."
    elif job_rows:
        next_action = "Check the queue and cancel any stuck job, then retry printing."
    elif not printer_payload.get("default_printer"):
        next_action = "Set a default printer or select one explicitly, then retry printing."
    elif offline_printers:
        next_action = "Bring the affected printer online or re-add it, then retry printing."
    else:
        next_action = "If the issue persists, capture a full doctor bundle or re-add the printer."

    return {
        "trace_id": trace_id,
        "generated_at": generated_at,
        "preset": "printer_cups",
        "printer": {
            "service": service_payload,
            "printers": printer_payload,
            "jobs": job_payload,
            "logs": {
                "source": "journalctl -u cups",
                "available": _diagnostics_command_ok(journal_result),
                "match_count": len(log_lines),
                "matches": log_lines,
            },
        },
        "summary": {
            "status": status,
            "assessment": assessment,
            "notes": notes,
            "next_action": next_action,
        },
    }


def collect_bluetooth_audio_diagnostics_snapshot(
    *,
    run_command_fn: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    trace_id = _doctor_trace_id()
    generated_at = _safe_now_iso()

    service_result = run_command_fn(["systemctl", "status", "bluetooth", "--no-pager", "--full"], timeout_s=1.5)
    controller_result = run_command_fn(["bluetoothctl", "show"], timeout_s=1.0)
    paired_result = run_command_fn(["bluetoothctl", "paired-devices"], timeout_s=1.0)
    journal_result = run_command_fn(["journalctl", "-u", "bluetooth", "--no-pager", "-n", "120"], timeout_s=2.0)

    service_payload = _bluetooth_service_snapshot(service_result)
    controller_payload = _bluetooth_controller_snapshot(controller_result)
    device_rows = _bluetooth_device_rows(paired_result, run_command_fn=run_command_fn)
    log_lines = _bluetooth_log_lines(journal_result)

    connected_devices = [row for row in device_rows if row.get("connected") is True]
    notes: list[str] = []
    if service_payload.get("active_state") not in {"active", "activating"}:
        notes.append("Bluetooth service is not active in the snapshot.")
    if controller_payload.get("powered") is False:
        notes.append("The Bluetooth controller appears powered off.")
    if device_rows and not connected_devices:
        notes.append("No connected Bluetooth device was visible in the compact snapshot.")
    if log_lines:
        lowered_logs = " ".join(line.lower() for line in log_lines)
        if any(token in lowered_logs for token in ("fail", "error", "disconnect", "drop", "timeout", "stuck")):
            notes.append("Recent Bluetooth logs contain failure markers.")
        else:
            notes.append("Recent Bluetooth activity is visible in the journal window.")
    else:
        notes.append("No recent Bluetooth journal matches were found in the compact window.")

    bluetooth_ok = (
        service_payload.get("active_state") in {"active", "activating"}
        and controller_payload.get("powered") is not False
        and not any(token in " ".join(log_lines).lower() for token in ("fail", "error", "disconnect", "drop", "timeout", "stuck"))
    )
    if bluetooth_ok and connected_devices:
        notes.append("At least one paired Bluetooth device is connected in the snapshot.")

    first_pass_assessment = "; ".join(notes) if notes else "No obvious Bluetooth/audio failure was visible in this compact snapshot."
    if service_payload.get("active_state") not in {"active", "activating"}:
        next_action = "Check the Bluetooth service and retry the device connection."
    elif controller_payload.get("powered") is False:
        next_action = "Turn Bluetooth on and retry the connection."
    elif log_lines:
        next_action = "Re-test the Bluetooth device and check whether it reconnects after suspend."
    else:
        next_action = "If the issue persists, capture a full doctor bundle or try a fresh Bluetooth reconnect."

    return {
        "trace_id": trace_id,
        "generated_at": generated_at,
        "preset": "bluetooth_audio",
        "bluetooth": {
            "service": service_payload,
            "controller": controller_payload,
            "devices": {
                "source": "bluetoothctl paired-devices",
                "available": _diagnostics_command_ok(paired_result),
                "paired_count": len(device_rows),
                "connected_count": len(connected_devices),
                "devices": device_rows,
                "error_kind": None
                if _diagnostics_command_ok(paired_result)
                else (
                    "permission_denied"
                    if paired_result.permission_denied
                    else "not_available"
                    if paired_result.not_available
                    else "command_failed"
                ),
            },
            "logs": {
                "source": "journalctl -u bluetooth",
                "available": _diagnostics_command_ok(journal_result),
                "match_count": len(log_lines),
                "matches": log_lines,
            },
        },
        "summary": {
            "status": "warn" if not bluetooth_ok or not connected_devices else "ok",
            "assessment": first_pass_assessment,
            "notes": notes,
            "next_action": next_action,
        },
    }


def collect_diagnostics_snapshot(
    *,
    run_command_fn: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    trace_id = _doctor_trace_id()
    generated_at = _safe_now_iso()

    uname_result = run_command_fn(["uname", "-a"], timeout_s=1.0)
    nmcli_result = run_command_fn(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"], timeout_s=1.0)
    journal_result = run_command_fn(["journalctl", "-b", "--no-pager", "-n", "120"], timeout_s=2.0)

    nmcli_rows = _diagnostics_nmc_rows(nmcli_result)
    suspend_resume_lines = _diagnostics_suspend_resume_lines(journal_result)
    if nmcli_rows:
        connected_devices = [row["device"] for row in nmcli_rows if str(row.get("state") or "").strip().lower() == "connected"]
        network_summary: dict[str, Any] = {
            "state": "up" if connected_devices else "degraded",
            "up_interfaces": connected_devices,
            "default_route": False,
            "dns_configured": False,
        }
    else:
        network_health = collect_system_health(sample_seconds=0.0).get("network")
        network_summary = network_health if isinstance(network_health, dict) else {}

    os_payload = {
        "available": _diagnostics_command_ok(uname_result),
        "source": "uname -a",
        "text": redact_secrets(uname_result.stdout).strip() or None,
        "error_kind": None
        if _diagnostics_command_ok(uname_result)
        else (
            "permission_denied"
            if uname_result.permission_denied
            else "not_available"
            if uname_result.not_available
            else "command_failed"
        ),
    }
    network_payload = {
        "source": "nmcli" if nmcli_rows else "system_health",
        "nmcli_available": bool(nmcli_rows),
        "system_summary": {
            "state": str(network_summary.get("state") or "unknown"),
            "up_interfaces": [str(item) for item in (network_summary.get("up_interfaces") if isinstance(network_summary.get("up_interfaces"), list) else []) if str(item).strip()],
            "default_route": bool(network_summary.get("default_route", False)),
            "dns_configured": bool(network_summary.get("dns_configured", False)),
        },
        "nmcli_rows": nmcli_rows,
    }
    suspend_resume_payload = {
        "source": "journalctl -b",
        "available": _diagnostics_command_ok(journal_result),
        "match_count": len(suspend_resume_lines),
        "matches": suspend_resume_lines,
    }

    notes: list[str] = []
    network_state = str(network_payload["system_summary"].get("state") or "unknown").lower()
    if network_state != "up":
        notes.append("Network is not fully up in the snapshot.")
    if suspend_resume_lines:
        lowered_matches = " ".join(line.lower() for line in suspend_resume_lines)
        if any(token in lowered_matches for token in ("fail", "error", "disconnect", "drop", "timeout")):
            notes.append("Suspend/resume logs contain failure markers.")
        else:
            notes.append("Suspend/resume activity is present in the recent journal window.")
    else:
        notes.append("No suspend/resume lines were found in the recent journal window.")

    first_pass_assessment = "; ".join(notes) if notes else "No obvious suspend/resume failure was visible in this compact snapshot."
    next_action = "Check NetworkManager and the Wi-Fi driver after suspend." if network_state != "up" or any(
        token in " ".join(suspend_resume_lines).lower() for token in ("fail", "error", "disconnect", "drop", "timeout")
    ) else "No obvious suspend/resume failure was visible in this compact snapshot."

    return {
        "trace_id": trace_id,
        "generated_at": generated_at,
        "os": os_payload,
        "network": network_payload,
        "suspend_resume": suspend_resume_payload,
        "summary": {
            "status": "warn" if network_state != "up" or any(
                token in " ".join(suspend_resume_lines).lower() for token in ("fail", "error", "disconnect", "drop", "timeout")
            ) else "ok",
            "assessment": first_pass_assessment,
            "notes": notes,
            "next_action": next_action,
        },
    }


def build_diagnostics_cards_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    os_payload = snapshot.get("os") if isinstance(snapshot.get("os"), dict) else {}
    network_payload = snapshot.get("network") if isinstance(snapshot.get("network"), dict) else {}
    network_summary = network_payload.get("system_summary") if isinstance(network_payload.get("system_summary"), dict) else {}
    suspend_resume = snapshot.get("suspend_resume") if isinstance(snapshot.get("suspend_resume"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    nmcli_rows = network_payload.get("nmcli_rows") if isinstance(network_payload.get("nmcli_rows"), list) else []
    network_lines = [
        f"state: {str(network_summary.get('state') or 'unknown')}",
        "up interfaces: " + (",".join(str(item) for item in (network_summary.get("up_interfaces") or [])) or "none"),
        f"default route: {'yes' if bool(network_summary.get('default_route', False)) else 'no'}",
        f"dns: {'yes' if bool(network_summary.get('dns_configured', False)) else 'no'}",
    ]
    if nmcli_rows:
        preview_rows: list[str] = []
        for row in nmcli_rows[:3]:
            if not isinstance(row, dict):
                continue
            preview_rows.append(
                "{device} {state} {connection}".format(
                    device=str(row.get("device") or "").strip() or "?",
                    state=str(row.get("state") or "").strip() or "unknown",
                    connection=str(row.get("connection") or "").strip() or "none",
                ).strip()
            )
        if preview_rows:
            network_lines.append("nmcli preview: " + "; ".join(preview_rows))
    assessment = str(summary.get("assessment") or summary.get("notes") or "").strip()
    if not assessment:
        assessment = "No obvious suspend/resume failure was visible in this compact snapshot."
    suspend_count = int(suspend_resume.get("match_count") or 0)
    suspend_preview = [str(item).strip() for item in (suspend_resume.get("matches") if isinstance(suspend_resume.get("matches"), list) else [])[:3] if str(item).strip()]
    suspend_lines = [f"matches: {suspend_count}"]
    suspend_lines.extend(f"- {item}" for item in suspend_preview)
    cards_payload = build_cards_payload(
        [
            {
                "key": "diagnostics-snapshot",
                "title": "Diagnostics snapshot",
                "severity": str(summary.get("status") or "ok"),
                "lines": [
                    f"OS/kernel: {str(os_payload.get('text') or 'unavailable')}",
                    f"Network: {'; '.join(network_lines)}",
                    f"Suspend/resume: {'; '.join(suspend_lines)}",
                    f"Assessment: {assessment}",
                    f"Next action: {str(summary.get('next_action') or 'No next action available.')}",
                ],
            }
        ],
        raw_available=False,
        summary="Compact diagnostics snapshot.",
        confidence=1.0,
        next_questions=[
            "Check NetworkManager and the Wi-Fi driver after suspend.",
            "Collect a full doctor bundle if you want deeper analysis.",
        ],
    )
    cards_payload["show_confidence"] = False
    return cards_payload


def build_bluetooth_audio_diagnostics_cards_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    bluetooth_payload = snapshot.get("bluetooth") if isinstance(snapshot.get("bluetooth"), dict) else {}
    service_payload = bluetooth_payload.get("service") if isinstance(bluetooth_payload.get("service"), dict) else {}
    controller_payload = bluetooth_payload.get("controller") if isinstance(bluetooth_payload.get("controller"), dict) else {}
    devices_payload = bluetooth_payload.get("devices") if isinstance(bluetooth_payload.get("devices"), dict) else {}
    logs_payload = bluetooth_payload.get("logs") if isinstance(bluetooth_payload.get("logs"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    device_rows = devices_payload.get("devices") if isinstance(devices_payload.get("devices"), list) else []
    device_preview: list[str] = []
    for row in device_rows[:3]:
        if not isinstance(row, dict):
            continue
        state_bits: list[str] = []
        if row.get("connected") is True:
            state_bits.append("connected")
        elif row.get("connected") is False:
            state_bits.append("disconnected")
        if row.get("paired") is True:
            state_bits.append("paired")
        if row.get("trusted") is True:
            state_bits.append("trusted")
        state_text = ", ".join(state_bits) if state_bits else "state unknown"
        device_preview.append(
            "{name} ({address}): {state}".format(
                name=str(row.get("name") or "").strip() or "?",
                address=str(row.get("address") or "").strip() or "?",
                state=state_text,
            )
        )
    log_lines = [str(item).strip() for item in (logs_payload.get("matches") if isinstance(logs_payload.get("matches"), list) else [])[:3] if str(item).strip()]
    service_bits = [
        f"active={str(service_payload.get('active_state') or 'unknown')}",
        f"loaded={str(service_payload.get('loaded_state') or 'unknown')}",
    ]
    controller_bits = [
        f"powered={'yes' if controller_payload.get('powered') is True else 'no' if controller_payload.get('powered') is False else 'unknown'}",
        f"pairable={'yes' if controller_payload.get('pairable') is True else 'no' if controller_payload.get('pairable') is False else 'unknown'}",
        f"discoverable={'yes' if controller_payload.get('discoverable') is True else 'no' if controller_payload.get('discoverable') is False else 'unknown'}",
    ]
    assessment = str(summary.get("assessment") or summary.get("notes") or "").strip()
    if not assessment:
        assessment = "No obvious Bluetooth/audio failure was visible in this compact snapshot."
    cards_payload = build_cards_payload(
        [
            {
                "key": "bluetooth-audio-diagnostics",
                "title": "Bluetooth/audio diagnostics",
                "severity": str(summary.get("status") or "ok"),
                "lines": [
                    f"Service: {'; '.join(service_bits)}",
                    f"Controller: address={str(controller_payload.get('address') or 'unknown')}; {'; '.join(controller_bits)}",
                    f"Devices: paired={int(devices_payload.get('paired_count') or 0)}; connected={int(devices_payload.get('connected_count') or 0)}"
                    + (f"; preview={'; '.join(device_preview)}" if device_preview else ""),
                    f"Logs: matches={int(logs_payload.get('match_count') or 0)}"
                    + (f"; preview={'; '.join(log_lines)}" if log_lines else ""),
                    f"Assessment: {assessment}",
                    f"Next action: {str(summary.get('next_action') or 'No next action available.')}",
                ],
            }
        ],
        raw_available=False,
        summary="Compact Bluetooth/audio diagnostics snapshot.",
        confidence=1.0,
        next_questions=[
            "Try reconnecting the Bluetooth device after suspend.",
            "Capture a full doctor bundle if you want deeper analysis.",
        ],
    )
    cards_payload["show_confidence"] = False
    return cards_payload


def _render_diagnostics_plain_text(snapshot: dict[str, Any]) -> str:
    os_payload = snapshot.get("os") if isinstance(snapshot.get("os"), dict) else {}
    network_payload = snapshot.get("network") if isinstance(snapshot.get("network"), dict) else {}
    network_summary = network_payload.get("system_summary") if isinstance(network_payload.get("system_summary"), dict) else {}
    nmcli_rows = network_payload.get("nmcli_rows") if isinstance(network_payload.get("nmcli_rows"), list) else []
    suspend_resume = snapshot.get("suspend_resume") if isinstance(snapshot.get("suspend_resume"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    lines = [
        "Diagnostics snapshot",
        f"OS/kernel: {str(os_payload.get('text') or 'unavailable')}",
        "Network: {state}; up={up}; default_route={default_route}; dns={dns}".format(
            state=str(network_summary.get("state") or "unknown"),
            up=",".join(str(item) for item in (network_summary.get("up_interfaces") or [])) or "none",
            default_route="yes" if bool(network_summary.get("default_route", False)) else "no",
            dns="yes" if bool(network_summary.get("dns_configured", False)) else "no",
        ),
        f"nmcli rows: {len(nmcli_rows)}",
        f"Suspend/resume matches: {int(suspend_resume.get('match_count') or 0)}",
    ]
    if nmcli_rows:
        preview_rows = []
        for row in nmcli_rows[:3]:
            if not isinstance(row, dict):
                continue
            preview_rows.append(
                "{device} {state} {connection}".format(
                    device=str(row.get("device") or "").strip() or "?",
                    state=str(row.get("state") or "").strip() or "unknown",
                    connection=str(row.get("connection") or "").strip() or "none",
                ).strip()
            )
        if preview_rows:
            lines.append("nmcli preview: " + "; ".join(preview_rows))
    matches = suspend_resume.get("matches") if isinstance(suspend_resume.get("matches"), list) else []
    if matches:
        lines.append("Recent suspend/resume lines:")
        for line in matches[:3]:
            text = str(line).strip()
            if text:
                lines.append(f"- {text}")
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    if notes:
        lines.append("Notes: " + "; ".join(str(item) for item in notes if str(item).strip()))
    next_action = str(summary.get("next_action") or "").strip()
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def render_diagnostics_snapshot(snapshot: dict[str, Any]) -> str:
    cards_payload = build_diagnostics_cards_payload(snapshot)
    ok, _error = validate_cards_payload(cards_payload)
    _ = ok
    return _render_diagnostics_plain_text(snapshot)


def _render_bluetooth_audio_plain_text(snapshot: dict[str, Any]) -> str:
    bluetooth_payload = snapshot.get("bluetooth") if isinstance(snapshot.get("bluetooth"), dict) else {}
    service_payload = bluetooth_payload.get("service") if isinstance(bluetooth_payload.get("service"), dict) else {}
    controller_payload = bluetooth_payload.get("controller") if isinstance(bluetooth_payload.get("controller"), dict) else {}
    devices_payload = bluetooth_payload.get("devices") if isinstance(bluetooth_payload.get("devices"), dict) else {}
    logs_payload = bluetooth_payload.get("logs") if isinstance(bluetooth_payload.get("logs"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    device_rows = devices_payload.get("devices") if isinstance(devices_payload.get("devices"), list) else []
    lines = [
        "Bluetooth/audio diagnostics",
        "Service: active={active}; loaded={loaded}".format(
            active=str(service_payload.get("active_state") or "unknown"),
            loaded=str(service_payload.get("loaded_state") or "unknown"),
        ),
        "Controller: address={address}; powered={powered}; pairable={pairable}; discoverable={discoverable}".format(
            address=str(controller_payload.get("address") or "unknown"),
            powered="yes" if controller_payload.get("powered") is True else "no" if controller_payload.get("powered") is False else "unknown",
            pairable="yes" if controller_payload.get("pairable") is True else "no" if controller_payload.get("pairable") is False else "unknown",
            discoverable="yes" if controller_payload.get("discoverable") is True else "no" if controller_payload.get("discoverable") is False else "unknown",
        ),
        "Devices: paired={paired}; connected={connected}".format(
            paired=int(devices_payload.get("paired_count") or 0),
            connected=int(devices_payload.get("connected_count") or 0),
        ),
        f"Logs: matches={int(logs_payload.get('match_count') or 0)}",
    ]
    if device_rows:
        preview_rows = []
        for row in device_rows[:3]:
            if not isinstance(row, dict):
                continue
            state_bits: list[str] = []
            if row.get("connected") is True:
                state_bits.append("connected")
            elif row.get("connected") is False:
                state_bits.append("disconnected")
            if row.get("paired") is True:
                state_bits.append("paired")
            if row.get("trusted") is True:
                state_bits.append("trusted")
            preview_rows.append(
                "{name} ({address}) {state}".format(
                    name=str(row.get("name") or "").strip() or "?",
                    address=str(row.get("address") or "").strip() or "?",
                    state=" ".join(state_bits) if state_bits else "state unknown",
                ).strip()
            )
        if preview_rows:
            lines.append("Devices preview: " + "; ".join(preview_rows))
    matches = logs_payload.get("matches") if isinstance(logs_payload.get("matches"), list) else []
    if matches:
        lines.append("Recent Bluetooth log lines:")
        for line in matches[:3]:
            text = str(line).strip()
            if text:
                lines.append(f"- {text}")
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    if notes:
        lines.append("Notes: " + "; ".join(str(item) for item in notes if str(item).strip()))
    next_action = str(summary.get("next_action") or "").strip()
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def render_bluetooth_audio_diagnostics_snapshot(snapshot: dict[str, Any]) -> str:
    cards_payload = build_bluetooth_audio_diagnostics_cards_payload(snapshot)
    ok, _error = validate_cards_payload(cards_payload)
    _ = ok
    return _render_bluetooth_audio_plain_text(snapshot)


def build_storage_disk_diagnostics_cards_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    storage_payload = snapshot.get("storage") if isinstance(snapshot.get("storage"), dict) else {}
    filesystems = storage_payload.get("filesystems") if isinstance(storage_payload.get("filesystems"), dict) else {}
    consumers = storage_payload.get("consumers") if isinstance(storage_payload.get("consumers"), dict) else {}
    logs_payload = storage_payload.get("logs") if isinstance(storage_payload.get("logs"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    fs_rows = filesystems.get("rows") if isinstance(filesystems.get("rows"), list) else []
    consumer_rows = consumers.get("entries") if isinstance(consumers.get("entries"), list) else []
    fs_preview: list[str] = []
    for row in fs_rows[:3]:
        if not isinstance(row, dict):
            continue
        fs_preview.append(
            "{mount} {used}% ({used_size}/{size})".format(
                mount=str(row.get("mountpoint") or "?"),
                used=f"{float(row.get('used_pct') or 0.0):.1f}",
                used_size=str(row.get("used") or "unknown"),
                size=str(row.get("size") or "unknown"),
            )
        )
    consumer_preview: list[str] = []
    for row in consumer_rows[:3]:
        if not isinstance(row, dict):
            continue
        consumer_preview.append(
            "{path} ({size})".format(
                path=str(row.get("path") or "").strip() or "?",
                size=_storage_format_bytes(int(row.get("size_bytes") or 0)),
            )
        )
    log_lines = [str(item).strip() for item in (logs_payload.get("matches") if isinstance(logs_payload.get("matches"), list) else [])[:3] if str(item).strip()]
    assessment = str(summary.get("assessment") or summary.get("notes") or "").strip()
    if not assessment:
        assessment = "No obvious storage failure was visible in this compact snapshot."
    cards_payload = build_cards_payload(
        [
            {
                "key": "storage-disk-diagnostics",
                "title": "Storage/disk diagnostics",
                "severity": str(summary.get("status") or "ok"),
                "lines": [
                    "Filesystem: " + ("; ".join(fs_preview) or "none"),
                    "Mount/device: "
                    + (
                        "; ".join(
                            "{device} -> {mount} ({fstype})".format(
                                device=str(row.get("device") or "").strip() or "?",
                                mount=str(row.get("mountpoint") or "").strip() or "?",
                                fstype=str(row.get("fstype") or "").strip() or "unknown",
                            )
                            for row in fs_rows[:3]
                            if isinstance(row, dict)
                        )
                        or "none"
                    ),
                    "Consumers: " + ("; ".join(consumer_preview) or "none"),
                    f"Logs: matches={int(logs_payload.get('match_count') or 0)}"
                    + (f"; preview={'; '.join(log_lines)}" if log_lines else ""),
                    f"Assessment: {assessment}",
                    f"Next action: {str(summary.get('next_action') or 'No next action available.')}",
                ],
            }
        ],
        raw_available=False,
        summary="Compact storage/disk diagnostics snapshot.",
        confidence=1.0,
        next_questions=[
            "Free space on the fullest filesystem and retry the save.",
            "Capture a full doctor bundle if you want deeper analysis.",
        ],
    )
    cards_payload["show_confidence"] = False
    return cards_payload


def _render_storage_disk_plain_text(snapshot: dict[str, Any]) -> str:
    storage_payload = snapshot.get("storage") if isinstance(snapshot.get("storage"), dict) else {}
    filesystems = storage_payload.get("filesystems") if isinstance(storage_payload.get("filesystems"), dict) else {}
    consumers = storage_payload.get("consumers") if isinstance(storage_payload.get("consumers"), dict) else {}
    logs_payload = storage_payload.get("logs") if isinstance(storage_payload.get("logs"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    fs_rows = filesystems.get("rows") if isinstance(filesystems.get("rows"), list) else []
    consumer_rows = consumers.get("entries") if isinstance(consumers.get("entries"), list) else []
    lines = [
        "Storage/disk diagnostics",
        "Filesystem: "
        + (
            "; ".join(
                "{mount} {used_pct:.1f}% used ({used}/{size})".format(
                    mount=str(row.get("mountpoint") or "?"),
                    used_pct=float(row.get("used_pct") or 0.0),
                    used=str(row.get("used") or "unknown"),
                    size=str(row.get("size") or "unknown"),
                )
                for row in fs_rows[:3]
                if isinstance(row, dict)
            )
            or "none"
        ),
        "Mount/device: "
        + (
            "; ".join(
                "{device} -> {mount} ({fstype})".format(
                    device=str(row.get("device") or "").strip() or "?",
                    mount=str(row.get("mountpoint") or "").strip() or "?",
                    fstype=str(row.get("fstype") or "").strip() or "unknown",
                )
                for row in fs_rows[:3]
                if isinstance(row, dict)
            )
            or "none"
        ),
        "Consumers: "
        + (
            "; ".join(
                "{path} ({size})".format(
                    path=str(row.get("path") or "").strip() or "?",
                    size=_storage_format_bytes(int(row.get("size_bytes") or 0)),
                )
                for row in consumer_rows[:3]
                if isinstance(row, dict)
            )
            or "none"
        ),
        f"Log matches: {int(logs_payload.get('match_count') or 0)}",
    ]
    matches = logs_payload.get("matches") if isinstance(logs_payload.get("matches"), list) else []
    if matches:
        lines.append("Recent log lines:")
        for line in matches[:3]:
            text = str(line).strip()
            if text:
                lines.append(f"- {text}")
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    if notes:
        lines.append("Notes: " + "; ".join(str(item) for item in notes if str(item).strip()))
    next_action = str(summary.get("next_action") or "").strip()
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def render_storage_disk_diagnostics_snapshot(snapshot: dict[str, Any]) -> str:
    cards_payload = build_storage_disk_diagnostics_cards_payload(snapshot)
    ok, _error = validate_cards_payload(cards_payload)
    _ = ok
    return _render_storage_disk_plain_text(snapshot)


def build_printer_cups_diagnostics_cards_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    printer_payload = snapshot.get("printer") if isinstance(snapshot.get("printer"), dict) else {}
    service_payload = printer_payload.get("service") if isinstance(printer_payload.get("service"), dict) else {}
    printers_payload = printer_payload.get("printers") if isinstance(printer_payload.get("printers"), dict) else {}
    jobs_payload = printer_payload.get("jobs") if isinstance(printer_payload.get("jobs"), dict) else {}
    logs_payload = printer_payload.get("logs") if isinstance(printer_payload.get("logs"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    printer_rows = printers_payload.get("rows") if isinstance(printers_payload.get("rows"), list) else []
    job_rows = jobs_payload.get("rows") if isinstance(jobs_payload.get("rows"), list) else []
    printer_preview: list[str] = []
    for row in printer_rows[:3]:
        if not isinstance(row, dict):
            continue
        bits = [str(row.get("state") or "unknown")]
        if row.get("enabled") is False:
            bits.append("disabled")
        elif row.get("enabled") is True:
            bits.append("enabled")
        if row.get("accepting") is False:
            bits.append("not accepting")
        elif row.get("accepting") is True:
            bits.append("accepting")
        detail = str(row.get("details") or "").strip()
        if detail:
            bits.append(detail[:80])
        printer_preview.append(f"{str(row.get('name') or '?')}: " + "; ".join(bits))
    job_preview: list[str] = []
    for row in job_rows[:3]:
        if not isinstance(row, dict):
            continue
        job_preview.append(str(row.get("text") or "").strip()[:120])
    log_lines = [str(item).strip() for item in (logs_payload.get("matches") if isinstance(logs_payload.get("matches"), list) else [])[:3] if str(item).strip()]
    assessment = str(summary.get("assessment") or summary.get("notes") or "").strip()
    if not assessment:
        assessment = "No obvious printer/CUPS failure was visible in this compact snapshot."
    cards_payload = build_cards_payload(
        [
            {
                "key": "printer-cups-diagnostics",
                "title": "Printer/CUPS diagnostics",
                "severity": str(summary.get("status") or "ok"),
                "lines": [
                    "Service: "
                    + "; ".join(
                        [
                            f"active={str(service_payload.get('active_state') or 'unknown')}",
                            f"loaded={str(service_payload.get('loaded_state') or 'unknown')}",
                        ]
                    ),
                    "Printers: "
                    + (
                        f"default={str(printers_payload.get('default_printer') or 'none')}; preview="
                        + ("; ".join(printer_preview) if printer_preview else "none")
                    ),
                    "Jobs: "
                    + (
                        f"count={int(jobs_payload.get('match_count') or 0)}"
                        + (f"; preview={'; '.join(job_preview)}" if job_preview else "")
                    ),
                    f"Logs: matches={int(logs_payload.get('match_count') or 0)}"
                    + (f"; preview={'; '.join(log_lines)}" if log_lines else ""),
                    f"Assessment: {assessment}",
                    f"Next action: {str(summary.get('next_action') or 'No next action available.')}",
                ],
            }
        ],
        raw_available=False,
        summary="Compact printer/CUPS diagnostics snapshot.",
        confidence=1.0,
        next_questions=[
            "Clear the queue and retry printing.",
            "Capture a full doctor bundle if you want deeper analysis.",
        ],
    )
    cards_payload["show_confidence"] = False
    return cards_payload


def _render_printer_cups_plain_text(snapshot: dict[str, Any]) -> str:
    printer_payload = snapshot.get("printer") if isinstance(snapshot.get("printer"), dict) else {}
    service_payload = printer_payload.get("service") if isinstance(printer_payload.get("service"), dict) else {}
    printers_payload = printer_payload.get("printers") if isinstance(printer_payload.get("printers"), dict) else {}
    jobs_payload = printer_payload.get("jobs") if isinstance(printer_payload.get("jobs"), dict) else {}
    logs_payload = printer_payload.get("logs") if isinstance(printer_payload.get("logs"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    printer_rows = printers_payload.get("rows") if isinstance(printers_payload.get("rows"), list) else []
    job_rows = jobs_payload.get("rows") if isinstance(jobs_payload.get("rows"), list) else []
    lines = [
        "Printer/CUPS diagnostics",
        f"Service: active={str(service_payload.get('active_state') or 'unknown')}; loaded={str(service_payload.get('loaded_state') or 'unknown')}",
        "Printers: default={default}; count={count}".format(
            default=str(printers_payload.get("default_printer") or "none"),
            count=int(printers_payload.get("printer_count") or len(printer_rows) or 0),
        ),
        "Jobs: count={count}".format(count=int(jobs_payload.get("match_count") or len(job_rows) or 0)),
        f"Log matches: {int(logs_payload.get('match_count') or 0)}",
    ]
    if printer_rows:
        preview_rows = []
        for row in printer_rows[:3]:
            if not isinstance(row, dict):
                continue
            bits = [str(row.get("state") or "unknown")]
            if row.get("enabled") is False:
                bits.append("disabled")
            elif row.get("enabled") is True:
                bits.append("enabled")
            if row.get("accepting") is False:
                bits.append("not accepting")
            elif row.get("accepting") is True:
                bits.append("accepting")
            detail = str(row.get("details") or "").strip()
            if detail:
                bits.append(detail[:80])
            preview_rows.append(f"{str(row.get('name') or '?')}: " + "; ".join(bits))
        if preview_rows:
            lines.append("Printer preview: " + "; ".join(preview_rows))
    if job_rows:
        preview_rows = []
        for row in job_rows[:3]:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if text:
                preview_rows.append(text[:120])
        if preview_rows:
            lines.append("Queue preview: " + "; ".join(preview_rows))
    matches = logs_payload.get("matches") if isinstance(logs_payload.get("matches"), list) else []
    if matches:
        lines.append("Recent CUPS log lines:")
        for line in matches[:3]:
            text = str(line).strip()
            if text:
                lines.append(f"- {text}")
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    if notes:
        lines.append("Notes: " + "; ".join(str(item) for item in notes if str(item).strip()))
    next_action = str(summary.get("next_action") or "").strip()
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def render_printer_cups_diagnostics_snapshot(snapshot: dict[str, Any]) -> str:
    cards_payload = build_printer_cups_diagnostics_cards_payload(snapshot)
    ok, _error = validate_cards_payload(cards_payload)
    _ = ok
    return _render_printer_cups_plain_text(snapshot)


def collect_generic_device_fallback_diagnostics_snapshot(
    *,
    run_command_fn: Callable[..., CommandResult] = run_command,
) -> dict[str, Any]:
    trace_id = _doctor_trace_id()
    generated_at = _safe_now_iso()

    uname_result = run_command_fn(["uname", "-a"], timeout_s=1.0)
    usb_result = run_command_fn(["lsusb"], timeout_s=1.0)
    pci_result = run_command_fn(["lspci", "-nn"], timeout_s=1.0)
    journal_result = run_command_fn(["journalctl", "-b", "-p", "warning", "--no-pager", "-n", "120"], timeout_s=2.0)
    dmesg_result = run_command_fn(["dmesg", "--ctime", "--level=err,warn"], timeout_s=1.0)

    usb_lines = _generic_device_snapshot_lines(usb_result)
    pci_lines = _generic_device_snapshot_lines(pci_result)
    journal_lines = _generic_device_log_lines(journal_result)
    dmesg_lines = _generic_device_log_lines(dmesg_result)
    log_text = " ".join(journal_lines + dmesg_lines).lower()
    issue_markers = ("fail", "error", "not found", "not detected", "not recognized", "timeout", "firmware", "disconnect", "reset", "blocked", "denied")

    notes: list[str] = []
    if usb_lines:
        notes.append(f"USB presence snapshot captured ({len(usb_lines)} line(s)).")
    else:
        notes.append("No USB presence snapshot was visible in the compact window.")
    if pci_lines:
        notes.append(f"PCI presence snapshot captured ({len(pci_lines)} line(s)).")
    if journal_lines:
        notes.append("Recent system journal warnings/errors were visible.")
    if dmesg_lines:
        notes.append("Recent kernel warnings/errors were visible in dmesg.")
    if any(token in log_text for token in issue_markers):
        notes.append("Recent system logs contain device or driver failure markers.")
    elif journal_lines or dmesg_lines:
        notes.append("Recent system logs are available in the compact snapshot.")

    if any(token in log_text for token in issue_markers):
        status = "warn"
        assessment = "; ".join(notes)
        next_action = "Retry the device action and check whether it appears after reconnecting or rebooting."
    elif not usb_lines and not pci_lines:
        status = "warn"
        assessment = "No obvious device-specific failure was visible in this compact snapshot."
        next_action = "If the issue persists, capture a full doctor bundle or share the exact device type."
    else:
        status = "ok"
        assessment = "; ".join(notes)
        next_action = "If the issue persists, capture a full doctor bundle or try the device again after reconnecting it."

    return {
        "trace_id": trace_id,
        "generated_at": generated_at,
        "preset": "generic_device_fallback",
        "device": {
            "os": {
                "available": _diagnostics_command_ok(uname_result),
                "source": "uname -a",
                "text": redact_secrets(uname_result.stdout).strip() or None,
                "error_kind": None
                if _diagnostics_command_ok(uname_result)
                else (
                    "permission_denied"
                    if uname_result.permission_denied
                    else "not_available"
                    if uname_result.not_available
                    else "command_failed"
                ),
            },
            "presence": {
                "usb": {
                    "available": _diagnostics_command_ok(usb_result),
                    "source": "lsusb",
                    "match_count": len(usb_lines),
                    "lines": usb_lines,
                    "error_kind": None
                    if _diagnostics_command_ok(usb_result)
                    else (
                        "permission_denied"
                        if usb_result.permission_denied
                        else "not_available"
                        if usb_result.not_available
                        else "command_failed"
                    ),
                },
                "pci": {
                    "available": _diagnostics_command_ok(pci_result),
                    "source": "lspci -nn",
                    "match_count": len(pci_lines),
                    "lines": pci_lines,
                    "error_kind": None
                    if _diagnostics_command_ok(pci_result)
                    else (
                        "permission_denied"
                        if pci_result.permission_denied
                        else "not_available"
                        if pci_result.not_available
                        else "command_failed"
                    ),
                },
            },
            "logs": {
                "journal": {
                    "available": _diagnostics_command_ok(journal_result),
                    "source": "journalctl -b -p warning",
                    "match_count": len(journal_lines),
                    "matches": journal_lines,
                    "error_kind": None
                    if _diagnostics_command_ok(journal_result)
                    else (
                        "permission_denied"
                        if journal_result.permission_denied
                        else "not_available"
                        if journal_result.not_available
                        else "command_failed"
                    ),
                },
                "dmesg": {
                    "available": _diagnostics_command_ok(dmesg_result),
                    "source": "dmesg --ctime --level=err,warn",
                    "match_count": len(dmesg_lines),
                    "matches": dmesg_lines,
                    "error_kind": None
                    if _diagnostics_command_ok(dmesg_result)
                    else (
                        "permission_denied"
                        if dmesg_result.permission_denied
                        else "not_available"
                        if dmesg_result.not_available
                        else "command_failed"
                    ),
                },
            },
        },
        "summary": {
            "status": status,
            "assessment": assessment,
            "notes": notes,
            "next_action": next_action,
        },
    }


def build_generic_device_fallback_diagnostics_cards_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    device_payload = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    os_payload = device_payload.get("os") if isinstance(device_payload.get("os"), dict) else {}
    presence_payload = device_payload.get("presence") if isinstance(device_payload.get("presence"), dict) else {}
    usb_payload = presence_payload.get("usb") if isinstance(presence_payload.get("usb"), dict) else {}
    pci_payload = presence_payload.get("pci") if isinstance(presence_payload.get("pci"), dict) else {}
    logs_payload = device_payload.get("logs") if isinstance(device_payload.get("logs"), dict) else {}
    journal_payload = logs_payload.get("journal") if isinstance(logs_payload.get("journal"), dict) else {}
    dmesg_payload = logs_payload.get("dmesg") if isinstance(logs_payload.get("dmesg"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    usb_lines = usb_payload.get("lines") if isinstance(usb_payload.get("lines"), list) else []
    pci_lines = pci_payload.get("lines") if isinstance(pci_payload.get("lines"), list) else []
    journal_lines = journal_payload.get("matches") if isinstance(journal_payload.get("matches"), list) else []
    dmesg_lines = dmesg_payload.get("matches") if isinstance(dmesg_payload.get("matches"), list) else []
    assessment = str(summary.get("assessment") or summary.get("notes") or "").strip()
    if not assessment:
        assessment = "No obvious device-specific failure was visible in this compact snapshot."
    cards_payload = build_cards_payload(
        [
            {
                "key": "generic-device-fallback-diagnostics",
                "title": "General device diagnostics",
                "severity": str(summary.get("status") or "ok"),
                "lines": [
                    f"OS/kernel: {str(os_payload.get('text') or 'unavailable')}",
                    "USB presence: "
                    + (
                        f"matches={int(usb_payload.get('match_count') or 0)}"
                        + (f"; preview={'; '.join(str(item) for item in usb_lines[:3] if str(item).strip())}" if usb_lines else "")
                    ),
                    "PCI presence: "
                    + (
                        f"matches={int(pci_payload.get('match_count') or 0)}"
                        + (f"; preview={'; '.join(str(item) for item in pci_lines[:3] if str(item).strip())}" if pci_lines else "")
                    ),
                    "Logs: journal="
                    + str(int(journal_payload.get("match_count") or 0))
                    + "; dmesg="
                    + str(int(dmesg_payload.get("match_count") or 0))
                    + (
                        f"; journal preview={'; '.join(str(item) for item in journal_lines[:3] if str(item).strip())}"
                        if journal_lines
                        else ""
                    )
                    + (
                        f"; dmesg preview={'; '.join(str(item) for item in dmesg_lines[:3] if str(item).strip())}"
                        if dmesg_lines
                        else ""
                    ),
                    f"Assessment: {assessment}",
                    f"Next action: {str(summary.get('next_action') or 'No next action available.')}",
                ],
            }
        ],
        raw_available=False,
        summary="Compact general device diagnostics snapshot.",
        confidence=1.0,
        next_questions=[
            "Try the device again after reconnecting it.",
            "Capture a full doctor bundle if you want deeper analysis.",
        ],
    )
    cards_payload["show_confidence"] = False
    return cards_payload


def _render_generic_device_fallback_plain_text(snapshot: dict[str, Any]) -> str:
    device_payload = snapshot.get("device") if isinstance(snapshot.get("device"), dict) else {}
    os_payload = device_payload.get("os") if isinstance(device_payload.get("os"), dict) else {}
    presence_payload = device_payload.get("presence") if isinstance(device_payload.get("presence"), dict) else {}
    usb_payload = presence_payload.get("usb") if isinstance(presence_payload.get("usb"), dict) else {}
    pci_payload = presence_payload.get("pci") if isinstance(presence_payload.get("pci"), dict) else {}
    logs_payload = device_payload.get("logs") if isinstance(device_payload.get("logs"), dict) else {}
    journal_payload = logs_payload.get("journal") if isinstance(logs_payload.get("journal"), dict) else {}
    dmesg_payload = logs_payload.get("dmesg") if isinstance(logs_payload.get("dmesg"), dict) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), dict) else {}
    lines = [
        "General device diagnostics",
        f"OS/kernel: {str(os_payload.get('text') or 'unavailable')}",
        f"USB presence: matches={int(usb_payload.get('match_count') or 0)}",
        f"PCI presence: matches={int(pci_payload.get('match_count') or 0)}",
        f"Logs: journal={int(journal_payload.get('match_count') or 0)}; dmesg={int(dmesg_payload.get('match_count') or 0)}",
    ]
    usb_lines = usb_payload.get("lines") if isinstance(usb_payload.get("lines"), list) else []
    if usb_lines:
        lines.append("USB preview: " + "; ".join(str(item) for item in usb_lines[:3] if str(item).strip()))
    pci_lines = pci_payload.get("lines") if isinstance(pci_payload.get("lines"), list) else []
    if pci_lines:
        lines.append("PCI preview: " + "; ".join(str(item) for item in pci_lines[:3] if str(item).strip()))
    journal_lines = journal_payload.get("matches") if isinstance(journal_payload.get("matches"), list) else []
    if journal_lines:
        lines.append("Recent journal lines:")
        for line in journal_lines[:3]:
            text = str(line).strip()
            if text:
                lines.append(f"- {text}")
    dmesg_lines = dmesg_payload.get("matches") if isinstance(dmesg_payload.get("matches"), list) else []
    if dmesg_lines:
        lines.append("Recent dmesg lines:")
        for line in dmesg_lines[:3]:
            text = str(line).strip()
            if text:
                lines.append(f"- {text}")
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    if notes:
        lines.append("Notes: " + "; ".join(str(item) for item in notes if str(item).strip()))
    next_action = str(summary.get("next_action") or "").strip()
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def render_generic_device_fallback_diagnostics_snapshot(snapshot: dict[str, Any]) -> str:
    cards_payload = build_generic_device_fallback_diagnostics_cards_payload(snapshot)
    ok, _error = validate_cards_payload(cards_payload)
    _ = ok
    return _render_generic_device_fallback_plain_text(snapshot)


def _startup_diagnostics_payload() -> dict[str, Any]:
    config: Any | None
    try:
        config = load_config(require_telegram_token=False)
    except Exception as exc:
        return {
            "api": {
                "trace_id": None,
                "component": "api.startup",
                "status": "FAIL",
                "failure_code": "config_load_failed",
                "next_action": next_step_for_failure("config_load_failed"),
                "checks": [],
                "message": f"config load failed: {exc.__class__.__name__}",
            },
            "telegram": {
                "trace_id": None,
                "component": "telegram.startup",
                "status": "WARN",
                "failure_code": "config_load_failed",
                "next_action": next_step_for_failure("config_load_failed"),
                "checks": [],
                "message": f"config load failed: {exc.__class__.__name__}",
            },
        }
    token = ""
    try:
        store = SecretStore(path=_effective_secret_store_path())
        token = str(store.get_secret("telegram:bot_token") or "").strip() or str(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    except Exception:
        token = str(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    return {
        "api": run_startup_checks(service="api", config=config),
        "telegram": run_startup_checks(service="telegram", config=config, token=token or None),
    }


def _safe_support_bundle(
    report: DoctorReport,
    *,
    repo_root: Path,
    api_base_url: str,
) -> str | None:
    try:
        bundle_dir = Path(tempfile.mkdtemp(prefix="agent-support-"))
        bundle_path = bundle_dir / "doctor_support_bundle.json"
        summary_path = bundle_dir / "SUMMARY.txt"
        payload = {
            "trace_id": report.trace_id,
            "generated_at": report.generated_at,
            "summary_status": report.summary_status,
            "doctor_report": report.to_dict(),
            "startup_checks": _startup_diagnostics_payload(),
            "runtime_api": _diagnostics_runtime_payloads(api_base_url),
            "paths": _diagnostics_paths_manifest(repo_root),
            "recovery": _recovery_manifest(),
        }
        safe_payload = _redact_bundle_value(payload)
        bundle_path.write_text(json.dumps(safe_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        summary_path.write_text(_render_text_report(report) + "\n", encoding="utf-8")
        return str(bundle_dir)
    except Exception:
        return None


def _apply_safe_fixes(report: DoctorReport, *, repo_root: Path) -> list[str]:
    changes: list[str] = []
    changes.extend(_safe_fix_dirs())
    changes.extend(_safe_fix_telegram_dropin())
    changes.extend(_safe_fix_stale_locks())
    changes.extend(_safe_fix_runtime_storage_upgrade(repo_root))
    return sorted(changes)


def run_doctor_report(
    *,
    repo_root: str | None = None,
    online: bool = False,
    fix: bool = False,
    collect_diagnostics: bool = False,
    api_base_url: str = f"http://127.0.0.1:{runtime_port()}",
    now_epoch: int | None = None,
) -> DoctorReport:
    configure_logging_if_needed()
    root = Path(repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    trace_id = _doctor_trace_id(now_epoch=now_epoch)
    checks = _doctor_checks(repo_root=root, online=bool(online), api_base_url=api_base_url)
    for item in checks:
        _log_check(trace_id, item)
    status = _status_from_checks(checks)
    next_action = _first_next_action(checks)
    preliminary = DoctorReport(
        trace_id=trace_id,
        generated_at=_safe_now_iso(now_epoch=now_epoch),
        summary_status=status,
        checks=checks,
        next_action=next_action,
        fixes_applied=[],
        support_bundle_path=None,
    )
    if not fix:
        support_bundle_path = None
        if collect_diagnostics:
            support_bundle_path = _safe_support_bundle(preliminary, repo_root=root, api_base_url=api_base_url)
        return DoctorReport(
            trace_id=preliminary.trace_id,
            generated_at=preliminary.generated_at,
            summary_status=preliminary.summary_status,
            checks=preliminary.checks,
            next_action=preliminary.next_action,
            fixes_applied=preliminary.fixes_applied,
            support_bundle_path=support_bundle_path,
        )
    changes = _apply_safe_fixes(preliminary, repo_root=root)
    post_checks = _doctor_checks(repo_root=root, online=bool(online), api_base_url=api_base_url)
    for item in post_checks:
        _log_check(trace_id, item)
    final_report = DoctorReport(
        trace_id=trace_id,
        generated_at=_safe_now_iso(now_epoch=now_epoch),
        summary_status=_status_from_checks(post_checks),
        checks=post_checks,
        next_action=_first_next_action(post_checks),
        fixes_applied=changes,
        support_bundle_path=None,
    )
    if fix or collect_diagnostics:
        return DoctorReport(
            trace_id=final_report.trace_id,
            generated_at=final_report.generated_at,
            summary_status=final_report.summary_status,
            checks=final_report.checks,
            next_action=final_report.next_action,
            fixes_applied=final_report.fixes_applied,
            support_bundle_path=_safe_support_bundle(final_report, repo_root=root, api_base_url=api_base_url),
        )
    return final_report


def _render_text_report(report: DoctorReport) -> str:
    lines = [
        f"Trace ID: {report.trace_id}",
        f"Status: {report.summary_status}",
        "Checks:",
    ]
    for item in report.checks:
        lines.append(f"- [{item.status}] {item.check_id}: {item.detail_short}")
        if item.status in {"WARN", "FAIL"} and item.next_action:
            lines.append(f"  next: {item.next_action}")
    if report.summary_status in {"WARN", "FAIL"}:
        lines.append(f"Next action: {report.next_action or next_step_for_failure('llm_unavailable')}")
    else:
        lines.append("Next action: none")
    if report.fixes_applied:
        lines.append("Fixes applied:")
        for row in report.fixes_applied:
            lines.append(f"- {row}")
    if report.support_bundle_path:
        lines.append(f"Support bundle: {report.support_bundle_path}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging_if_needed()
    parser = argparse.ArgumentParser(description="Personal Agent doctor checks")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    parser.add_argument("--fix", action="store_true", help="apply deterministic safe fixes")
    parser.add_argument(
        "--collect-diagnostics",
        action="store_true",
        help="write one redacted local diagnostics bundle and print its path",
    )
    parser.add_argument("--online", action="store_true", help="allow online Telegram token validation")
    parser.add_argument("--api-base-url", default=f"http://127.0.0.1:{runtime_port()}", help="local API base URL")
    parser.add_argument("--repo-root", default=None, help="override repo root")
    args = parser.parse_args(argv)

    report = run_doctor_report(
        repo_root=args.repo_root,
        online=bool(args.online),
        fix=bool(args.fix),
        collect_diagnostics=bool(args.collect_diagnostics),
        api_base_url=str(args.api_base_url),
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=True, sort_keys=True, indent=2))
    else:
        print(_render_text_report(report))

    return 1 if report.summary_status == "FAIL" else 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
