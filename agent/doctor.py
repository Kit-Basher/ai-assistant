from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
from skills.observe_now.handler import observe_now
from agent.audit_log import redact
from agent.golden_path import next_step_for_failure
from agent.secret_store import SecretStore


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
    return os.path.join(repo_root, "memory", "agent.db")


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
    logger = logging.getLogger("agent.doctor")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


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


def _effective_secret_store_path() -> str:
    configured = os.getenv("AGENT_SECRET_STORE_PATH", "").strip()
    if configured:
        return configured
    return str(Path.home() / ".local" / "share" / "personal-agent" / "secrets.enc.json")


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
        _ = store.get_secret("telegram:bot_token")
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
        Path.home() / ".local" / "share" / "personal-agent",
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
    status_ok, payload_or_error = _api_get_json(f"{api_base.rstrip('/')}/llm/status", timeout_seconds=0.8)
    if not status_ok:
        return DoctorCheck(
            check_id="llm.availability",
            status="WARN",
            detail_short=f"/llm/status unavailable: {payload_or_error}",
            next_action="Run: systemctl --user restart personal-agent-api.service",
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
    return DoctorCheck(check_id="logging.stdout", status="OK", detail_short=f"handlers={len(handlers)}")


def _doctor_checks(
    *,
    repo_root: Path,
    online: bool,
    api_base_url: str,
) -> list[DoctorCheck]:
    checks = [
        _check_python_runtime(),
        _check_repo_readable(repo_root),
        _check_secret_store_path(),
        _check_required_dirs(),
        _check_telegram_dropin(),
        _check_write_mode_safe(),
        _check_systemd_service("personal-agent-api.service", "systemd.api_service"),
        _check_systemd_service("personal-agent-telegram.service", "systemd.telegram_service"),
        _check_telegram_poller_singleton(),
        _check_llm_availability(api_base_url),
        _check_telegram_token(online=online),
        _check_logging_to_stdout(),
    ]
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
        Path.home() / ".local" / "share" / "personal-agent",
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
    root = Path.home() / ".local" / "share" / "personal-agent"
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


def _safe_support_bundle(report: DoctorReport) -> str | None:
    try:
        bundle_dir = Path(tempfile.mkdtemp(prefix="agent-support-"))
        bundle_path = bundle_dir / "doctor_support_bundle.json"
        payload = {
            "trace_id": report.trace_id,
            "generated_at": report.generated_at,
            "summary_status": report.summary_status,
            "checks": [item.to_dict() for item in report.checks],
            "next_action": report.next_action,
        }
        safe_payload = redact(payload)
        bundle_path.write_text(json.dumps(safe_payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return str(bundle_dir)
    except Exception:
        return None


def _apply_safe_fixes(report: DoctorReport) -> tuple[list[str], str | None]:
    changes: list[str] = []
    changes.extend(_safe_fix_dirs())
    changes.extend(_safe_fix_telegram_dropin())
    changes.extend(_safe_fix_stale_locks())
    bundle_path = _safe_support_bundle(report)
    if bundle_path:
        changes.append(f"support_bundle:{bundle_path}")
    return sorted(changes), bundle_path


def run_doctor_report(
    *,
    repo_root: str | None = None,
    online: bool = False,
    fix: bool = False,
    api_base_url: str = "http://127.0.0.1:8765",
    now_epoch: int | None = None,
) -> DoctorReport:
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
        return preliminary
    changes, bundle_path = _apply_safe_fixes(preliminary)
    post_checks = _doctor_checks(repo_root=root, online=bool(online), api_base_url=api_base_url)
    for item in post_checks:
        _log_check(trace_id, item)
    return DoctorReport(
        trace_id=trace_id,
        generated_at=_safe_now_iso(now_epoch=now_epoch),
        summary_status=_status_from_checks(post_checks),
        checks=post_checks,
        next_action=_first_next_action(post_checks),
        fixes_applied=changes,
        support_bundle_path=bundle_path,
    )


def _render_text_report(report: DoctorReport) -> str:
    lines = [
        f"Trace ID: {report.trace_id}",
        f"Status: {report.summary_status}",
        "Checks:",
    ]
    for item in report.checks:
        lines.append(f"- [{item.status}] {item.check_id}: {item.detail_short}")
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
    parser = argparse.ArgumentParser(description="Personal Agent doctor checks")
    parser.add_argument("--json", action="store_true", help="emit JSON output")
    parser.add_argument("--fix", action="store_true", help="apply deterministic safe fixes")
    parser.add_argument("--online", action="store_true", help="allow online Telegram token validation")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8765", help="local API base URL")
    parser.add_argument("--repo-root", default=None, help="override repo root")
    args = parser.parse_args(argv)

    report = run_doctor_report(
        repo_root=args.repo_root,
        online=bool(args.online),
        fix=bool(args.fix),
        api_base_url=str(args.api_base_url),
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=True, sort_keys=True, indent=2))
    else:
        print(_render_text_report(report))

    return 1 if report.summary_status == "FAIL" else 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
