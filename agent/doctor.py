from __future__ import annotations

import os
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from memory.db import MemoryDB
from skills.observe_now.handler import observe_now


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


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
