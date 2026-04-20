from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from agent.config import runtime_service_name
from agent.diagnostics import CommandResult, redact_secrets, run_command
from memory.db import MemoryDB


_SERVICE_UNITS = (
    runtime_service_name(),
    "personal-agent-telegram.service",
)


def _parse_systemctl_show(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _service_state_text(result: CommandResult) -> tuple[str, str | None, str | None, str]:
    if result.permission_denied:
        return "not available (permission)", None, None, "not available (permission)"
    if result.not_available:
        return "not available", None, None, "not available"
    if result.error:
        return "not available", None, None, "not available"
    info = _parse_systemctl_show(result.stdout)
    active_state = info.get("ActiveState") or "unknown"
    enabled_state = info.get("UnitFileState") or None
    main_pid = info.get("MainPID")
    if not main_pid or main_pid == "0":
        main_pid = None
    result_code = info.get("Result")
    exec_code = info.get("ExecMainCode")
    exec_status = info.get("ExecMainStatus")
    exit_ts = info.get("ExecMainExitTimestamp")
    last_exit = "not available"
    parts = []
    if result_code:
        parts.append(f"result={result_code}")
    if exec_code:
        parts.append(f"code={exec_code}")
    if exec_status:
        parts.append(f"status={exec_status}")
    if exit_ts and exit_ts != "n/a":
        parts.append(f"exited_at={exit_ts}")
    if parts:
        last_exit = ", ".join(parts)
    return active_state, enabled_state, main_pid, last_exit


def _journal_lines(result: CommandResult) -> tuple[list[str], str | None]:
    if result.permission_denied:
        return [], "not available (permission)"
    if result.not_available or result.error:
        return [], "not available"
    output = redact_secrets(result.stdout)
    lines = output.splitlines()
    if not lines:
        return ["(no entries)"], None
    return lines[-50:], None


def _table_exists(db: MemoryDB, table: str) -> bool:
    cur = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    )
    return cur.fetchone() is not None


def _service_snapshot(unit_name: str, runner: Callable[[list[str], float], CommandResult]) -> dict[str, str | None]:
    result = runner(
        [
            "systemctl",
            "--user",
            "show",
            unit_name,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "UnitFileState",
            "-p",
            "MainPID",
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
            "-p",
            "ExecMainCode",
            "-p",
            "ExecMainExitTimestamp",
        ],
        2.0,
    )
    active_state, enabled_state, main_pid, last_exit = _service_state_text(result)
    return {
        "unit": unit_name,
        "active_state": active_state,
        "enabled_state": enabled_state,
        "main_pid": main_pid,
        "last_exit": last_exit,
    }


def build_runtime_status_report(
    db: MemoryDB,
    run_command_fn: Callable[[list[str], float], CommandResult] | None = None,
    now: datetime | None = None,
) -> str:
    runner = run_command_fn or run_command
    now_dt = now or datetime.now(timezone.utc)

    services = [_service_snapshot(unit_name, runner) for unit_name in _SERVICE_UNITS]
    service_logs = {
        unit_name: _journal_lines(
            runner(
                ["journalctl", "--user", "-u", unit_name, "-n", "25", "--no-pager"],
                2.0,
            )
        )
        for unit_name in _SERVICE_UNITS
    }

    reminders_status = "reminders table not found"
    reminders_counts: dict[str, int] | None = None
    if _table_exists(db, "reminders"):
        cur = db._conn.execute("SELECT COUNT(*) AS total FROM reminders")
        total = int(cur.fetchone()["total"])
        def _count(status: str) -> int:
            cur = db._conn.execute(
                "SELECT COUNT(*) AS count FROM reminders WHERE status = ?",
                (status,),
            )
            return int(cur.fetchone()["count"])
        reminders_counts = {
            "total": total,
            "pending": _count("pending"),
            "sent": _count("sent"),
            "failed": _count("failed"),
        }
        reminders_status = "ok"

    audit_count_text = "not available"
    if _table_exists(db, "audit_log"):
        start_ts = (now_dt - timedelta(hours=24)).isoformat()
        cur = db._conn.execute(
            "SELECT COUNT(*) AS count FROM audit_log WHERE created_at >= ?",
            (start_ts,),
        )
        audit_count = int(cur.fetchone()["count"])
        audit_count_text = str(audit_count)

    notes: list[str] = []
    api_service = next((row for row in services if row["unit"] == runtime_service_name()), None)
    telegram_service = next((row for row in services if row["unit"] == "personal-agent-telegram.service"), None)
    api_state = str((api_service or {}).get("active_state") or "unknown")
    telegram_state = str((telegram_service or {}).get("active_state") or "unknown")
    if api_state != "active":
        notes.append("API service is not active.")
    if telegram_state == "failed":
        notes.append("Telegram service failed.")
    elif telegram_state == "active" and api_state != "active":
        notes.append("Telegram service is active while the API service is not active.")

    lines: list[str] = []
    lines.append("1. Service State")
    for service in services:
        lines.append(
            "- {unit}: status={status} enabled={enabled} main_pid={main_pid} last_exit={last_exit}".format(
                unit=str(service.get("unit") or ""),
                status=str(service.get("active_state") or "unknown"),
                enabled=str(service.get("enabled_state") or "not available"),
                main_pid=str(service.get("main_pid") or "not available"),
                last_exit=str(service.get("last_exit") or "not available"),
            )
        )
    lines.append("2. Recent Logs")
    for unit_name in _SERVICE_UNITS:
        log_lines, log_note = service_logs.get(unit_name, ([], "not available"))
        lines.append(f"- {unit_name}")
        if log_note:
            lines.append(f"  {log_note}")
        else:
            lines.extend(f"  {entry}" for entry in log_lines)
    lines.append("3. Database State")
    lines.append(f"- db_path: {db.db_path}")
    if reminders_status == "ok" and reminders_counts:
        lines.append(
            "- reminders: total={total}, pending={pending}, sent={sent}, failed={failed}".format(
                **reminders_counts
            )
        )
    else:
        lines.append("- reminders: reminders table not found")
    lines.append(f"- audit_log last_24h: {audit_count_text}")
    lines.append("4. Notes")
    if notes:
        for item in notes:
            lines.append(f"- {item}")
    else:
        lines.append("- none detected")
    return redact_secrets("\n".join(lines))
