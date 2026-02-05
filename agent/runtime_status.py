from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from agent.diagnostics import CommandResult, redact_secrets, run_command
from memory.db import MemoryDB


def _parse_systemctl_show(output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _service_state_text(result: CommandResult) -> tuple[str, str | None, str]:
    if result.permission_denied:
        return "not available (permission)", None, "not available (permission)"
    if result.not_available:
        return "not available", None, "not available"
    if result.error:
        return "not available", None, "not available"
    info = _parse_systemctl_show(result.stdout)
    active_state = info.get("ActiveState") or "unknown"
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
    return active_state, main_pid, last_exit


def _process_list(result: CommandResult) -> tuple[list[dict[str, str]], bool, str | None]:
    if result.permission_denied:
        return [], False, "not available (permission)"
    if result.not_available or result.error:
        return [], False, "not available"
    processes: list[dict[str, str]] = []
    orphan_found = False
    lines = result.stdout.splitlines()
    for line in lines[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        if "telegram_adapter" not in cmd:
            continue
        if len(cmd) > 120:
            cmd = cmd[:117] + "..."
        processes.append({"pid": pid, "ppid": ppid, "cmd": cmd})
        if ppid == "1":
            orphan_found = True
    return processes, orphan_found, None


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


def build_runtime_status_report(
    db: MemoryDB,
    run_command_fn: Callable[[list[str], float], CommandResult] | None = None,
    now: datetime | None = None,
) -> str:
    runner = run_command_fn or run_command
    now_dt = now or datetime.now(timezone.utc)

    svc_result = runner(
        [
            "systemctl",
            "show",
            "personal-agent",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
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
    service_state, main_pid, last_exit = _service_state_text(svc_result)

    ps_result = runner(["ps", "-eo", "pid,ppid,args"], 2.0)
    processes, orphan_found, ps_note = _process_list(ps_result)

    journal_result = runner(
        ["journalctl", "-u", "personal-agent", "-n", "50", "--no-pager"],
        2.0,
    )
    journal_lines, journal_note = _journal_lines(journal_result)

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

    conflicts: list[str] = []
    if service_state in {"inactive", "failed"} and processes:
        conflicts.append("service inactive but telegram_adapter process running")

    lines: list[str] = []
    lines.append("1. Service State")
    lines.append(f"- status: {service_state}")
    lines.append(f"- main_pid: {main_pid or 'not available'}")
    lines.append(f"- last_exit: {last_exit}")
    lines.append("2. Process State")
    if ps_note:
        lines.append(f"- {ps_note}")
    elif not processes:
        lines.append("- none found")
    else:
        for proc in processes:
            lines.append(f"- pid={proc['pid']} ppid={proc['ppid']} cmd={proc['cmd']}")
        if orphan_found:
            lines.append("- orphan detected: ppid=1")
    lines.append("3. Recent Logs")
    if journal_note:
        lines.append(f"- {journal_note}")
    else:
        lines.extend(journal_lines)
    lines.append("4. Database State")
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
    lines.append("5. Conflicts Detected")
    if conflicts:
        for item in conflicts:
            lines.append(f"- {item}")
    else:
        lines.append("- none detected")
    return redact_secrets("\n".join(lines))
