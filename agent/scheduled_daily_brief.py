from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request
from zoneinfo import ZoneInfo

from agent.daily_brief import should_send_daily_brief
from memory.db import MemoryDB


_TRUE_VALUES = {"on", "true", "1", "yes"}


def _log(message: str) -> None:
    print(f"[scheduled_daily_brief] {message}", flush=True)


def _is_enabled(value: str | None, default: str = "off") -> bool:
    return (value or default).strip().lower() in _TRUE_VALUES


def _safe_int(value: object, fallback: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return fallback


def _parse_iso_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_due_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = text[:10]
    try:
        return date.fromisoformat(candidate)
    except Exception:
        return None


def _format_due(value: object) -> str:
    parsed = _parse_due_date(value)
    return parsed.isoformat() if parsed is not None else "none"


def _format_task_line(row: dict[str, object]) -> str:
    return (
        f"- [{_safe_int(row.get('id'), 0)}] {str(row.get('title') or '').strip()} "
        f"(due {_format_due(row.get('due_date'))} | "
        f"impact {_safe_int(row.get('impact_1to5'), -1) if row.get('impact_1to5') is not None else '?'} | "
        f"effort {_safe_int(row.get('effort_mins'), -1) if row.get('effort_mins') is not None else '?'}m)"
    )


def _format_open_loop_line(row: dict[str, object]) -> str:
    return (
        f"- [{_safe_int(row.get('id'), 0)}] {str(row.get('title') or '').strip()} "
        f"(due {_format_due(row.get('due_date'))} | priority {_safe_int(row.get('priority'), 0)})"
    )


def _sort_tasks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    max_due = date.max
    max_created = datetime.max.replace(tzinfo=timezone.utc)

    def key(row: dict[str, object]) -> tuple[int, date, int, datetime, int]:
        due = _parse_due_date(row.get("due_date"))
        due_key = due or max_due
        due_null_rank = 1 if due is None else 0
        impact = _safe_int(row.get("impact_1to5"), -1)
        created = _parse_iso_dt(row.get("created_at")) or max_created
        row_id = _safe_int(row.get("id"), 2**31 - 1)
        return (due_null_rank, due_key, -impact, created, row_id)

    return sorted(rows, key=key)


def _sort_open_loops(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    max_due = date.max
    max_created = datetime.max.replace(tzinfo=timezone.utc)

    def key(row: dict[str, object]) -> tuple[int, date, int, datetime, int]:
        due = _parse_due_date(row.get("due_date"))
        due_key = due or max_due
        due_null_rank = 1 if due is None else 0
        priority = _safe_int(row.get("priority"), 0)
        created = _parse_iso_dt(row.get("created_at")) or max_created
        row_id = _safe_int(row.get("id"), 2**31 - 1)
        return (due_null_rank, due_key, -priority, created, row_id)

    return sorted(rows, key=key)


def _build_brief_markdown(
    local_date: date,
    open_tasks: list[dict[str, object]],
    due_soon_tasks: list[dict[str, object]],
    open_loops: list[dict[str, object]],
    nudge: str | None,
) -> str:
    lines: list[str] = [f"**Today:** {local_date.isoformat()}"]

    if open_tasks:
        lines.extend(["", "**Top tasks:**"])
        for row in open_tasks[:3]:
            lines.append(_format_task_line(row))

    if due_soon_tasks:
        lines.extend(["", "**Due soon:**"])
        for row in due_soon_tasks:
            lines.append(_format_task_line(row))

    if open_loops:
        lines.extend(["", "**Open loops:**"])
        for row in open_loops[:3]:
            lines.append(_format_open_loop_line(row))

    if nudge:
        lines.extend(["", f"**Nudge:** {nudge}"])

    return "\n".join(lines).strip()


def _choose_nudge(open_tasks: list[dict[str, object]], open_loops: list[dict[str, object]]) -> str | None:
    max_created = datetime.max.replace(tzinfo=timezone.utc)
    tasks_by_age = sorted(
        open_tasks,
        key=lambda row: (
            _parse_iso_dt(row.get("created_at")) or max_created,
            _safe_int(row.get("id"), 2**31 - 1),
        ),
    )
    loops_by_age = sorted(
        open_loops,
        key=lambda row: (
            _parse_iso_dt(row.get("created_at")) or max_created,
            _safe_int(row.get("id"), 2**31 - 1),
        ),
    )

    first_task = tasks_by_age[0] if tasks_by_age else None
    first_loop = loops_by_age[0] if loops_by_age else None

    if first_task is None and first_loop is None:
        return None
    if first_loop is None:
        return f"[{_safe_int(first_task.get('id'), 0)}] {str(first_task.get('title') or '').strip()}"
    if first_task is None:
        return f"[{_safe_int(first_loop.get('id'), 0)}] {str(first_loop.get('title') or '').strip()}"

    task_key = (
        _parse_iso_dt(first_task.get("created_at")) or max_created,
        _safe_int(first_task.get("id"), 2**31 - 1),
    )
    loop_key = (
        _parse_iso_dt(first_loop.get("created_at")) or max_created,
        _safe_int(first_loop.get("id"), 2**31 - 1),
    )

    if task_key <= loop_key:
        return f"[{_safe_int(first_task.get('id'), 0)}] {str(first_task.get('title') or '').strip()}"
    return f"[{_safe_int(first_loop.get('id'), 0)}] {str(first_loop.get('title') or '').strip()}"


def _send_telegram_message(token: str, chat_id: str, text: str) -> None:
    payload = url_parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
    ).encode("utf-8")
    req = url_request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        method="POST",
    )
    try:
        with url_request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except url_error.URLError as exc:
        raise RuntimeError(f"telegram send failed: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("telegram send failed: invalid API response") from exc
    if not bool(parsed.get("ok")):
        description = str(parsed.get("description") or "unknown error")
        raise RuntimeError(f"telegram send failed: {description}")


def run_once(now_utc: datetime | None = None) -> int:
    db: MemoryDB | None = None
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        db_path = os.getenv("AGENT_DB_PATH", os.path.join(repo_root, "memory", "agent.db"))
        timezone_name = os.getenv("AGENT_TIMEZONE", "UTC")
        schema_path = os.path.join(repo_root, "memory", "schema.sql")

        db = MemoryDB(db_path)
        db.init_schema(schema_path)

        chat_id = db.get_preference("telegram_chat_id")
        if not chat_id:
            _log("skip: telegram_chat_id not configured")
            return 0

        enabled = _is_enabled(db.get_preference("daily_brief_enabled"), default="off")
        local_time = (db.get_preference("daily_brief_time") or "09:00").strip()
        quiet_mode = _is_enabled(db.get_preference("daily_brief_quiet_mode"), default="off")
        threshold_pref = db.get_preference("disk_delta_threshold_mb")
        disk_delta_threshold_mb = float(threshold_pref) if (threshold_pref and threshold_pref.isdigit()) else 250.0
        svc_gate = _is_enabled(db.get_preference("only_send_if_service_unhealthy"), default="off")
        include_due_pref = db.get_preference("include_open_loops_due_within_days")
        include_due_days = int(include_due_pref) if (include_due_pref and include_due_pref.isdigit()) else 3
        last_sent = db.get_preference("daily_brief_last_sent_date")

        current_now = now_utc or datetime.now(timezone.utc)
        local_now = current_now.astimezone(ZoneInfo(timezone_name))
        local_date = local_now.date()

        base_decision = should_send_daily_brief(
            now_utc=current_now,
            timezone_name=timezone_name,
            enabled=enabled,
            local_time_hhmm=local_time,
            last_sent_local_date=last_sent,
            quiet_mode=False,
            disk_delta_mb=None,
            disk_delta_threshold_mb=disk_delta_threshold_mb,
            service_unhealthy=False,
            only_send_if_service_unhealthy=False,
            has_due_open_loops=False,
        )
        if base_decision.reason in {"disabled", "invalid_time", "before_time", "already_sent_today"}:
            _log(f"skip: reason={base_decision.reason} local_date={base_decision.local_date}")
            return 0

        task_rows = [
            row
            for row in db.list_tasks(limit=500)
            if str(row.get("status") or "").strip().lower() in {"todo", "doing"}
        ]
        open_tasks = _sort_tasks(task_rows)

        due_soon_tasks = []
        for row in open_tasks:
            due = _parse_due_date(row.get("due_date"))
            if due is None:
                continue
            delta_days = (due - local_date).days
            if 0 <= delta_days <= include_due_days:
                due_soon_tasks.append(row)

        open_loop_rows = db.list_open_loops(status="open", limit=500, order="created")
        open_loops = _sort_open_loops(open_loop_rows)

        due_open_loops_count = 0
        for row in open_loops:
            due = _parse_due_date(row.get("due_date"))
            if due is None:
                continue
            delta_days = (due - local_date).days
            if 0 <= delta_days <= include_due_days:
                due_open_loops_count += 1

        decision = should_send_daily_brief(
            now_utc=current_now,
            timezone_name=timezone_name,
            enabled=enabled,
            local_time_hhmm=local_time,
            last_sent_local_date=last_sent,
            quiet_mode=quiet_mode,
            disk_delta_mb=None,
            disk_delta_threshold_mb=disk_delta_threshold_mb,
            service_unhealthy=False,
            only_send_if_service_unhealthy=svc_gate,
            has_due_open_loops=due_open_loops_count > 0,
        )
        if not decision.should_send:
            _log(f"skip: reason={decision.reason} local_date={decision.local_date}")
            return 0

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            _log("error: TELEGRAM_BOT_TOKEN is required when daily brief is due")
            return 1

        nudge = _choose_nudge(open_tasks, open_loops)
        text = _build_brief_markdown(local_date, open_tasks, due_soon_tasks, open_loops, nudge)

        for attempt in (1, 2):
            try:
                _send_telegram_message(token, str(chat_id), text)
                db.set_preference("daily_brief_last_sent_date", decision.local_date)
                _log(f"sent: local_date={decision.local_date}")
                return 0
            except Exception as exc:
                if attempt == 1:
                    _log(f"send attempt 1 failed: {exc}; retrying")
                    time.sleep(1.0)
                    continue
                _log(f"send failed: {exc}")
                return 1
        return 1
    except Exception as exc:
        _log(f"error: {exc}")
        return 1
    finally:
        if db is not None:
            db.close()


def main() -> None:
    raise SystemExit(run_once())


if __name__ == "__main__":
    main()
