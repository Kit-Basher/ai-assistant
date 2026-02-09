from __future__ import annotations

import asyncio
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import os
import json
import uuid

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
except ModuleNotFoundError:  # pragma: no cover - testing without telegram installed
    Update = object  # type: ignore
    Application = object  # type: ignore
    CommandHandler = object  # type: ignore
    ContextTypes = object  # type: ignore
    MessageHandler = object  # type: ignore
    filters = object()  # type: ignore

from agent.config import load_config
from agent.llm_router import LLMRouter
from agent.llm_client import build_llm_broker
from agent.logging_utils import log_event
from agent.logging_utils import log_audit_event
from agent.scheduled_snapshots import (
    safe_run_scheduled_snapshot,
    safe_run_storage_snapshot,
    safe_run_resource_snapshot,
    safe_run_network_snapshot,
)
from agent.debug_protocol import DebugProtocol
from agent.orchestrator import Orchestrator
from agent.digest_runner import ensure_daily_digest_row, attempt_send_due_digests
from agent.daily_digest import build_daily_health_digest
from agent.journal import make_journal_line
from memory.db import MemoryDB
from agent.permissions import canonical_user_id, is_owner, role_for_user
from agent.settings_export import export_settings, dumps_export_json


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = (update.effective_message.text or "").strip()
    if not text:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]
    config = context.application.bot_data.get("config")

    platform_user_id = _platform_user_id_from_update(update)
    user_id = db.get_or_create_user_id("telegram", platform_user_id) or canonical_user_id("telegram", platform_user_id)

    # Remember the owner's chat id for reminders/jobs, but do not overwrite it based on non-owner chats.
    owner_id = getattr(config, "owner_user_id", None) if config else None
    if is_owner(user_id, owner_id):
        db.set_preference("telegram_chat_id", str(platform_user_id))

    # Route ALL non-command text through the orchestrator (single brain).
    response = orchestrator.handle_message(text, user_id=user_id)
    reply_text = response.text.strip() if response and response.text else ""
    if not reply_text:
        reply_text = "Sorry, I didn't catch that."
    await update.effective_message.reply_text(reply_text)

    log_event(log_path, "telegram_message", {"chat_id": chat_id, "user_id": user_id, "text": text})


def _command_payload(text: str, command: str) -> str:
    if not text:
        return ""
    parts = text.split(maxsplit=1)
    if not parts:
        return ""
    token = parts[0]
    if token.startswith(command):
        return parts[1] if len(parts) > 1 else ""
    return ""

def _platform_user_id_from_update(update: Update) -> str:
    u = getattr(update, "effective_user", None)
    if u is not None:
        try:
            uid = getattr(u, "id", None)
            if uid is not None:
                return str(uid)
        except Exception:
            pass
    # Fallback to chat id (works for private chats; keeps tests simple).
    if getattr(update, "effective_chat", None) is not None:
        return str(getattr(update.effective_chat, "id", ""))
    return ""


def _sender_user_id(update: Update, db: MemoryDB | None = None) -> str:
    pid = _platform_user_id_from_update(update)
    if db is not None and hasattr(db, "get_or_create_user_id"):
        try:
            uid = db.get_or_create_user_id("telegram", pid)
            if uid:
                return uid
        except Exception:
            pass
    return canonical_user_id("telegram", pid)


def _owner_allowed(user_id: str, config) -> bool:  # noqa: ANN001
    owner_id = getattr(config, "owner_user_id", None) or getattr(config, "digest_owner_user_id", None)
    return is_owner(user_id, owner_id)


def _digest_owner_allowed(chat_id: str, owner_user_id: str | None) -> bool:
    """
    Legacy helper (kept for backwards compatibility with older tests/callers).

    chat_id is a Telegram chat/user id string; owner_user_id may be raw or "tg:"-prefixed.
    """
    cid = str(chat_id or "").strip()
    if not cid:
        return False
    return is_owner(canonical_user_id("telegram", cid), owner_user_id)


def _chat_id_for_user_id(db: MemoryDB, user_id: str) -> str:
    # Prefer stored identity mapping; fallback to "tg:<id>" suffix.
    try:
        pid = db.get_platform_user_id("telegram", user_id)
        if pid:
            return str(pid)
    except Exception:
        pass
    s = str(user_id or "")
    return s.split(":", 1)[1] if s.startswith("tg:") and ":" in s else s


async def _handle_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/remind")
    prompt = f"remind me {content}".strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path: str = context.application.bot_data["log_path"]

    response = orchestrator.handle_message(prompt, user_id=user_id)
    await update.effective_message.reply_text(response.text)
    log_event(log_path, "telegram_command", {"user_id": user_id, "text": text, "forwarded": prompt})

async def _handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    response = orchestrator.handle_message("/status", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_runtime_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    response = orchestrator.handle_message("/runtime_status", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_disk_grow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/disk_grow")
    prompt = "/disk_grow {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    response = orchestrator.handle_message(prompt, user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    if not config or role_for_user(user_id, getattr(config, "owner_user_id", None)) != "admin":
        await update.effective_message.reply_text("Not authorized.")
        return

    response = orchestrator.handle_message("/audit", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_storage_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/storage_snapshot", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_storage_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/storage_report", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_resource_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/resource_report", user_id=user_id)
    await update.effective_message.reply_text(response.text)

async def _handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/brief", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_observe_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/observe_now", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_hardware_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/hardware_report", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_llm_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/llm_ping")
    prompt = "/llm_ping {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    if not config or role_for_user(user_id, getattr(config, "owner_user_id", None)) != "admin":
        await update.effective_message.reply_text("Not authorized.")
        return
    response = orchestrator.handle_message(prompt, user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_llm_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    if not config or role_for_user(user_id, getattr(config, "owner_user_id", None)) != "admin":
        await update.effective_message.reply_text("Not authorized.")
        return
    response = orchestrator.handle_message("/llm_status", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_llm_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    if not config or role_for_user(user_id, getattr(config, "owner_user_id", None)) != "admin":
        await update.effective_message.reply_text("Not authorized.")
        return
    response = orchestrator.handle_message("/llm_mode", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_network_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)

    response = orchestrator.handle_message("/network_report", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_weekly_reflection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    if not config or role_for_user(user_id, getattr(config, "owner_user_id", None)) != "admin":
        await update.effective_message.reply_text("Not authorized.")
        return
    response = orchestrator.handle_message("/weekly_reflection", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask")
    prompt = "/ask {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    response = orchestrator.handle_message(prompt, user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_ask_opinion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask_opinion")
    prompt = "/ask_opinion {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    response = orchestrator.handle_message(prompt, user_id=user_id)
    await update.effective_message.reply_text(response.text)

def _parse_hhmm(value: str, default: str) -> time:
    s = (value or "").strip() or default
    try:
        hh, mm = s.split(":", 1)
        return time(hour=int(hh), minute=int(mm))
    except Exception:
        hh, mm = default.split(":", 1)
        return time(hour=int(hh), minute=int(mm))


def _today_run_date(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).date().isoformat()

def _effective_pref(db: MemoryDB, user_id: str, key: str):
    try:
        prefs = db.get_preferences(user_id)
    except Exception:
        prefs = {}
    return prefs.get(key)


def _effective_bool(db: MemoryDB, user_id: str, key: str, default: bool) -> bool:
    v = _effective_pref(db, user_id, key)
    return bool(v) if isinstance(v, bool) else bool(default)


def _effective_str(db: MemoryDB, user_id: str, key: str, default: str) -> str:
    v = _effective_pref(db, user_id, key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return default


async def _handle_digest_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    kind = getattr(config, "digest_kind", "daily_health_digest")
    latest = db.get_latest_digest_run(user_id, kind)
    deferred = 0
    due = 0
    try:
        cur = db._conn.execute(
            "SELECT COUNT(*) AS c FROM digest_runs WHERE user_id = ? AND kind = ? AND sent = 0 AND deferred = 1",
            (user_id, kind),
        )
        deferred = int(cur.fetchone()["c"] or 0)
        cur2 = db._conn.execute(
            "SELECT COUNT(*) AS c FROM digest_runs WHERE user_id = ? AND kind = ? AND sent = 0",
            (user_id, kind),
        )
        due = int(cur2.fetchone()["c"] or 0)
    except Exception:
        deferred = deferred
        due = due

    enabled = bool(getattr(config, "digest_enabled", False))
    weekdays_only = bool(getattr(config, "digest_weekdays_only", False))
    quiet_start = str(getattr(config, "digest_quiet_hours_start", "09:00"))
    quiet_end = str(getattr(config, "digest_quiet_hours_end", "21:00"))
    digest_time = str(getattr(config, "digest_time_local", "09:00"))

    lines = [
        f"DIGEST_ENABLED={'1' if enabled else '0'}",
        f"TIMEZONE={tz_name}",
        f"RUN_TIME_LOCAL={digest_time}",
        f"QUIET_HOURS={quiet_start}-{quiet_end}",
        f"WEEKDAYS_ONLY={'1' if weekdays_only else '0'}",
    ]
    if latest:
        lines.append(f"LAST_RUN_DATE={latest.get('run_date') or ''}")
        lines.append(f"LAST_SENT={'1' if bool(latest.get('sent')) else '0'}")
        lines.append(f"LAST_SENT_AT={latest.get('sent_at') or ''}")
    else:
        lines.append("LAST_RUN_DATE=")
    lines.append(f"UNSENT_COUNT={due}")
    lines.append(f"DEFERRED_COUNT={deferred}")

    await update.effective_message.reply_text("\n".join(lines).strip())


def _digest_status_label(*, sent: bool, deferred: bool, notable: bool) -> str:
    if sent:
        return "sent"
    if deferred:
        return "deferred"
    if not notable:
        return "skipped"
    return "unsent"


def _digest_summary_from_message_text(message_text: str) -> str:
    text = (message_text or "").strip()
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            return s[2:].strip()
    # Fallback: first line.
    return text.splitlines()[0].strip()


def _format_digest_inbox_lines(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for row in rows or []:
        run_date = str(row.get("run_date") or "").strip()
        sent = bool(row.get("sent"))
        deferred = bool(row.get("deferred"))
        ms_json = str(row.get("machine_summary_json") or "")
        severity = "ok"
        notable = False
        try:
            ms = json.loads(ms_json or "{}")
        except Exception:
            ms = {}
        if isinstance(ms, dict):
            severity = str(ms.get("severity") or "ok")
            sig = ms.get("signals") if isinstance(ms.get("signals"), dict) else {}
            if isinstance(sig, dict):
                notable = bool(sig.get("notable"))
        status = _digest_status_label(sent=sent, deferred=deferred, notable=notable)
        summary = _digest_summary_from_message_text(str(row.get("message_text") or ""))
        if not summary:
            summary = "No message text."
        lines.append(f"{run_date} — {status} — {severity} — {summary}")
    return lines


async def _handle_digest_inbox(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/digest_inbox")
    try:
        n = int((content or "").strip() or 7)
    except Exception:
        n = 7
    n = max(1, min(n, 30))

    kind = getattr(config, "digest_kind", "daily_health_digest")
    rows = db.list_recent_digests(user_id, kind, limit=n)
    if not rows:
        await update.effective_message.reply_text("No digests yet.")
        return
    lines = _format_digest_inbox_lines(rows)
    await update.effective_message.reply_text("\n".join(lines).strip())


async def _handle_digest_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/digest_show").strip()
    kind = getattr(config, "digest_kind", "daily_health_digest")

    if content:
        row = db.get_digest_by_date(user_id, kind, content)
    else:
        row = db.get_latest_digest_run(user_id, kind)
    if not row:
        await update.effective_message.reply_text("No digest for that date.")
        return
    msg = str(row.get("message_text") or "").strip()
    if not msg:
        await update.effective_message.reply_text("No saved message text for that digest.")
        return
    await update.effective_message.reply_text(msg)


async def _handle_digest_deferred(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    kind = getattr(config, "digest_kind", "daily_health_digest")
    rows = db.list_deferred_digests(user_id, kind, limit=10)
    if not rows:
        await update.effective_message.reply_text("No deferred digests.")
        return
    lines = []
    for r in rows:
        run_date = str(r.get("run_date") or "")
        last_attempt = str(r.get("last_attempt_at") or "")
        lines.append(f"{run_date} — deferred (last_attempt_at={last_attempt})")
    await update.effective_message.reply_text("\n".join(lines).strip())


async def _handle_digest_resend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    run_date = _command_payload(text, "/digest_resend").strip()
    if not run_date:
        await update.effective_message.reply_text("Usage: /digest_resend YYYY-MM-DD")
        return

    kind = getattr(config, "digest_kind", "daily_health_digest")
    row = db.get_digest_by_date(user_id, kind, run_date)
    if not row:
        await update.effective_message.reply_text("No digest for that date.")
        return

    ms_json = str(row.get("machine_summary_json") or "")
    notable = False
    try:
        ms = json.loads(ms_json or "{}")
    except Exception:
        ms = {}
    if isinstance(ms, dict):
        sig = ms.get("signals") if isinstance(ms.get("signals"), dict) else {}
        if isinstance(sig, dict):
            notable = bool(sig.get("notable"))
    if not notable:
        await update.effective_message.reply_text("That digest was not notable; nothing to resend.")
        return

    msg = str(row.get("message_text") or "").strip()
    if not msg:
        await update.effective_message.reply_text("No saved message text for that digest.")
        return

    try:
        send_chat_id = _chat_id_for_user_id(db, user_id)
        await context.bot.send_message(chat_id=send_chat_id, text=msg)
        sent_at = datetime.now(timezone.utc).isoformat()
        db.mark_digest_sent(str(row.get("id") or ""), sent_at=sent_at)
        try:
            log_audit_event(
                context.application.bot_data.get("log_path") or "",
                event="digest_resent",
                user_id=user_id,
                snapshot_id=str(row.get("id") or ""),
                error="resent",
                probe="telegram_digest",
                target="telegram",
                severity="info",
            )
        except Exception:
            pass
        # Phase 10B: journal entry for resend.
        try:
            sev = str((ms.get("severity") if isinstance(ms, dict) else "") or "ok").strip().lower()
            if sev not in {"ok", "watch", "act_soon"}:
                sev = "ok"
            line = make_journal_line("digest_resent", ms if isinstance(ms, dict) else {})
            db.insert_journal_entry(
                id=str(uuid.uuid4()),
                user_id=user_id,
                kind="digest_resent",
                severity=sev,
                created_at=sent_at,
                line=line or f"Digest resent: {sev}",
                machine_summary_json=ms_json or "{}",
            )
        except Exception as excj:
            try:
                log_audit_event(
                    context.application.bot_data.get("log_path") or "",
                    event="journal_insert_failed",
                    user_id=user_id,
                    snapshot_id=str(row.get("id") or ""),
                    error=str(excj) or "insert_failed",
                    probe="change_journal",
                    target="db",
                    severity="warn",
                )
            except Exception:
                pass
        await update.effective_message.reply_text("Resent.")
    except Exception as exc:
        await update.effective_message.reply_text(f"Resend failed: {exc}")


async def _handle_digest_details(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    run_date = _command_payload(text, "/digest_details").strip()
    if not run_date:
        await update.effective_message.reply_text("Usage: /digest_details YYYY-MM-DD")
        return

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    kind = getattr(config, "digest_kind", "daily_health_digest")
    row = db.get_digest_by_date(user_id, kind, run_date)
    if not row:
        await update.effective_message.reply_text("No digest for that date.")
        return
    if not str(row.get("details_cache_key") or "").strip():
        await update.effective_message.reply_text("No cached details for that digest.")
        return

    # Rebuild details deterministically from the stored digest_runs row (DB-only).
    report = build_daily_health_digest(db, user_id, run_date, tz_name, kind=kind)
    details = (report.details_text or "").strip()
    if not details:
        await update.effective_message.reply_text("No cached details for that digest.")
        return
    await update.effective_message.reply_text("Full report:\n" + details)


async def _handle_digest_build_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    kind = getattr(config, "digest_kind", "daily_health_digest")
    run_date = _today_run_date(tz_name)

    res = ensure_daily_digest_row(db, user_id=user_id, kind=kind, run_date=run_date, timezone_name=tz_name, dry_run=False)
    await update.effective_message.reply_text(
        f"digest_id={res.digest_id} run_date={res.run_date} notable={'true' if res.notable else 'false'}"
    )


async def _handle_digest_send_due(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    kind = getattr(config, "digest_kind", "daily_health_digest")
    weekdays_only = _effective_bool(db, user_id, "DIGEST_WEEKDAYS_ONLY", bool(getattr(config, "digest_weekdays_only", False)))
    quiet_start_s = _effective_str(db, user_id, "DIGEST_QUIET_HOURS_START", str(getattr(config, "digest_quiet_hours_start", "09:00")))
    quiet_end_s = _effective_str(db, user_id, "DIGEST_QUIET_HOURS_END", str(getattr(config, "digest_quiet_hours_end", "21:00")))
    quiet_start = _parse_hhmm(quiet_start_s, "09:00")
    quiet_end = _parse_hhmm(quiet_end_s, "21:00")

    now_iso = datetime.now(timezone.utc).isoformat()
    due = attempt_send_due_digests(
        db,
        user_id=user_id,
        kind=kind,
        timezone_name=tz_name,
        now_iso=now_iso,
        weekdays_only=weekdays_only,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        dry_run=False,
        limit=5,
    )
    if not due:
        await update.effective_message.reply_text("No due digests.")
        return

    sent = 0
    for item in due:
        try:
            send_chat_id = _chat_id_for_user_id(db, user_id)
            await context.bot.send_message(chat_id=send_chat_id, text=item.text.strip())
            sent_at = datetime.now(timezone.utc).isoformat()
            db.mark_digest_sent(item.digest_id, sent_at=sent_at)
            # Phase 10B: journal entry when a digest is actually delivered.
            try:
                row = db.get_digest_by_id(user_id, item.digest_id)
                ms_json = str((row or {}).get("machine_summary_json") or "")
                try:
                    ms = json.loads(ms_json or "{}")
                except Exception:
                    ms = {}
                sev = str((ms.get("severity") if isinstance(ms, dict) else "") or "ok").strip().lower()
                if sev not in {"ok", "watch", "act_soon"}:
                    sev = "ok"
                line = make_journal_line("daily_digest", ms if isinstance(ms, dict) else {})
                db.insert_journal_entry(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    kind="daily_digest",
                    severity=sev,
                    created_at=sent_at,
                    line=line or f"Digest sent: {sev}",
                    machine_summary_json=ms_json or "{}",
                )
            except Exception as excj:
                try:
                    log_audit_event(
                        context.application.bot_data.get("log_path") or "",
                        event="journal_insert_failed",
                        user_id=user_id,
                        snapshot_id=str(item.digest_id or ""),
                        error=str(excj) or "insert_failed",
                        probe="change_journal",
                        target="db",
                        severity="warn",
                    )
                except Exception:
                    pass
            sent += 1
        except Exception as exc:
            log_event(
                context.application.bot_data.get("log_path") or "",
                "digest_send_failed",
                {"digest_id": item.digest_id, "error": str(exc)},
            )
            # Do not mark sent if delivery failed.
            continue

    await update.effective_message.reply_text(f"Sent {sent} digest(s).")


def _fmt_local_yyyy_mm_dd_hhmm(ts_iso: str, tz_name: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(ZoneInfo(tz_name))
        return local.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return (ts_iso or "").strip()[:16]


def _format_journal_lines(rows: list[dict], tz_name: str) -> list[str]:
    out: list[str] = []
    for r in rows or []:
        ts = _fmt_local_yyyy_mm_dd_hhmm(str(r.get("created_at") or ""), tz_name)
        line = str(r.get("line") or "").strip()
        out.append(f"{ts} — {line}")
    return out


async def _handle_journal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/journal").strip()
    try:
        n = int(content or 10)
    except Exception:
        n = 10
    n = max(1, min(n, 50))

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    rows = db.list_journal_entries(user_id, limit=n)
    if not rows:
        await update.effective_message.reply_text("No journal entries yet.")
        return
    lines = _format_journal_lines(rows, tz_name)
    await update.effective_message.reply_text("\n".join(lines).strip())


async def _handle_journal_kind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    content = _command_payload(text, "/journal_kind").strip()
    parts = [p for p in content.split() if p]
    if not parts:
        await update.effective_message.reply_text("Usage: /journal_kind <kind> [n]")
        return
    kind = parts[0].strip().lower()
    try:
        n = int(parts[1]) if len(parts) > 1 else 10
    except Exception:
        n = 10
    n = max(1, min(n, 50))

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    rows = db.list_journal_entries(user_id, limit=n, kind=kind)
    if not rows:
        await update.effective_message.reply_text("No journal entries for that kind.")
        return
    lines = _format_journal_lines(rows, tz_name)
    await update.effective_message.reply_text("\n".join(lines).strip())


async def _handle_journal_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    text = update.effective_message.text or ""
    entry_id = _command_payload(text, "/journal_show").strip()
    if not entry_id:
        await update.effective_message.reply_text("Usage: /journal_show <entry_id>")
        return

    tz_name = getattr(config, "agent_timezone", "America/Regina")
    row = db.get_journal_entry(user_id, entry_id)
    if not row:
        await update.effective_message.reply_text("No journal entry with that id.")
        return
    ts = _fmt_local_yyyy_mm_dd_hhmm(str(row.get("created_at") or ""), tz_name)
    kind = str(row.get("kind") or "")
    severity = str(row.get("severity") or "")
    line = str(row.get("line") or "")
    ms_json = str(row.get("machine_summary_json") or "")
    out = "\n".join(
        [
            f"id={entry_id}",
            f"created_at={ts}",
            f"kind={kind}",
            f"severity={severity}",
            f"line={line}",
            f"machine_summary_json={ms_json}",
        ]
    ).strip()
    await update.effective_message.reply_text(out)


def _settings_ui_running(pid_path: str) -> tuple[bool, int | None]:
    p = (pid_path or "").strip()
    if not p or not os.path.exists(p):
        return False, None
    try:
        pid_s = open(p, "r", encoding="utf-8").read().strip()
        pid = int(pid_s)
    except Exception:
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except Exception:
        return False, pid


async def _handle_settings_ui(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db) if db is not None else canonical_user_id("telegram", _platform_user_id_from_update(update))
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    port = (os.getenv("SETTINGS_UI_PORT", "8765") or "8765").strip()
    token_path = (os.getenv("SETTINGS_UI_TOKEN_PATH", "/etc/personal-agent/ui.token") or "/etc/personal-agent/ui.token").strip()
    pid_path = (os.getenv("SETTINGS_UI_PID_PATH", "/run/personal-agent/settings_ui.pid") or "/run/personal-agent/settings_ui.pid").strip()

    running, pid = _settings_ui_running(pid_path)
    status = "running" if running else "not running"
    pid_part = f" (pid {pid})" if pid else ""

    lines = [
        f"Settings URL: http://127.0.0.1:{port}/settings",
        f"Token path: {token_path}",
        f"PID file: {pid_path} ({status}{pid_part})",
        "Start UI: python -m agent.settings_ui",
        "Note: Local-only (127.0.0.1).",
    ]
    await update.effective_message.reply_text("\n".join(lines).strip())

async def _handle_doctor(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/doctor", user_id=user_id)
    await update.effective_message.reply_text(response.text)


async def _handle_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    config = context.application.bot_data.get("config")
    db: MemoryDB | None = context.application.bot_data.get("db")
    user_id = _sender_user_id(update, db)
    owner_id = getattr(config, "owner_user_id", None) if config else None
    role = role_for_user(user_id, owner_id)
    await update.effective_message.reply_text(f"user_id={user_id}\nrole={role}")


async def _handle_settings_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    config = context.application.bot_data.get("config")
    db: MemoryDB = context.application.bot_data["db"]
    user_id = _sender_user_id(update, db)
    if not config or not _owner_allowed(user_id, config):
        await update.effective_message.reply_text("Not authorized.")
        return
    env_path = (os.getenv("AGENT_ENV_PATH", "/etc/personal-agent/agent.env") or "/etc/personal-agent/agent.env").strip()
    data = export_settings(db, user_id, env_path, include_all_users=True)
    text = dumps_export_json(data)
    # Telegram has message size limits; keep it simple for now.
    await update.effective_message.reply_text(text.strip())


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    import logging

    logger = logging.getLogger(__name__)
    error = context.error
    try:
        from telegram.error import Conflict
    except Exception:  # pragma: no cover - defensive import
        Conflict = None
    if Conflict is not None and isinstance(error, Conflict):
        logger.error(
            "Telegram polling conflict detected. Another instance may be running or getUpdates is active elsewhere.",
            exc_info=error,
        )
        return
    logger.exception("Telegram handler error", exc_info=error)


async def _check_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]
    debug_protocol: DebugProtocol | None = context.application.bot_data.get("debug_protocol")
    orchestrator: Orchestrator | None = context.application.bot_data.get("orchestrator")

    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return

    now_ts = datetime.now(timezone.utc).isoformat()
    reminders = db.list_due_reminders(now_ts)
    for reminder in reminders:
        try:
            audit_id = db.audit_log_create(
                user_id=str(chat_id),
                action_type="reminder_send",
                action_id=str(reminder.id),
                status="attempted",
                details={
                    "event_type": "reminder_send",
                    "reminder_id": reminder.id,
                    "claim_attempted": True,
                    "claim_succeeded": False,
                    "send_succeeded": False,
                },
            )
        except Exception:
            return

        claim_ok = db.claim_reminder_sent(reminder.id, now_ts)
        if not claim_ok:
            try:
                db.audit_log_update_status(
                    audit_id,
                    "skipped",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": False,
                        "send_succeeded": False,
                        "status_transition": "pending->skipped",
                    },
                )
            except Exception:
                return
            continue

        if debug_protocol and orchestrator:
            if debug_protocol.record_reminder(
                str(chat_id),
                reminder.text,
                datetime.now(timezone.utc),
            ):
                response = orchestrator.handle_message("/runtime_status", user_id=str(chat_id))
                await context.bot.send_message(chat_id=chat_id, text=response.text)
                db.mark_reminder_failed(reminder.id, "debug_protocol_triggered")
                try:
                    db.audit_log_update_status(
                        audit_id,
                        "failed",
                        details={
                            "event_type": "reminder_send",
                            "reminder_id": reminder.id,
                            "claim_attempted": True,
                            "claim_succeeded": True,
                            "send_succeeded": False,
                            "status_transition": "sent->failed",
                            "error": "debug_protocol_triggered",
                        },
                    )
                except Exception:
                    return
                continue

        try:
            await context.bot.send_message(chat_id=chat_id, text=f"Reminder: {reminder.text}")
            try:
                db.audit_log_update_status(
                    audit_id,
                    "executed",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": True,
                        "send_succeeded": True,
                        "status_transition": "pending->sent",
                    },
                )
            except Exception:
                return
            log_event(log_path, "reminder_sent", {"reminder_id": reminder.id})
        except Exception as exc:
            db.mark_reminder_failed(reminder.id, str(exc))
            try:
                db.audit_log_update_status(
                    audit_id,
                    "failed",
                    details={
                        "event_type": "reminder_send",
                        "reminder_id": reminder.id,
                        "claim_attempted": True,
                        "claim_succeeded": True,
                        "send_succeeded": False,
                        "status_transition": "sent->failed",
                        "error": str(exc),
                    },
                )
            except Exception:
                return
            if debug_protocol and orchestrator:
                if debug_protocol.record_audit_event(
                    "reminder_send",
                    str(reminder.id),
                    "failed",
                    datetime.now(timezone.utc),
                ):
                    response = orchestrator.handle_message("/runtime_status", user_id=str(chat_id))
                    await context.bot.send_message(chat_id=chat_id, text=response.text)


async def _scheduled_disk_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    home_path: str = context.application.bot_data["home_path"]
    safe_run_scheduled_snapshot(db, home_path, "/")


async def _scheduled_storage_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    home_path: str = context.application.bot_data["home_path"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_storage_snapshot(db, timezone, home_path, user_id)


async def _scheduled_resource_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_resource_snapshot(db, timezone, user_id)


async def _scheduled_network_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone: str = context.application.bot_data["timezone"]
    user_id = db.get_preference("telegram_chat_id") or "system"
    safe_run_network_snapshot(db, timezone, user_id)

async def _scheduled_digest_build_today(context: ContextTypes.DEFAULT_TYPE) -> None:
    config = context.application.bot_data.get("config")
    if not config:
        return
    owner_id = getattr(config, "digest_owner_user_id", None) or getattr(config, "owner_user_id", None)
    if not owner_id:
        return

    db: MemoryDB = context.application.bot_data["db"]
    tz_name = getattr(config, "agent_timezone", "America/Regina")
    if not _effective_bool(db, str(owner_id), "DIGEST_ENABLED", bool(getattr(config, "digest_enabled", False))):
        return
    kind = getattr(config, "digest_kind", "daily_health_digest")
    run_date = _today_run_date(tz_name)
    ensure_daily_digest_row(db, user_id=str(owner_id), kind=kind, run_date=run_date, timezone_name=tz_name, dry_run=False)


async def _scheduled_digest_send_due(context: ContextTypes.DEFAULT_TYPE) -> None:
    config = context.application.bot_data.get("config")
    if not config:
        return
    owner_id = getattr(config, "digest_owner_user_id", None) or getattr(config, "owner_user_id", None)
    if not owner_id:
        return

    db: MemoryDB = context.application.bot_data["db"]
    tz_name = getattr(config, "agent_timezone", "America/Regina")
    if not _effective_bool(db, str(owner_id), "DIGEST_ENABLED", bool(getattr(config, "digest_enabled", False))):
        return
    kind = getattr(config, "digest_kind", "daily_health_digest")
    weekdays_only = _effective_bool(db, str(owner_id), "DIGEST_WEEKDAYS_ONLY", bool(getattr(config, "digest_weekdays_only", False)))
    quiet_start_s = _effective_str(db, str(owner_id), "DIGEST_QUIET_HOURS_START", str(getattr(config, "digest_quiet_hours_start", "09:00")))
    quiet_end_s = _effective_str(db, str(owner_id), "DIGEST_QUIET_HOURS_END", str(getattr(config, "digest_quiet_hours_end", "21:00")))
    quiet_start = _parse_hhmm(quiet_start_s, "09:00")
    quiet_end = _parse_hhmm(quiet_end_s, "21:00")

    now_iso = datetime.now(timezone.utc).isoformat()
    due = attempt_send_due_digests(
        db,
        user_id=str(owner_id),
        kind=kind,
        timezone_name=tz_name,
        now_iso=now_iso,
        weekdays_only=weekdays_only,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        dry_run=False,
        limit=5,
    )
    for item in due:
        try:
            chat_id = _chat_id_for_user_id(db, str(owner_id))
            await context.bot.send_message(chat_id=chat_id, text=item.text.strip())
            sent_at = datetime.now(timezone.utc).isoformat()
            db.mark_digest_sent(item.digest_id, sent_at=sent_at)
            # Phase 10B: journal entry when digest is actually delivered.
            try:
                row = db.get_digest_by_id(str(owner_id), item.digest_id)
                ms_json = str((row or {}).get("machine_summary_json") or "")
                try:
                    ms = json.loads(ms_json or "{}")
                except Exception:
                    ms = {}
                sev = str((ms.get("severity") if isinstance(ms, dict) else "") or "ok").strip().lower()
                if sev not in {"ok", "watch", "act_soon"}:
                    sev = "ok"
                line = make_journal_line("daily_digest", ms if isinstance(ms, dict) else {})
                db.insert_journal_entry(
                    id=str(uuid.uuid4()),
                    user_id=str(owner_id),
                    kind="daily_digest",
                    severity=sev,
                    created_at=sent_at,
                    line=line or f"Digest sent: {sev}",
                    machine_summary_json=ms_json or "{}",
                )
            except Exception as excj:
                try:
                    log_audit_event(
                        context.application.bot_data.get("log_path") or "",
                        event="journal_insert_failed",
                        user_id=str(owner_id),
                        snapshot_id=str(item.digest_id or ""),
                        error=str(excj) or "insert_failed",
                        probe="change_journal",
                        target="db",
                        severity="warn",
                    )
                except Exception:
                    pass
        except Exception as exc:
            log_event(
                context.application.bot_data.get("log_path") or "",
                "digest_send_failed",
                {"digest_id": item.digest_id, "error": str(exc)},
            )
            continue


def build_app() -> Application:
    config = load_config()
    db = MemoryDB(config.db_path)

    schema_path = "{}/memory/schema.sql".format(
        __import__("os").path.abspath(
            __import__("os").path.join(__import__("os").path.dirname(__file__), "..")
        )
    )
    db.init_schema(schema_path)

    llm_client = LLMRouter(config, log_path=config.log_path)

    llm_broker, llm_broker_error = build_llm_broker(config)
    if config.llm_selector_requested == "broker" and config.llm_broker_fallback_reason:
        log_event(
            config.log_path,
            "llm_broker_fallback",
            {
                "reason": config.llm_broker_fallback_reason,
                "selector": "direct",
                "provider": config.llm_provider,
            },
        )

    orchestrator = Orchestrator(
        db=db,
        skills_path=config.skills_path,
        log_path=config.log_path,
        timezone=config.agent_timezone,
        llm_client=llm_client,
        enable_writes=config.enable_writes,
        llm_broker=llm_broker,
        llm_broker_error=llm_broker_error,
    )
    debug_protocol = DebugProtocol()

    app = Application.builder().token(config.telegram_bot_token).build()

    # Log exceptions to journalctl instead of swallowing them.
    app.add_error_handler(_on_error)

    # Explicit command handlers (commands should NOT go through the generic text handler).
    app.add_handler(CommandHandler("remind", _handle_remind))
    app.add_handler(CommandHandler("status", _handle_status))
    app.add_handler(CommandHandler("runtime_status", _handle_runtime_status))
    app.add_handler(CommandHandler("disk_grow", _handle_disk_grow))
    app.add_handler(CommandHandler("audit", _handle_audit))
    app.add_handler(CommandHandler("storage_snapshot", _handle_storage_snapshot))
    app.add_handler(CommandHandler("storage_report", _handle_storage_report))
    app.add_handler(CommandHandler("resource_report", _handle_resource_report))
    app.add_handler(CommandHandler("brief", _handle_brief))
    app.add_handler(CommandHandler("observe_now", _handle_observe_now))
    app.add_handler(CommandHandler("hardware_report", _handle_hardware_report))
    app.add_handler(CommandHandler("network_report", _handle_network_report))
    app.add_handler(CommandHandler("weekly_reflection", _handle_weekly_reflection))
    app.add_handler(CommandHandler("ask", _handle_ask))
    app.add_handler(CommandHandler("ask_opinion", _handle_ask_opinion))
    app.add_handler(CommandHandler("llm_ping", _handle_llm_ping))
    app.add_handler(CommandHandler("llm_status", _handle_llm_status))
    app.add_handler(CommandHandler("llm_mode", _handle_llm_mode))
    app.add_handler(CommandHandler("digest_status", _handle_digest_status))
    app.add_handler(CommandHandler("digest_build_today", _handle_digest_build_today))
    app.add_handler(CommandHandler("digest_send_due", _handle_digest_send_due))
    app.add_handler(CommandHandler("digest_inbox", _handle_digest_inbox))
    app.add_handler(CommandHandler("digest_show", _handle_digest_show))
    app.add_handler(CommandHandler("digest_deferred", _handle_digest_deferred))
    app.add_handler(CommandHandler("digest_resend", _handle_digest_resend))
    app.add_handler(CommandHandler("digest_details", _handle_digest_details))
    app.add_handler(CommandHandler("journal", _handle_journal))
    app.add_handler(CommandHandler("journal_kind", _handle_journal_kind))
    app.add_handler(CommandHandler("journal_show", _handle_journal_show))
    app.add_handler(CommandHandler("settings_ui", _handle_settings_ui))
    app.add_handler(CommandHandler("doctor", _handle_doctor))
    app.add_handler(CommandHandler("whoami", _handle_whoami))
    app.add_handler(CommandHandler("settings_export", _handle_settings_export))

    # Non-command messages only.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    app.bot_data["db"] = db
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["debug_protocol"] = debug_protocol
    app.bot_data["log_path"] = config.log_path
    app.bot_data["home_path"] = os.path.expanduser("~")
    app.bot_data["timezone"] = config.agent_timezone
    app.bot_data["config"] = config

    app.job_queue.run_repeating(_check_reminders, interval=30, first=5)
    if config.enable_scheduled_snapshots:
        run_time = time(9, 0, tzinfo=ZoneInfo(config.agent_timezone))
        app.job_queue.run_daily(_scheduled_disk_snapshot, time=run_time, name="disk_snapshot_daily")
        app.job_queue.run_daily(_scheduled_storage_snapshot, time=run_time, name="storage_snapshot_daily")
        app.job_queue.run_daily(_scheduled_resource_snapshot, time=run_time, name="resource_snapshot_daily")
        app.job_queue.run_daily(_scheduled_network_snapshot, time=run_time, name="network_snapshot_daily")
    if getattr(config, "digest_owner_user_id", None):
        try:
            hh, mm = str(config.digest_time_local or "09:00").split(":", 1)
            digest_run_time = time(int(hh), int(mm), tzinfo=ZoneInfo(config.agent_timezone))
        except Exception:
            digest_run_time = time(9, 0, tzinfo=ZoneInfo(config.agent_timezone))
        app.job_queue.run_daily(_scheduled_digest_build_today, time=digest_run_time, name="digest_build_daily")
        app.job_queue.run_repeating(_scheduled_digest_send_due, interval=1800, first=15, name="digest_send_due")
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
