from __future__ import annotations

import asyncio
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import os

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
from agent.scheduled_snapshots import (
    safe_run_scheduled_snapshot,
    safe_run_storage_snapshot,
    safe_run_resource_snapshot,
    safe_run_network_snapshot,
)
from agent.debug_protocol import DebugProtocol
from agent.orchestrator import Orchestrator
from agent.cards import render_cards_markdown, validate_cards_payload
from agent.daily_brief import should_send_daily_brief
from memory.db import MemoryDB


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

    # Remember which chat we're talking to (useful for reminders/jobs).
    db.set_preference("telegram_chat_id", chat_id)

    # Route ALL non-command text through the orchestrator (single brain).
    response = orchestrator.handle_message(text, user_id=chat_id)
    reply_text = response.text.strip() if response and response.text else ""
    parse_mode = None
    if response and isinstance(response.data, dict):
        ok, _ = validate_cards_payload(response.data)
        if ok:
            reply_text = render_cards_markdown(response.data)
            parse_mode = "Markdown"
    if not reply_text:
        reply_text = "Ask about disk, CPU, memory, or process changes."
    await update.effective_message.reply_text(reply_text, parse_mode=parse_mode)

    log_event(log_path, "telegram_message", {"chat_id": chat_id, "text": text})


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


async def _handle_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/remind")
    prompt = f"remind me {content}".strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    log_path: str = context.application.bot_data["log_path"]

    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(response.text)
    log_event(log_path, "telegram_command", {"chat_id": chat_id, "text": text, "forwarded": prompt})

async def _handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/status", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_runtime_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/runtime_status", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_disk_grow(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/disk_grow")
    prompt = "/disk_grow {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/audit", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_storage_snapshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/storage_snapshot", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_storage_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/storage_report", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_resource_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/resource_report", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/brief", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_network_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/network_report", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_weekly_reflection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]

    response = orchestrator.handle_message("/weekly_reflection", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/today", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_open_loops(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/open_loops")
    prompt = f"/open_loops {content}".strip()
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/health", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_daily_brief_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return
    chat_id = str(update.effective_chat.id)
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message("/daily_brief_status", user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask")
    prompt = "/ask {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(response.text)


async def _handle_ask_opinion(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.effective_message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.effective_message.text or ""
    content = _command_payload(text, "/ask_opinion")
    prompt = "/ask_opinion {}".format(content).strip()

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    response = orchestrator.handle_message(prompt, user_id=chat_id)
    await update.effective_message.reply_text(response.text)


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


async def _scheduled_daily_brief(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    timezone_name: str = context.application.bot_data["timezone"]
    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return
    enabled = (db.get_preference("daily_brief_enabled") or "off").strip().lower() in {"on", "true", "1", "yes"}
    local_time = (db.get_preference("daily_brief_time") or "09:00").strip()
    quiet_mode = (db.get_preference("daily_brief_quiet_mode") or "off").strip().lower() in {"on", "true", "1", "yes"}
    threshold_pref = db.get_preference("disk_delta_threshold_mb")
    disk_delta_threshold_mb = float(threshold_pref) if (threshold_pref and threshold_pref.isdigit()) else 250.0
    svc_gate = (db.get_preference("only_send_if_service_unhealthy") or "off").strip().lower() in {
        "on",
        "true",
        "1",
        "yes",
    }
    include_due_pref = db.get_preference("include_open_loops_due_within_days")
    include_due_days = int(include_due_pref) if (include_due_pref and include_due_pref.isdigit()) else 2
    last_sent = db.get_preference("daily_brief_last_sent_date")
    payload = orchestrator.build_daily_brief_cards(str(chat_id))
    signals = payload.get("daily_brief_signals") if isinstance(payload, dict) else {}
    disk_delta_mb = signals.get("disk_delta_mb") if isinstance(signals, dict) else None
    service_unhealthy = bool(signals.get("service_unhealthy")) if isinstance(signals, dict) else False
    due_open_loops = int(signals.get("due_open_loops_count") or 0) if isinstance(signals, dict) else 0
    decision = should_send_daily_brief(
        now_utc=datetime.now(timezone.utc),
        timezone_name=timezone_name,
        enabled=enabled,
        local_time_hhmm=local_time,
        last_sent_local_date=last_sent,
        quiet_mode=quiet_mode,
        disk_delta_mb=disk_delta_mb,
        disk_delta_threshold_mb=disk_delta_threshold_mb,
        service_unhealthy=service_unhealthy,
        only_send_if_service_unhealthy=svc_gate,
        has_due_open_loops=due_open_loops > 0 and include_due_days > 0,
    )
    if not decision.should_send:
        return
    text = render_cards_markdown(payload)
    send_ok = False
    for attempt in (1, 2):
        try:
            await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            send_ok = True
            break
        except Exception:
            if attempt == 1:
                await asyncio.sleep(1.0)
                continue
            return
    if send_ok:
        db.set_preference("daily_brief_last_sent_date", decision.local_date)


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
    app.add_handler(CommandHandler("network_report", _handle_network_report))
    app.add_handler(CommandHandler("weekly_reflection", _handle_weekly_reflection))
    app.add_handler(CommandHandler("today", _handle_today))
    app.add_handler(CommandHandler("open_loops", _handle_open_loops))
    app.add_handler(CommandHandler("health", _handle_health))
    app.add_handler(CommandHandler("daily_brief_status", _handle_daily_brief_status))
    app.add_handler(CommandHandler("ask", _handle_ask))
    app.add_handler(CommandHandler("ask_opinion", _handle_ask_opinion))

    # Non-command messages only.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

    app.bot_data["db"] = db
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["debug_protocol"] = debug_protocol
    app.bot_data["log_path"] = config.log_path
    app.bot_data["home_path"] = os.path.expanduser("~")
    app.bot_data["timezone"] = config.agent_timezone

    app.job_queue.run_repeating(_check_reminders, interval=30, first=5)
    app.job_queue.run_repeating(_scheduled_daily_brief, interval=60, first=10)
    if config.enable_scheduled_snapshots:
        run_time = time(9, 0, tzinfo=ZoneInfo(config.agent_timezone))
        app.job_queue.run_daily(_scheduled_disk_snapshot, time=run_time, name="disk_snapshot_daily")
        app.job_queue.run_daily(_scheduled_storage_snapshot, time=run_time, name="storage_snapshot_daily")
        app.job_queue.run_daily(_scheduled_resource_snapshot, time=run_time, name="resource_snapshot_daily")
        app.job_queue.run_daily(_scheduled_network_snapshot, time=run_time, name="network_snapshot_daily")
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
