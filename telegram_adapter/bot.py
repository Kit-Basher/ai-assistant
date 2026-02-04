from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from agent.config import load_config
from agent.intent_router import route_message
from agent.llm_router import LLMRouter
from agent.logging_utils import log_event
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat is None or update.message is None:
        return

    chat_id = str(update.effective_chat.id)
    text = update.message.text or ""

    orchestrator: Orchestrator = context.application.bot_data["orchestrator"]
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]

    db.set_preference("telegram_chat_id", chat_id)

    if text.startswith("/"):
        response = orchestrator.handle_message(text, user_id=chat_id)
        await update.message.reply_text(response.text)
        log_event(log_path, "telegram_message", {"chat_id": chat_id, "text": text})
        return

    decision = route_message(
        chat_id,
        text,
        {"db": db, "timezone": orchestrator.timezone, "chat_id": chat_id},
    )
    log_event(
        log_path,
        "intent_decision",
        {
            "user_id": chat_id,
            "text": text,
            "decision_type": decision.get("type"),
            "decision_function": decision.get("function"),
            "confidence": decision.get("confidence"),
            "explanation": decision.get("explanation"),
        },
    )
    if decision.get("type") == "skill_call":
        log_event(
            log_path,
            "intent_metric",
            {"metric": "intent_matched", "intent": decision.get("function")},
        )
    elif decision.get("type") == "clarify":
        log_event(
            log_path,
            "intent_metric",
            {"metric": "clarification_triggered", "intent": decision.get("intent")},
        )
    elif decision.get("type") == "noop":
        log_event(
            log_path,
            "intent_metric",
            {"metric": "no_intent", "intent": None},
        )

    if decision.get("type") == "skill_call":
        response = orchestrator.handle_intent(decision, user_id=chat_id)
        await update.message.reply_text(response.text)
        return

    if decision.get("type") == "clarify":
        question = decision.get("question", "")
        options = decision.get("options") or []
        if options:
            question = "{}\nOptions: {}".format(question, ", ".join(options))
        await update.message.reply_text(question)
        return

    if decision.get("type") in {"respond", "noop"}:
        await update.message.reply_text(decision.get("text", ""))
        return

    await update.message.reply_text("Try /help")


async def _check_reminders(context: ContextTypes.DEFAULT_TYPE) -> None:
    db: MemoryDB = context.application.bot_data["db"]
    log_path: str = context.application.bot_data["log_path"]

    chat_id = db.get_preference("telegram_chat_id")
    if not chat_id:
        return

    now_ts = datetime.now(timezone.utc).isoformat()
    reminders = db.list_due_reminders(now_ts)
    for reminder in reminders:
        await context.bot.send_message(chat_id=chat_id, text=f"Reminder: {reminder.text}")
        db.mark_reminder_sent(reminder.id)
        log_event(log_path, "reminder_sent", {"reminder_id": reminder.id})


def build_app() -> Application:
    config = load_config()
    db = MemoryDB(config.db_path)
    schema_path = "{}/memory/schema.sql".format(
        __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
    )
    db.init_schema(schema_path)

    llm_client = LLMRouter(config, log_path=config.log_path)

    orchestrator = Orchestrator(
        db=db,
        skills_path=config.skills_path,
        log_path=config.log_path,
        timezone=config.agent_timezone,
        llm_client=llm_client,
    )

    app = Application.builder().token(config.telegram_bot_token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_handler(MessageHandler(filters.COMMAND, _handle_message))

    app.bot_data["db"] = db
    app.bot_data["orchestrator"] = orchestrator
    app.bot_data["log_path"] = config.log_path

    app.job_queue.run_repeating(_check_reminders, interval=30, first=5)
    return app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
