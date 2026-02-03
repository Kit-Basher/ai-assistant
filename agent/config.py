from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str | None
    openai_model: str
    agent_timezone: str
    db_path: str
    log_path: str
    skills_path: str


def load_config() -> Config:
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    agent_timezone = os.getenv("AGENT_TIMEZONE", "America/Regina")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.getenv("AGENT_DB_PATH", os.path.join(base_dir, "memory", "agent.db"))
    log_path = os.getenv("AGENT_LOG_PATH", os.path.join(base_dir, "logs", "agent.jsonl"))
    skills_path = os.getenv("AGENT_SKILLS_PATH", os.path.join(base_dir, "skills"))

    return Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        agent_timezone=agent_timezone,
        db_path=db_path,
        log_path=log_path,
        skills_path=skills_path,
    )
