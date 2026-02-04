from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    openai_api_key: str | None
    openai_model: str
    openai_model_worker: str | None
    agent_timezone: str
    db_path: str
    log_path: str
    skills_path: str
    ollama_host: str | None
    ollama_model: str | None
    ollama_model_sentinel: str | None
    ollama_model_worker: str | None
    allow_cloud: bool
    prefer_local: bool
    llm_timeout_seconds: int


def load_config() -> Config:
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_model_worker = os.getenv("OPENAI_MODEL_WORKER", "").strip() or None
    agent_timezone = os.getenv("AGENT_TIMEZONE", "America/Regina")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_path = os.getenv("AGENT_DB_PATH", os.path.join(base_dir, "memory", "agent.db"))
    log_path = os.getenv("AGENT_LOG_PATH", os.path.join(base_dir, "logs", "agent.jsonl"))
    skills_path = os.getenv("AGENT_SKILLS_PATH", os.path.join(base_dir, "skills"))

    ollama_host = os.getenv("OLLAMA_HOST", "").strip() or None
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip() or None
    ollama_model_sentinel = os.getenv("OLLAMA_MODEL_SENTINEL", "").strip() or None
    ollama_model_worker = os.getenv("OLLAMA_MODEL_WORKER", "").strip() or None
    allow_cloud = os.getenv("ALLOW_CLOUD", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    prefer_local = os.getenv("PREFER_LOCAL", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    llm_timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "20") or 20)

    return Config(
        telegram_bot_token=telegram_bot_token,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_model_worker=openai_model_worker,
        agent_timezone=agent_timezone,
        db_path=db_path,
        log_path=log_path,
        skills_path=skills_path,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        ollama_model_sentinel=ollama_model_sentinel,
        ollama_model_worker=ollama_model_worker,
        allow_cloud=allow_cloud,
        prefer_local=prefer_local,
        llm_timeout_seconds=llm_timeout_seconds,
    )
