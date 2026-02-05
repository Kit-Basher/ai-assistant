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
    enable_writes: bool = False
    llm_provider: str = "none"
    enable_llm_presentation: bool = False
    openai_base_url: str | None = None
    ollama_base_url: str | None = None
    anthropic_api_key: str | None = None
    llm_selector: str = "single"
    llm_broker_policy_path: str | None = None
    llm_allow_remote: bool = False
    openrouter_api_key: str | None = None
    openrouter_base_url: str | None = None
    openrouter_model: str | None = None
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None


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

    llm_provider = os.getenv("LLM_PROVIDER", "none").strip().lower() or "none"
    enable_llm_presentation = (
        os.getenv("ENABLE_LLM_PRESENTATION", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    )
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").strip() or None
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
    llm_selector = os.getenv("LLM_SELECTOR", "single").strip().lower() or "single"
    llm_broker_policy_path = os.getenv("LLM_BROKER_POLICY_PATH", "").strip() or None
    llm_allow_remote = os.getenv("LLM_ALLOW_REMOTE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or None
    openrouter_base_url = (
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip() or None
    )
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip() or None
    openrouter_site_url = os.getenv("OPENROUTER_SITE_URL", "").strip() or None
    openrouter_app_name = os.getenv("OPENROUTER_APP_NAME", "").strip() or None
    enable_writes = os.getenv("ENABLE_WRITES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    if enable_llm_presentation and llm_provider == "none":
        raise RuntimeError("ENABLE_LLM_PRESENTATION=1 requires LLM_PROVIDER to be set explicitly.")

    if llm_provider == "openai" and not openai_api_key:
        raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY.")
    if llm_provider == "ollama" and not ollama_base_url:
        raise RuntimeError("LLM_PROVIDER=ollama requires OLLAMA_BASE_URL.")
    if llm_provider == "openrouter" and not openrouter_api_key:
        raise RuntimeError("LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY.")
    if llm_provider not in {"none", "openai", "ollama", "anthropic", "openrouter"}:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {llm_provider}")
    if llm_provider == "anthropic" and not anthropic_api_key:
        raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY.")

    if llm_selector not in {"single", "broker"}:
        raise RuntimeError(f"Unsupported LLM_SELECTOR: {llm_selector}")
    if llm_selector == "broker":
        if not llm_broker_policy_path:
            raise RuntimeError("LLM_SELECTOR=broker requires LLM_BROKER_POLICY_PATH.")
        if not os.path.isfile(llm_broker_policy_path):
            raise RuntimeError("LLM_BROKER_POLICY_PATH is missing or not readable.")

    ollama_host = ollama_base_url or os.getenv("OLLAMA_HOST", "").strip() or None
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
        enable_writes=enable_writes,
        llm_provider=llm_provider,
        enable_llm_presentation=enable_llm_presentation,
        openai_base_url=openai_base_url,
        ollama_base_url=ollama_base_url,
        anthropic_api_key=anthropic_api_key,
        llm_selector=llm_selector,
        llm_broker_policy_path=llm_broker_policy_path,
        llm_allow_remote=llm_allow_remote,
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=openrouter_base_url,
        openrouter_model=openrouter_model,
        openrouter_site_url=openrouter_site_url,
        openrouter_app_name=openrouter_app_name,
    )
