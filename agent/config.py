from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


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
    enable_scheduled_snapshots: bool = False
    telegram_enabled: bool = False
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
    llm_registry_path: str | None = None
    llm_routing_mode: str = "auto"
    llm_retry_attempts: int = 3
    llm_retry_base_delay_ms: int = 200
    llm_circuit_breaker_failures: int = 3
    llm_circuit_breaker_window_seconds: int = 60
    llm_circuit_breaker_cooldown_seconds: int = 45
    llm_usage_stats_path: str | None = None
    model_scout_enabled: bool = True
    model_scout_notify_delta: float = 15.0
    model_scout_absolute_threshold: float = 80.0
    model_scout_max_suggestions_per_notify: int = 2
    model_scout_license_allowlist: tuple[str, ...] = ("apache-2.0", "mit", "bsd-3-clause")
    model_scout_size_max_b: float = 12.0
    model_scout_state_path: str | None = None
    model_watch_enabled: bool = True
    model_watch_interval_seconds: int = 86400
    model_watch_startup_grace_seconds: int = 300
    model_watch_state_path: str | None = None
    model_watch_config_path: str | None = None
    model_watch_catalog_path: str | None = None
    provider_catalog_state_path: str | None = None
    model_watch_min_improvement: float = 0.08
    model_watch_buzz_enabled: bool = False
    model_watch_buzz_sources_allowlist: tuple[str, ...] = (
        "huggingface_trending",
        "openrouter_models",
    )
    model_watch_hf_enabled: bool = False
    model_watch_hf_allowlist_repos: tuple[str, ...] = ()
    model_watch_hf_allowlist_orgs: tuple[str, ...] = ()
    model_watch_hf_require_gguf_for_install: bool = True
    model_watch_hf_max_total_bytes: int = 40 * 1024 * 1024 * 1024
    model_watch_hf_state_path: str | None = None
    model_watch_hf_download_base_path: str | None = None
    default_policy: dict[str, object] = field(
        default_factory=lambda: {
            "cost_cap_per_1m": 6.0,
            "allowlist": [],
            "quality_weight": 1.0,
            "price_weight": 0.04,
            "latency_weight": 0.25,
            "instability_weight": 0.5,
        }
    )
    premium_policy: dict[str, object] = field(
        default_factory=lambda: {
            "cost_cap_per_1m": 12.0,
            "allowlist": [],
            "quality_weight": 1.35,
            "price_weight": 0.025,
            "latency_weight": 0.2,
            "instability_weight": 0.45,
        }
    )
    memory_v2_enabled: bool = False
    intent_llm_rerank_enabled: bool = False
    perception_enabled: bool = True
    perception_roots: tuple[str, ...] = ("/home", "/data/projects")
    perception_interval_seconds: int = 5
    llm_health_interval_seconds: int = 900
    llm_health_max_probes_per_run: int = 6
    llm_health_probe_timeout_seconds: float = 6.0
    llm_health_state_path: str | None = None
    llm_catalog_path: str | None = None
    llm_catalog_refresh_interval_seconds: int = 21600
    llm_automation_enabled: bool = True
    llm_model_scout_interval_seconds: int = 86400
    llm_autoconfig_interval_seconds: int = 604800
    llm_autoconfig_run_on_startup: bool = False
    llm_hygiene_interval_seconds: int = 86400
    llm_hygiene_unavailable_days: int = 7
    llm_hygiene_remove_empty_disabled_providers: bool = True
    llm_hygiene_disable_repeatedly_failing_providers: bool = False
    llm_hygiene_provider_failure_streak: int = 8
    llm_registry_prune_allow_apply: bool | None = None
    llm_registry_prune_unused_days: int = 30
    llm_registry_prune_disable_failing_provider: bool = False
    llm_self_heal_interval_seconds: int = 86400
    llm_self_heal_allow_apply: bool | None = None
    llm_capabilities_reconcile_allow_apply: bool | None = None
    autopilot_notify_enabled: bool = True
    autopilot_notify_rate_limit_seconds: int = 1800
    autopilot_notify_dedupe_window_seconds: int = 86400
    autopilot_notify_store_path: str | None = None
    autopilot_notify_quiet_start_hour: int | None = None
    autopilot_notify_quiet_end_hour: int | None = None
    llm_notifications_allow_test: bool | None = None
    llm_notifications_allow_send: bool | None = None
    llm_notifications_max_items: int = 200
    llm_notifications_max_age_days: int = 30
    llm_notifications_compact: bool = True
    llm_registry_snapshots_dir: str | None = None
    llm_registry_snapshot_max_items: int = 40
    llm_registry_rollback_allow: bool | None = None
    llm_autopilot_safe_mode: bool = True
    llm_autopilot_state_path: str | None = None
    llm_autopilot_churn_window_seconds: int = 1800
    llm_autopilot_churn_min_applies: int = 4
    llm_autopilot_churn_recent_limit: int = 80
    llm_autopilot_bootstrap_allow_apply: bool | None = None
    llm_autopilot_ledger_path: str | None = None
    llm_autopilot_ledger_max_items: int = 400


@dataclass(frozen=True)
class ObserveConfig:
    db_path: str


def load_observe_config() -> ObserveConfig:
    db_path = os.environ.get("AGENT_DB_PATH")
    if not db_path:
        repo_root = Path(__file__).resolve().parents[1]
        db_path = str(repo_root / "memory" / "agent.db")
    return ObserveConfig(db_path=db_path)


def load_config(*, require_telegram_token: bool = True) -> Config:
    telegram_enabled = os.getenv("TELEGRAM_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if require_telegram_token and telegram_enabled and not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
    if not telegram_bot_token:
        telegram_bot_token = "local-api"

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
    enable_scheduled_snapshots = os.getenv("ENABLE_SCHEDULED_SNAPSHOTS", "0").strip().lower() in {
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
    registry_env = os.getenv("LLM_REGISTRY_PATH", "").strip()
    default_registry_path = os.path.join(base_dir, "llm_registry.json")
    if registry_env:
        llm_registry_path = registry_env
    elif os.path.isfile(default_registry_path):
        llm_registry_path = default_registry_path
    else:
        llm_registry_path = None
    llm_routing_mode = os.getenv("LLM_ROUTING_MODE", "auto").strip().lower() or "auto"
    llm_retry_attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "3") or 3)
    llm_retry_base_delay_ms = int(os.getenv("LLM_RETRY_BASE_DELAY_MS", "200") or 200)
    llm_circuit_breaker_failures = int(os.getenv("LLM_CIRCUIT_BREAKER_FAILURES", "3") or 3)
    llm_circuit_breaker_window_seconds = int(os.getenv("LLM_CIRCUIT_BREAKER_WINDOW_SECONDS", "60") or 60)
    llm_circuit_breaker_cooldown_seconds = int(
        os.getenv("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "45") or 45
    )
    llm_usage_stats_path = os.getenv("LLM_USAGE_STATS_PATH", "").strip() or None
    model_scout_enabled = os.getenv("MODEL_SCOUT_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    model_scout_notify_delta = float(os.getenv("MODEL_SCOUT_NOTIFY_DELTA", "15") or 15)
    model_scout_absolute_threshold = float(os.getenv("MODEL_SCOUT_ABSOLUTE_THRESHOLD", "80") or 80)
    model_scout_max_suggestions_per_notify = int(os.getenv("MODEL_SCOUT_MAX_SUGGESTIONS_PER_NOTIFY", "2") or 2)
    model_scout_license_allowlist_raw = os.getenv(
        "MODEL_SCOUT_LICENSE_ALLOWLIST",
        "apache-2.0,mit,bsd-3-clause",
    )
    model_scout_license_allowlist = tuple(
        sorted(
            {
                item.strip().lower()
                for item in model_scout_license_allowlist_raw.split(",")
                if item.strip()
            }
        )
    ) or ("apache-2.0", "mit", "bsd-3-clause")
    model_scout_size_max_b = float(os.getenv("MODEL_SCOUT_SIZE_MAX_B", "12") or 12)
    model_scout_state_path = os.getenv("AGENT_MODEL_SCOUT_STATE_PATH", "").strip() or None
    model_watch_enabled = os.getenv("AGENT_MODEL_WATCH_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    model_watch_interval_raw = os.getenv("AGENT_MODEL_WATCH_INTERVAL_SECONDS", "").strip()
    if model_watch_interval_raw:
        model_watch_interval_seconds = int(model_watch_interval_raw or 86400)
    else:
        model_watch_interval_seconds = int(
            os.getenv("AGENT_MODEL_WATCH_MIN_INTERVAL_SECONDS", str(24 * 60 * 60)) or (24 * 60 * 60)
        )
    model_watch_startup_grace_seconds = int(os.getenv("AGENT_MODEL_WATCH_STARTUP_GRACE_SECONDS", "300") or 300)
    model_watch_state_path = os.getenv("AGENT_MODEL_WATCH_STATE_PATH", "").strip() or None
    model_watch_config_path = os.getenv("AGENT_MODEL_WATCH_CONFIG_PATH", "").strip() or None
    model_watch_catalog_path = os.getenv("AGENT_MODEL_WATCH_CATALOG_PATH", "").strip() or None
    provider_catalog_state_path = os.getenv("AGENT_PROVIDER_CATALOG_STATE_PATH", "").strip() or None
    model_watch_min_improvement = float(os.getenv("AGENT_MODEL_WATCH_MIN_IMPROVEMENT", "0.08") or 0.08)
    model_watch_buzz_enabled = os.getenv("AGENT_MODEL_WATCH_BUZZ_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    model_watch_buzz_sources_allowlist_raw = os.getenv(
        "AGENT_MODEL_WATCH_BUZZ_SOURCES_ALLOWLIST",
        "openrouter_models,huggingface_trending",
    ).strip()
    model_watch_buzz_sources_allowlist = tuple(
        sorted(
            {
                item.strip().lower()
                for item in model_watch_buzz_sources_allowlist_raw.split(",")
                if item.strip()
            }
        )
    ) or (
        "huggingface_trending",
        "openrouter_models",
    )
    model_watch_hf_enabled = os.getenv("AGENT_MODEL_WATCH_HF_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    model_watch_hf_allowlist_repos_raw = os.getenv(
        "AGENT_MODEL_WATCH_HF_ALLOWLIST_REPOS",
        "",
    ).strip()
    model_watch_hf_allowlist_repos = tuple(
        sorted(
            {
                item.strip()
                for item in model_watch_hf_allowlist_repos_raw.split(",")
                if item.strip()
            }
        )
    )
    model_watch_hf_allowlist_orgs_raw = os.getenv(
        "AGENT_MODEL_WATCH_HF_ALLOWLIST_ORGS",
        "",
    ).strip()
    model_watch_hf_allowlist_orgs = tuple(
        sorted(
            {
                item.strip()
                for item in model_watch_hf_allowlist_orgs_raw.split(",")
                if item.strip()
            }
        )
    )
    model_watch_hf_require_gguf_for_install = os.getenv(
        "AGENT_MODEL_WATCH_HF_REQUIRE_GGUF_FOR_INSTALL",
        "1",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    model_watch_hf_max_total_bytes = int(
        os.getenv("AGENT_MODEL_WATCH_HF_MAX_TOTAL_BYTES", str(40 * 1024 * 1024 * 1024))
        or (40 * 1024 * 1024 * 1024)
    )
    model_watch_hf_state_path = os.getenv("AGENT_MODEL_WATCH_HF_STATE_PATH", "").strip() or None
    model_watch_hf_download_base_path = (
        os.getenv("AGENT_MODEL_WATCH_HF_DOWNLOAD_BASE_PATH", "").strip() or None
    )

    def _policy_from_env(prefix: str, fallback: dict[str, object]) -> dict[str, object]:
        policy = dict(fallback)
        raw_json = os.getenv(f"AGENT_{prefix}_POLICY", "").strip()
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"AGENT_{prefix}_POLICY must be valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(f"AGENT_{prefix}_POLICY must be a JSON object.")
            policy.update(parsed)
        cost_cap_raw = os.getenv(f"AGENT_{prefix}_POLICY_COST_CAP_PER_1M", "").strip()
        if cost_cap_raw:
            policy["cost_cap_per_1m"] = float(cost_cap_raw)
        allowlist_raw = os.getenv(f"AGENT_{prefix}_POLICY_ALLOWLIST", "").strip()
        if allowlist_raw:
            policy["allowlist"] = sorted({item.strip() for item in allowlist_raw.split(",") if item.strip()})
        for key in ("quality_weight", "price_weight", "latency_weight", "instability_weight"):
            value_raw = os.getenv(f"AGENT_{prefix}_POLICY_{key.upper()}", "").strip()
            if value_raw:
                policy[key] = float(value_raw)
        allowlist = policy.get("allowlist")
        if isinstance(allowlist, (list, tuple, set, frozenset)):
            policy["allowlist"] = sorted({str(item).strip() for item in allowlist if str(item).strip()})
        else:
            policy["allowlist"] = []
        policy["cost_cap_per_1m"] = max(0.0, float(policy.get("cost_cap_per_1m", fallback["cost_cap_per_1m"])))
        for key in ("quality_weight", "price_weight", "latency_weight", "instability_weight"):
            policy[key] = max(0.0, float(policy.get(key, fallback[key])))
        return policy

    default_policy = _policy_from_env(
        "DEFAULT",
        {
            "cost_cap_per_1m": 6.0,
            "allowlist": [],
            "quality_weight": 1.0,
            "price_weight": 0.04,
            "latency_weight": 0.25,
            "instability_weight": 0.5,
        },
    )
    premium_policy = _policy_from_env(
        "PREMIUM",
        {
            "cost_cap_per_1m": 12.0,
            "allowlist": [],
            "quality_weight": 1.35,
            "price_weight": 0.025,
            "latency_weight": 0.2,
            "instability_weight": 0.45,
        },
    )
    memory_v2_enabled = os.getenv("AGENT_MEMORY_V2_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    intent_llm_rerank_enabled = os.getenv("AGENT_INTENT_LLM_RERANK_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    perception_enabled = os.getenv("PERCEPTION_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    perception_roots_raw = os.getenv("PERCEPTION_ROOTS", "/home,/data/projects").strip()
    perception_roots = tuple(
        item.strip()
        for item in perception_roots_raw.split(",")
        if item.strip()
    ) or ("/home", "/data/projects")
    perception_interval_seconds = int(os.getenv("PERCEPTION_INTERVAL_SECONDS", "5") or 5)
    llm_health_interval_seconds = int(os.getenv("LLM_HEALTH_INTERVAL_SECONDS", "900") or 900)
    llm_health_max_probes_per_run = int(os.getenv("LLM_HEALTH_MAX_PROBES_PER_RUN", "6") or 6)
    llm_health_probe_timeout_seconds = float(os.getenv("LLM_HEALTH_PROBE_TIMEOUT_SECONDS", "6") or 6)
    llm_health_state_path = os.getenv("LLM_HEALTH_STATE_PATH", "").strip() or None
    llm_catalog_path = os.getenv("LLM_CATALOG_PATH", "").strip() or None
    llm_catalog_refresh_interval_seconds = int(os.getenv("LLM_CATALOG_REFRESH_INTERVAL_S", "21600") or 21600)
    llm_automation_enabled = os.getenv("LLM_AUTOMATION_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    llm_model_scout_interval_seconds = int(os.getenv("LLM_MODEL_SCOUT_INTERVAL_SECONDS", "86400") or 86400)
    llm_autoconfig_interval_seconds = int(os.getenv("LLM_AUTOCONFIG_INTERVAL_SECONDS", "604800") or 604800)
    llm_autoconfig_run_on_startup = os.getenv("LLM_AUTOCONFIG_RUN_ON_STARTUP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    llm_hygiene_interval_seconds = int(os.getenv("LLM_HYGIENE_INTERVAL_SECONDS", "86400") or 86400)
    llm_hygiene_unavailable_days = int(os.getenv("LLM_HYGIENE_UNAVAILABLE_DAYS", "7") or 7)
    llm_hygiene_remove_empty_disabled_providers = (
        os.getenv("LLM_HYGIENE_REMOVE_EMPTY_DISABLED_PROVIDERS", "1").strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )
    llm_hygiene_disable_repeatedly_failing_providers = (
        os.getenv("LLM_HYGIENE_DISABLE_REPEATEDLY_FAILING_PROVIDERS", "0").strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )
    llm_hygiene_provider_failure_streak = int(os.getenv("LLM_HYGIENE_PROVIDER_FAILURE_STREAK", "8") or 8)
    llm_registry_prune_allow_apply_raw = os.getenv("LLM_REGISTRY_PRUNE_ALLOW_APPLY", "").strip().lower()
    if not llm_registry_prune_allow_apply_raw:
        llm_registry_prune_allow_apply: bool | None = None
    elif llm_registry_prune_allow_apply_raw in {"1", "true", "yes", "y", "on"}:
        llm_registry_prune_allow_apply = True
    elif llm_registry_prune_allow_apply_raw in {"0", "false", "no", "n", "off"}:
        llm_registry_prune_allow_apply = False
    else:
        raise RuntimeError("LLM_REGISTRY_PRUNE_ALLOW_APPLY must be true/false when set.")
    llm_registry_prune_unused_days = int(os.getenv("LLM_REGISTRY_PRUNE_UNUSED_DAYS", "30") or 30)
    llm_registry_prune_disable_failing_provider = (
        os.getenv("LLM_REGISTRY_PRUNE_DISABLE_FAILING_PROVIDER", "0").strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )
    llm_self_heal_interval_seconds = int(os.getenv("LLM_SELF_HEAL_INTERVAL_S", "86400") or 86400)
    llm_self_heal_allow_apply_raw = os.getenv("LLM_SELF_HEAL_ALLOW_APPLY", "").strip().lower()
    if not llm_self_heal_allow_apply_raw:
        llm_self_heal_allow_apply: bool | None = None
    elif llm_self_heal_allow_apply_raw in {"1", "true", "yes", "y", "on"}:
        llm_self_heal_allow_apply = True
    elif llm_self_heal_allow_apply_raw in {"0", "false", "no", "n", "off"}:
        llm_self_heal_allow_apply = False
    else:
        raise RuntimeError("LLM_SELF_HEAL_ALLOW_APPLY must be true/false when set.")
    llm_capabilities_reconcile_allow_apply_raw = (
        os.getenv("LLM_CAPABILITIES_RECONCILE_ALLOW_APPLY", "").strip().lower()
    )
    if not llm_capabilities_reconcile_allow_apply_raw:
        llm_capabilities_reconcile_allow_apply: bool | None = None
    elif llm_capabilities_reconcile_allow_apply_raw in {"1", "true", "yes", "y", "on"}:
        llm_capabilities_reconcile_allow_apply = True
    elif llm_capabilities_reconcile_allow_apply_raw in {"0", "false", "no", "n", "off"}:
        llm_capabilities_reconcile_allow_apply = False
    else:
        raise RuntimeError("LLM_CAPABILITIES_RECONCILE_ALLOW_APPLY must be true/false when set.")
    autopilot_notify_enabled = os.getenv("AUTOPILOT_NOTIFY_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    autopilot_notify_rate_limit_seconds = int(os.getenv("AUTOPILOT_NOTIFY_RATE_LIMIT_SECONDS", "1800") or 1800)
    autopilot_notify_dedupe_window_seconds = int(
        os.getenv("AUTOPILOT_NOTIFY_DEDUPE_WINDOW_SECONDS", "86400") or 86400
    )
    autopilot_notify_store_path = os.getenv("AUTOPILOT_NOTIFY_STORE_PATH", "").strip() or None
    quiet_start_raw = os.getenv("AUTOPILOT_NOTIFY_QUIET_START_HOUR", "").strip()
    quiet_end_raw = os.getenv("AUTOPILOT_NOTIFY_QUIET_END_HOUR", "").strip()
    autopilot_notify_quiet_start_hour = int(quiet_start_raw) if quiet_start_raw else None
    autopilot_notify_quiet_end_hour = int(quiet_end_raw) if quiet_end_raw else None
    llm_notifications_allow_test_raw = os.getenv("LLM_NOTIFICATIONS_ALLOW_TEST", "").strip().lower()
    if not llm_notifications_allow_test_raw:
        llm_notifications_allow_test: bool | None = None
    elif llm_notifications_allow_test_raw in {"1", "true", "yes", "y", "on"}:
        llm_notifications_allow_test = True
    elif llm_notifications_allow_test_raw in {"0", "false", "no", "n", "off"}:
        llm_notifications_allow_test = False
    else:
        raise RuntimeError("LLM_NOTIFICATIONS_ALLOW_TEST must be true/false when set.")
    llm_notifications_allow_send_raw = os.getenv("LLM_NOTIFICATIONS_ALLOW_SEND", "").strip().lower()
    if not llm_notifications_allow_send_raw:
        llm_notifications_allow_send: bool | None = None
    elif llm_notifications_allow_send_raw in {"1", "true", "yes", "y", "on"}:
        llm_notifications_allow_send = True
    elif llm_notifications_allow_send_raw in {"0", "false", "no", "n", "off"}:
        llm_notifications_allow_send = False
    else:
        raise RuntimeError("LLM_NOTIFICATIONS_ALLOW_SEND must be true/false when set.")
    llm_notifications_max_items = int(os.getenv("LLM_NOTIFICATIONS_MAX_ITEMS", "200") or 200)
    llm_notifications_max_age_days = int(os.getenv("LLM_NOTIFICATIONS_MAX_AGE_DAYS", "30") or 30)
    llm_notifications_compact = os.getenv("LLM_NOTIFICATIONS_COMPACT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    llm_registry_snapshots_dir = os.getenv("LLM_REGISTRY_SNAPSHOTS_DIR", "").strip() or None
    llm_registry_snapshot_max_items = int(os.getenv("LLM_REGISTRY_SNAPSHOT_MAX_ITEMS", "40") or 40)
    llm_registry_rollback_allow_raw = os.getenv("LLM_REGISTRY_ROLLBACK_ALLOW", "").strip().lower()
    if not llm_registry_rollback_allow_raw:
        llm_registry_rollback_allow: bool | None = None
    elif llm_registry_rollback_allow_raw in {"1", "true", "yes", "y", "on"}:
        llm_registry_rollback_allow = True
    elif llm_registry_rollback_allow_raw in {"0", "false", "no", "n", "off"}:
        llm_registry_rollback_allow = False
    else:
        raise RuntimeError("LLM_REGISTRY_ROLLBACK_ALLOW must be true/false when set.")
    llm_autopilot_safe_mode = os.getenv("LLM_AUTOPILOT_SAFE_MODE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    llm_autopilot_state_path = os.getenv("LLM_AUTOPILOT_STATE_PATH", "").strip() or None
    llm_autopilot_churn_window_seconds = int(
        os.getenv("LLM_AUTOPILOT_CHURN_WINDOW_SECONDS", "1800") or 1800
    )
    llm_autopilot_churn_min_applies = int(os.getenv("LLM_AUTOPILOT_CHURN_MIN_APPLIES", "4") or 4)
    llm_autopilot_churn_recent_limit = int(os.getenv("LLM_AUTOPILOT_CHURN_RECENT_LIMIT", "80") or 80)
    llm_autopilot_bootstrap_allow_apply_raw = os.getenv("LLM_AUTOPILOT_BOOTSTRAP_ALLOW_APPLY", "").strip().lower()
    if not llm_autopilot_bootstrap_allow_apply_raw:
        llm_autopilot_bootstrap_allow_apply: bool | None = None
    elif llm_autopilot_bootstrap_allow_apply_raw in {"1", "true", "yes", "y", "on"}:
        llm_autopilot_bootstrap_allow_apply = True
    elif llm_autopilot_bootstrap_allow_apply_raw in {"0", "false", "no", "n", "off"}:
        llm_autopilot_bootstrap_allow_apply = False
    else:
        raise RuntimeError("LLM_AUTOPILOT_BOOTSTRAP_ALLOW_APPLY must be true/false when set.")
    llm_autopilot_ledger_path = os.getenv("LLM_AUTOPILOT_LEDGER_PATH", "").strip() or None
    llm_autopilot_ledger_max_items = int(os.getenv("LLM_AUTOPILOT_LEDGER_MAX_ITEMS", "400") or 400)

    if llm_routing_mode not in {
        "auto",
        "prefer_cheap",
        "prefer_best",
        "prefer_local_lowest_cost_capable",
    }:
        raise RuntimeError(f"Unsupported LLM_ROUTING_MODE: {llm_routing_mode}")
    if llm_retry_attempts < 1:
        raise RuntimeError("LLM_RETRY_ATTEMPTS must be >= 1.")
    if llm_retry_base_delay_ms < 0:
        raise RuntimeError("LLM_RETRY_BASE_DELAY_MS must be >= 0.")
    if llm_circuit_breaker_failures < 1:
        raise RuntimeError("LLM_CIRCUIT_BREAKER_FAILURES must be >= 1.")
    if llm_circuit_breaker_window_seconds < 1:
        raise RuntimeError("LLM_CIRCUIT_BREAKER_WINDOW_SECONDS must be >= 1.")
    if llm_circuit_breaker_cooldown_seconds < 1:
        raise RuntimeError("LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS must be >= 1.")
    if llm_registry_path and not os.path.isfile(llm_registry_path):
        raise RuntimeError("LLM_REGISTRY_PATH is missing or not readable.")
    if model_scout_notify_delta < 0:
        raise RuntimeError("MODEL_SCOUT_NOTIFY_DELTA must be >= 0.")
    if model_scout_absolute_threshold < 0:
        raise RuntimeError("MODEL_SCOUT_ABSOLUTE_THRESHOLD must be >= 0.")
    if model_scout_max_suggestions_per_notify < 1:
        raise RuntimeError("MODEL_SCOUT_MAX_SUGGESTIONS_PER_NOTIFY must be >= 1.")
    if model_scout_size_max_b <= 0:
        raise RuntimeError("MODEL_SCOUT_SIZE_MAX_B must be > 0.")
    if model_watch_interval_seconds < 1:
        raise RuntimeError("AGENT_MODEL_WATCH_INTERVAL_SECONDS must be >= 1.")
    if model_watch_startup_grace_seconds < 0:
        raise RuntimeError("AGENT_MODEL_WATCH_STARTUP_GRACE_SECONDS must be >= 0.")
    if model_watch_min_improvement < 0:
        raise RuntimeError("AGENT_MODEL_WATCH_MIN_IMPROVEMENT must be >= 0.")
    if model_watch_hf_max_total_bytes < 0:
        raise RuntimeError("AGENT_MODEL_WATCH_HF_MAX_TOTAL_BYTES must be >= 0.")
    if float(default_policy.get("cost_cap_per_1m", 0.0)) < 0:
        raise RuntimeError("AGENT_DEFAULT_POLICY_COST_CAP_PER_1M must be >= 0.")
    if float(premium_policy.get("cost_cap_per_1m", 0.0)) < 0:
        raise RuntimeError("AGENT_PREMIUM_POLICY_COST_CAP_PER_1M must be >= 0.")
    if perception_interval_seconds < 1:
        raise RuntimeError("PERCEPTION_INTERVAL_SECONDS must be >= 1.")
    if llm_health_interval_seconds < 1:
        raise RuntimeError("LLM_HEALTH_INTERVAL_SECONDS must be >= 1.")
    if llm_health_max_probes_per_run < 1:
        raise RuntimeError("LLM_HEALTH_MAX_PROBES_PER_RUN must be >= 1.")
    if llm_health_probe_timeout_seconds <= 0:
        raise RuntimeError("LLM_HEALTH_PROBE_TIMEOUT_SECONDS must be > 0.")
    if llm_catalog_refresh_interval_seconds < 1:
        raise RuntimeError("LLM_CATALOG_REFRESH_INTERVAL_S must be >= 1.")
    if llm_model_scout_interval_seconds < 1:
        raise RuntimeError("LLM_MODEL_SCOUT_INTERVAL_SECONDS must be >= 1.")
    if llm_autoconfig_interval_seconds < 1:
        raise RuntimeError("LLM_AUTOCONFIG_INTERVAL_SECONDS must be >= 1.")
    if llm_hygiene_interval_seconds < 1:
        raise RuntimeError("LLM_HYGIENE_INTERVAL_SECONDS must be >= 1.")
    if llm_hygiene_unavailable_days < 1:
        raise RuntimeError("LLM_HYGIENE_UNAVAILABLE_DAYS must be >= 1.")
    if llm_hygiene_provider_failure_streak < 1:
        raise RuntimeError("LLM_HYGIENE_PROVIDER_FAILURE_STREAK must be >= 1.")
    if llm_registry_prune_unused_days < 1:
        raise RuntimeError("LLM_REGISTRY_PRUNE_UNUSED_DAYS must be >= 1.")
    if llm_self_heal_interval_seconds < 1:
        raise RuntimeError("LLM_SELF_HEAL_INTERVAL_S must be >= 1.")
    if autopilot_notify_rate_limit_seconds < 0:
        raise RuntimeError("AUTOPILOT_NOTIFY_RATE_LIMIT_SECONDS must be >= 0.")
    if autopilot_notify_dedupe_window_seconds < 0:
        raise RuntimeError("AUTOPILOT_NOTIFY_DEDUPE_WINDOW_SECONDS must be >= 0.")
    if autopilot_notify_quiet_start_hour is not None and not 0 <= autopilot_notify_quiet_start_hour <= 23:
        raise RuntimeError("AUTOPILOT_NOTIFY_QUIET_START_HOUR must be between 0 and 23.")
    if autopilot_notify_quiet_end_hour is not None and not 0 <= autopilot_notify_quiet_end_hour <= 23:
        raise RuntimeError("AUTOPILOT_NOTIFY_QUIET_END_HOUR must be between 0 and 23.")
    if llm_notifications_max_items < 1:
        raise RuntimeError("LLM_NOTIFICATIONS_MAX_ITEMS must be >= 1.")
    if llm_notifications_max_age_days < 0:
        raise RuntimeError("LLM_NOTIFICATIONS_MAX_AGE_DAYS must be >= 0.")
    if llm_registry_snapshot_max_items < 1:
        raise RuntimeError("LLM_REGISTRY_SNAPSHOT_MAX_ITEMS must be >= 1.")
    if llm_autopilot_churn_window_seconds < 60:
        raise RuntimeError("LLM_AUTOPILOT_CHURN_WINDOW_SECONDS must be >= 60.")
    if llm_autopilot_churn_min_applies < 2:
        raise RuntimeError("LLM_AUTOPILOT_CHURN_MIN_APPLIES must be >= 2.")
    if llm_autopilot_churn_recent_limit < 1:
        raise RuntimeError("LLM_AUTOPILOT_CHURN_RECENT_LIMIT must be >= 1.")
    if llm_autopilot_ledger_max_items < 1:
        raise RuntimeError("LLM_AUTOPILOT_LEDGER_MAX_ITEMS must be >= 1.")

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
        enable_scheduled_snapshots=enable_scheduled_snapshots,
        telegram_enabled=telegram_enabled,
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
        llm_registry_path=llm_registry_path,
        llm_routing_mode=llm_routing_mode,
        llm_retry_attempts=llm_retry_attempts,
        llm_retry_base_delay_ms=llm_retry_base_delay_ms,
        llm_circuit_breaker_failures=llm_circuit_breaker_failures,
        llm_circuit_breaker_window_seconds=llm_circuit_breaker_window_seconds,
        llm_circuit_breaker_cooldown_seconds=llm_circuit_breaker_cooldown_seconds,
        llm_usage_stats_path=llm_usage_stats_path,
        model_scout_enabled=model_scout_enabled,
        model_scout_notify_delta=model_scout_notify_delta,
        model_scout_absolute_threshold=model_scout_absolute_threshold,
        model_scout_max_suggestions_per_notify=model_scout_max_suggestions_per_notify,
        model_scout_license_allowlist=model_scout_license_allowlist,
        model_scout_size_max_b=model_scout_size_max_b,
        model_scout_state_path=model_scout_state_path,
        model_watch_enabled=model_watch_enabled,
        model_watch_interval_seconds=model_watch_interval_seconds,
        model_watch_startup_grace_seconds=model_watch_startup_grace_seconds,
        model_watch_state_path=model_watch_state_path,
        model_watch_config_path=model_watch_config_path,
        model_watch_catalog_path=model_watch_catalog_path,
        provider_catalog_state_path=provider_catalog_state_path,
        model_watch_min_improvement=model_watch_min_improvement,
        model_watch_buzz_enabled=model_watch_buzz_enabled,
        model_watch_buzz_sources_allowlist=model_watch_buzz_sources_allowlist,
        model_watch_hf_enabled=model_watch_hf_enabled,
        model_watch_hf_allowlist_repos=model_watch_hf_allowlist_repos,
        model_watch_hf_allowlist_orgs=model_watch_hf_allowlist_orgs,
        model_watch_hf_require_gguf_for_install=model_watch_hf_require_gguf_for_install,
        model_watch_hf_max_total_bytes=model_watch_hf_max_total_bytes,
        model_watch_hf_state_path=model_watch_hf_state_path,
        model_watch_hf_download_base_path=model_watch_hf_download_base_path,
        default_policy=default_policy,
        premium_policy=premium_policy,
        memory_v2_enabled=memory_v2_enabled,
        intent_llm_rerank_enabled=intent_llm_rerank_enabled,
        perception_enabled=perception_enabled,
        perception_roots=perception_roots,
        perception_interval_seconds=perception_interval_seconds,
        llm_health_interval_seconds=llm_health_interval_seconds,
        llm_health_max_probes_per_run=llm_health_max_probes_per_run,
        llm_health_probe_timeout_seconds=llm_health_probe_timeout_seconds,
        llm_health_state_path=llm_health_state_path,
        llm_catalog_path=llm_catalog_path,
        llm_catalog_refresh_interval_seconds=llm_catalog_refresh_interval_seconds,
        llm_automation_enabled=llm_automation_enabled,
        llm_model_scout_interval_seconds=llm_model_scout_interval_seconds,
        llm_autoconfig_interval_seconds=llm_autoconfig_interval_seconds,
        llm_autoconfig_run_on_startup=llm_autoconfig_run_on_startup,
        llm_hygiene_interval_seconds=llm_hygiene_interval_seconds,
        llm_hygiene_unavailable_days=llm_hygiene_unavailable_days,
        llm_hygiene_remove_empty_disabled_providers=llm_hygiene_remove_empty_disabled_providers,
        llm_hygiene_disable_repeatedly_failing_providers=llm_hygiene_disable_repeatedly_failing_providers,
        llm_hygiene_provider_failure_streak=llm_hygiene_provider_failure_streak,
        llm_registry_prune_allow_apply=llm_registry_prune_allow_apply,
        llm_registry_prune_unused_days=llm_registry_prune_unused_days,
        llm_registry_prune_disable_failing_provider=llm_registry_prune_disable_failing_provider,
        llm_self_heal_interval_seconds=llm_self_heal_interval_seconds,
        llm_self_heal_allow_apply=llm_self_heal_allow_apply,
        llm_capabilities_reconcile_allow_apply=llm_capabilities_reconcile_allow_apply,
        autopilot_notify_enabled=autopilot_notify_enabled,
        autopilot_notify_rate_limit_seconds=autopilot_notify_rate_limit_seconds,
        autopilot_notify_dedupe_window_seconds=autopilot_notify_dedupe_window_seconds,
        autopilot_notify_store_path=autopilot_notify_store_path,
        autopilot_notify_quiet_start_hour=autopilot_notify_quiet_start_hour,
        autopilot_notify_quiet_end_hour=autopilot_notify_quiet_end_hour,
        llm_notifications_allow_test=llm_notifications_allow_test,
        llm_notifications_allow_send=llm_notifications_allow_send,
        llm_notifications_max_items=llm_notifications_max_items,
        llm_notifications_max_age_days=llm_notifications_max_age_days,
        llm_notifications_compact=llm_notifications_compact,
        llm_registry_snapshots_dir=llm_registry_snapshots_dir,
        llm_registry_snapshot_max_items=llm_registry_snapshot_max_items,
        llm_registry_rollback_allow=llm_registry_rollback_allow,
        llm_autopilot_safe_mode=llm_autopilot_safe_mode,
        llm_autopilot_state_path=llm_autopilot_state_path,
        llm_autopilot_churn_window_seconds=llm_autopilot_churn_window_seconds,
        llm_autopilot_churn_min_applies=llm_autopilot_churn_min_applies,
        llm_autopilot_churn_recent_limit=llm_autopilot_churn_recent_limit,
        llm_autopilot_bootstrap_allow_apply=llm_autopilot_bootstrap_allow_apply,
        llm_autopilot_ledger_path=llm_autopilot_ledger_path,
        llm_autopilot_ledger_max_items=llm_autopilot_ledger_max_items,
    )
