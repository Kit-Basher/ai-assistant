from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from agent.config import Config


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    provider_type: str
    base_url: str | None
    auth_env_var: str | None
    enabled: bool


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    model: str
    capabilities: frozenset[str]
    quality_rank: int
    cost_rank: int
    default_for: tuple[str, ...]
    enabled: bool


@dataclass(frozen=True)
class Registry:
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    routing_defaults: dict[str, Any]

    def sorted_models(self) -> list[ModelConfig]:
        return sorted(self.models.values(), key=lambda item: item.id)

    def provider_env_status(self, env: dict[str, str] | None = None) -> dict[str, bool]:
        env_map = env if env is not None else dict(os.environ)
        status: dict[str, bool] = {}
        for provider in self.providers.values():
            if provider.auth_env_var:
                status[provider.auth_env_var] = bool(env_map.get(provider.auth_env_var, "").strip())
        return status


_DEFAULT_REGISTRY: dict[str, Any] = {
    "providers": {
        "openai": {
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "auth_env_var": "OPENAI_API_KEY",
            "enabled": True,
        },
        "openrouter": {
            "provider_type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "auth_env_var": "OPENROUTER_API_KEY",
            "enabled": True,
        },
        "ollama": {
            "provider_type": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "auth_env_var": None,
            "enabled": True,
        },
    },
    "models": {
        "openai:gpt-4.1-mini": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "capabilities": ["chat", "json", "tools"],
            "quality_rank": 7,
            "cost_rank": 4,
            "default_for": ["chat", "presentation_rewrite"],
            "enabled": True,
        },
        "openai:gpt-4.1": {
            "provider": "openai",
            "model": "gpt-4.1",
            "capabilities": ["chat", "json", "tools", "vision"],
            "quality_rank": 9,
            "cost_rank": 8,
            "default_for": ["best_quality"],
            "enabled": True,
        },
        "openrouter:openai/gpt-4o-mini": {
            "provider": "openrouter",
            "model": "openai/gpt-4o-mini",
            "capabilities": ["chat", "json", "tools"],
            "quality_rank": 6,
            "cost_rank": 3,
            "default_for": ["chat"],
            "enabled": True,
        },
        "ollama:llama3": {
            "provider": "ollama",
            "model": "llama3",
            "capabilities": ["chat"],
            "quality_rank": 3,
            "cost_rank": 1,
            "default_for": ["cheap_local"],
            "enabled": True,
        },
    },
    "routing": {
        "mode": "auto",
        "fallback_chain": [],
    },
}


def _read_registry_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM registry must be a JSON object.")
    return parsed


def _merge_registry(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "providers": dict(base.get("providers") or {}),
        "models": dict(base.get("models") or {}),
        "routing": dict(base.get("routing") or {}),
    }
    if isinstance(override.get("providers"), dict):
        for name, payload in (override.get("providers") or {}).items():
            if isinstance(payload, dict):
                merged["providers"][str(name)] = {
                    **(merged["providers"].get(str(name)) or {}),
                    **payload,
                }
    if isinstance(override.get("models"), dict):
        for model_id, payload in (override.get("models") or {}).items():
            if isinstance(payload, dict):
                merged["models"][str(model_id)] = {
                    **(merged["models"].get(str(model_id)) or {}),
                    **payload,
                }
    if isinstance(override.get("routing"), dict):
        merged["routing"].update(override.get("routing") or {})
    return merged


def _default_capabilities(provider: str, model: str) -> frozenset[str]:
    provider_name = (provider or "").strip().lower()
    model_name = (model or "").strip().lower()
    if provider_name == "openai":
        caps = {"chat", "json", "tools"}
        if "vision" in model_name or "gpt-4.1" in model_name:
            caps.add("vision")
        return frozenset(caps)
    if provider_name == "openrouter":
        return frozenset({"chat", "json", "tools"})
    if provider_name == "ollama":
        return frozenset({"chat"})
    return frozenset({"chat"})


def _upsert_config_model(data: dict[str, Any], provider: str, model: str | None) -> None:
    if not model:
        return
    model_name = model.strip()
    if not model_name:
        return
    model_id = f"{provider}:{model_name}"
    existing = data["models"].get(model_id) or {}
    if provider == "openai":
        quality_rank = int(existing.get("quality_rank", 7))
        cost_rank = int(existing.get("cost_rank", 4))
    elif provider == "openrouter":
        quality_rank = int(existing.get("quality_rank", 6))
        cost_rank = int(existing.get("cost_rank", 3))
    else:
        quality_rank = int(existing.get("quality_rank", 3))
        cost_rank = int(existing.get("cost_rank", 1))
    default_for = set(existing.get("default_for") or [])
    default_for.update({"chat", "presentation_rewrite"})
    data["models"][model_id] = {
        **existing,
        "provider": provider,
        "model": model_name,
        "capabilities": list(existing.get("capabilities") or _default_capabilities(provider, model_name)),
        "quality_rank": quality_rank,
        "cost_rank": cost_rank,
        "default_for": sorted(default_for),
        "enabled": bool(existing.get("enabled", True)),
    }


def load_registry(config: Config) -> Registry:
    data = _merge_registry(_DEFAULT_REGISTRY, {})

    registry_path = config.llm_registry_path
    if registry_path:
        override = _read_registry_file(registry_path)
        data = _merge_registry(data, override)

    # Keep legacy config fields authoritative.
    provider_openai = data["providers"].get("openai") or {}
    data["providers"]["openai"] = {
        **provider_openai,
        "provider_type": "openai",
        "base_url": config.openai_base_url or provider_openai.get("base_url"),
        "auth_env_var": provider_openai.get("auth_env_var", "OPENAI_API_KEY"),
        "enabled": bool(provider_openai.get("enabled", True)),
    }

    provider_openrouter = data["providers"].get("openrouter") or {}
    data["providers"]["openrouter"] = {
        **provider_openrouter,
        "provider_type": "openai",
        "base_url": config.openrouter_base_url or provider_openrouter.get("base_url"),
        "auth_env_var": provider_openrouter.get("auth_env_var", "OPENROUTER_API_KEY"),
        "enabled": bool(provider_openrouter.get("enabled", True)),
    }

    provider_ollama = data["providers"].get("ollama") or {}
    data["providers"]["ollama"] = {
        **provider_ollama,
        "provider_type": "ollama",
        "base_url": config.ollama_base_url or config.ollama_host or provider_ollama.get("base_url"),
        "auth_env_var": provider_ollama.get("auth_env_var"),
        "enabled": bool(provider_ollama.get("enabled", True)),
    }

    _upsert_config_model(data, "openai", config.openai_model)
    _upsert_config_model(data, "openrouter", config.openrouter_model)
    _upsert_config_model(data, "ollama", config.ollama_model)

    providers: dict[str, ProviderConfig] = {}
    for provider_name, payload in sorted((data.get("providers") or {}).items()):
        if not isinstance(payload, dict):
            continue
        providers[provider_name] = ProviderConfig(
            name=provider_name,
            provider_type=str(payload.get("provider_type") or provider_name),
            base_url=(payload.get("base_url") or None),
            auth_env_var=(payload.get("auth_env_var") or None),
            enabled=bool(payload.get("enabled", True)),
        )

    models: dict[str, ModelConfig] = {}
    for model_id, payload in sorted((data.get("models") or {}).items()):
        if not isinstance(payload, dict):
            continue
        provider = str(payload.get("provider") or "").strip().lower()
        model_name = str(payload.get("model") or "").strip()
        if not provider or not model_name:
            continue
        capabilities_raw = payload.get("capabilities") or _default_capabilities(provider, model_name)
        capabilities = frozenset(str(item).strip().lower() for item in capabilities_raw if str(item).strip())
        models[model_id] = ModelConfig(
            id=str(model_id),
            provider=provider,
            model=model_name,
            capabilities=capabilities,
            quality_rank=int(payload.get("quality_rank", 0) or 0),
            cost_rank=int(payload.get("cost_rank", 0) or 0),
            default_for=tuple(str(item) for item in (payload.get("default_for") or [])),
            enabled=bool(payload.get("enabled", True)),
        )

    routing_defaults = {
        "mode": str((data.get("routing") or {}).get("mode") or "auto").strip().lower(),
        "fallback_chain": tuple(str(item) for item in ((data.get("routing") or {}).get("fallback_chain") or [])),
    }

    return Registry(
        providers=providers,
        models=models,
        routing_defaults=routing_defaults,
    )
