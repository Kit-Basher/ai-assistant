from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import Config


_REGISTRY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class APIKeySource:
    source_type: str
    name: str


@dataclass(frozen=True)
class ProviderConfig:
    id: str
    provider_type: str
    base_url: str
    chat_path: str
    api_key_source: APIKeySource | None
    default_headers: dict[str, Any]
    default_query_params: dict[str, Any]
    enabled: bool
    local: bool

    @property
    def name(self) -> str:
        return self.id

    @property
    def auth_env_var(self) -> str | None:
        if self.api_key_source and self.api_key_source.source_type == "env":
            return self.api_key_source.name
        return None


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
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    max_context_tokens: int | None = None


@dataclass(frozen=True)
class DefaultsConfig:
    routing_mode: str
    default_provider: str | None
    default_model: str | None
    allow_remote_fallback: bool


@dataclass(frozen=True)
class Registry:
    schema_version: int
    path: str | None
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    defaults: DefaultsConfig
    fallback_chain: tuple[str, ...]

    def sorted_models(self) -> list[ModelConfig]:
        return sorted(self.models.values(), key=lambda item: item.id)

    def provider_env_status(self, env: dict[str, str] | None = None) -> dict[str, bool]:
        env_map = env if env is not None else dict(os.environ)
        status: dict[str, bool] = {}
        for provider in self.providers.values():
            if provider.auth_env_var:
                status[provider.auth_env_var] = bool(env_map.get(provider.auth_env_var, "").strip())
        return status

    def to_document(self) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for provider_id, provider in sorted(self.providers.items()):
            providers[provider_id] = {
                "provider_type": provider.provider_type,
                "base_url": provider.base_url,
                "chat_path": provider.chat_path,
                "api_key_source": (
                    {
                        "type": provider.api_key_source.source_type,
                        "name": provider.api_key_source.name,
                    }
                    if provider.api_key_source
                    else None
                ),
                "default_headers": copy.deepcopy(provider.default_headers),
                "default_query_params": copy.deepcopy(provider.default_query_params),
                "enabled": provider.enabled,
                "local": provider.local,
            }

        models: dict[str, Any] = {}
        for model_id, model in sorted(self.models.items()):
            models[model_id] = {
                "provider": model.provider,
                "model": model.model,
                "capabilities": sorted(model.capabilities),
                "quality_rank": model.quality_rank,
                "cost_rank": model.cost_rank,
                "default_for": list(model.default_for),
                "enabled": model.enabled,
                "pricing": {
                    "input_per_million_tokens": model.input_cost_per_million_tokens,
                    "output_per_million_tokens": model.output_cost_per_million_tokens,
                },
                "max_context_tokens": model.max_context_tokens,
            }

        return {
            "schema_version": _REGISTRY_SCHEMA_VERSION,
            "providers": providers,
            "models": models,
            "defaults": {
                "routing_mode": self.defaults.routing_mode,
                "default_provider": self.defaults.default_provider,
                "default_model": self.defaults.default_model,
                "allow_remote_fallback": self.defaults.allow_remote_fallback,
                "fallback_chain": list(self.fallback_chain),
            },
        }


def _default_registry_document() -> dict[str, Any]:
    return {
        "schema_version": _REGISTRY_SCHEMA_VERSION,
        "providers": {
            "openai": {
                "provider_type": "openai_compat",
                "base_url": "https://api.openai.com",
                "chat_path": "/v1/chat/completions",
                "api_key_source": {"type": "env", "name": "OPENAI_API_KEY"},
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
            "openrouter": {
                "provider_type": "openai_compat",
                "base_url": "https://openrouter.ai/api",
                "chat_path": "/v1/chat/completions",
                "api_key_source": {"type": "env", "name": "OPENROUTER_API_KEY"},
                "default_headers": {
                    "HTTP-Referer": {"from_env": "OPENROUTER_SITE_URL"},
                    "X-Title": {"from_env": "OPENROUTER_APP_NAME"},
                },
                "default_query_params": {},
                "enabled": True,
                "local": False,
            },
            "ollama": {
                "provider_type": "openai_compat",
                "base_url": "http://127.0.0.1:11434",
                "chat_path": "/v1/chat/completions",
                "api_key_source": None,
                "default_headers": {},
                "default_query_params": {},
                "enabled": True,
                "local": True,
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
                "pricing": {
                    "input_per_million_tokens": 0.4,
                    "output_per_million_tokens": 1.6,
                },
                "max_context_tokens": 128000,
            },
            "openai:gpt-4.1": {
                "provider": "openai",
                "model": "gpt-4.1",
                "capabilities": ["chat", "json", "tools", "vision"],
                "quality_rank": 9,
                "cost_rank": 8,
                "default_for": ["best_quality"],
                "enabled": True,
                "pricing": {
                    "input_per_million_tokens": 2.0,
                    "output_per_million_tokens": 8.0,
                },
                "max_context_tokens": 128000,
            },
            "openai:gpt-4o-mini": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "capabilities": ["chat", "json", "tools"],
                "quality_rank": 6,
                "cost_rank": 3,
                "default_for": ["chat"],
                "enabled": True,
                "pricing": {
                    "input_per_million_tokens": 0.15,
                    "output_per_million_tokens": 0.6,
                },
                "max_context_tokens": 128000,
            },
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat", "json", "tools"],
                "quality_rank": 6,
                "cost_rank": 3,
                "default_for": ["chat"],
                "enabled": True,
                "pricing": {
                    "input_per_million_tokens": 0.15,
                    "output_per_million_tokens": 0.6,
                },
                "max_context_tokens": 128000,
            },
            "ollama:llama3": {
                "provider": "ollama",
                "model": "llama3",
                "capabilities": ["chat"],
                "quality_rank": 3,
                "cost_rank": 1,
                "default_for": ["cheap_local"],
                "enabled": True,
                "pricing": {
                    "input_per_million_tokens": None,
                    "output_per_million_tokens": None,
                },
                "max_context_tokens": 8192,
            },
        },
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": None,
            "default_model": None,
            "allow_remote_fallback": True,
            "fallback_chain": [
                "ollama:llama3",
                "openai:gpt-4o-mini",
                "openrouter:openai/gpt-4o-mini",
                "openai:gpt-4.1-mini",
                "openai:gpt-4.1",
            ],
        },
    }


def _read_registry_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM registry must be a JSON object.")
    return parsed


def _normalize_api_key_source(payload: Any) -> APIKeySource | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        source_type = str(payload.get("type") or payload.get("source_type") or "").strip().lower()
        name = str(payload.get("name") or "").strip()
        if source_type in {"env", "secret"} and name:
            return APIKeySource(source_type=source_type, name=name)
    return None


def _default_capabilities(provider: str, model: str) -> frozenset[str]:
    provider_name = (provider or "").strip().lower()
    model_name = (model or "").strip().lower()
    if provider_name in {"openai", "openrouter"}:
        caps = {"chat", "json", "tools"}
        if "vision" in model_name or "gpt-4.1" in model_name:
            caps.add("vision")
        return frozenset(caps)
    if provider_name == "ollama":
        return frozenset({"chat"})
    return frozenset({"chat"})


def _migrate_v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    providers_v1 = raw.get("providers") or {}
    models_v1 = raw.get("models") or {}
    routing_v1 = raw.get("routing") or {}

    migrated = _default_registry_document()

    providers_out: dict[str, Any] = {}
    for provider_id, payload in providers_v1.items():
        if not isinstance(payload, dict):
            continue
        provider_type = str(payload.get("provider_type") or "openai_compat").strip().lower()
        if provider_type in {"openai", "ollama", "openrouter"}:
            provider_type = "openai_compat"

        auth_env_var = str(payload.get("auth_env_var") or "").strip() or None
        api_key_source = {"type": "env", "name": auth_env_var} if auth_env_var else None

        provider_id_norm = str(provider_id).strip().lower()
        local = provider_id_norm == "ollama"

        providers_out[provider_id_norm] = {
            "provider_type": provider_type,
            "base_url": str(payload.get("base_url") or migrated["providers"].get(provider_id_norm, {}).get("base_url") or "").strip(),
            "chat_path": "/v1/chat/completions",
            "api_key_source": api_key_source,
            "default_headers": {},
            "default_query_params": {},
            "enabled": bool(payload.get("enabled", True)),
            "local": bool(payload.get("local", local)),
        }

    models_out: dict[str, Any] = {}
    for model_id, payload in models_v1.items():
        if not isinstance(payload, dict):
            continue
        provider = str(payload.get("provider") or "").strip().lower()
        model_name = str(payload.get("model") or "").strip()
        if not provider or not model_name:
            continue
        models_out[str(model_id)] = {
            "provider": provider,
            "model": model_name,
            "capabilities": list(payload.get("capabilities") or _default_capabilities(provider, model_name)),
            "quality_rank": int(payload.get("quality_rank", 0) or 0),
            "cost_rank": int(payload.get("cost_rank", 0) or 0),
            "default_for": list(payload.get("default_for") or []),
            "enabled": bool(payload.get("enabled", True)),
            "pricing": {
                "input_per_million_tokens": None,
                "output_per_million_tokens": None,
            },
            "max_context_tokens": None,
        }

    migrated["providers"].update(providers_out)
    migrated["models"].update(models_out)
    migrated["defaults"] = {
        "routing_mode": str(routing_v1.get("mode") or "auto").strip().lower() or "auto",
        "default_provider": None,
        "default_model": None,
        "allow_remote_fallback": True,
        "fallback_chain": list(routing_v1.get("fallback_chain") or []),
    }

    return migrated


def _normalize_document(raw: dict[str, Any]) -> dict[str, Any]:
    if int(raw.get("schema_version", 0) or 0) >= _REGISTRY_SCHEMA_VERSION:
        merged = _default_registry_document()
        merged.update({"schema_version": _REGISTRY_SCHEMA_VERSION})

        providers = merged.get("providers") or {}
        providers_raw = raw.get("providers") if isinstance(raw.get("providers"), dict) else {}
        for provider_id, payload in providers_raw.items():
            if isinstance(payload, dict):
                existing = providers.get(provider_id, {})
                providers[provider_id] = {**existing, **payload}

        models = merged.get("models") or {}
        models_raw = raw.get("models") if isinstance(raw.get("models"), dict) else {}
        for model_id, payload in models_raw.items():
            if isinstance(payload, dict):
                existing = models.get(model_id, {})
                models[model_id] = {**existing, **payload}

        defaults = merged.get("defaults") or {}
        defaults_raw = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
        defaults.update(defaults_raw)

        merged["providers"] = providers
        merged["models"] = models
        merged["defaults"] = defaults
        return merged

    return _migrate_v1_to_v2(raw)


def _upsert_config_model(data: dict[str, Any], provider: str, model: str | None) -> None:
    if not model:
        return
    model_name = model.strip()
    if not model_name:
        return

    model_id = f"{provider}:{model_name}"
    existing = (data.get("models") or {}).get(model_id) or {}
    if provider == "openai":
        quality_rank = int(existing.get("quality_rank", 7) or 7)
        cost_rank = int(existing.get("cost_rank", 4) or 4)
    elif provider == "openrouter":
        quality_rank = int(existing.get("quality_rank", 6) or 6)
        cost_rank = int(existing.get("cost_rank", 3) or 3)
    else:
        quality_rank = int(existing.get("quality_rank", 3) or 3)
        cost_rank = int(existing.get("cost_rank", 1) or 1)

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
        "pricing": existing.get("pricing")
        or {
            "input_per_million_tokens": None,
            "output_per_million_tokens": None,
        },
        "max_context_tokens": existing.get("max_context_tokens"),
    }


def _apply_legacy_env_overrides(data: dict[str, Any], config: Config) -> None:
    providers = data.get("providers") or {}

    if "openai" in providers:
        providers["openai"] = {
            **providers["openai"],
            "provider_type": "openai_compat",
            "base_url": config.openai_base_url or providers["openai"].get("base_url"),
            "api_key_source": providers["openai"].get("api_key_source") or {"type": "env", "name": "OPENAI_API_KEY"},
            "enabled": bool(providers["openai"].get("enabled", True)),
            "local": bool(providers["openai"].get("local", False)),
        }

    if "openrouter" in providers:
        providers["openrouter"] = {
            **providers["openrouter"],
            "provider_type": "openai_compat",
            "base_url": config.openrouter_base_url or providers["openrouter"].get("base_url"),
            "api_key_source": providers["openrouter"].get("api_key_source")
            or {"type": "env", "name": "OPENROUTER_API_KEY"},
            "enabled": bool(providers["openrouter"].get("enabled", True)),
            "local": bool(providers["openrouter"].get("local", False)),
        }

    if "ollama" in providers:
        providers["ollama"] = {
            **providers["ollama"],
            "provider_type": "openai_compat",
            "base_url": config.ollama_base_url or config.ollama_host or providers["ollama"].get("base_url"),
            "api_key_source": None,
            "enabled": bool(providers["ollama"].get("enabled", True)),
            "local": True,
        }

    _upsert_config_model(data, "openai", config.openai_model)
    _upsert_config_model(data, "openrouter", config.openrouter_model)
    _upsert_config_model(data, "ollama", config.ollama_model)


def _parse_registry(data: dict[str, Any], path: str | None) -> Registry:
    providers: dict[str, ProviderConfig] = {}
    for provider_id, payload in sorted((data.get("providers") or {}).items()):
        if not isinstance(payload, dict):
            continue
        provider_key = str(provider_id).strip().lower()
        if not provider_key:
            continue
        provider_type = str(payload.get("provider_type") or "openai_compat").strip().lower() or "openai_compat"
        base_url = str(payload.get("base_url") or "").strip()
        chat_path = str(payload.get("chat_path") or "/v1/chat/completions").strip() or "/v1/chat/completions"
        if not chat_path.startswith("/"):
            chat_path = "/" + chat_path
        providers[provider_key] = ProviderConfig(
            id=provider_key,
            provider_type=provider_type,
            base_url=base_url,
            chat_path=chat_path,
            api_key_source=_normalize_api_key_source(payload.get("api_key_source")),
            default_headers=copy.deepcopy(payload.get("default_headers") or {}),
            default_query_params=copy.deepcopy(payload.get("default_query_params") or {}),
            enabled=bool(payload.get("enabled", True)),
            local=bool(payload.get("local", False)),
        )

    models: dict[str, ModelConfig] = {}
    for model_id, payload in sorted((data.get("models") or {}).items()):
        if not isinstance(payload, dict):
            continue
        model_key = str(model_id).strip()
        provider = str(payload.get("provider") or "").strip().lower()
        model_name = str(payload.get("model") or "").strip()
        if not model_key or not provider or not model_name:
            continue

        capabilities_raw = payload.get("capabilities") or _default_capabilities(provider, model_name)
        capabilities = frozenset(str(item).strip().lower() for item in capabilities_raw if str(item).strip())

        pricing = payload.get("pricing") if isinstance(payload.get("pricing"), dict) else {}
        input_price = pricing.get("input_per_million_tokens")
        output_price = pricing.get("output_per_million_tokens")

        max_context_tokens = payload.get("max_context_tokens")
        max_ctx = int(max_context_tokens) if isinstance(max_context_tokens, int) else None

        models[model_key] = ModelConfig(
            id=model_key,
            provider=provider,
            model=model_name,
            capabilities=capabilities,
            quality_rank=int(payload.get("quality_rank", 0) or 0),
            cost_rank=int(payload.get("cost_rank", 0) or 0),
            default_for=tuple(str(item) for item in (payload.get("default_for") or [])),
            enabled=bool(payload.get("enabled", True)),
            input_cost_per_million_tokens=(float(input_price) if input_price is not None else None),
            output_cost_per_million_tokens=(float(output_price) if output_price is not None else None),
            max_context_tokens=max_ctx,
        )

    defaults_raw = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    defaults = DefaultsConfig(
        routing_mode=str(defaults_raw.get("routing_mode") or "auto").strip().lower() or "auto",
        default_provider=str(defaults_raw.get("default_provider") or "").strip().lower() or None,
        default_model=str(defaults_raw.get("default_model") or "").strip() or None,
        allow_remote_fallback=bool(defaults_raw.get("allow_remote_fallback", True)),
    )

    fallback_chain_raw = defaults_raw.get("fallback_chain") if isinstance(defaults_raw.get("fallback_chain"), list) else []
    fallback_chain = tuple(str(item) for item in fallback_chain_raw if str(item))

    return Registry(
        schema_version=_REGISTRY_SCHEMA_VERSION,
        path=path,
        providers=providers,
        models=models,
        defaults=defaults,
        fallback_chain=fallback_chain,
    )


def load_registry_document(path: str | None) -> dict[str, Any]:
    if not path:
        return _default_registry_document()
    if not os.path.isfile(path):
        return _default_registry_document()
    raw = _read_registry_file(path)
    return _normalize_document(raw)


def load_registry(config: Config) -> Registry:
    data = load_registry_document(config.llm_registry_path)
    _apply_legacy_env_overrides(data, config)
    return _parse_registry(data, config.llm_registry_path)


def save_registry_document(path: str, document: dict[str, Any]) -> None:
    normalized = _normalize_document(document)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


class RegistryStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def load(self, config: Config) -> Registry:
        data = load_registry_document(self.path)
        _apply_legacy_env_overrides(data, config)
        return _parse_registry(data, self.path)

    def save(self, registry: Registry) -> None:
        save_registry_document(self.path, registry.to_document())

    def read_document(self) -> dict[str, Any]:
        return load_registry_document(self.path)

    def write_document(self, document: dict[str, Any]) -> None:
        save_registry_document(self.path, document)
