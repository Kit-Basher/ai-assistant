from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from agent.config import Config, load_config
from agent.llm.registry import RegistryStore
from agent.llm.router import LLMRouter
from agent.secret_store import SecretStore


_PROVIDER_ID_RE = re.compile(r"^[a-z0-9_-]{2,64}$")


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


class AgentRuntime:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.secret_store = SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
        self._repo_root = Path(__file__).resolve().parents[1]
        self.started_at = datetime.now(timezone.utc)
        self.started_at_iso = self.started_at.isoformat()
        self.pid = os.getpid()
        self.version = self._read_version()
        self.git_commit = self._read_git_commit()

        registry_path = config.llm_registry_path
        if not registry_path:
            registry_path = str(self._repo_root / "llm_registry.json")
        self.registry_store = RegistryStore(registry_path)
        self.webui_dist_path = Path(
            os.getenv("AGENT_WEBUI_DIST_PATH", "").strip() or str(self._repo_root / "agent" / "webui" / "dist")
        ).resolve()
        self.webui_dev_proxy = _is_truthy(os.getenv("WEBUI_DEV_PROXY"))
        self.webui_dev_url = os.getenv("WEBUI_DEV_URL", "http://127.0.0.1:1420").strip() or "http://127.0.0.1:1420"
        self.listening_url = self._default_listening_url()

        self.router: LLMRouter | None = None
        self.registry_document: dict[str, Any] = {}

        self._request_log: deque[dict[str, Any]] = deque(maxlen=100)
        self._reload_router()

    def _default_listening_url(self) -> str:
        host = os.getenv("AGENT_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
        port = os.getenv("AGENT_API_PORT", "8765").strip() or "8765"
        return f"http://{host}:{port}"

    def set_listening(self, host: str, port: int) -> None:
        self.listening_url = f"http://{host}:{int(port)}"

    def _read_version(self) -> str:
        version_file = self._repo_root / "VERSION"
        try:
            if version_file.is_file():
                version = version_file.read_text(encoding="utf-8").strip()
                if version:
                    return version
        except Exception:
            pass
        return "unknown"

    def _read_git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(self._repo_root), "rev-parse", "--short", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                commit = (result.stdout or "").strip()
                if commit:
                    return commit
        except Exception:
            pass
        return None

    def _reload_router(self) -> None:
        self.registry_document = self.registry_store.read_document()
        registry = self.registry_store.load(self.config)
        self.router = LLMRouter(
            self.config,
            registry=registry,
            log_path=self.config.log_path,
            secret_store=self.secret_store,
        )

    @property
    def _router(self) -> LLMRouter:
        assert self.router is not None
        return self.router

    def _save_registry_document(self, document: dict[str, Any]) -> None:
        self.registry_store.write_document(document)
        self._reload_router()

    def _log_request(self, endpoint: str, ok: bool, payload: dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "ok": bool(ok),
            "payload": payload,
        }
        self._request_log.appendleft(record)

    @staticmethod
    def _provider_public_payload(provider_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else None
        return {
            "id": provider_id,
            "provider_type": payload.get("provider_type"),
            "base_url": payload.get("base_url"),
            "chat_path": payload.get("chat_path"),
            "enabled": bool(payload.get("enabled", True)),
            "local": bool(payload.get("local", False)),
            "api_key_source": {
                "type": (api_source or {}).get("type"),
                "name": (api_source or {}).get("name"),
            }
            if api_source
            else None,
            "default_headers": payload.get("default_headers") or {},
            "default_query_params": payload.get("default_query_params") or {},
        }

    def _sorted_provider_ids(self) -> list[str]:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        return sorted(str(provider_id) for provider_id in providers.keys())

    def _models_for_provider(self, provider_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            if str(payload.get("provider") or "").strip().lower() != provider_id:
                continue
            rows.append({"id": model_id, **payload})
        return rows

    def _ensure_defaults(self, document: dict[str, Any]) -> dict[str, Any]:
        defaults = document.get("defaults") if isinstance(document.get("defaults"), dict) else {}
        if not isinstance(defaults, dict):
            defaults = {}
        defaults.setdefault("routing_mode", "auto")
        defaults.setdefault("default_provider", None)
        defaults.setdefault("default_model", None)
        defaults.setdefault("allow_remote_fallback", True)
        defaults.setdefault("fallback_chain", [])
        document["defaults"] = defaults
        return defaults

    def health(self) -> dict[str, Any]:
        snapshot = self._router.doctor_snapshot()
        return {
            "ok": True,
            "service": "personal-agent-api",
            "time": datetime.now(timezone.utc).isoformat(),
            "routing_mode": snapshot.get("routing_mode"),
            "configured_providers": [item.get("id") for item in snapshot.get("providers") or []],
            "registry_path": self.registry_store.path,
        }

    def models(self) -> dict[str, Any]:
        snapshot = self._router.doctor_snapshot()
        return {
            "providers": snapshot.get("providers") or [],
            "models": snapshot.get("models") or [],
            "routing_mode": snapshot.get("routing_mode"),
            "defaults": snapshot.get("defaults") or {},
            "circuits": snapshot.get("circuits") or {},
        }

    def version_info(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": self.version,
            "git_commit": self.git_commit,
            "started_at": self.started_at_iso,
            "pid": self.pid,
            "listening": self.listening_url,
        }

    def list_providers(self) -> dict[str, Any]:
        providers = self.registry_document.get("providers") if isinstance(self.registry_document.get("providers"), dict) else {}
        rows = [
            {
                **self._provider_public_payload(provider_id, payload),
                "models": self._models_for_provider(provider_id),
            }
            for provider_id, payload in sorted(providers.items())
            if isinstance(payload, dict)
        ]
        return {"providers": rows}

    @staticmethod
    def _normalize_model_payload(provider_id: str, raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model_name = str(raw.get("model") or raw.get("name") or "").strip()
        if not model_name:
            raise ValueError("model is required")
        model_id = str(raw.get("id") or f"{provider_id}:{model_name}").strip()
        capabilities = raw.get("capabilities") if isinstance(raw.get("capabilities"), list) else ["chat"]
        pricing_raw = raw.get("pricing") if isinstance(raw.get("pricing"), dict) else {}
        payload = {
            "provider": provider_id,
            "model": model_name,
            "capabilities": [str(item).strip().lower() for item in capabilities if str(item).strip()],
            "quality_rank": int(raw.get("quality_rank", 5) or 5),
            "cost_rank": int(raw.get("cost_rank", 5) or 5),
            "default_for": [str(item) for item in (raw.get("default_for") or ["chat"])],
            "enabled": bool(raw.get("enabled", True)),
            "pricing": {
                "input_per_million_tokens": pricing_raw.get("input_per_million_tokens"),
                "output_per_million_tokens": pricing_raw.get("output_per_million_tokens"),
            },
            "max_context_tokens": raw.get("max_context_tokens"),
        }
        return model_id, payload

    def add_provider(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_id = str(payload.get("id") or payload.get("provider_id") or "").strip().lower()
        if not _PROVIDER_ID_RE.match(provider_id):
            return False, {"ok": False, "error": "provider id must match [a-z0-9_-]{2,64}"}

        provider_type = str(payload.get("provider_type") or "openai_compat").strip().lower()
        if provider_type not in {"openai_compat"}:
            return False, {"ok": False, "error": "provider_type must be openai_compat"}

        base_url = str(payload.get("base_url") or "").strip()
        if not base_url:
            return False, {"ok": False, "error": "base_url is required"}

        chat_path = str(payload.get("chat_path") or "/v1/chat/completions").strip() or "/v1/chat/completions"
        if not chat_path.startswith("/"):
            chat_path = "/" + chat_path

        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        if provider_id in providers:
            return False, {"ok": False, "error": f"provider already exists: {provider_id}"}

        api_key_source = payload.get("api_key_source") if isinstance(payload.get("api_key_source"), dict) else None
        if api_key_source is None:
            auth_env_var = str(payload.get("auth_env_var") or "").strip()
            if auth_env_var:
                api_key_source = {"type": "env", "name": auth_env_var}
            elif bool(payload.get("requires_api_key", True)):
                api_key_source = {"type": "secret", "name": f"provider:{provider_id}:api_key"}

        providers[provider_id] = {
            "provider_type": provider_type,
            "base_url": base_url,
            "chat_path": chat_path,
            "api_key_source": api_key_source,
            "default_headers": payload.get("default_headers") if isinstance(payload.get("default_headers"), dict) else {},
            "default_query_params": payload.get("default_query_params")
            if isinstance(payload.get("default_query_params"), dict)
            else {},
            "enabled": bool(payload.get("enabled", True)),
            "local": bool(payload.get("local", False)),
        }

        model_items = payload.get("models") if isinstance(payload.get("models"), list) else []
        single_model = str(payload.get("model") or "").strip()
        if single_model:
            model_items.append({"model": single_model, "id": payload.get("model_id")})

        for model_raw in model_items:
            if not isinstance(model_raw, dict):
                continue
            try:
                model_id, model_payload = self._normalize_model_payload(provider_id, model_raw)
            except ValueError:
                continue
            models[model_id] = model_payload

        document["providers"] = providers
        document["models"] = models
        self._ensure_defaults(document)
        self._save_registry_document(document)

        return True, {
            "ok": True,
            "provider": self._provider_public_payload(provider_id, providers[provider_id]),
            "models": self._models_for_provider(provider_id),
        }

    def update_provider(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        if provider_key not in providers:
            return False, {"ok": False, "error": "provider not found"}

        current = dict(providers[provider_key])
        if "base_url" in payload:
            current["base_url"] = str(payload.get("base_url") or "").strip() or current.get("base_url")
        if "chat_path" in payload:
            chat_path = str(payload.get("chat_path") or "").strip() or current.get("chat_path")
            if not str(chat_path).startswith("/"):
                chat_path = "/" + str(chat_path)
            current["chat_path"] = chat_path
        if "enabled" in payload:
            current["enabled"] = bool(payload.get("enabled"))
        if "local" in payload:
            current["local"] = bool(payload.get("local"))
        if isinstance(payload.get("default_headers"), dict):
            current["default_headers"] = payload.get("default_headers")
        if isinstance(payload.get("default_query_params"), dict):
            current["default_query_params"] = payload.get("default_query_params")
        if isinstance(payload.get("api_key_source"), dict):
            current["api_key_source"] = payload.get("api_key_source")

        providers[provider_key] = current
        document["providers"] = providers
        self._save_registry_document(document)
        return True, {
            "ok": True,
            "provider": self._provider_public_payload(provider_key, current),
        }

    def delete_provider(self, provider_id: str) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        defaults = self._ensure_defaults(document)

        if provider_key not in providers:
            return False, {"ok": False, "error": "provider not found"}

        warning = None
        if str(defaults.get("default_provider") or "").strip().lower() == provider_key:
            warning = "removed provider was default_provider"
            defaults["default_provider"] = None
            if str(defaults.get("default_model") or "").startswith(f"{provider_key}:"):
                defaults["default_model"] = None

        providers.pop(provider_key, None)
        for model_id in list(models.keys()):
            model_payload = models.get(model_id)
            if isinstance(model_payload, dict) and str(model_payload.get("provider") or "").strip().lower() == provider_key:
                models.pop(model_id, None)

        document["providers"] = providers
        document["models"] = models
        document["defaults"] = defaults
        self._save_registry_document(document)

        response = {"ok": True, "deleted": provider_key}
        if warning:
            response["warning"] = warning
        return True, response

    def set_provider_secret(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        api_key = str(payload.get("api_key") or "").strip()
        if not api_key:
            return False, {"ok": False, "error": "api_key is required"}

        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_payload = providers.get(provider_key)
        if not isinstance(provider_payload, dict):
            return False, {"ok": False, "error": "provider not found"}

        source = provider_payload.get("api_key_source") if isinstance(provider_payload.get("api_key_source"), dict) else None
        if source is None or source.get("type") != "secret":
            source = {"type": "secret", "name": f"provider:{provider_key}:api_key"}
            provider_payload["api_key_source"] = source
            providers[provider_key] = provider_payload
            document["providers"] = providers
            self._save_registry_document(document)

        self.secret_store.set_secret(str(source.get("name") or ""), api_key)
        self._router.set_provider_api_key(provider_key, api_key)
        return True, {"ok": True, "provider": provider_key}

    def _provider_default_model(self, provider_id: str) -> str | None:
        models = self.registry_document.get("models") if isinstance(self.registry_document.get("models"), dict) else {}
        for model_id, payload in sorted(models.items()):
            if not isinstance(payload, dict):
                continue
            if str(payload.get("provider") or "").strip().lower() == provider_id and bool(payload.get("enabled", True)):
                return str(model_id)
        return None

    def test_provider(self, provider_id: str, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        provider_key = provider_id.strip().lower()
        model_override = str(payload.get("model") or "").strip() or self._provider_default_model(provider_key)
        if not model_override:
            return False, {"ok": False, "error": "No enabled model for provider"}

        key_override = str(payload.get("api_key") or "").strip()
        previous_source_key = None
        provider_entry = (
            self.registry_document.get("providers", {}).get(provider_key)
            if isinstance(self.registry_document.get("providers"), dict)
            else None
        )
        if isinstance(provider_entry, dict):
            source = provider_entry.get("api_key_source") if isinstance(provider_entry.get("api_key_source"), dict) else None
            if isinstance(source, dict) and source.get("type") == "secret":
                previous_source_key = self.secret_store.get_secret(str(source.get("name") or ""))

        if key_override:
            self._router.set_provider_api_key(provider_key, key_override)

        result = self._router.chat(
            [
                {"role": "system", "content": "Reply with PONG."},
                {"role": "user", "content": "ping"},
            ],
            purpose="diagnostics",
            task_type="diagnostics",
            provider_override=provider_key,
            model_override=model_override,
            timeout_seconds=float(payload.get("timeout_seconds") or 6.0),
        )

        if not result.get("ok"):
            if key_override and previous_source_key is not None:
                self._router.set_provider_api_key(provider_key, previous_source_key)
            response = {
                "ok": False,
                "provider": provider_key,
                "model": model_override,
                "error": result.get("error_class") or "provider_error",
                "message": result.get("error") or "connectivity test failed",
            }
            self._log_request(f"/providers/{provider_key}/test", False, response)
            return False, response

        if key_override and previous_source_key is None:
            # ephemeral override, do not persist unless caller uses /secret
            self._router.set_provider_api_key(provider_key, "")

        response = {
            "ok": True,
            "provider": provider_key,
            "model": result.get("model"),
            "duration_ms": int(result.get("duration_ms") or 0),
        }
        self._log_request(f"/providers/{provider_key}/test", True, response)
        return True, response

    def get_defaults(self) -> dict[str, Any]:
        defaults = self._ensure_defaults(self.registry_document)
        return {
            "routing_mode": defaults.get("routing_mode") or self._router.policy.mode,
            "default_provider": defaults.get("default_provider"),
            "default_model": defaults.get("default_model"),
            "allow_remote_fallback": bool(defaults.get("allow_remote_fallback", True)),
        }

    def update_defaults(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        valid_modes = {
            "auto",
            "prefer_cheap",
            "prefer_best",
            "prefer_local_lowest_cost_capable",
        }

        document = self.registry_document
        defaults = self._ensure_defaults(document)
        models = document.get("models") if isinstance(document.get("models"), dict) else {}
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        provider_ids = {str(provider_id).strip().lower() for provider_id in providers.keys()}
        provider_override = str(payload.get("default_provider") or "").strip().lower() if "default_provider" in payload else None

        if "routing_mode" in payload:
            mode = str(payload.get("routing_mode") or "").strip().lower()
            if mode not in valid_modes:
                return False, {"ok": False, "error": "invalid routing_mode"}
            defaults["routing_mode"] = mode

        if "default_provider" in payload:
            provider = provider_override or None
            if provider and provider not in self._sorted_provider_ids():
                return False, {"ok": False, "error": "default_provider not found"}
            defaults["default_provider"] = provider

        if "default_model" in payload:
            model = str(payload.get("default_model") or "").strip() or None
            if model is None:
                defaults["default_model"] = None
            else:
                provider_for_model = defaults.get("default_provider")
                provider_for_model = str(provider_for_model).strip().lower() if provider_for_model else None
                canonical_model = self._normalize_default_model_id(
                    model,
                    provider_for_model=provider_for_model,
                    models=models,
                    provider_ids=provider_ids,
                )
                if canonical_model is None:
                    return False, {"ok": False, "error": "default_model not found"}
                defaults["default_model"] = canonical_model
        elif (
            "default_provider" in payload
            and defaults.get("default_model")
            and str(defaults.get("default_provider") or "").strip()
        ):
            model_id = str(defaults.get("default_model") or "").strip()
            if model_id and model_id in models:
                existing_provider = str((models.get(model_id) or {}).get("provider") or "").strip().lower()
                selected_provider = str(defaults.get("default_provider") or "").strip().lower()
                if existing_provider and selected_provider and existing_provider != selected_provider:
                    defaults["default_model"] = None

        if "allow_remote_fallback" in payload:
            defaults["allow_remote_fallback"] = bool(payload.get("allow_remote_fallback"))

        document["defaults"] = defaults
        self._save_registry_document(document)
        return True, {"ok": True, **self.get_defaults()}

    @staticmethod
    def _normalize_default_model_id(
        model_value: str,
        *,
        provider_for_model: str | None,
        models: dict[str, Any],
        provider_ids: set[str],
    ) -> str | None:
        candidate = (model_value or "").strip()
        if not candidate:
            return None

        # Accept canonical full ids as-is.
        if candidate in models:
            return candidate

        # Only treat "<provider>:<name>" as canonical if the prefix is a known provider id.
        prefix = candidate.split(":", 1)[0].strip().lower() if ":" in candidate else ""
        if prefix and prefix in provider_ids:
            return None

        if not provider_for_model:
            return None

        scoped_model_id = f"{provider_for_model}:{candidate}"
        if scoped_model_id in models:
            return scoped_model_id
        return None

    @staticmethod
    def _default_refreshed_capabilities(model_name: str) -> list[str]:
        normalized_name = (model_name or "").strip().lower()
        if "embed" in normalized_name:
            return ["embedding"]
        return ["chat"]

    @staticmethod
    def _normalize_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
        raw = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return []
        messages: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user").strip() or "user"
            content = str(item.get("content") or "")
            messages.append({"role": role, "content": content})
        return messages

    def chat(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        messages = self._normalize_messages(payload)
        if not messages:
            return False, {"ok": False, "error": "messages must be a non-empty list"}

        defaults = self.get_defaults()
        model_override = str(payload.get("model") or "").strip() or defaults.get("default_model")
        provider_override = str(payload.get("provider") or "").strip().lower() or defaults.get("default_provider")

        result = self._router.chat(
            messages,
            purpose=str(payload.get("purpose") or "chat"),
            task_type=str(payload.get("task_type") or payload.get("purpose") or "chat"),
            provider_override=provider_override,
            model_override=model_override,
            require_tools=bool(payload.get("require_tools")),
            require_json=bool(payload.get("require_json")),
            require_vision=bool(payload.get("require_vision")),
            min_context_tokens=int(payload.get("min_context_tokens") or 0) or None,
            timeout_seconds=float(payload.get("timeout_seconds") or 0) or None,
        )

        response = {
            "ok": bool(result.get("ok")),
            "assistant": {
                "role": "assistant",
                "content": result.get("text") or "",
            },
            "meta": {
                "provider": result.get("provider"),
                "model": result.get("model"),
                "fallback_used": bool(result.get("fallback_used")),
                "attempts": result.get("attempts") or [],
                "duration_ms": int(result.get("duration_ms") or 0),
                "error": result.get("error_class"),
            },
        }
        self._log_request("/chat", bool(result.get("ok")), response["meta"])
        return bool(result.get("ok")), response

    @staticmethod
    def _http_get_json(url: str, timeout_seconds: float = 4.0) -> dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw or "{}")
        if isinstance(parsed, dict):
            return parsed
        return {}

    def refresh_models(self) -> tuple[bool, dict[str, Any]]:
        document = self.registry_document
        providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
        models = document.get("models") if isinstance(document.get("models"), dict) else {}

        refreshed: dict[str, list[str]] = {}

        for provider_id, payload in sorted(providers.items()):
            if not isinstance(payload, dict):
                continue
            if not bool(payload.get("enabled", True)):
                continue
            if not bool(payload.get("local", False)):
                continue

            base_url = str(payload.get("base_url") or "").rstrip("/")
            if not base_url:
                continue

            discovered_models: list[str] = []
            # Try OpenAI-compatible model listing first.
            try:
                parsed = self._http_get_json(base_url + "/v1/models")
                data = parsed.get("data") if isinstance(parsed.get("data"), list) else []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    model_name = str(item.get("id") or "").strip()
                    if model_name:
                        discovered_models.append(model_name)
            except Exception:
                discovered_models = []

            if not discovered_models:
                try:
                    parsed = self._http_get_json(base_url + "/api/tags")
                    tags = parsed.get("models") if isinstance(parsed.get("models"), list) else []
                    for item in tags:
                        if not isinstance(item, dict):
                            continue
                        model_name = str(item.get("name") or "").strip()
                        if model_name:
                            discovered_models.append(model_name)
                except Exception:
                    discovered_models = []

            if not discovered_models:
                continue

            refreshed[provider_id] = []
            for model_name in sorted(set(discovered_models)):
                model_id = f"{provider_id}:{model_name}"
                refreshed[provider_id].append(model_id)
                existing = models.get(model_id) if isinstance(models.get(model_id), dict) else {}
                models[model_id] = {
                    **existing,
                    "provider": provider_id,
                    "model": model_name,
                    "capabilities": list(existing.get("capabilities") or self._default_refreshed_capabilities(model_name)),
                    "quality_rank": int(existing.get("quality_rank", 2) or 2),
                    "cost_rank": int(existing.get("cost_rank", 0) or 0),
                    "default_for": list(existing.get("default_for") or ["chat"]),
                    "enabled": bool(existing.get("enabled", True)),
                    "pricing": existing.get("pricing")
                    or {
                        "input_per_million_tokens": None,
                        "output_per_million_tokens": None,
                    },
                    "max_context_tokens": existing.get("max_context_tokens"),
                }

        document["models"] = models
        self._save_registry_document(document)
        return True, {"ok": True, "refreshed": refreshed, "models": self.models().get("models")}

    def get_config(self) -> dict[str, Any]:
        defaults = self.get_defaults()
        return {
            "routing_mode": defaults.get("routing_mode"),
            "retry_attempts": self._router.policy.retry_attempts,
            "timeout_seconds": self._router.policy.default_timeout_seconds,
            "secret_storage": self.secret_store.backend_name,
        }

    def update_config(self, payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if "routing_mode" not in payload:
            return False, {"ok": False, "error": "routing_mode is required"}
        ok, updated = self.update_defaults({"routing_mode": payload.get("routing_mode")})
        if not ok:
            return False, updated
        return True, {"ok": True, "routing_mode": updated.get("routing_mode")}

    def webui_dev_landing_html(self) -> str:
        return (
            "<!doctype html>"
            "<html><head><meta charset='utf-8'><meta name='personal-agent-webui' content='1'>"
            "<title>Personal Agent Web UI (Dev)</title></head>"
            "<body style='font-family: sans-serif; padding: 2rem;'>"
            "<h1>Personal Agent Web UI (Dev Mode)</h1>"
            f"<p>WEBUI_DEV_PROXY=1 is enabled. Open <a href='{self.webui_dev_url}'>{self.webui_dev_url}</a>.</p>"
            "<p>The API is still available at this host for /health, /providers, /chat, and related endpoints.</p>"
            "</body></html>"
        )

    def webui_missing_html(self) -> str:
        return (
            "<!doctype html>"
            "<html><head><meta charset='utf-8'><meta name='personal-agent-webui' content='1'>"
            "<title>Personal Agent Web UI</title></head>"
            "<body style='font-family: sans-serif; padding: 2rem;'>"
            "<h1>Personal Agent Web UI</h1>"
            "<p>UI assets are missing.</p>"
            "<p>Build them with:</p>"
            "<pre>./scripts/build_webui.sh</pre>"
            "</body></html>"
        )

    def resolve_webui_file(self, request_path: str) -> tuple[Path, str] | None:
        clean_path = (request_path or "/").split("?", 1)[0].split("#", 1)[0]
        rel_path = clean_path.lstrip("/") or "index.html"

        # Keep serving scope narrow and explicit.
        if rel_path != "index.html" and not rel_path.startswith("assets/") and "/" in rel_path:
            return None
        if ".." in rel_path.split("/"):
            return None

        candidate = (self.webui_dist_path / rel_path).resolve()
        try:
            candidate.relative_to(self.webui_dist_path)
        except ValueError:
            return None

        if not candidate.is_file():
            return None

        if rel_path.startswith("assets/"):
            cache_control = "public, max-age=31536000, immutable"
        elif rel_path == "index.html":
            cache_control = "no-cache"
        else:
            cache_control = "public, max-age=3600"
        return candidate, cache_control


class APIServerHandler(BaseHTTPRequestHandler):
    runtime: AgentRuntime

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - avoid noisy stdout in tests
        _ = format
        _ = args

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
        return {}

    def _handle_internal_error(self, method: str, exc: Exception) -> None:
        print(
            f"api_server internal_error method={method} path={getattr(self, 'path', '')} "
            f"error={exc.__class__.__name__}",
            file=sys.stderr,
            flush=True,
        )
        try:
            self._send_json(500, {"ok": False, "error": "internal_error"})
        except Exception:
            pass

    def _path_parts(self) -> tuple[str, list[str]]:
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path or "/"
        parts = [part for part in clean_path.split("/") if part]
        return clean_path, parts

    def do_OPTIONS(self) -> None:  # noqa: N802
        try:
            self._send_json(200, {"ok": True})
        except Exception as exc:
            self._handle_internal_error("OPTIONS", exc)

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            if path == "/health":
                self._send_json(200, self.runtime.health())
                return
            if path == "/version":
                self._send_json(200, self.runtime.version_info())
                return
            if path == "/models":
                self._send_json(200, self.runtime.models())
                return
            if path == "/config":
                self._send_json(200, self.runtime.get_config())
                return
            if path == "/defaults":
                self._send_json(200, self.runtime.get_defaults())
                return
            if path == "/providers":
                self._send_json(200, self.runtime.list_providers())
                return

            if self._try_serve_webui(path):
                return
            self._send_json(404, {"ok": False, "error": "not_found", "path": path, "parts": parts})
        except Exception as exc:
            self._handle_internal_error("GET", exc)

    def _try_serve_webui(self, path: str) -> bool:
        if path == "/" and self.runtime.webui_dev_proxy:
            html = self.runtime.webui_dev_landing_html().encode("utf-8")
            self._send_bytes(
                200,
                html,
                content_type="text/html; charset=utf-8",
                cache_control="no-store",
            )
            return True

        if path == "/":
            resolved = self.runtime.resolve_webui_file(path)
            if resolved is not None:
                self._send_static_file(*resolved)
            else:
                fallback = self.runtime.webui_missing_html().encode("utf-8")
                self._send_bytes(
                    200,
                    fallback,
                    content_type="text/html; charset=utf-8",
                    cache_control="no-store",
                )
            return True

        if path.startswith("/assets/") or (path.count("/") == 1 and "." in path.rsplit("/", 1)[-1]):
            resolved = self.runtime.resolve_webui_file(path)
            if resolved is None:
                return False
            self._send_static_file(*resolved)
            return True

        return False

    def _send_static_file(self, file_path: Path, cache_control: str) -> None:
        try:
            body = file_path.read_bytes()
        except Exception:
            self._send_json(500, {"ok": False, "error": "static_read_failed"})
            return

        guessed_content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = guessed_content_type or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type = f"{content_type}; charset=utf-8"

        self._send_bytes(200, body, content_type=content_type, cache_control=cache_control)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            payload = self._read_json()

            if path == "/chat":
                ok, body = self.runtime.chat(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/providers":
                ok, body = self.runtime.add_provider(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/providers/test":
                # backward-compatible endpoint
                provider_id = str(payload.get("provider") or "").strip().lower()
                if not provider_id:
                    self._send_json(400, {"ok": False, "error": "provider is required"})
                    return
                ok, body = self.runtime.test_provider(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/models/refresh":
                ok, body = self.runtime.refresh_models()
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 3 and parts[0] == "providers" and parts[2] == "secret":
                provider_id = parts[1]
                ok, body = self.runtime.set_provider_secret(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 3 and parts[0] == "providers" and parts[2] == "test":
                provider_id = parts[1]
                ok, body = self.runtime.test_provider(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            self._send_json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._handle_internal_error("POST", exc)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            payload = self._read_json()

            if path == "/config":
                ok, body = self.runtime.update_config(payload)
                self._send_json(200 if ok else 400, body)
                return

            if path == "/defaults":
                ok, body = self.runtime.update_defaults(payload)
                self._send_json(200 if ok else 400, body)
                return

            if len(parts) == 2 and parts[0] == "providers":
                provider_id = parts[1]
                ok, body = self.runtime.update_provider(provider_id, payload)
                self._send_json(200 if ok else 400, body)
                return

            self._send_json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._handle_internal_error("PUT", exc)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            path, parts = self._path_parts()
            if len(parts) == 2 and parts[0] == "providers":
                provider_id = parts[1]
                ok, body = self.runtime.delete_provider(provider_id)
                self._send_json(200 if ok else 400, body)
                return
            self._send_json(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._handle_internal_error("DELETE", exc)


def build_runtime(config: Config | None = None) -> AgentRuntime:
    loaded = config or load_config(require_telegram_token=False)
    return AgentRuntime(loaded)


def run_server(host: str, port: int) -> None:
    runtime = build_runtime()
    runtime.set_listening(host, port)

    class _Handler(APIServerHandler):
        pass

    _Handler.runtime = runtime

    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        print(
            f"Failed to bind Personal Agent API on {runtime.listening_url}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc

    print(
        f"Personal Agent API started pid={runtime.pid} listening={runtime.listening_url} "
        f"registry_path={runtime.registry_store.path} version={runtime.version} "
        f"git_commit={runtime.git_commit or 'unknown'}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Personal Agent HTTP API")
    parser.add_argument("--host", default=os.getenv("AGENT_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AGENT_API_PORT", "8765")))
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
