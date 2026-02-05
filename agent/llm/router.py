from __future__ import annotations

import os

from agent.llm.base import LLMResult
from agent.llm.providers.ollama_provider import OllamaProvider, ping_ollama
from agent.llm.providers.openai_provider import OpenAIProvider
from agent.llm.providers.openrouter_provider import OpenRouterProvider


class LLMNarrationRouter:
    def __init__(self) -> None:
        narration_flag = os.getenv("ENABLE_NARRATION", "").strip().lower()
        legacy_flag = os.getenv("LLM_NARRATION_ENABLED", "").strip().lower()
        self.narration_enabled = (narration_flag or legacy_flag) in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.routing = os.getenv("LLM_ROUTING", "auto").strip().lower() or "auto"
        self.timeout_s = int(os.getenv("LLM_TIMEOUT_SECONDS", "15") or 15)

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "").strip()

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openrouter_base_url = (
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        )
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
        self.openrouter_site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        self.openrouter_app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()

    def _available_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)

    def _available_ollama(self) -> bool:
        return bool(self.ollama_host and self.ollama_model)

    def _available_openrouter(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_base_url and self.openrouter_model)

    def _upgrade_requested(self, payload: dict | None) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("upgrade") or payload.get("require_big_model"):
            return True
        tier = payload.get("tier_required")
        return tier == "high"

    def _cloud_order(self, upgrade: bool) -> list[str]:
        if upgrade:
            return ["openrouter", "openai"]
        return ["openai", "openrouter"]

    def _provider_order(self, payload: dict | None = None) -> list[str]:
        upgrade = self._upgrade_requested(payload)
        if self.routing == "local_only":
            return ["ollama"]
        if self.routing == "cloud_only":
            return self._cloud_order(upgrade)
        if self.routing == "cloud_first":
            return self._cloud_order(upgrade) + ["ollama"]
        if self.routing == "local_first":
            return ["ollama"] + self._cloud_order(upgrade)
        if upgrade:
            return self._cloud_order(upgrade) + ["ollama"]
        return ["ollama", "openai"]

    def summarize(self, kind: str, payload: dict) -> LLMResult | None:
        if not self.narration_enabled:
            return None
        order = self._provider_order(payload)
        last_error = None
        for provider_name in order:
            if provider_name == "ollama" and not self._available_ollama():
                last_error = "ollama_unavailable"
                continue
            if provider_name == "openai" and not self._available_openai():
                last_error = "openai_unavailable"
                continue
            if provider_name == "openrouter" and not self._available_openrouter():
                last_error = "openrouter_unavailable"
                continue

            if provider_name == "ollama":
                provider = OllamaProvider(self.ollama_host, self.ollama_model)
            elif provider_name == "openrouter":
                provider = OpenRouterProvider(
                    self.openrouter_api_key,
                    self.openrouter_model,
                    self.openrouter_base_url,
                    site_url=self.openrouter_site_url or None,
                    app_name=self.openrouter_app_name or None,
                )
            else:
                provider = OpenAIProvider(self.openai_api_key, self.openai_model)

            result = provider.summarize(kind, payload, timeout_s=self.timeout_s)
            if result.text and not result.degraded:
                return result
            if result.text:
                return result
            last_error = result.error or "provider_failed"

        return LLMResult("", "", "", True, last_error or "no_providers")

    def status(self) -> dict[str, str]:
        ollama_reachable = ping_ollama(self.ollama_host, timeout_s=2)
        return {
            "narration_enabled": "true" if self.narration_enabled else "false",
            "routing": self.routing,
            "openai_configured": "yes" if self._available_openai() else "no",
            "openai_model": self.openai_model or "none",
            "ollama_reachable": "yes" if ollama_reachable else "no",
            "ollama_model": self.ollama_model or "none",
            "openrouter_configured": "yes" if self._available_openrouter() else "no",
            "openrouter_model": self.openrouter_model or "none",
        }
