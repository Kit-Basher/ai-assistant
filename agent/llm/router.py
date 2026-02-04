from __future__ import annotations

import os

from agent.llm.base import LLMResult
from agent.llm.providers.ollama_provider import OllamaProvider, ping_ollama
from agent.llm.providers.openai_provider import OpenAIProvider


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

    def _available_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)

    def _available_ollama(self) -> bool:
        return bool(self.ollama_host and self.ollama_model)

    def _provider_order(self) -> list[str]:
        if self.routing == "local_only":
            return ["ollama"]
        if self.routing == "cloud_only":
            return ["openai"]
        if self.routing == "cloud_first":
            return ["openai", "ollama"]
        if self.routing == "local_first":
            return ["ollama", "openai"]
        return ["ollama", "openai"]

    def summarize(self, kind: str, payload: dict) -> LLMResult | None:
        if not self.narration_enabled:
            return None
        order = self._provider_order()
        last_error = None
        for provider_name in order:
            if provider_name == "ollama" and not self._available_ollama():
                last_error = "ollama_unavailable"
                continue
            if provider_name == "openai" and not self._available_openai():
                last_error = "openai_unavailable"
                continue

            if provider_name == "ollama":
                provider = OllamaProvider(self.ollama_host, self.ollama_model)
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
        }
