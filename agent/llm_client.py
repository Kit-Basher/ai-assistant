from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.request
from typing import Any

from agent.config import Config
from agent.llm.broker_policy import load_policy


class LLMClient:
    def enabled(self) -> bool:
        return False

    def intent_from_text(self, text: str) -> dict[str, Any] | None:
        return None

    def generate(self, messages):
        return ""

    def summarize(self, prompt: str) -> str:
        return prompt


@dataclass
class OpenAIClient(LLMClient):
    api_key: str
    model: str
    base_url: str | None = None

    def enabled(self) -> bool:
        return True

    def intent_from_text(self, text: str) -> dict[str, Any] | None:
        from openai import OpenAI

        if self.base_url:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            client = OpenAI(api_key=self.api_key)
        system = (
            "You are a parser that converts user text into a JSON intent. "
            "Return a JSON object with keys: command, args. "
            "If uncertain, return null. "
            "Output strict JSON only; no prose, no markdown."
        )
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        content = response.output_text.strip()
        if not content:
            return None
        try:
            import json

            return json.loads(content)
        except Exception:
            return None

    def summarize(self, prompt: str) -> str:
        from openai import OpenAI

        if self.base_url:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "Write a concise, friendly summary."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text.strip()

    def generate(self, messages):
        return ""


class DummyClient(LLMClient):
    def generate(self, messages):
        return ""


@dataclass
class OpenRouterClient(LLMClient):
    api_key: str
    model: str
    base_url: str
    site_url: str | None = None
    app_name: str | None = None
    timeout_seconds: int = 20
    provider: str = "openrouter"

    def enabled(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def generate(self, prompt: str) -> str:
        if not self.enabled():
            return ""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Follow the user's instructions exactly."},
                {"role": "user", "content": prompt},
            ],
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(body, ensure_ascii=True).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw or "{}")
            choices = parsed.get("choices") or []
            message = choices[0].get("message") if choices else None
            return (message or {}).get("content") or ""
        except Exception:
            return ""


@dataclass
class OllamaClient(LLMClient):
    host: str
    model: str
    timeout_seconds: int = 20

    def enabled(self) -> bool:
        return bool(self.host and self.model)

    def generate(self, messages):
        # Minimal placeholder; real usage is routed via injected clients in tests.
        return ""


class UnsupportedClient(LLMClient):
    def __init__(self, provider: str) -> None:
        self.provider = provider


def build_client_for_provider(config: Config, provider_policy) -> LLMClient:
    provider = (provider_policy.provider or "").strip().lower()
    model = provider_policy.model
    if provider == "openai":
        return OpenAIClient(
            api_key=config.openai_api_key or "",
            model=model,
            base_url=config.openai_base_url,
        )
    if provider == "ollama":
        return OllamaClient(
            host=config.ollama_base_url or config.ollama_host or "",
            model=model or "",
            timeout_seconds=config.llm_timeout_seconds,
        )
    if provider == "openrouter":
        return OpenRouterClient(
            api_key=config.openrouter_api_key or "",
            model=model or (config.openrouter_model or ""),
            base_url=config.openrouter_base_url or "",
            site_url=config.openrouter_site_url,
            app_name=config.openrouter_app_name,
            timeout_seconds=config.llm_timeout_seconds,
        )
    return UnsupportedClient(provider)


def build_llm_client(config: Config) -> LLMClient:
    provider = (config.llm_provider or "none").strip().lower()
    if provider == "none":
        return DummyClient()
    if provider == "openai":
        return OpenAIClient(
            api_key=config.openai_api_key or "",
            model=config.openai_model,
            base_url=config.openai_base_url,
        )
    if provider == "ollama":
        return OllamaClient(
            host=config.ollama_base_url or config.ollama_host or "",
            model=config.ollama_model or "",
            timeout_seconds=config.llm_timeout_seconds,
        )
    if provider == "openrouter":
        return OpenRouterClient(
            api_key=config.openrouter_api_key or "",
            model=config.openrouter_model or "",
            base_url=config.openrouter_base_url or "",
            site_url=config.openrouter_site_url,
            app_name=config.openrouter_app_name,
            timeout_seconds=config.llm_timeout_seconds,
        )
    return UnsupportedClient(provider)


def build_llm_broker(config: Config):
    if (config.llm_selector or "single") != "broker":
        return None, None
    if not config.llm_broker_policy_path:
        return None, "missing_policy_path"
    try:
        policy = load_policy(config.llm_broker_policy_path)
    except Exception as exc:  # pragma: no cover - exercised via tests
        return None, f"policy_error:{exc}"
    from agent.llm.broker import LLMBroker

    broker = LLMBroker(config, policy)
    return broker, None


def _openai_generate_fallback() -> str:
    return ""


def _openai_generate_messages(messages) -> str:
    return _openai_generate_fallback()
