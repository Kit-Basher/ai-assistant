from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def enabled(self) -> bool:
        return True

    def intent_from_text(self, text: str) -> dict[str, Any] | None:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        system = (
            "You are a parser that converts user text into a JSON intent. "
            "Return a JSON object with keys: command, args. "
            "If uncertain, return null."
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
class OllamaClient(LLMClient):
    host: str
    model: str
    timeout_seconds: int = 20

    def enabled(self) -> bool:
        return bool(self.host and self.model)

    def generate(self, messages):
        # Minimal placeholder; real usage is routed via injected clients in tests.
        return ""


def _openai_generate_fallback() -> str:
    return ""


def _openai_generate_messages(messages) -> str:
    return _openai_generate_fallback()
