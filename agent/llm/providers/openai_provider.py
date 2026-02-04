from __future__ import annotations

from agent.llm.base import LLMResult


class OpenAIProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def summarize(self, _kind: str, _payload: dict, timeout_s: int = 15) -> LLMResult:
        return LLMResult("", "openai", self.model or "", True, "openai_unavailable")
