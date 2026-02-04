from __future__ import annotations

from agent.llm.base import LLMResult


def ping_ollama(_host: str, timeout_s: int = 2) -> bool:
    return False


class OllamaProvider:
    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model

    def summarize(self, _kind: str, _payload: dict, timeout_s: int = 15) -> LLMResult:
        return LLMResult("", "ollama", self.model or "", True, "ollama_unavailable")
