from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    degraded: bool
    error: str | None = None
