from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class Request:
    messages: tuple[Message, ...]
    purpose: str = "chat"
    provider: str | None = None
    model: str | None = None
    require_tools: bool = False
    require_json: bool = False
    require_vision: bool = False
    tools: tuple[dict[str, Any], ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Response:
    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = None


@dataclass
class LLMError(Exception):
    kind: str
    retriable: bool
    provider: str
    status_code: int | None
    message: str
    raw: Any = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        status = f" status={self.status_code}" if self.status_code is not None else ""
        return f"{self.provider}:{self.kind}{status} {self.message}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "retriable": self.retriable,
            "provider": self.provider,
            "status_code": self.status_code,
            "message": self.message,
        }
