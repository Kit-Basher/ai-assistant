from __future__ import annotations

from abc import ABC, abstractmethod

from agent.llm.types import Request, Response


class Provider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def chat(self, request: Request, *, model: str, timeout_seconds: float) -> Response:
        raise NotImplementedError

    def test_connection(self, *, model: str, timeout_seconds: float = 5.0) -> tuple[bool, str | None]:
        try:
            self.chat(
                Request(messages=()),
                model=model,
                timeout_seconds=timeout_seconds,
            )
            return True, None
        except Exception as exc:  # pragma: no cover - default path
            return False, str(exc)
