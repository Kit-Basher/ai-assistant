from __future__ import annotations

from typing import Any, Callable

from agent.llm.providers.base import Provider
from agent.llm.types import LLMError, Message, Request, Response, ToolCall, Usage


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class OpenAIProvider(Provider):
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None = None,
        provider_name: str = "openai",
        enabled: bool = True,
        client_factory: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = (base_url or "").strip() or None
        self._provider_name = provider_name
        self._enabled = bool(enabled)
        self._client_factory = client_factory

    @property
    def name(self) -> str:
        return self._provider_name

    def set_api_key(self, api_key: str | None) -> None:
        self._api_key = (api_key or "").strip()

    def available(self) -> bool:
        return self._enabled and bool(self._api_key)

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory(self._api_key, self._base_url)
        from openai import OpenAI

        if self._base_url:
            return OpenAI(api_key=self._api_key, base_url=self._base_url)
        return OpenAI(api_key=self._api_key)

    @staticmethod
    def _to_input(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        if not messages:
            return [{"role": "user", "content": "ping"}]
        payload: list[dict[str, Any]] = []
        for msg in messages:
            row: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                row["name"] = msg.name
            if msg.tool_call_id:
                row["tool_call_id"] = msg.tool_call_id
            payload.append(row)
        return payload

    @staticmethod
    def _extract_usage(raw: Any) -> Usage:
        usage = getattr(raw, "usage", None)
        if usage is None:
            return Usage()
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _extract_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
        output = getattr(raw, "output", None)
        if not isinstance(output, list):
            return ()
        calls: list[ToolCall] = []
        for item in output:
            if getattr(item, "type", None) != "function_call":
                continue
            call_id = _as_text(getattr(item, "id", ""))
            name = _as_text(getattr(item, "name", ""))
            args = _as_text(getattr(item, "arguments", ""))
            if call_id and name:
                calls.append(ToolCall(id=call_id, name=name, arguments=args))
        return tuple(calls)

    @staticmethod
    def _extract_text(raw: Any) -> str:
        output_text = _as_text(getattr(raw, "output_text", ""))
        if output_text.strip():
            return output_text.strip()

        output = getattr(raw, "output", None)
        if not isinstance(output, list):
            return ""

        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                if getattr(part, "type", None) in {"output_text", "text"}:
                    text = _as_text(getattr(part, "text", ""))
                    if text:
                        chunks.append(text)
        return "\n".join(chunks).strip()

    def _normalize_exception(self, exc: Exception) -> LLMError:
        status_code = getattr(exc, "status_code", None)
        message = _as_text(exc) or exc.__class__.__name__
        exc_name = exc.__class__.__name__.lower()

        kind = "provider_error"
        retriable = False

        if isinstance(exc, TimeoutError) or "timeout" in exc_name:
            kind = "timeout"
            retriable = True
        elif "rate" in exc_name and "limit" in exc_name:
            kind = "rate_limit"
            retriable = True
        elif "connection" in exc_name or "apierror" in exc_name and status_code is None:
            kind = "network"
            retriable = True
        elif status_code == 429:
            kind = "rate_limit"
            retriable = True
        elif status_code is not None and 500 <= int(status_code) <= 599:
            kind = "server_error"
            retriable = True
        elif status_code in {401, 403}:
            kind = "auth_error"
        elif status_code is not None and 400 <= int(status_code) <= 499:
            kind = "bad_request"

        return LLMError(
            kind=kind,
            retriable=retriable,
            provider=self.name,
            status_code=int(status_code) if status_code is not None else None,
            message=message,
            raw={"type": exc.__class__.__name__},
        )

    def chat(self, request: Request, *, model: str, timeout_seconds: float) -> Response:
        if not self.available():
            raise LLMError(
                kind="auth_error",
                retriable=False,
                provider=self.name,
                status_code=None,
                message="Provider is not configured.",
                raw=None,
            )

        payload: dict[str, Any] = {
            "model": model,
            "input": self._to_input(request.messages),
        }
        if request.temperature is not None:
            payload["temperature"] = float(request.temperature)
        if request.max_tokens is not None:
            payload["max_output_tokens"] = int(request.max_tokens)
        if request.tools:
            payload["tools"] = list(request.tools)
        if request.require_json:
            payload["text"] = {"format": {"type": "json_object"}}

        client = self._client()
        try:
            raw = client.responses.create(timeout=timeout_seconds, **payload)
        except Exception as exc:
            raise self._normalize_exception(exc) from exc

        text = self._extract_text(raw)
        return Response(
            text=text,
            provider=self.name,
            model=model,
            usage=self._extract_usage(raw),
            tool_calls=self._extract_tool_calls(raw),
            raw=raw,
        )

    def test_connection(self, *, model: str, timeout_seconds: float = 5.0) -> tuple[bool, str | None]:
        try:
            self.chat(
                Request(messages=(Message(role="user", content="ping"),), purpose="diagnostics"),
                model=model,
                timeout_seconds=timeout_seconds,
            )
            return True, None
        except LLMError as exc:
            return False, exc.message
