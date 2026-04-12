from __future__ import annotations

from dataclasses import dataclass
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from agent.llm.providers.base import Provider
from agent.llm.registry import APIKeySource, ProviderConfig
from agent.llm.types import EmbeddingResponse, LLMError, Message, Request, Response, ToolCall, Usage
from agent.secret_store import SecretStore


@dataclass
class OpenAICompatProvider(Provider):
    config: ProviderConfig
    secret_store: SecretStore | None = None

    def __post_init__(self) -> None:
        self._override_api_key: str | None = None

    @property
    def name(self) -> str:
        return self.config.id

    def set_api_key(self, api_key: str | None) -> None:
        self._override_api_key = (api_key or "").strip() or None

    def _resolve_key_source(self, source: APIKeySource | None) -> str | None:
        if source is None:
            return None
        if source.source_type == "env":
            return (os.environ.get(source.name, "") or "").strip() or None
        if source.source_type == "secret":
            if not self.secret_store:
                return None
            return (self.secret_store.get_secret(source.name) or "").strip() or None
        return None

    def _resolve_reference_value(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, str):
            return value
        if not isinstance(value, dict):
            return str(value)

        if "value" in value:
            static_value = value.get("value")
            return str(static_value) if static_value is not None else None
        if "from_env" in value:
            env_name = str(value.get("from_env") or "").strip()
            if not env_name:
                return None
            return (os.environ.get(env_name, "") or "").strip() or None
        if "from_secret" in value:
            secret_key = str(value.get("from_secret") or "").strip()
            if not secret_key or not self.secret_store:
                return None
            return (self.secret_store.get_secret(secret_key) or "").strip() or None
        if "from_secret_store" in value:
            secret_key = str(value.get("from_secret_store") or "").strip()
            if not secret_key or not self.secret_store:
                return None
            return (self.secret_store.get_secret(secret_key) or "").strip() or None
        return None

    def _resolve_api_key(self) -> str | None:
        if self._override_api_key is not None:
            return self._override_api_key
        return self._resolve_key_source(self.config.api_key_source)

    def available(self) -> bool:
        if not self.config.enabled:
            return False
        if not self.config.base_url:
            return False
        if self.config.api_key_source is None:
            return True
        return bool(self._resolve_api_key())

    @staticmethod
    def _normalize_content(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            chunks: list[str] = []
            for item in raw:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        chunks.append(item["text"])
                    elif isinstance(item.get("content"), str):
                        chunks.append(item["content"])
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunks)
        if isinstance(raw, dict):
            if isinstance(raw.get("text"), str):
                return raw["text"]
            if isinstance(raw.get("content"), str):
                return raw["content"]
        return str(raw)

    @staticmethod
    def _to_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
        if not messages:
            return [{"role": "user", "content": "ping"}]
        output: list[dict[str, Any]] = []
        for item in messages:
            row: dict[str, Any] = {
                "role": item.role,
                "content": item.content,
            }
            if item.name:
                row["name"] = item.name
            if item.tool_call_id:
                row["tool_call_id"] = item.tool_call_id
            output.append(row)
        return output

    @staticmethod
    def _parse_usage(parsed: dict[str, Any]) -> Usage:
        usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
        raw_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        tool_calls: list[ToolCall] = []
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            call_id = str(item.get("id") or "").strip()
            function_payload = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = str(function_payload.get("name") or "").strip()
            arguments_raw = function_payload.get("arguments")
            if isinstance(arguments_raw, str):
                arguments = arguments_raw
            elif arguments_raw is None:
                arguments = ""
            else:
                arguments = json.dumps(arguments_raw, ensure_ascii=True)
            if call_id and name:
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return tuple(tool_calls)

    def _build_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        chat_path = self.config.chat_path if self.config.chat_path.startswith("/") else f"/{self.config.chat_path}"
        url = base_url + chat_path

        query_params: dict[str, str] = {}
        for key, raw_value in (self.config.default_query_params or {}).items():
            value = self._resolve_reference_value(raw_value)
            if value is None:
                continue
            query_params[str(key)] = value

        if not query_params:
            return url

        parsed = urllib.parse.urlparse(url)
        existing = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        existing.update(query_params)
        query = urllib.parse.urlencode(existing)
        rebuilt = parsed._replace(query=query)
        return urllib.parse.urlunparse(rebuilt)

    def _embedding_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        chat_path = self.config.chat_path if self.config.chat_path.startswith("/") else f"/{self.config.chat_path}"
        if chat_path.endswith("/chat/completions"):
            embedding_path = f"{chat_path[: -len('/chat/completions')]}/embeddings"
        elif chat_path.endswith("/completions"):
            embedding_path = f"{chat_path[: -len('/completions')]}/embeddings"
        else:
            embedding_path = "/v1/embeddings"
        return base_url + embedding_path

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }

        key = self._resolve_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        for header_name, raw_value in (self.config.default_headers or {}).items():
            value = self._resolve_reference_value(raw_value)
            if value is None or value == "":
                continue
            headers[str(header_name)] = value

        return headers

    @staticmethod
    def _normalize_status_error(provider: str, status_code: int, message: str, raw: Any = None) -> LLMError:
        if status_code == 429:
            return LLMError(
                kind="rate_limit",
                retriable=True,
                provider=provider,
                status_code=status_code,
                message=message,
                raw=raw,
            )
        if status_code in {401, 403}:
            return LLMError(
                kind="auth_error",
                retriable=False,
                provider=provider,
                status_code=status_code,
                message=message,
                raw=raw,
            )
        if 500 <= status_code <= 599:
            return LLMError(
                kind="server_error",
                retriable=True,
                provider=provider,
                status_code=status_code,
                message=message,
                raw=raw,
            )
        return LLMError(
            kind="bad_request",
            retriable=False,
            provider=provider,
            status_code=status_code,
            message=message,
            raw=raw,
        )

    @staticmethod
    def _normalize_other_error(provider: str, exc: Exception) -> LLMError:
        if isinstance(exc, TimeoutError) or isinstance(exc, socket.timeout):
            return LLMError(
                kind="timeout",
                retriable=True,
                provider=provider,
                status_code=None,
                message=str(exc) or "timeout",
                raw={"type": exc.__class__.__name__},
            )
        if isinstance(exc, urllib.error.URLError):
            return LLMError(
                kind="network",
                retriable=True,
                provider=provider,
                status_code=None,
                message=str(exc.reason) if getattr(exc, "reason", None) else str(exc),
                raw={"type": exc.__class__.__name__},
            )
        return LLMError(
            kind="provider_error",
            retriable=False,
            provider=provider,
            status_code=None,
            message=str(exc) or exc.__class__.__name__,
            raw={"type": exc.__class__.__name__},
        )

    def chat(self, request: Request, *, model: str, timeout_seconds: float) -> Response:
        if not self.available():
            raise LLMError(
                kind="provider_unavailable",
                retriable=False,
                provider=self.name,
                status_code=None,
                message="Provider is unavailable or missing credentials.",
                raw=None,
            )

        body: dict[str, Any] = {
            "model": model,
            "messages": self._to_messages(request.messages),
        }
        if request.temperature is not None:
            body["temperature"] = float(request.temperature)
        if request.max_tokens is not None:
            body["max_tokens"] = int(request.max_tokens)
        if request.tools:
            body["tools"] = list(request.tools)
        if request.require_json:
            body["response_format"] = {"type": "json_object"}

        data = json.dumps(body, ensure_ascii=True).encode("utf-8")
        url = self._build_url()
        headers = self._build_headers()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                raw_text = response.read().decode("utf-8")
                status_code = int(getattr(response, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            message = body_text or str(exc)
            raise self._normalize_status_error(
                self.name,
                int(getattr(exc, "code", 500) or 500),
                message,
                raw={"body": body_text},
            ) from exc
        except Exception as exc:
            raise self._normalize_other_error(self.name, exc) from exc

        if status_code >= 400:
            raise self._normalize_status_error(self.name, status_code, f"HTTP {status_code}")

        try:
            parsed = json.loads(raw_text or "{}")
        except Exception as exc:
            raise LLMError(
                kind="provider_error",
                retriable=False,
                provider=self.name,
                status_code=status_code,
                message="Invalid JSON response from provider.",
                raw={"body": raw_text[:500]},
            ) from exc

        choices = parsed.get("choices") if isinstance(parsed.get("choices"), list) else []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        message = message if isinstance(message, dict) else {}
        text = self._normalize_content(message.get("content")).strip()
        tool_calls = self._parse_tool_calls(message)

        return Response(
            text=text,
            provider=self.name,
            model=model,
            usage=self._parse_usage(parsed),
            tool_calls=tool_calls,
            raw=parsed,
        )

    def embed_texts(self, texts: tuple[str, ...], *, model: str, timeout_seconds: float) -> EmbeddingResponse:
        if not self.available():
            raise LLMError(
                kind="provider_unavailable",
                retriable=False,
                provider=self.name,
                status_code=None,
                message="Provider is unavailable or missing credentials.",
                raw=None,
            )

        body: dict[str, Any] = {
            "model": model,
            "input": [str(item) for item in texts] if texts else ["ping"],
        }
        data = json.dumps(body, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(self._embedding_url(), data=data, headers=self._build_headers(), method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                raw_text = response.read().decode("utf-8")
                status_code = int(getattr(response, "status", 200) or 200)
        except urllib.error.HTTPError as exc:
            body_bytes = exc.read() if hasattr(exc, "read") else b""
            body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
            raise self._normalize_status_error(
                self.name,
                int(getattr(exc, "code", 500) or 500),
                body_text or str(exc),
                raw={"body": body_text},
            ) from exc
        except Exception as exc:
            raise self._normalize_other_error(self.name, exc) from exc

        if status_code >= 400:
            raise self._normalize_status_error(self.name, status_code, f"HTTP {status_code}")

        try:
            parsed = json.loads(raw_text or "{}")
        except Exception as exc:
            raise LLMError(
                kind="provider_error",
                retriable=False,
                provider=self.name,
                status_code=status_code,
                message="Invalid JSON response from provider.",
                raw={"body": raw_text[:500]},
            ) from exc

        rows = parsed.get("data") if isinstance(parsed.get("data"), list) else []
        vectors: list[tuple[float, ...]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            embedding = row.get("embedding")
            if not isinstance(embedding, list):
                continue
            vector: list[float] = []
            for value in embedding:
                try:
                    vector.append(float(value))
                except (TypeError, ValueError):
                    continue
            if vector:
                vectors.append(tuple(vector))
        if not vectors:
            raise LLMError(
                kind="provider_error",
                retriable=False,
                provider=self.name,
                status_code=status_code,
                message="Embedding response did not include vectors.",
                raw=parsed,
            )
        usage = self._parse_usage(parsed)
        return EmbeddingResponse(
            provider=self.name,
            model=model,
            vectors=tuple(vectors),
            usage=usage,
            raw=parsed,
        )
