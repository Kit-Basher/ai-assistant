from __future__ import annotations

import json
import urllib.request
from typing import Any

from agent.llm.base import LLMResult


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.site_url = site_url
        self.app_name = app_name

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _build_payload(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        model_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        content = json.dumps({"kind": kind, "payload": payload}, ensure_ascii=True)
        messages = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": content},
        ]
        body: dict[str, Any] = {
            "model": model_override or self.model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = float(temperature)
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        return body

    def summarize(self, kind: str, payload: dict, timeout_s: int = 15) -> LLMResult:
        if not self.api_key or not self.model or not self.base_url:
            return LLMResult("", "openrouter", self.model or "", True, "openrouter_unavailable")

        model_override = payload.get("model") if isinstance(payload, dict) else None
        temperature = None
        max_tokens = None
        if isinstance(payload, dict):
            temperature = payload.get("temperature")
            max_tokens = payload.get("max_tokens")

        body = self._build_payload(
            kind,
            payload,
            model_override=model_override,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        url = self.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(body, ensure_ascii=True).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=self._build_headers(), method="POST")

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw or "{}")
            choices = parsed.get("choices") or []
            message = choices[0].get("message") if choices else None
            text = (message or {}).get("content") if message else ""
            if not text:
                return LLMResult("", "openrouter", body["model"], True, "openrouter_empty")
            return LLMResult(text, "openrouter", body["model"], False, None)
        except Exception:
            return LLMResult("", "openrouter", body.get("model") or "", True, "openrouter_unavailable")
