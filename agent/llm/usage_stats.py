from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from agent.llm.types import Message


@dataclass(frozen=True)
class UsageEstimate:
    prompt_tokens: float
    completion_tokens: float
    samples: int


class UsageStatsStore:
    def __init__(self, path: str | None, ema_alpha: float = 0.35) -> None:
        self._path = Path(path) if path else None
        self._ema_alpha = max(0.01, min(1.0, float(ema_alpha)))
        self._entries: dict[str, UsageEstimate] = {}
        self._load()

    @staticmethod
    def _key(task_type: str, provider_id: str, model_id: str) -> str:
        return f"{task_type}::{provider_id}::{model_id}"

    def _load(self) -> None:
        if not self._path or not self._path.is_file():
            return
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(parsed, dict):
            return
        for key, payload in parsed.items():
            if not isinstance(payload, dict):
                continue
            try:
                prompt_tokens = float(payload.get("prompt_tokens") or 0.0)
                completion_tokens = float(payload.get("completion_tokens") or 0.0)
                samples = int(payload.get("samples") or 0)
            except Exception:
                continue
            self._entries[str(key)] = UsageEstimate(
                prompt_tokens=max(0.0, prompt_tokens),
                completion_tokens=max(0.0, completion_tokens),
                samples=max(0, samples),
            )

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        for key, estimate in self._entries.items():
            payload[key] = {
                "prompt_tokens": estimate.prompt_tokens,
                "completion_tokens": estimate.completion_tokens,
                "samples": estimate.samples,
            }
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    def get(self, task_type: str, provider_id: str, model_id: str) -> UsageEstimate | None:
        return self._entries.get(self._key(task_type, provider_id, model_id))

    def update(
        self,
        task_type: str,
        provider_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        key = self._key(task_type, provider_id, model_id)
        current = self._entries.get(key)
        prompt = max(0, int(prompt_tokens))
        completion = max(0, int(completion_tokens))

        if current is None:
            self._entries[key] = UsageEstimate(
                prompt_tokens=float(prompt),
                completion_tokens=float(completion),
                samples=1,
            )
            self._save()
            return

        alpha = self._ema_alpha
        next_prompt = (alpha * prompt) + ((1.0 - alpha) * current.prompt_tokens)
        next_completion = (alpha * completion) + ((1.0 - alpha) * current.completion_tokens)
        self._entries[key] = UsageEstimate(
            prompt_tokens=float(next_prompt),
            completion_tokens=float(next_completion),
            samples=current.samples + 1,
        )
        self._save()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for key, estimate in sorted(self._entries.items()):
            output[key] = {
                "prompt_tokens": estimate.prompt_tokens,
                "completion_tokens": estimate.completion_tokens,
                "samples": estimate.samples,
            }
        return output


def estimate_prompt_tokens(messages: tuple[Message, ...]) -> int:
    if not messages:
        return 16
    total_chars = 0
    for item in messages:
        total_chars += len(item.role) + len(item.content)
        if item.name:
            total_chars += len(item.name)
    # Conservative char->token estimate
    tokens = max(16, int(total_chars / 3.5))
    return tokens


def estimate_completion_tokens(prompt_tokens: int, response_text: str | None = None) -> int:
    if response_text:
        by_text = max(16, int(len(response_text) / 3.5))
        return by_text
    return max(96, int(prompt_tokens * 0.7))


def default_usage_stats_path(db_path: str, override_path: str | None = None) -> str | None:
    if override_path:
        return override_path
    if not db_path:
        return None
    parent = Path(db_path).resolve().parent
    return str(parent / "llm_usage_stats.json")
