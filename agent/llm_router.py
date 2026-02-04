from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.config import Config
from agent.logging_utils import log_event
from agent.llm_client import DummyClient, OllamaClient, OpenAIClient


_TIER_ORDER = {"low": 1, "mid": 2, "high": 3}


@dataclass
class Candidate:
    provider: str
    model: str | None
    privacy: str
    tier: str
    cost_rank: int
    timeout_seconds: int


def _tier_at_least(candidate: str, required: str) -> bool:
    return _TIER_ORDER.get(candidate, 0) >= _TIER_ORDER.get(required, 0)


def _infer_tier(provider: str, model: str | None, role: str, from_override: bool) -> str:
    if from_override and role == "sentinel":
        return "low"
    if from_override and role == "worker":
        return "mid"
    if provider == "openai" and model and "gpt-4" in model:
        return "high"
    return "mid"


class LLMRouter:
    def __init__(
        self,
        config: Config,
        clients: dict[str, Any] | None = None,
        log_path: str | None = None,
    ) -> None:
        self.config = config
        self._clients = clients or {}
        self._log_path = log_path

    def enabled(self) -> bool:
        return False

    def intent_from_text(self, text: str) -> dict[str, Any] | None:
        return None

    def _build_candidates(self, task_kind: str, tier_required: str) -> list[Candidate]:
        role = "sentinel" if task_kind == "watchdog" else "worker"
        candidates: list[Candidate] = []

        ollama_model = None
        ollama_override = False
        if role == "sentinel" and self.config.ollama_model_sentinel:
            ollama_model = self.config.ollama_model_sentinel
            ollama_override = True
        elif role == "worker" and self.config.ollama_model_worker:
            ollama_model = self.config.ollama_model_worker
            ollama_override = True
        else:
            ollama_model = self.config.ollama_model

        if self.config.ollama_host:
            candidates.append(
                Candidate(
                    provider="ollama",
                    model=ollama_model,
                    privacy="local",
                    tier=_infer_tier("ollama", ollama_model, role, ollama_override),
                    cost_rank=1,
                    timeout_seconds=self.config.llm_timeout_seconds,
                )
            )

        openai_model = None
        openai_override = False
        if role == "worker" and self.config.openai_model_worker:
            openai_model = self.config.openai_model_worker
            openai_override = True
        else:
            openai_model = self.config.openai_model

        if self.config.openai_api_key and self.config.allow_cloud:
            candidates.append(
                Candidate(
                    provider="openai",
                    model=openai_model,
                    privacy="cloud",
                    tier=_infer_tier("openai", openai_model, role, openai_override),
                    cost_rank=3,
                    timeout_seconds=self.config.llm_timeout_seconds,
                )
            )

        if task_kind == "watchdog":
            candidates = [c for c in candidates if c.privacy == "local"]

        candidates = [c for c in candidates if _tier_at_least(c.tier, tier_required)]
        if self.config.prefer_local:
            candidates.sort(key=lambda c: (0 if c.privacy == "local" else 1, c.cost_rank))
        else:
            candidates.sort(key=lambda c: c.cost_rank)
        return candidates

    def _client_for(self, candidate: Candidate):
        if candidate.provider in self._clients:
            return self._clients[candidate.provider]
        if candidate.provider == "ollama" and candidate.model:
            return OllamaClient(
                host=self.config.ollama_host or "",
                model=candidate.model,
                timeout_seconds=candidate.timeout_seconds,
            )
        if candidate.provider == "openai" and candidate.model:
            return OpenAIClient(
                api_key=self.config.openai_api_key or "",
                model=candidate.model,
                timeout_seconds=candidate.timeout_seconds,
            )
        if candidate.provider == "dummy":
            return DummyClient()
        return None

    def generate(
        self,
        task_kind: str,
        tier_required: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        fallbacks: list[dict[str, Any]] = []
        candidates = self._build_candidates(task_kind, tier_required)
        log_payload_base = {
            "task_kind": task_kind,
            "tier_required": tier_required,
            "allow_cloud": self.config.allow_cloud,
            "prefer_local": self.config.prefer_local,
        }
        if task_kind == "watchdog" and not candidates:
            dummy = DummyClient()
            result = {
                "text": dummy.generate(messages),
                "meta": {
                    "selected_provider": "dummy",
                    "selected_model": None,
                    "tier_used": "low",
                    "fallbacks_attempted": fallbacks,
                },
            }
            if self._log_path:
                log_event(
                    self._log_path,
                    "llm_routing_decision",
                    {
                        **log_payload_base,
                        "selected_provider": "dummy",
                        "selected_model": None,
                        "fallbacks_attempted": fallbacks,
                    },
                )
            return result

        for candidate in candidates:
            if not candidate.model:
                fallbacks.append(
                    {
                        "provider": candidate.provider,
                        "model": None,
                        "reason": "missing_model",
                    }
                )
                continue
            client = self._client_for(candidate)
            if client is None:
                fallbacks.append(
                    {
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "reason": "no_client",
                    }
                )
                continue
            try:
                text = client.generate(messages)
                result = {
                    "text": text,
                    "meta": {
                        "selected_provider": candidate.provider,
                        "selected_model": candidate.model,
                        "tier_used": candidate.tier,
                        "fallbacks_attempted": fallbacks,
                    },
                }
                if self._log_path:
                    log_event(
                        self._log_path,
                        "llm_routing_decision",
                        {
                            **log_payload_base,
                            "selected_provider": candidate.provider,
                            "selected_model": candidate.model,
                            "fallbacks_attempted": fallbacks,
                        },
                    )
                return result
            except Exception as exc:
                fallbacks.append(
                    {
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "reason": str(exc),
                    }
                )

        dummy = DummyClient()
        result = {
            "text": dummy.generate(messages),
            "meta": {
                "selected_provider": "dummy",
                "selected_model": None,
                "tier_used": "low",
                "fallbacks_attempted": fallbacks,
            },
        }
        if self._log_path:
            log_event(
                self._log_path,
                "llm_routing_decision",
                {
                    **log_payload_base,
                    "selected_provider": "dummy",
                    "selected_model": None,
                    "fallbacks_attempted": fallbacks,
                },
            )
        return result
