from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.config import Config
from agent.llm.broker_policy import BrokerPolicy, ProviderPolicy
from agent.llm_client import DummyClient, build_client_for_provider


@dataclass
class TaskSpec:
    task: str
    require_local: bool = False
    max_cost: int | None = None


class LLMBroker:
    def __init__(
        self,
        config: Config,
        policy: BrokerPolicy,
        client_factory: Callable[[Config, ProviderPolicy], Any] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.client_factory = client_factory or build_client_for_provider

    def select(self, task_spec: TaskSpec) -> tuple[Any, dict[str, Any]]:
        candidates = []
        reasons = []

        for provider in self.policy.providers:
            allowed, reason = self._candidate_allowed(provider, task_spec)
            if not allowed:
                reasons.append({"id": provider.id, "reason": reason})
                continue
            score = self._score(provider)
            candidates.append(
                {
                    "id": provider.id,
                    "provider": provider.provider,
                    "model": provider.model,
                    "remote": provider.remote,
                    "score": score,
                    "cost": provider.cost,
                    "latency": provider.latency,
                    "reliability": provider.reliability,
                }
            )

        if not candidates:
            return DummyClient(), {
                "winner": None,
                "winner_id": None,
                "candidates": [],
                "candidates_count": 0,
                "rejected": reasons,
                "failure_reason": "no_candidates",
            }

        candidates_sorted = sorted(
            candidates,
            key=lambda c: (
                -c["score"],
                *self._tie_breaker_key(c),
            ),
        )
        winner = candidates_sorted[0]
        provider_policy = self._provider_by_id(winner["id"])
        client = self.client_factory(self.config, provider_policy)
        return client, {
            "winner": winner,
            "winner_id": winner["id"],
            "candidates": candidates_sorted,
            "candidates_count": len(candidates_sorted),
            "rejected": reasons,
        }

    def _candidate_allowed(self, provider: ProviderPolicy, task_spec: TaskSpec) -> tuple[bool, str]:
        if task_spec.task not in provider.capabilities:
            return False, "missing_capability"
        if task_spec.max_cost is not None and provider.cost > task_spec.max_cost:
            return False, "cost_exceeded"
        if task_spec.require_local or not self.config.llm_allow_remote:
            if provider.remote:
                return False, "remote_disallowed"
        if not self._provider_available(provider):
            return False, "provider_unavailable"
        return True, "ok"

    def _provider_available(self, provider: ProviderPolicy) -> bool:
        if provider.provider == "openai":
            return bool(self.config.openai_api_key)
        if provider.provider == "ollama":
            return bool(self.config.ollama_base_url or self.config.ollama_host) and bool(provider.model)
        if provider.provider == "openrouter":
            return bool(self.config.openrouter_api_key and self.config.openrouter_base_url and provider.model)
        if provider.provider == "anthropic":
            return bool(self.config.anthropic_api_key)
        return False

    def _score(self, provider: ProviderPolicy) -> int:
        w = self.policy.weights
        return int(
            w.get("cost", 0) * provider.cost
            + w.get("latency", 0) * provider.latency
            + w.get("reliability", 0) * provider.reliability
        )

    def _tie_breaker_key(self, candidate: dict[str, Any]) -> tuple:
        key_parts = []
        for key in self.policy.tie_breaker:
            if key == "reliability":
                key_parts.append(-candidate["reliability"])
            elif key == "latency":
                key_parts.append(candidate["latency"])
            elif key == "cost":
                key_parts.append(candidate["cost"])
            elif key == "id":
                key_parts.append(candidate["id"])
            else:
                key_parts.append(candidate.get(key))
        return tuple(key_parts)

    def _provider_by_id(self, provider_id: str) -> ProviderPolicy:
        for provider in self.policy.providers:
            if provider.id == provider_id:
                return provider
        raise ValueError("Provider id not found.")
