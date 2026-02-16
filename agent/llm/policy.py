from __future__ import annotations

from dataclasses import dataclass

from agent.config import Config
from agent.llm.registry import ModelConfig, Registry
from agent.llm.types import Request


_VALID_ROUTING_MODES = {"auto", "prefer_cheap", "prefer_best"}


@dataclass(frozen=True)
class RoutingPolicy:
    mode: str
    retry_attempts: int
    retry_base_delay_ms: int
    circuit_breaker_failures: int
    circuit_breaker_window_seconds: int
    circuit_breaker_cooldown_seconds: int
    default_timeout_seconds: int
    fallback_chain: tuple[str, ...] = ()

    def required_capabilities(self, request: Request) -> frozenset[str]:
        required = {"chat"}
        if request.require_tools:
            required.add("tools")
        if request.require_json:
            required.add("json")
        if request.require_vision:
            required.add("vision")
        return frozenset(required)

    def allows_model(self, request: Request, model: ModelConfig) -> bool:
        if request.provider and model.provider != request.provider:
            return False
        if request.model and request.model not in {model.id, model.model}:
            return False
        required = self.required_capabilities(request)
        return required.issubset(model.capabilities)

    def ordered_candidates(self, request: Request, models: list[ModelConfig]) -> list[ModelConfig]:
        candidates = [model for model in models if self.allows_model(request, model) and model.enabled]

        chain_index = {model_id: idx for idx, model_id in enumerate(self.fallback_chain)}

        def sort_key(model: ModelConfig) -> tuple[int, int, int, int, str]:
            in_chain = model.id in chain_index
            chain_rank = chain_index.get(model.id, 10_000)
            purpose_penalty = 0 if request.purpose in model.default_for else 1
            if self.mode == "prefer_cheap":
                return (
                    0 if in_chain else 1,
                    chain_rank,
                    model.cost_rank,
                    -model.quality_rank,
                    model.id,
                )
            if self.mode == "prefer_best":
                return (
                    0 if in_chain else 1,
                    chain_rank,
                    -model.quality_rank,
                    model.cost_rank,
                    model.id,
                )
            # auto
            score = (model.quality_rank * 2) - model.cost_rank
            return (
                0 if in_chain else 1,
                chain_rank,
                purpose_penalty,
                -score,
                model.id,
            )

        return sorted(candidates, key=sort_key)


def load_routing_policy(config: Config, registry: Registry) -> RoutingPolicy:
    mode = config.llm_routing_mode or str(registry.routing_defaults.get("mode") or "auto")
    mode = mode.strip().lower()
    if mode not in _VALID_ROUTING_MODES:
        mode = "auto"

    fallback_chain = tuple(
        item for item in registry.routing_defaults.get("fallback_chain", ()) if item in registry.models
    )

    return RoutingPolicy(
        mode=mode,
        retry_attempts=max(1, int(config.llm_retry_attempts)),
        retry_base_delay_ms=max(0, int(config.llm_retry_base_delay_ms)),
        circuit_breaker_failures=max(1, int(config.llm_circuit_breaker_failures)),
        circuit_breaker_window_seconds=max(1, int(config.llm_circuit_breaker_window_seconds)),
        circuit_breaker_cooldown_seconds=max(1, int(config.llm_circuit_breaker_cooldown_seconds)),
        default_timeout_seconds=max(1, int(config.llm_timeout_seconds)),
        fallback_chain=fallback_chain,
    )
