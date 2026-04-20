from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import json
import os
import random
import time
from typing import Any

from agent.config import Config
from agent.llm.policy import RoutingPolicy, load_routing_policy
from agent.llm.providers.base import Provider
from agent.llm.providers.openai_compat import OpenAICompatProvider
from agent.llm.registry import ModelConfig, Registry, load_registry
from agent.llm.types import LLMError, Message, Request
from agent.llm.usage_stats import (
    UsageStatsStore,
    default_usage_stats_path,
    estimate_completion_tokens,
    estimate_prompt_tokens,
)
from agent.logging_utils import log_event
from agent.secret_store import SecretStore


@dataclass
class CircuitBreakerState:
    failures: deque[float] = field(default_factory=deque)
    opened_until: float | None = None


@dataclass
class OutcomeState:
    successes: int = 0
    failures: int = 0
    last_error_kind: str | None = None
    last_status_code: int | None = None
    last_ts: float | None = None
    cooldown_until: float | None = None
    cooldown_reason: str | None = None
    auth_down: bool = False


class _UnavailableProvider(Provider):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return False

    def chat(self, request: Request, *, model: str, timeout_seconds: float):  # pragma: no cover - defensive
        _ = request
        _ = model
        _ = timeout_seconds
        raise LLMError(
            kind="provider_unavailable",
            retriable=False,
            provider=self._name,
            status_code=None,
            message="Provider is unavailable.",
            raw=None,
        )

    def embed_texts(self, texts: tuple[str, ...], *, model: str, timeout_seconds: float):  # pragma: no cover - defensive
        _ = texts
        _ = model
        _ = timeout_seconds
        raise LLMError(
            kind="provider_unavailable",
            retriable=False,
            provider=self._name,
            status_code=None,
            message="Provider is unavailable.",
            raw=None,
        )


class LLMRouter:
    def __init__(
        self,
        config: Config,
        *,
        providers: dict[str, Provider] | None = None,
        registry: Registry | None = None,
        policy: RoutingPolicy | None = None,
        log_path: str | None = None,
        rng: random.Random | None = None,
        time_fn=None,
        sleep_fn=None,
        secret_store: SecretStore | None = None,
        usage_stats: UsageStatsStore | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or load_registry(config)
        self.policy = policy or load_routing_policy(config, self.registry)
        self._secret_store = secret_store or SecretStore(path=os.getenv("AGENT_SECRET_STORE_PATH", "").strip() or None)
        self._usage_stats = usage_stats or UsageStatsStore(
            default_usage_stats_path(config.db_path, config.llm_usage_stats_path)
        )
        self._providers = providers or self._build_default_providers()
        self._circuits: dict[str, CircuitBreakerState] = {}
        self._outcomes: dict[str, OutcomeState] = {}
        self._external_health: dict[str, Any] = {
            "providers": {},
            "models": {},
            "last_run_at": None,
        }
        self._rng = rng or random.Random(0)
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._log_path = log_path

    def _build_default_providers(self) -> dict[str, Provider]:
        providers: dict[str, Provider] = {}
        for provider_id, provider_cfg in self.registry.providers.items():
            if provider_cfg.provider_type in {"openai_compat", "openai"}:
                providers[provider_id] = OpenAICompatProvider(provider_cfg, secret_store=self._secret_store)
            else:
                providers[provider_id] = _UnavailableProvider(provider_id)
        return providers

    def provider_for_id(self, provider_id: str) -> Provider | None:
        provider_key = str(provider_id or "").strip().lower()
        if not provider_key:
            return None
        provider = self._providers.get(provider_key)
        return provider

    def model_config(self, model_id: str) -> ModelConfig | None:
        model_key = str(model_id or "").strip()
        if not model_key:
            return None
        model = self.registry.models.get(model_key)
        return model

    @staticmethod
    def _normalize_messages(messages: list[dict[str, str]] | tuple[dict[str, str], ...]) -> tuple[Message, ...]:
        normalized: list[Message] = []
        for row in messages:
            role = str((row or {}).get("role") or "user").strip() or "user"
            content = str((row or {}).get("content") or "")
            name = (row or {}).get("name")
            tool_call_id = (row or {}).get("tool_call_id")
            normalized.append(
                Message(
                    role=role,
                    content=content,
                    name=str(name) if name else None,
                    tool_call_id=str(tool_call_id) if tool_call_id else None,
                )
            )
        return tuple(normalized)

    def enabled(self) -> bool:
        request = Request(messages=(), purpose="chat")
        return bool(self._ordered_models(request))

    def intent_from_text(self, text: str) -> dict[str, Any] | None:
        _ = text
        return None

    def set_routing_mode(self, mode: str) -> None:
        normalized = (mode or "").strip().lower()
        if normalized not in {
            "auto",
            "prefer_cheap",
            "prefer_best",
            "prefer_local_lowest_cost_capable",
        }:
            raise ValueError(
                "routing mode must be one of: auto, prefer_cheap, prefer_best, prefer_local_lowest_cost_capable"
            )
        self.policy = replace(self.policy, mode=normalized)

    def set_registry(self, registry: Registry) -> None:
        self.registry = registry
        self.policy = load_routing_policy(self.config, self.registry)
        self._providers = self._build_default_providers()
        active_model_ids = set(self.registry.models.keys())
        self._circuits = {model_id: state for model_id, state in self._circuits.items() if model_id in active_model_ids}
        self._outcomes = {model_id: state for model_id, state in self._outcomes.items() if model_id in active_model_ids}
        external_models = self._external_health.get("models") if isinstance(self._external_health.get("models"), dict) else {}
        self._external_health["models"] = {
            model_id: payload
            for model_id, payload in external_models.items()
            if model_id in active_model_ids and isinstance(payload, dict)
        }

    def set_external_health_state(self, payload: dict[str, Any] | None) -> None:
        def _safe_int(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        parsed = payload if isinstance(payload, dict) else {}
        providers_raw = parsed.get("providers") if isinstance(parsed.get("providers"), dict) else {}
        models_raw = parsed.get("models") if isinstance(parsed.get("models"), dict) else {}
        providers: dict[str, dict[str, Any]] = {}
        models: dict[str, dict[str, Any]] = {}

        for provider_id, row in sorted(providers_raw.items()):
            if not isinstance(row, dict):
                continue
            pid = str(provider_id).strip().lower()
            if not pid:
                continue
            providers[pid] = {
                "status": str(row.get("status") or "unknown").strip().lower(),
                "last_error_kind": str(row.get("last_error_kind") or "").strip().lower() or None,
                "status_code": _safe_int(row.get("status_code")),
                "last_checked_at": _safe_int(row.get("last_checked_at")),
                "cooldown_until": _safe_int(row.get("cooldown_until")),
                "down_since": _safe_int(row.get("down_since")),
            }

        for model_id, row in sorted(models_raw.items()):
            if not isinstance(row, dict):
                continue
            mid = str(model_id).strip()
            if not mid:
                continue
            models[mid] = {
                "provider_id": str(row.get("provider_id") or "").strip().lower() or (mid.split(":", 1)[0].strip().lower() if ":" in mid else ""),
                "status": str(row.get("status") or "unknown").strip().lower(),
                "last_error_kind": str(row.get("last_error_kind") or "").strip().lower() or None,
                "status_code": _safe_int(row.get("status_code")),
                "last_checked_at": _safe_int(row.get("last_checked_at")),
                "cooldown_until": _safe_int(row.get("cooldown_until")),
                "down_since": _safe_int(row.get("down_since")),
            }

        self._external_health = {
            "providers": providers,
            "models": models,
            "last_run_at": _safe_int(parsed.get("last_run_at")),
        }

    def external_health_snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._external_health, ensure_ascii=True))

    def _external_provider_health(self, provider_id: str) -> dict[str, Any] | None:
        providers = self._external_health.get("providers")
        if not isinstance(providers, dict):
            return None
        row = providers.get(provider_id)
        return row if isinstance(row, dict) else None

    def _external_model_health(self, model_id: str) -> dict[str, Any] | None:
        models = self._external_health.get("models")
        if not isinstance(models, dict):
            return None
        row = models.get(model_id)
        return row if isinstance(row, dict) else None

    @staticmethod
    def _external_blocked(row: dict[str, Any] | None) -> bool:
        if not isinstance(row, dict):
            return False
        status = str(row.get("status") or "unknown").strip().lower()
        if status in {"down", "degraded"}:
            return True
        return False

    def set_provider_api_key(self, provider: str, api_key: str) -> bool:
        provider_name = (provider or "").strip().lower()
        if not provider_name:
            return False

        provider_cfg = self.registry.providers.get(provider_name)
        if not provider_cfg:
            return False

        impl = self._providers.get(provider_name)
        if impl is None:
            return False

        key = (api_key or "").strip()
        if hasattr(impl, "set_api_key"):
            getattr(impl, "set_api_key")(key)

        source = provider_cfg.api_key_source
        if source and source.source_type == "env":
            if key:
                os.environ[source.name] = key
            else:
                os.environ.pop(source.name, None)
        elif source and source.source_type == "secret":
            self._secret_store.set_secret(source.name, key)

        self._clear_provider_auth_down(provider_name)
        return True

    def _outcome(self, model_id: str) -> OutcomeState:
        if model_id not in self._outcomes:
            self._outcomes[model_id] = OutcomeState()
        return self._outcomes[model_id]

    def _clear_provider_auth_down(self, provider_id: str) -> None:
        provider_key = (provider_id or "").strip().lower()
        if not provider_key:
            return
        for model_id, model in self.registry.models.items():
            if model.provider != provider_key:
                continue
            state = self._outcomes.get(model_id)
            if state is None:
                continue
            state.auth_down = False
            if state.last_error_kind == "auth_error":
                state.last_error_kind = None
                state.last_status_code = None
            state.cooldown_until = None
            state.cooldown_reason = None

    def _is_auth_down(self, model_id: str) -> bool:
        state = self._outcomes.get(model_id)
        return bool(state.auth_down) if state else False

    def _circuit(self, model_id: str) -> CircuitBreakerState:
        if model_id not in self._circuits:
            self._circuits[model_id] = CircuitBreakerState()
        return self._circuits[model_id]

    def _is_circuit_open(self, model_id: str) -> bool:
        state = self._circuit(model_id)
        if state.opened_until is None:
            return False
        now = float(self._time_fn())
        if now >= state.opened_until:
            state.opened_until = None
            state.failures.clear()
            return False
        return True

    def _record_success(self, model_id: str) -> None:
        state = self._circuit(model_id)
        state.failures.clear()
        state.opened_until = None

    def _record_failure(self, model_id: str) -> None:
        state = self._circuit(model_id)
        now = float(self._time_fn())
        window = float(self.policy.circuit_breaker_window_seconds)
        state.failures.append(now)
        while state.failures and now - state.failures[0] > window:
            state.failures.popleft()
        if len(state.failures) >= int(self.policy.circuit_breaker_failures):
            state.opened_until = now + float(self.policy.circuit_breaker_cooldown_seconds)

    def _rate_limit_cooldown_seconds(self) -> float:
        return max(10.0, float(self.policy.circuit_breaker_cooldown_seconds))

    def _server_error_cooldown_seconds(self) -> float:
        return max(5.0, float(self.policy.circuit_breaker_cooldown_seconds) / 2.0)

    def _record_outcome_success(self, model_id: str) -> None:
        state = self._outcome(model_id)
        state.successes += 1
        state.last_ts = float(self._time_fn())
        state.cooldown_until = None
        state.cooldown_reason = None
        state.last_error_kind = None
        state.last_status_code = None

    def _record_outcome_failure(self, model_id: str, error: LLMError) -> None:
        state = self._outcome(model_id)
        now = float(self._time_fn())
        state.failures += 1
        state.last_ts = now
        state.last_error_kind = error.kind
        state.last_status_code = error.status_code

        if error.kind == "auth_error":
            state.auth_down = True
            state.cooldown_until = None
            state.cooldown_reason = "auth_error"
            return

        status_code = int(error.status_code) if error.status_code is not None else None
        if error.kind == "rate_limit" or status_code == 429:
            state.cooldown_until = now + self._rate_limit_cooldown_seconds()
            state.cooldown_reason = "rate_limit"
            return
        if (status_code is not None and 500 <= status_code <= 599) or error.kind == "server_error":
            state.cooldown_until = now + self._server_error_cooldown_seconds()
            state.cooldown_reason = "server_error"
            return

        state.cooldown_until = None
        state.cooldown_reason = None

    def _sleep_retry(self, attempt_idx: int) -> None:
        base = float(self.policy.retry_base_delay_ms) / 1000.0
        if base <= 0:
            return
        delay = base * (2**attempt_idx)
        jitter = self._rng.random() * base
        self._sleep_fn(delay + jitter)

    @staticmethod
    def _normalize_unexpected_error(provider: str, exc: Exception) -> LLMError:
        message = str(exc) or exc.__class__.__name__
        retriable = isinstance(exc, (TimeoutError, ConnectionError))
        kind = "timeout" if isinstance(exc, TimeoutError) else "provider_error"
        return LLMError(
            kind=kind,
            retriable=retriable,
            provider=provider,
            status_code=None,
            message=message,
            raw={"type": exc.__class__.__name__},
        )

    def _call_with_retry(self, provider: Provider, model: ModelConfig, request: Request):
        attempts = max(1, int(self.policy.retry_attempts))
        timeout = float(request.timeout_seconds or self.policy.default_timeout_seconds)

        for attempt_idx in range(attempts):
            try:
                return provider.chat(request, model=model.model, timeout_seconds=timeout)
            except LLMError as exc:
                if not exc.retriable or attempt_idx == attempts - 1:
                    raise
                self._sleep_retry(attempt_idx)
            except Exception as exc:
                normalized = self._normalize_unexpected_error(provider.name, exc)
                if not normalized.retriable or attempt_idx == attempts - 1:
                    raise normalized
                self._sleep_retry(attempt_idx)

        raise LLMError(
            kind="provider_error",
            retriable=False,
            provider=provider.name,
            status_code=None,
            message="request_failed",
            raw=None,
        )

    def _context_capable(self, model: ModelConfig, request: Request) -> bool:
        min_context_tokens = request.min_context_tokens
        if min_context_tokens is None:
            return True
        if model.max_context_tokens is None:
            return True
        return int(model.max_context_tokens) >= int(min_context_tokens)

    def _candidate_models(self, request: Request) -> list[ModelConfig]:
        base_candidates: list[ModelConfig] = []
        for model in self.registry.sorted_models():
            provider_cfg = self.registry.providers.get(model.provider)
            if provider_cfg is None or not provider_cfg.enabled:
                continue
            if not model.enabled or not model.available:
                continue
            if self._is_auth_down(model.id):
                continue
            if self._external_blocked(self._external_provider_health(model.provider)):
                continue
            if self._external_blocked(self._external_model_health(model.id)):
                continue
            if not self.policy.allows_model(request, model) or not self._context_capable(model, request):
                continue
            base_candidates.append(model)

        if not self.policy.allow_remote_fallback and not request.provider and not request.model:
            base_candidates = [
                model
                for model in base_candidates
                if self.registry.providers.get(model.provider) and self.registry.providers[model.provider].local
            ]

        return base_candidates

    def _expected_tokens(self, request: Request, model: ModelConfig) -> tuple[float, float]:
        task_type = request.task_type or request.purpose or "chat"
        stats = self._usage_stats.get(task_type, model.provider, model.id)
        if stats:
            return stats.prompt_tokens, stats.completion_tokens

        prompt_est = float(estimate_prompt_tokens(request.messages))
        completion_est = float(estimate_completion_tokens(int(prompt_est)))
        return prompt_est, completion_est

    def _expected_cost(self, request: Request, model: ModelConfig) -> float:
        provider_cfg = self.registry.providers.get(model.provider)
        if provider_cfg is None:
            return 10_000_000.0

        prompt_tokens, completion_tokens = self._expected_tokens(request, model)

        in_price = model.input_cost_per_million_tokens
        out_price = model.output_cost_per_million_tokens

        if provider_cfg.local and (in_price is None or out_price is None):
            return 0.0
        if in_price is None or out_price is None:
            return 1_000_000.0 + float(model.cost_rank)

        return (prompt_tokens * in_price / 1_000_000.0) + (completion_tokens * out_price / 1_000_000.0)

    def _health_penalty(self, model: ModelConfig) -> float:
        provider_cfg = self.registry.providers.get(model.provider)
        if provider_cfg is None or not provider_cfg.enabled:
            return 1_000_000_000.0
        if not model.enabled or not model.available:
            return 1_000_000_000.0

        outcome = self._outcomes.get(model.id)
        if outcome is None:
            return 0.0
        if outcome.auth_down:
            return 1_000_000_000.0

        penalty = 0.0
        now = float(self._time_fn())

        if outcome.cooldown_until is not None and now < outcome.cooldown_until:
            if outcome.cooldown_reason == "rate_limit":
                penalty += 10_000.0
            elif outcome.cooldown_reason == "server_error":
                penalty += 1_000.0
            else:
                penalty += 500.0

        total = outcome.successes + outcome.failures
        if total > 0:
            penalty += (float(outcome.failures) / float(total)) * 100.0
        penalty += float(min(outcome.failures, 20))
        return penalty

    def _ordered_models(self, request: Request) -> list[ModelConfig]:
        candidates = self._candidate_models(request)
        if self.policy.mode != "prefer_local_lowest_cost_capable":
            return self.policy.ordered_candidates(request, candidates)

        def sort_key(model: ModelConfig) -> tuple[int, float, float, str, str]:
            provider_cfg = self.registry.providers.get(model.provider)
            is_local = bool(provider_cfg.local) if provider_cfg else False
            return (
                0 if is_local else 1,
                self._health_penalty(model),
                self._expected_cost(request, model),
                model.provider,
                model.id,
            )

        return sorted(candidates, key=sort_key)

    def _record_usage(self, request: Request, model: ModelConfig, response_text: str, usage: dict[str, Any]) -> None:
        task_type = request.task_type or request.purpose or "chat"
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        if prompt_tokens <= 0:
            prompt_tokens = estimate_prompt_tokens(request.messages)
        if completion_tokens <= 0:
            completion_tokens = estimate_completion_tokens(prompt_tokens, response_text)

        self._usage_stats.update(task_type, model.provider, model.id, prompt_tokens, completion_tokens)

    def chat(
        self,
        messages: list[dict[str, str]] | tuple[dict[str, str], ...],
        *,
        purpose: str = "chat",
        compute_tier: str = "mid",
        provider_override: str | None = None,
        model_override: str | None = None,
        require_tools: bool = False,
        require_json: bool = False,
        require_vision: bool = False,
        min_context_tokens: int | None = None,
        task_type: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = compute_tier
        start = float(self._time_fn())

        request = Request(
            messages=self._normalize_messages(messages),
            purpose=purpose,
            task_type=task_type,
            provider=(provider_override or "").strip().lower() or None,
            model=(model_override or "").strip() or None,
            require_tools=bool(require_tools),
            require_json=bool(require_json),
            require_vision=bool(require_vision),
            min_context_tokens=min_context_tokens,
            tools=tuple(tools or ()),
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )

        ordered_models = self._ordered_models(request)
        attempts: list[dict[str, Any]] = []
        last_error: LLMError | None = None

        for model in ordered_models:
            provider = self._providers.get(model.provider)
            if provider is None:
                attempts.append(
                    {
                        "provider": model.provider,
                        "model": model.model,
                        "reason": "provider_not_implemented",
                    }
                )
                continue
            if self._is_circuit_open(model.id):
                attempts.append(
                    {
                        "provider": model.provider,
                        "model": model.model,
                        "reason": "circuit_open",
                    }
                )
                continue
            if not provider.available():
                attempts.append(
                    {
                        "provider": model.provider,
                        "model": model.model,
                        "reason": "provider_unavailable",
                    }
                )
                continue

            provider_started = float(self._time_fn())
            metadata = request.metadata if isinstance(request.metadata, dict) else {}
            if self._log_path:
                log_event(
                    self._log_path,
                    "llm_provider_request_start",
                    {
                        "trace_id": str(metadata.get("trace_id") or "").strip() or None,
                        "selection_reason": str(metadata.get("selection_reason") or "").strip() or None,
                        "purpose": purpose,
                        "task_type": task_type or purpose,
                        "provider": provider.name,
                        "model": model.model,
                        "timeout_seconds": timeout_seconds,
                    },
                )
            try:
                response = self._call_with_retry(provider, model, request)
                self._record_success(model.id)
                self._record_outcome_success(model.id)
                duration_ms = int((float(self._time_fn()) - start) * 1000)
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                self._record_usage(request, model, response.text, usage)
                payload = {
                    "ok": True,
                    "text": response.text,
                    "provider": response.provider,
                    "model": response.model,
                    "fallback_used": bool(attempts),
                    "attempts": attempts,
                    "usage": usage,
                    "duration_ms": duration_ms,
                    "error_class": None,
                }
                if self._log_path:
                    log_event(
                        self._log_path,
                        "llm_routing_decision",
                        {
                            "trace_id": str(metadata.get("trace_id") or "").strip() or None,
                            "selection_reason": str(metadata.get("selection_reason") or "").strip() or None,
                            "purpose": purpose,
                            "task_type": task_type or purpose,
                            "required_capabilities": sorted(self.policy.required_capabilities(request)),
                            "selected_provider": response.provider,
                            "selected_model": response.model,
                            "fallback_used": bool(attempts),
                            "attempts": attempts,
                            "routing_mode": self.policy.mode,
                        },
                    )
                    log_event(
                        self._log_path,
                        "llm_provider_request_end",
                        {
                            "trace_id": str(metadata.get("trace_id") or "").strip() or None,
                            "selection_reason": str(metadata.get("selection_reason") or "").strip() or None,
                            "purpose": purpose,
                            "task_type": task_type or purpose,
                            "provider": response.provider,
                            "model": response.model,
                            "timeout_seconds": timeout_seconds,
                            "duration_ms": int(max(0.0, float(self._time_fn()) - provider_started) * 1000),
                            "ok": True,
                            "error_kind": None,
                        },
                    )
                return payload
            except LLMError as exc:
                last_error = exc
                self._record_failure(model.id)
                self._record_outcome_failure(model.id, exc)
                if self._log_path:
                    log_event(
                        self._log_path,
                        "llm_provider_request_end",
                        {
                            "trace_id": str(metadata.get("trace_id") or "").strip() or None,
                            "selection_reason": str(metadata.get("selection_reason") or "").strip() or None,
                            "purpose": purpose,
                            "task_type": task_type or purpose,
                            "provider": provider.name,
                            "model": model.model,
                            "timeout_seconds": timeout_seconds,
                            "duration_ms": int(max(0.0, float(self._time_fn()) - provider_started) * 1000),
                            "ok": False,
                            "error_kind": exc.kind,
                            "status_code": exc.status_code,
                            "retriable": exc.retriable,
                        },
                    )
                attempts.append(
                    {
                        "provider": model.provider,
                        "model": model.model,
                        "reason": exc.kind,
                        "retriable": exc.retriable,
                        "status_code": exc.status_code,
                    }
                )

        duration_ms = int((float(self._time_fn()) - start) * 1000)
        error_class = last_error.kind if last_error else "no_candidates"
        error_message = last_error.message if last_error else "No compatible model available."

        if self._log_path:
            metadata = request.metadata if isinstance(request.metadata, dict) else {}
            log_event(
                self._log_path,
                "llm_routing_decision",
                {
                    "trace_id": str(metadata.get("trace_id") or "").strip() or None,
                    "selection_reason": str(metadata.get("selection_reason") or "").strip() or None,
                    "purpose": purpose,
                    "task_type": task_type or purpose,
                    "required_capabilities": sorted(self.policy.required_capabilities(request)),
                    "selected_provider": "none",
                    "selected_model": None,
                    "fallback_used": False,
                    "attempts": attempts,
                    "routing_mode": self.policy.mode,
                    "error": error_class,
                },
            )

        return {
            "ok": False,
            "text": "",
            "provider": None,
            "model": None,
            "fallback_used": False,
            "attempts": attempts,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "duration_ms": duration_ms,
            "error_class": error_class,
            "error": error_message,
        }

    def generate(
        self,
        task_kind: str,
        tier_required: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        _ = tier_required
        result = self.chat(messages, purpose=task_kind, task_type=task_kind)
        if not result.get("ok"):
            return {
                "text": "",
                "meta": {
                    "selected_provider": "dummy",
                    "selected_model": None,
                    "tier_used": "low",
                    "fallbacks_attempted": result.get("attempts", []),
                },
            }
        return {
            "text": result.get("text") or "",
            "meta": {
                "selected_provider": result.get("provider"),
                "selected_model": result.get("model"),
                "tier_used": tier_required,
                "fallbacks_attempted": result.get("attempts", []),
            },
        }

    def circuit_states(self) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        now = float(self._time_fn())
        for model_id, state in sorted(self._circuits.items()):
            open_now = state.opened_until is not None and now < state.opened_until
            states[model_id] = {
                "open": bool(open_now),
                "failure_count": len(state.failures),
                "opened_until": state.opened_until,
            }
        return states

    def usage_stats_snapshot(self) -> dict[str, dict[str, Any]]:
        return self._usage_stats.snapshot()

    def _model_health(self, model: ModelConfig, *, provider_available: bool) -> dict[str, Any]:
        state = self._outcomes.get(model.id)
        external_state = self._external_model_health(model.id)
        now = float(self._time_fn())

        status = "ok"
        if not model.enabled or not model.available:
            status = "down"
        elif self._is_auth_down(model.id):
            status = "down"
        elif not provider_available:
            status = "down"
        elif state and state.cooldown_until is not None and now < state.cooldown_until:
            status = "degraded"
        elif state and state.failures > state.successes:
            status = "degraded"

        if isinstance(external_state, dict):
            external_status = str(external_state.get("status") or "unknown").strip().lower()
            if external_status == "down":
                status = "down"
            elif external_status == "degraded" and status != "down":
                status = "degraded"

        return {
            "status": status,
            "last_error_kind": (
                (external_state.get("last_error_kind") if isinstance(external_state, dict) else None)
                or (state.last_error_kind if state else None)
            ),
            "status_code": (
                (external_state.get("status_code") if isinstance(external_state, dict) else None)
                or (state.last_status_code if state else None)
            ),
            "last_status_code": (
                (external_state.get("status_code") if isinstance(external_state, dict) else None)
                or (state.last_status_code if state else None)
            ),
            "last_ts": (
                (float(external_state.get("last_checked_at")) if isinstance(external_state, dict) and external_state.get("last_checked_at") is not None else None)
                or (state.last_ts if state else None)
            ),
            "last_checked_at": external_state.get("last_checked_at") if isinstance(external_state, dict) else None,
            "cooldown_until": external_state.get("cooldown_until") if isinstance(external_state, dict) else None,
            "down_since": external_state.get("down_since") if isinstance(external_state, dict) else None,
            "successes": int(state.successes) if state else 0,
            "failures": int(state.failures) if state else 0,
        }

    def _provider_health(self, provider_id: str, *, provider_available: bool) -> dict[str, Any]:
        provider_cfg = self.registry.providers.get(provider_id)
        external_state = self._external_provider_health(provider_id)
        provider_models = [model for model in self.registry.sorted_models() if model.provider == provider_id]
        model_ids = sorted(model.id for model in provider_models)
        model_states = [self._outcomes.get(model_id) for model_id in model_ids]

        latest_state: OutcomeState | None = None
        for state in model_states:
            if state is None:
                continue
            if latest_state is None:
                latest_state = state
                continue
            latest_ts = latest_state.last_ts if latest_state.last_ts is not None else -1.0
            state_ts = state.last_ts if state.last_ts is not None else -1.0
            if state_ts > latest_ts:
                latest_state = state

        status = "ok"
        if provider_cfg is None or not provider_cfg.enabled or not provider_available:
            status = "down"
        else:
            down_models = 0
            degraded_models = 0
            for model in provider_models:
                health = self._model_health(model, provider_available=provider_available)
                if health["status"] == "down":
                    down_models += 1
                elif health["status"] == "degraded":
                    degraded_models += 1
            total_models = len(provider_models)
            if total_models > 0 and down_models > 0 and down_models == total_models:
                status = "down"
            elif down_models > 0 or degraded_models > 0:
                status = "degraded"

        if isinstance(external_state, dict):
            external_status = str(external_state.get("status") or "unknown").strip().lower()
            if external_status == "down":
                status = "down"
            elif external_status == "degraded" and status != "down":
                status = "degraded"

        return {
            "status": status,
            "last_error_kind": (
                (external_state.get("last_error_kind") if isinstance(external_state, dict) else None)
                or (latest_state.last_error_kind if latest_state else None)
            ),
            "status_code": (
                (external_state.get("status_code") if isinstance(external_state, dict) else None)
                or (latest_state.last_status_code if latest_state else None)
            ),
            "last_status_code": (
                (external_state.get("status_code") if isinstance(external_state, dict) else None)
                or (latest_state.last_status_code if latest_state else None)
            ),
            "last_ts": (
                (float(external_state.get("last_checked_at")) if isinstance(external_state, dict) and external_state.get("last_checked_at") is not None else None)
                or (latest_state.last_ts if latest_state else None)
            ),
            "last_checked_at": external_state.get("last_checked_at") if isinstance(external_state, dict) else None,
            "cooldown_until": external_state.get("cooldown_until") if isinstance(external_state, dict) else None,
            "down_since": external_state.get("down_since") if isinstance(external_state, dict) else None,
            "successes": sum(int(state.successes) for state in model_states if state),
            "failures": sum(int(state.failures) for state in model_states if state),
        }

    def doctor_snapshot(self) -> dict[str, Any]:
        provider_rows: list[dict[str, Any]] = []
        for provider_id, spec in sorted(self.registry.providers.items()):
            impl = self._providers.get(provider_id)
            provider_available = bool(impl.available()) if impl else False
            env_present = None
            if spec.auth_env_var:
                env_present = bool(os.getenv(spec.auth_env_var, "").strip())
            provider_rows.append(
                {
                    "id": provider_id,
                    "name": provider_id,
                    "type": spec.provider_type,
                    "base_url": spec.base_url,
                    "chat_path": spec.chat_path,
                    "auth_env_var": spec.auth_env_var,
                    "env_present": env_present,
                    "enabled": spec.enabled,
                    "local": spec.local,
                    "available": provider_available,
                    "health": self._provider_health(provider_id, provider_available=provider_available),
                }
            )

        model_rows: list[dict[str, Any]] = []
        for model in self.registry.sorted_models():
            impl = self._providers.get(model.provider)
            provider_available = bool(impl.available()) if impl else False
            model_rows.append(
                {
                    "id": model.id,
                    "provider": model.provider,
                    "model": model.model,
                    "capabilities": sorted(model.capabilities),
                    "enabled": model.enabled,
                    "available": model.available,
                    "routable": bool(model.enabled and model.available and provider_available),
                    "input_cost_per_million_tokens": model.input_cost_per_million_tokens,
                    "output_cost_per_million_tokens": model.output_cost_per_million_tokens,
                    "max_context_tokens": model.max_context_tokens,
                    "health": self._model_health(model, provider_available=provider_available),
                }
            )

        env_status = self.registry.provider_env_status()
        present = sorted([name for name, ok in env_status.items() if ok])
        missing = sorted([name for name, ok in env_status.items() if not ok])

        return {
            "routing_mode": self.policy.mode,
            "defaults": {
                "default_provider": self.policy.default_provider,
                "default_model": self.policy.default_model,
                "allow_remote_fallback": self.policy.allow_remote_fallback,
            },
            "providers": provider_rows,
            "models": model_rows,
            "env": {
                "present": present,
                "missing": missing,
            },
            "circuits": self.circuit_states(),
            "usage_stats": self._usage_stats.snapshot(),
        }
