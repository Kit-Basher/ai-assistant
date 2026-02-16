from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import json
import os
import random
import time
from typing import Any

from agent.config import Config, load_config
from agent.llm.base import LLMResult
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

        return True

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
        base_candidates = [
            model
            for model in self.registry.sorted_models()
            if model.enabled and self.policy.allows_model(request, model) and self._context_capable(model, request)
        ]

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

    def _reliability_penalty(self, model: ModelConfig) -> float:
        state = self._circuits.get(model.id)
        if not state:
            return 0.0
        return float(len(state.failures)) / max(1, float(self.policy.circuit_breaker_failures))

    def _ordered_models(self, request: Request) -> list[ModelConfig]:
        candidates = self._candidate_models(request)
        if self.policy.mode != "prefer_local_lowest_cost_capable":
            return self.policy.ordered_candidates(request, candidates)

        def sort_key(model: ModelConfig) -> tuple[int, float, float, str, str]:
            provider_cfg = self.registry.providers.get(model.provider)
            is_local = bool(provider_cfg.local) if provider_cfg else False
            return (
                0 if is_local else 1,
                self._expected_cost(request, model),
                self._reliability_penalty(model),
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

            try:
                response = self._call_with_retry(provider, model, request)
                self._record_success(model.id)
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
                return payload
            except LLMError as exc:
                last_error = exc
                self._record_failure(model.id)
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
            log_event(
                self._log_path,
                "llm_routing_decision",
                {
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

    def doctor_snapshot(self) -> dict[str, Any]:
        provider_rows: list[dict[str, Any]] = []
        for provider_id, spec in sorted(self.registry.providers.items()):
            impl = self._providers.get(provider_id)
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
                    "available": bool(impl.available()) if impl else False,
                }
            )

        model_rows: list[dict[str, Any]] = []
        for model in self.registry.sorted_models():
            impl = self._providers.get(model.provider)
            model_rows.append(
                {
                    "id": model.id,
                    "provider": model.provider,
                    "model": model.model,
                    "capabilities": sorted(model.capabilities),
                    "enabled": model.enabled,
                    "available": bool(impl.available()) if impl else False,
                    "input_cost_per_million_tokens": model.input_cost_per_million_tokens,
                    "output_cost_per_million_tokens": model.output_cost_per_million_tokens,
                    "max_context_tokens": model.max_context_tokens,
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


class LLMNarrationRouter:
    """Backward-compatible narration helper.

    New runtime flows should use LLMRouter directly.
    """

    def __init__(self) -> None:
        narration_flag = os.getenv("ENABLE_NARRATION", "").strip().lower()
        legacy_flag = os.getenv("LLM_NARRATION_ENABLED", "").strip().lower()
        self.narration_enabled = (narration_flag or legacy_flag) in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.routing = os.getenv("LLM_ROUTING", "auto").strip().lower() or "auto"
        self.timeout_s = int(os.getenv("LLM_TIMEOUT_SECONDS", "15") or 15)

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
        self.ollama_model = os.getenv("OLLAMA_MODEL", "").strip()

        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openrouter_base_url = (
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
        )
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()

    def _available_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)

    def _available_ollama(self) -> bool:
        return bool(self.ollama_host and self.ollama_model)

    def _available_openrouter(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_base_url and self.openrouter_model)

    def _upgrade_requested(self, payload: dict | None) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("upgrade") or payload.get("require_big_model"):
            return True
        tier = payload.get("tier_required")
        return tier == "high"

    def _cloud_order(self, upgrade: bool) -> list[str]:
        if upgrade:
            return ["openrouter", "openai"]
        return ["openai", "openrouter"]

    def _provider_order(self, payload: dict | None = None) -> list[str]:
        upgrade = self._upgrade_requested(payload)
        if self.routing == "local_only":
            return ["ollama"]
        if self.routing == "cloud_only":
            return self._cloud_order(upgrade)
        if self.routing == "cloud_first":
            return self._cloud_order(upgrade) + ["ollama"]
        if self.routing == "local_first":
            return ["ollama"] + self._cloud_order(upgrade)
        if upgrade:
            return self._cloud_order(upgrade) + ["ollama"]
        return ["ollama", "openai"]

    def _build_router(self) -> LLMRouter | None:
        try:
            cfg = load_config(require_telegram_token=False)
            return LLMRouter(cfg)
        except Exception:
            return None

    def summarize(self, kind: str, payload: dict) -> LLMResult | None:
        if not self.narration_enabled:
            return None
        router = self._build_router()
        if router is None:
            return LLMResult("", "", "", True, "router_unavailable")

        provider_order = self._provider_order(payload)
        last_error = "no_providers"
        for provider_name in provider_order:
            result = router.chat(
                [
                    {"role": "system", "content": "Summarize the payload in 2-4 short bullet lines."},
                    {
                        "role": "user",
                        "content": json.dumps({"kind": kind, "payload": payload}, ensure_ascii=True),
                    },
                ],
                purpose="narration",
                provider_override=provider_name,
                timeout_seconds=self.timeout_s,
            )
            if result.get("ok") and result.get("text"):
                return LLMResult(
                    result.get("text") or "",
                    result.get("provider") or provider_name,
                    result.get("model") or "",
                    bool(result.get("fallback_used")),
                    None,
                )
            if result.get("error_class"):
                last_error = str(result.get("error_class"))

        return LLMResult("", "", "", True, last_error)

    def status(self) -> dict[str, str]:
        router = self._build_router()
        if router is None:
            return {
                "narration_enabled": "true" if self.narration_enabled else "false",
                "routing": self.routing,
                "router": "unavailable",
            }
        snapshot = router.doctor_snapshot()
        return {
            "narration_enabled": "true" if self.narration_enabled else "false",
            "routing": self.routing,
            "routing_mode": str(snapshot.get("routing_mode") or "auto"),
            "providers": str(len(snapshot.get("providers") or [])),
            "models": str(len(snapshot.get("models") or [])),
        }
