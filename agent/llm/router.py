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
from agent.llm.providers.openai import OpenAIProvider
from agent.llm.registry import ModelConfig, ProviderConfig, Registry, load_registry
from agent.llm.types import LLMError, Message, Request
from agent.logging_utils import log_event


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
    ) -> None:
        self.config = config
        self.registry = registry or load_registry(config)
        self.policy = policy or load_routing_policy(config, self.registry)
        self._providers = providers or self._build_default_providers()
        self._circuits: dict[str, CircuitBreakerState] = {}
        self._rng = rng or random.Random()
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._log_path = log_path

    def _build_default_providers(self) -> dict[str, Provider]:
        providers: dict[str, Provider] = {}

        openai_spec = self.registry.providers.get("openai")
        if openai_spec:
            providers["openai"] = OpenAIProvider(
                api_key=self.config.openai_api_key,
                base_url=openai_spec.base_url,
                provider_name="openai",
                enabled=openai_spec.enabled,
            )

        openrouter_spec = self.registry.providers.get("openrouter")
        if openrouter_spec:
            providers["openrouter"] = OpenAIProvider(
                api_key=self.config.openrouter_api_key,
                base_url=openrouter_spec.base_url,
                provider_name="openrouter",
                enabled=openrouter_spec.enabled,
            )

        # Reserved slot for future provider implementations.
        if "ollama" in self.registry.providers and "ollama" not in providers:
            providers["ollama"] = _UnavailableProvider("ollama")

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
        candidates = self._candidate_models(request)
        return bool(candidates)

    def intent_from_text(self, text: str) -> dict[str, Any] | None:
        _ = text
        return None

    def set_routing_mode(self, mode: str) -> None:
        normalized = (mode or "").strip().lower()
        if normalized not in {"auto", "prefer_cheap", "prefer_best"}:
            raise ValueError("routing mode must be one of: auto, prefer_cheap, prefer_best")
        self.policy = replace(self.policy, mode=normalized)

    def set_provider_api_key(self, provider: str, api_key: str) -> bool:
        provider_name = (provider or "").strip().lower()
        secret = (api_key or "").strip()
        if not provider_name:
            return False

        spec = self.registry.providers.get(provider_name)
        if not spec:
            return False

        impl = self._providers.get(provider_name)
        if impl is None:
            return False

        if hasattr(impl, "set_api_key"):
            getattr(impl, "set_api_key")(secret)

        if spec.auth_env_var:
            if secret:
                os.environ[spec.auth_env_var] = secret
            else:
                os.environ.pop(spec.auth_env_var, None)

        return True

    def _candidate_models(self, request: Request) -> list[ModelConfig]:
        all_models = self.registry.sorted_models()
        allowed = self.policy.ordered_candidates(request, all_models)
        candidates: list[ModelConfig] = []
        for model in allowed:
            if self._is_circuit_open(model.id):
                continue
            provider = self._providers.get(model.provider)
            if provider is None:
                continue
            if not provider.available():
                continue
            candidates.append(model)
        return candidates

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

    def _sleep_retry(self, attempt_idx: int) -> None:
        base = float(self.policy.retry_base_delay_ms) / 1000.0
        if base <= 0:
            return
        delay = base * (2**attempt_idx)
        jitter = self._rng.random() * base
        self._sleep_fn(delay + jitter)

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
        tools: list[dict[str, Any]] | None = None,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = compute_tier
        start = float(self._time_fn())

        request = Request(
            messages=self._normalize_messages(messages),
            purpose=purpose,
            provider=(provider_override or "").strip().lower() or None,
            model=(model_override or "").strip() or None,
            require_tools=bool(require_tools),
            require_json=bool(require_json),
            require_vision=bool(require_vision),
            tools=tuple(tools or ()),
            timeout_seconds=timeout_seconds,
            metadata=metadata or {},
        )

        all_candidates = self.policy.ordered_candidates(request, self.registry.sorted_models())
        attempts: list[dict[str, Any]] = []
        last_error: LLMError | None = None

        for model in all_candidates:
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
                payload = {
                    "ok": True,
                    "text": response.text,
                    "provider": response.provider,
                    "model": response.model,
                    "fallback_used": bool(attempts),
                    "attempts": attempts,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "duration_ms": duration_ms,
                    "error_class": None,
                }
                if self._log_path:
                    log_event(
                        self._log_path,
                        "llm_routing_decision",
                        {
                            "purpose": purpose,
                            "required_capabilities": sorted(self.policy.required_capabilities(request)),
                            "selected_provider": response.provider,
                            "selected_model": response.model,
                            "fallback_used": bool(attempts),
                            "attempts": attempts,
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
                    "required_capabilities": sorted(self.policy.required_capabilities(request)),
                    "selected_provider": "none",
                    "selected_model": None,
                    "fallback_used": False,
                    "attempts": attempts,
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
        result = self.chat(messages, purpose=task_kind, compute_tier=tier_required)
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
        for name, spec in sorted(self.registry.providers.items()):
            impl = self._providers.get(name)
            env_present = None
            if spec.auth_env_var:
                env_present = bool(os.getenv(spec.auth_env_var, "").strip())
            provider_rows.append(
                {
                    "name": name,
                    "type": spec.provider_type,
                    "base_url": spec.base_url,
                    "auth_env_var": spec.auth_env_var,
                    "env_present": env_present,
                    "enabled": spec.enabled,
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
                }
            )

        env_status = self.registry.provider_env_status()
        present = sorted([name for name, ok in env_status.items() if ok])
        missing = sorted([name for name, ok in env_status.items() if not ok])

        return {
            "routing_mode": self.policy.mode,
            "providers": provider_rows,
            "models": model_rows,
            "env": {
                "present": present,
                "missing": missing,
            },
            "circuits": self.circuit_states(),
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
