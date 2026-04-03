from __future__ import annotations

import time
import unittest

from agent.config import Config
from agent.llm.policy import RoutingPolicy
from agent.llm.providers.base import Provider
from agent.llm.registry import DefaultsConfig, ModelConfig, ProviderConfig, Registry
from agent.llm.router import LLMRouter
from agent.llm.types import EmbeddingResponse, LLMError, Request, Response, Usage
from agent.llm.usage_stats import UsageStatsStore


class FakeProvider(Provider):
    def __init__(self, name: str, outcomes: list[Response | LLMError], available: bool = True) -> None:
        self._name = name
        self._outcomes = list(outcomes)
        self._available = available
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def available(self) -> bool:
        return self._available

    def chat(self, request: Request, *, model: str, timeout_seconds: float) -> Response:
        _ = request
        _ = timeout_seconds
        self.calls += 1
        if not self._outcomes:
            return Response(text="ok", provider=self._name, model=model, usage=Usage(20, 10, 30))
        value = self._outcomes.pop(0)
        if isinstance(value, LLMError):
            raise value
        return value

    def embed_texts(self, texts: tuple[str, ...], *, model: str, timeout_seconds: float) -> EmbeddingResponse:
        _ = texts
        _ = timeout_seconds
        self.calls += 1
        return EmbeddingResponse(provider=self._name, model=model, vectors=((0.1, 0.2, 0.3),), usage=Usage(3, 0, 3))


def _config(**overrides) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key="sk-test",
        openai_model="gpt-4.1-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=":memory:",
        log_path="/tmp/log.jsonl",
        skills_path="/tmp/skills",
        ollama_host=None,
        ollama_model=None,
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=12,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url=None,
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=False,
        openrouter_api_key=None,
        openrouter_base_url=None,
        openrouter_model=None,
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=None,
        llm_routing_mode="prefer_local_lowest_cost_capable",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=None,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry() -> Registry:
    providers = {
        "local": ProviderConfig(
            id="local",
            provider_type="openai_compat",
            base_url="http://127.0.0.1:11434",
            chat_path="/v1/chat/completions",
            api_key_source=None,
            default_headers={},
            default_query_params={},
            enabled=True,
            local=True,
        ),
        "remote_a": ProviderConfig(
            id="remote_a",
            provider_type="openai_compat",
            base_url="https://a.example/v1",
            chat_path="/v1/chat/completions",
            api_key_source=None,
            default_headers={},
            default_query_params={},
            enabled=True,
            local=False,
        ),
        "remote_b": ProviderConfig(
            id="remote_b",
            provider_type="openai_compat",
            base_url="https://b.example/v1",
            chat_path="/v1/chat/completions",
            api_key_source=None,
            default_headers={},
            default_query_params={},
            enabled=True,
            local=False,
        ),
    }

    models = {
        "local:chat": ModelConfig(
            id="local:chat",
            provider="local",
            model="chat",
            capabilities=frozenset({"chat"}),
            task_types=("chat",),
            quality_rank=2,
            cost_rank=0,
            default_for=("chat",),
            enabled=True,
            input_cost_per_million_tokens=None,
            output_cost_per_million_tokens=None,
            max_context_tokens=8192,
        ),
        "remote_a:cheap": ModelConfig(
            id="remote_a:cheap",
            provider="remote_a",
            model="cheap",
            capabilities=frozenset({"chat", "vision"}),
            task_types=("chat",),
            quality_rank=5,
            cost_rank=2,
            default_for=("chat",),
            enabled=True,
            input_cost_per_million_tokens=0.2,
            output_cost_per_million_tokens=0.8,
            max_context_tokens=128000,
        ),
        "remote_b:expensive": ModelConfig(
            id="remote_b:expensive",
            provider="remote_b",
            model="expensive",
            capabilities=frozenset({"chat", "vision"}),
            task_types=("chat",),
            quality_rank=6,
            cost_rank=4,
            default_for=("chat",),
            enabled=True,
            input_cost_per_million_tokens=1.2,
            output_cost_per_million_tokens=4.0,
            max_context_tokens=128000,
        ),
        "remote_b:tool": ModelConfig(
            id="remote_b:tool",
            provider="remote_b",
            model="tool",
            capabilities=frozenset({"chat", "tools", "json"}),
            task_types=("chat",),
            quality_rank=7,
            cost_rank=3,
            default_for=("chat",),
            enabled=True,
            input_cost_per_million_tokens=0.4,
            output_cost_per_million_tokens=1.2,
            max_context_tokens=128000,
        ),
    }

    defaults = DefaultsConfig(
        routing_mode="prefer_local_lowest_cost_capable",
        default_provider=None,
        default_model=None,
        allow_remote_fallback=True,
    )

    return Registry(
        schema_version=2,
        path=None,
        providers=providers,
        models=models,
        defaults=defaults,
        fallback_chain=("local:chat", "remote_a:cheap", "remote_b:expensive", "remote_b:tool"),
    )


def _policy(mode: str = "prefer_local_lowest_cost_capable") -> RoutingPolicy:
    return RoutingPolicy(
        mode=mode,
        retry_attempts=1,
        retry_base_delay_ms=0,
        circuit_breaker_failures=2,
        circuit_breaker_window_seconds=60,
        circuit_breaker_cooldown_seconds=30,
        default_timeout_seconds=10,
        allow_remote_fallback=True,
        fallback_chain=("local:chat", "remote_a:cheap", "remote_b:expensive", "remote_b:tool"),
    )


class TestLLMRouter(unittest.TestCase):
    def test_retriable_error_retries_same_candidate(self) -> None:
        local = FakeProvider(
            "local",
            [
                LLMError(
                    kind="timeout",
                    retriable=True,
                    provider="local",
                    status_code=None,
                    message="slow",
                ),
                Response("after-retry", "local", "chat", usage=Usage(20, 10, 30)),
            ],
        )
        router = LLMRouter(
            _config(llm_retry_attempts=2),
            providers={
                "local": local,
                "remote_a": FakeProvider("remote_a", []),
                "remote_b": FakeProvider("remote_b", []),
            },
            registry=_registry(),
            policy=RoutingPolicy(
                mode="prefer_local_lowest_cost_capable",
                retry_attempts=2,
                retry_base_delay_ms=0,
                circuit_breaker_failures=2,
                circuit_breaker_window_seconds=60,
                circuit_breaker_cooldown_seconds=30,
                default_timeout_seconds=10,
                allow_remote_fallback=True,
                fallback_chain=("local:chat", "remote_a:cheap", "remote_b:expensive", "remote_b:tool"),
            ),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat([{"role": "user", "content": "hi"}], purpose="chat", task_type="chat")
        self.assertTrue(result["ok"])
        self.assertEqual("local", result["provider"])
        self.assertEqual(2, local.calls)
        self.assertFalse(result["fallback_used"])

    def test_circuit_breaker_opens_and_resets_after_cooldown(self) -> None:
        now = [10.0]
        local = FakeProvider(
            "local",
            [
                LLMError(
                    kind="server_error",
                    retriable=False,
                    provider="local",
                    status_code=500,
                    message="boom-1",
                ),
                LLMError(
                    kind="server_error",
                    retriable=False,
                    provider="local",
                    status_code=500,
                    message="boom-2",
                ),
                Response("local-recovered", "local", "chat", usage=Usage(18, 8, 26)),
            ],
        )
        remote = FakeProvider(
            "remote_a",
            [
                Response("remote-1", "remote_a", "cheap", usage=Usage(20, 10, 30)),
                Response("remote-2", "remote_a", "cheap", usage=Usage(20, 10, 30)),
                Response("remote-3", "remote_a", "cheap", usage=Usage(20, 10, 30)),
            ],
        )
        router = LLMRouter(
            _config(),
            providers={
                "local": local,
                "remote_a": remote,
                "remote_b": FakeProvider("remote_b", []),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
            time_fn=lambda: now[0],
            sleep_fn=lambda _: None,
        )

        first = router.chat([{"role": "user", "content": "one"}], purpose="chat", task_type="chat")
        self.assertTrue(first["ok"])
        self.assertEqual("remote_a", first["provider"])

        second = router.chat([{"role": "user", "content": "two"}], purpose="chat", task_type="chat")
        self.assertTrue(second["ok"])
        self.assertEqual("remote_a", second["provider"])

        third = router.chat([{"role": "user", "content": "three"}], purpose="chat", task_type="chat")
        self.assertTrue(third["ok"])
        self.assertEqual("remote_a", third["provider"])
        self.assertEqual("circuit_open", third["attempts"][0]["reason"])

        now[0] += 31.0
        fourth = router.chat([{"role": "user", "content": "four"}], purpose="chat", task_type="chat")
        self.assertTrue(fourth["ok"])
        self.assertEqual("local", fourth["provider"])
        self.assertEqual(3, local.calls)

    def test_local_model_wins_when_capable(self) -> None:
        router = LLMRouter(
            _config(),
            providers={
                "local": FakeProvider("local", [Response("local", "local", "chat", usage=Usage(20, 10, 30))]),
                "remote_a": FakeProvider("remote_a", []),
                "remote_b": FakeProvider("remote_b", []),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat([{"role": "user", "content": "hi"}], purpose="chat", task_type="chat")
        self.assertTrue(result["ok"])
        self.assertEqual("local", result["provider"])

    def test_cheapest_remote_wins_when_no_local_capable(self) -> None:
        router = LLMRouter(
            _config(),
            providers={
                "local": FakeProvider("local", []),
                "remote_a": FakeProvider("remote_a", [Response("cheap", "remote_a", "cheap", usage=Usage(100, 20, 120))]),
                "remote_b": FakeProvider("remote_b", [Response("exp", "remote_b", "expensive", usage=Usage(100, 20, 120))]),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat(
            [{"role": "user", "content": "vision request"}],
            purpose="chat",
            task_type="chat",
            require_vision=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("remote_a", result["provider"])
        self.assertEqual("cheap", result["model"])

    def test_capability_gating_enforced(self) -> None:
        router = LLMRouter(
            _config(),
            providers={
                "local": FakeProvider("local", []),
                "remote_a": FakeProvider("remote_a", []),
                "remote_b": FakeProvider("remote_b", [Response("tool", "remote_b", "tool")]),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat(
            [{"role": "user", "content": "need tool"}],
            purpose="chat",
            task_type="chat",
            require_tools=True,
            require_json=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("remote_b", result["provider"])
        self.assertEqual("tool", result["model"])

    def test_deterministic_tie_break_by_provider_and_model(self) -> None:
        registry = _registry()
        # Make remote costs equal to force tie-break.
        model_a = registry.models["remote_a:cheap"]
        model_b = registry.models["remote_b:expensive"]
        registry = Registry(
            schema_version=registry.schema_version,
            path=registry.path,
            providers=registry.providers,
            models={
                **registry.models,
                "remote_a:cheap": ModelConfig(
                    **{**model_a.__dict__, "input_cost_per_million_tokens": 1.0, "output_cost_per_million_tokens": 1.0}
                ),
                "remote_b:expensive": ModelConfig(
                    **{**model_b.__dict__, "input_cost_per_million_tokens": 1.0, "output_cost_per_million_tokens": 1.0}
                ),
                "local:chat": ModelConfig(
                    **{**registry.models["local:chat"].__dict__, "enabled": False}
                ),
            },
            defaults=registry.defaults,
            fallback_chain=registry.fallback_chain,
        )

        router = LLMRouter(
            _config(),
            providers={
                "local": FakeProvider("local", []),
                "remote_a": FakeProvider("remote_a", [Response("a", "remote_a", "cheap")]),
                "remote_b": FakeProvider("remote_b", [Response("b", "remote_b", "expensive")]),
            },
            registry=registry,
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat(
            [{"role": "user", "content": "vision request"}],
            purpose="chat",
            task_type="chat",
            require_vision=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("remote_a", result["provider"])
        self.assertEqual("cheap", result["model"])

    def test_fallback_to_next_candidate_when_first_fails(self) -> None:
        router = LLMRouter(
            _config(),
            providers={
                "local": FakeProvider(
                    "local",
                    [
                        LLMError(
                            kind="server_error",
                            retriable=False,
                            provider="local",
                            status_code=500,
                            message="boom",
                        )
                    ],
                ),
                "remote_a": FakeProvider("remote_a", [Response("remote", "remote_a", "cheap")]),
                "remote_b": FakeProvider("remote_b", []),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat([{"role": "user", "content": "hello"}], purpose="chat", task_type="chat")
        self.assertTrue(result["ok"])
        self.assertEqual("remote_a", result["provider"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual("server_error", result["attempts"][0]["reason"])

    def test_disabled_or_unavailable_models_in_fallback_chain_are_never_attempted(self) -> None:
        registry = _registry()
        local_model = registry.models["local:chat"]
        remote_a_model = registry.models["remote_a:cheap"]
        remote_b_model = registry.models["remote_b:expensive"]
        registry = Registry(
            schema_version=registry.schema_version,
            path=registry.path,
            providers=registry.providers,
            models={
                **registry.models,
                "local:chat": ModelConfig(
                    **{**local_model.__dict__, "enabled": False}
                ),
                "remote_a:cheap": ModelConfig(
                    **{**remote_a_model.__dict__, "available": False}
                ),
                "remote_b:expensive": ModelConfig(
                    **{**remote_b_model.__dict__, "enabled": True, "available": True}
                ),
            },
            defaults=DefaultsConfig(
                routing_mode="auto",
                default_provider=None,
                default_model=None,
                allow_remote_fallback=True,
            ),
            fallback_chain=("local:chat", "remote_a:cheap", "remote_b:expensive"),
        )

        local = FakeProvider("local", [Response("should-not-run", "local", "chat")])
        remote_a = FakeProvider("remote_a", [Response("should-not-run", "remote_a", "cheap")])
        remote_b = FakeProvider("remote_b", [Response("ok", "remote_b", "expensive")])

        router = LLMRouter(
            _config(),
            providers={
                "local": local,
                "remote_a": remote_a,
                "remote_b": remote_b,
            },
            registry=registry,
            policy=RoutingPolicy(
                mode="auto",
                retry_attempts=1,
                retry_base_delay_ms=0,
                circuit_breaker_failures=2,
                circuit_breaker_window_seconds=60,
                circuit_breaker_cooldown_seconds=30,
                default_timeout_seconds=10,
                allow_remote_fallback=True,
                fallback_chain=("local:chat", "remote_a:cheap", "remote_b:expensive"),
            ),
            usage_stats=UsageStatsStore(None),
        )

        result = router.chat([{"role": "user", "content": "hello"}], purpose="chat", task_type="chat")
        self.assertTrue(result["ok"])
        self.assertEqual("remote_b", result["provider"])
        self.assertEqual(0, local.calls)
        self.assertEqual(0, remote_a.calls)
        self.assertEqual(1, remote_b.calls)

    def test_server_error_triggers_cooldown_and_degraded_health(self) -> None:
        remote = FakeProvider(
            "remote_a",
            [
                LLMError(
                    kind="server_error",
                    retriable=False,
                    provider="remote_a",
                    status_code=502,
                    message="bad_gateway",
                ),
            ],
        )
        local = FakeProvider("local", [Response("local-ok", "local", "chat"), Response("local-ok-2", "local", "chat")])
        router = LLMRouter(
            _config(),
            providers={
                "local": local,
                "remote_a": remote,
                "remote_b": FakeProvider("remote_b", []),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
            time_fn=lambda: 100.0,
            sleep_fn=lambda _: None,
        )

        first = router.chat(
            [{"role": "user", "content": "force remote"}],
            purpose="chat",
            task_type="chat",
            provider_override="remote_a",
        )
        self.assertFalse(first["ok"])

        second = router.chat([{"role": "user", "content": "normal"}], purpose="chat", task_type="chat")
        self.assertTrue(second["ok"])
        self.assertEqual("local", second["provider"])

        state = router._outcomes["remote_a:cheap"]  # type: ignore[attr-defined]
        self.assertIsNotNone(state.cooldown_until)
        self.assertGreater(float(state.cooldown_until or 0.0), 100.0)
        self.assertEqual("server_error", state.last_error_kind)

        snapshot = router.doctor_snapshot()
        provider_health = {item["id"]: item["health"] for item in snapshot["providers"]}
        self.assertEqual("degraded", provider_health["remote_a"]["status"])

    def test_rate_limit_candidate_is_deprioritized_when_alternative_exists(self) -> None:
        registry = _registry()
        local_model = registry.models["local:chat"]
        registry = Registry(
            schema_version=registry.schema_version,
            path=registry.path,
            providers=registry.providers,
            models={
                **registry.models,
                "local:chat": ModelConfig(
                    **{**local_model.__dict__, "enabled": False}
                ),
            },
            defaults=registry.defaults,
            fallback_chain=registry.fallback_chain,
        )

        remote_a = FakeProvider(
            "remote_a",
            [
                LLMError(
                    kind="rate_limit",
                    retriable=False,
                    provider="remote_a",
                    status_code=429,
                    message="too_many_requests",
                ),
                Response("should-not-be-picked-next", "remote_a", "cheap"),
            ],
        )
        remote_b = FakeProvider(
            "remote_b",
            [
                Response("fallback-success", "remote_b", "expensive"),
                Response("preferred-after-rate-limit", "remote_b", "expensive"),
            ],
        )

        router = LLMRouter(
            _config(),
            providers={
                "local": FakeProvider("local", []),
                "remote_a": remote_a,
                "remote_b": remote_b,
            },
            registry=registry,
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
            time_fn=lambda: 200.0,
            sleep_fn=lambda _: None,
        )

        first = router.chat([{"role": "user", "content": "first"}], purpose="chat", task_type="chat")
        self.assertTrue(first["ok"])
        self.assertEqual("remote_b", first["provider"])

        second = router.chat([{"role": "user", "content": "second"}], purpose="chat", task_type="chat")
        self.assertTrue(second["ok"])
        self.assertEqual("remote_b", second["provider"])
        self.assertEqual(1, remote_a.calls)

    def test_external_health_skip_list_avoids_recently_down_candidate(self) -> None:
        remote_a = FakeProvider("remote_a", [Response("remote-ok", "remote_a", "cheap")])
        local = FakeProvider("local", [Response("local-should-not-run", "local", "chat")])
        router = LLMRouter(
            _config(),
            providers={
                "local": local,
                "remote_a": remote_a,
                "remote_b": FakeProvider("remote_b", []),
            },
            registry=_registry(),
            policy=_policy(),
            usage_stats=UsageStatsStore(None),
        )
        router.set_external_health_state(
            {
                "providers": {
                    "local": {
                        "status": "down",
                        "cooldown_until": int(time.time()) + 120,
                    }
                },
                "models": {
                    "local:chat": {
                        "provider_id": "local",
                        "status": "down",
                        "cooldown_until": int(time.time()) + 120,
                    }
                },
                "last_run_at": int(time.time()),
            }
        )

        result = router.chat([{"role": "user", "content": "hello"}], purpose="chat", task_type="chat")
        self.assertTrue(result["ok"])
        self.assertEqual("remote_a", result["provider"])
        self.assertEqual(0, local.calls)
        self.assertEqual(1, remote_a.calls)


if __name__ == "__main__":
    unittest.main()
