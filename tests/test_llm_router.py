from __future__ import annotations

import unittest

from agent.config import Config
from agent.llm.policy import RoutingPolicy
from agent.llm.providers.base import Provider
from agent.llm.registry import ModelConfig, ProviderConfig, Registry
from agent.llm.router import LLMRouter
from agent.llm.types import LLMError, Request, Response, Usage


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
            return Response(text="ok", provider=self._name, model=model, usage=Usage())
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, LLMError):
            raise outcome
        return outcome


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
        prefer_local=False,
        llm_timeout_seconds=10,
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
        llm_routing_mode="auto",
        llm_retry_attempts=2,
        llm_retry_base_delay_ms=50,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
    )
    return base.__class__(**{**base.__dict__, **overrides})


def _registry() -> Registry:
    providers = {
        "openai": ProviderConfig(
            name="openai",
            provider_type="openai",
            base_url="https://api.openai.com/v1",
            auth_env_var="OPENAI_API_KEY",
            enabled=True,
        ),
        "backup": ProviderConfig(
            name="backup",
            provider_type="openai",
            base_url="https://api.example.test/v1",
            auth_env_var="BACKUP_API_KEY",
            enabled=True,
        ),
    }
    models = {
        "openai:cheap-chat": ModelConfig(
            id="openai:cheap-chat",
            provider="openai",
            model="cheap-chat",
            capabilities=frozenset({"chat"}),
            quality_rank=5,
            cost_rank=1,
            default_for=("chat",),
            enabled=True,
        ),
        "openai:tool-json": ModelConfig(
            id="openai:tool-json",
            provider="openai",
            model="tool-json",
            capabilities=frozenset({"chat", "tools", "json"}),
            quality_rank=7,
            cost_rank=3,
            default_for=("chat", "presentation_rewrite"),
            enabled=True,
        ),
        "openai:vision": ModelConfig(
            id="openai:vision",
            provider="openai",
            model="vision",
            capabilities=frozenset({"chat", "vision"}),
            quality_rank=8,
            cost_rank=6,
            default_for=("best_quality",),
            enabled=True,
        ),
        "backup:stable": ModelConfig(
            id="backup:stable",
            provider="backup",
            model="stable",
            capabilities=frozenset({"chat"}),
            quality_rank=6,
            cost_rank=2,
            default_for=("chat",),
            enabled=True,
        ),
    }
    return Registry(providers=providers, models=models, routing_defaults={"mode": "auto", "fallback_chain": ()})


def _policy(**overrides) -> RoutingPolicy:
    base = RoutingPolicy(
        mode="prefer_cheap",
        retry_attempts=2,
        retry_base_delay_ms=25,
        circuit_breaker_failures=2,
        circuit_breaker_window_seconds=60,
        circuit_breaker_cooldown_seconds=30,
        default_timeout_seconds=10,
        fallback_chain=("openai:cheap-chat", "backup:stable", "openai:tool-json", "openai:vision"),
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestLLMRouter(unittest.TestCase):
    def test_deterministic_selection_with_same_health_state(self) -> None:
        provider = FakeProvider("openai", [Response("a", "openai", "cheap-chat"), Response("b", "openai", "cheap-chat")])
        router = LLMRouter(
            _config(),
            providers={"openai": provider, "backup": FakeProvider("backup", [])},
            registry=_registry(),
            policy=_policy(),
        )

        res1 = router.chat([{"role": "user", "content": "hi"}], purpose="chat")
        res2 = router.chat([{"role": "user", "content": "hi"}], purpose="chat")

        self.assertTrue(res1["ok"])
        self.assertTrue(res2["ok"])
        self.assertEqual(res1["provider"], res2["provider"])
        self.assertEqual(res1["model"], res2["model"])

    def test_capability_gating_prefers_tool_model(self) -> None:
        provider = FakeProvider("openai", [Response("tool", "openai", "tool-json")])
        router = LLMRouter(
            _config(),
            providers={"openai": provider, "backup": FakeProvider("backup", [])},
            registry=_registry(),
            policy=_policy(),
        )

        result = router.chat(
            [{"role": "user", "content": "call a tool"}],
            purpose="chat",
            require_tools=True,
            require_json=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("openai", result["provider"])
        self.assertEqual("tool-json", result["model"])

    def test_fallback_when_primary_fails(self) -> None:
        failing = FakeProvider(
            "openai",
            [
                LLMError(
                    kind="server_error",
                    retriable=False,
                    provider="openai",
                    status_code=500,
                    message="boom",
                )
            ],
        )
        backup = FakeProvider("backup", [Response("from backup", "backup", "stable")])

        router = LLMRouter(
            _config(),
            providers={"openai": failing, "backup": backup},
            registry=_registry(),
            policy=_policy(),
        )

        result = router.chat([{"role": "user", "content": "hello"}], purpose="chat")
        self.assertTrue(result["ok"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual("backup", result["provider"])
        self.assertEqual("stable", result["model"])
        self.assertEqual("server_error", result["attempts"][0]["reason"])

    def test_circuit_breaker_opens_and_resets_after_cooldown(self) -> None:
        clock = FakeClock()
        primary = FakeProvider(
            "openai",
            [
                LLMError(
                    kind="server_error",
                    retriable=False,
                    provider="openai",
                    status_code=500,
                    message="fail-1",
                ),
                LLMError(
                    kind="server_error",
                    retriable=False,
                    provider="openai",
                    status_code=500,
                    message="fail-2",
                ),
                Response("primary recovered", "openai", "cheap-chat"),
            ],
        )
        backup = FakeProvider(
            "backup",
            [
                Response("backup1", "backup", "stable"),
                Response("backup2", "backup", "stable"),
                Response("backup3", "backup", "stable"),
            ],
        )

        router = LLMRouter(
            _config(),
            providers={"openai": primary, "backup": backup},
            registry=_registry(),
            policy=_policy(circuit_breaker_failures=2, circuit_breaker_cooldown_seconds=30),
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        first = router.chat([{"role": "user", "content": "hi"}], purpose="chat")
        second = router.chat([{"role": "user", "content": "hi"}], purpose="chat")
        third = router.chat([{"role": "user", "content": "hi"}], purpose="chat")

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(third["ok"])
        self.assertEqual("backup", third["provider"])
        self.assertEqual("circuit_open", third["attempts"][0]["reason"])

        clock.advance(31)
        fourth = router.chat([{"role": "user", "content": "hi"}], purpose="chat")
        self.assertTrue(fourth["ok"])
        self.assertEqual("openai", fourth["provider"])
        self.assertEqual("cheap-chat", fourth["model"])

    def test_retries_retriable_errors_with_backoff(self) -> None:
        clock = FakeClock()
        provider = FakeProvider(
            "openai",
            [
                LLMError(
                    kind="timeout",
                    retriable=True,
                    provider="openai",
                    status_code=None,
                    message="timeout",
                ),
                Response("ok", "openai", "cheap-chat"),
            ],
        )

        router = LLMRouter(
            _config(),
            providers={"openai": provider, "backup": FakeProvider("backup", [])},
            registry=_registry(),
            policy=_policy(retry_attempts=2, retry_base_delay_ms=50),
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        result = router.chat([{"role": "user", "content": "hello"}], purpose="chat")
        self.assertTrue(result["ok"])
        self.assertEqual(2, provider.calls)
        self.assertEqual(1, len(clock.sleep_calls))
        self.assertGreater(clock.sleep_calls[0], 0)


if __name__ == "__main__":
    unittest.main()
