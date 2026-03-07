from __future__ import annotations

import types
import unittest

from agent.config import Config
from agent.llm.control_contract import (
    normalize_model_inventory,
    normalize_selection_result,
    normalize_task_request,
)
from agent.llm.install_planner import build_install_plan
from agent.llm.model_health_check import check_model_health, check_provider_health
from agent.llm.model_inventory import build_model_inventory
from agent.llm.model_selector import select_model_for_task
from agent.llm.registry import DefaultsConfig, ModelConfig, ProviderConfig, Registry
from agent.llm.task_classifier import classify_task_request


def _config() -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path="/tmp/agent.db",
        log_path="/tmp/agent.log",
        skills_path="/tmp/skills",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen2.5:3b-instruct",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=None,
        default_policy={
            "cost_cap_per_1m": 1.0,
            "allowlist": ["openrouter:cheap-chat"],
            "quality_weight": 1.0,
            "price_weight": 0.04,
            "latency_weight": 0.25,
            "instability_weight": 0.5,
        },
        premium_policy={
            "cost_cap_per_1m": 10.0,
            "allowlist": [],
            "quality_weight": 1.35,
            "price_weight": 0.025,
            "latency_weight": 0.2,
            "instability_weight": 0.45,
        },
    )


def _registry() -> Registry:
    return Registry(
        schema_version=2,
        path=None,
        providers={
            "ollama": ProviderConfig(
                id="ollama",
                provider_type="openai_compat",
                base_url="http://127.0.0.1:11434",
                chat_path="/v1/chat/completions",
                api_key_source=None,
                default_headers={},
                default_query_params={},
                enabled=True,
                local=True,
            ),
            "openrouter": ProviderConfig(
                id="openrouter",
                provider_type="openai_compat",
                base_url="https://openrouter.ai/api/v1",
                chat_path="/v1/chat/completions",
                api_key_source=None,
                default_headers={},
                default_query_params={},
                enabled=True,
                local=False,
            ),
        },
        models={
            "ollama:llama3": ModelConfig(
                id="ollama:llama3",
                provider="ollama",
                model="llama3",
                capabilities=frozenset({"chat"}),
                quality_rank=3,
                cost_rank=1,
                default_for=("chat",),
                enabled=True,
                available=True,
                input_cost_per_million_tokens=None,
                output_cost_per_million_tokens=None,
                max_context_tokens=8192,
            ),
            "openrouter:cheap-chat": ModelConfig(
                id="openrouter:cheap-chat",
                provider="openrouter",
                model="cheap-chat",
                capabilities=frozenset({"chat", "json"}),
                quality_rank=6,
                cost_rank=2,
                default_for=("chat",),
                enabled=True,
                available=True,
                input_cost_per_million_tokens=0.1,
                output_cost_per_million_tokens=0.2,
                max_context_tokens=65536,
            ),
        },
        defaults=DefaultsConfig(
            routing_mode="prefer_local_lowest_cost_capable",
            default_provider="ollama",
            default_model="ollama:llama3",
            allow_remote_fallback=True,
            chat_model="ollama:llama3",
            embed_model=None,
            last_chat_model=None,
        ),
        fallback_chain=(),
    )


class TestLLMControlPlane(unittest.TestCase):
    def test_control_contract_normalizes_shapes(self) -> None:
        inventory = normalize_model_inventory(
            [
                {"id": "openrouter:cheap-chat", "provider": "OPENROUTER", "local": False},
                {"id": "ollama:llama3", "provider": "Ollama", "local": True, "capabilities": ["chat", "chat"]},
            ]
        )
        self.assertEqual(["ollama:llama3", "openrouter:cheap-chat"], [row["id"] for row in inventory])
        self.assertEqual(["chat"], inventory[0]["capabilities"])
        task = normalize_task_request({"task_type": "UNKNOWN", "requirements": ["json", "chat", "json"]})
        self.assertEqual("chat", task["task_type"])
        self.assertEqual(["chat", "json"], task["requirements"])
        selection = normalize_selection_result({"selected_model": " ollama:llama3 ", "fallbacks": ["a", "a", "b"]})
        self.assertEqual("ollama:llama3", selection["selected_model"])
        self.assertEqual(["a", "b"], selection["fallbacks"])

    def test_task_classifier_is_deterministic(self) -> None:
        self.assertEqual("health", classify_task_request("how is my pc")["task_type"])
        coding = classify_task_request("debug this python traceback")
        self.assertEqual("coding", coding["task_type"])
        self.assertEqual(["chat"], coding["requirements"])
        self.assertEqual("vision", classify_task_request("analyze this image")["task_type"])
        reasoning = classify_task_request("compare these approaches and reason deeply")
        self.assertEqual("reasoning", reasoning["task_type"])
        self.assertIn("long_context", reasoning["requirements"])

    def test_model_health_classifies_provider_and_install_failures(self) -> None:
        cfg = _config()
        registry = _registry()
        provider = check_provider_health(
            config=cfg,
            registry=registry,
            provider_id="ollama",
            provider_probe_fn=lambda *_args, **_kwargs: {"status": "down", "error_kind": "connection_refused"},
        )
        self.assertFalse(provider["healthy"])
        self.assertEqual("provider_down", provider["failure_kind"])

        model = check_model_health(
            config=cfg,
            registry=registry,
            model=registry.models["ollama:llama3"],
            installed=False,
            provider_health={"status": "ok"},
        )
        self.assertFalse(model["healthy"])
        self.assertEqual("not_installed", model["failure_kind"])

    def test_inventory_merges_discovered_local_models_and_orders_deterministically(self) -> None:
        cfg = _config()
        registry = _registry()
        from unittest.mock import patch

        with patch("agent.llm.model_inventory.check_provider_health", return_value={"provider_health": {"status": "ok"}}), patch(
            "agent.llm.model_inventory.check_model_health",
            side_effect=lambda **kwargs: {"healthy": kwargs.get("installed", False) or kwargs["model"].provider != "ollama", "failure_kind": None if (kwargs.get("installed", False) or kwargs["model"].provider != "ollama") else "not_installed"},
        ):
            inventory = build_model_inventory(
                config=cfg,
                registry=registry,
                discovered_local_models=["llama3", "qwen2.5:3b-instruct"],
            )
        self.assertEqual(
            ["ollama:llama3", "ollama:qwen2.5:3b-instruct", "openrouter:cheap-chat"],
            [row["id"] for row in inventory],
        )
        self.assertTrue(bool(inventory[1]["installed"]))
        self.assertEqual("ollama_list", inventory[1]["source"])

    def test_selector_prefers_healthy_local_model(self) -> None:
        inventory = [
            {
                "id": "openrouter:cheap-chat",
                "provider": "openrouter",
                "installed": False,
                "available": True,
                "healthy": True,
                "capabilities": ["chat", "json"],
                "context_window": 65536,
                "local": False,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 6,
                "cost_rank": 2,
                "price_in": 0.1,
                "price_out": 0.2,
            },
            {
                "id": "ollama:llama3",
                "provider": "ollama",
                "installed": True,
                "available": True,
                "healthy": True,
                "capabilities": ["chat"],
                "context_window": 8192,
                "local": True,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 3,
                "cost_rank": 1,
                "price_in": None,
                "price_out": None,
            },
        ]
        selection = select_model_for_task(
            inventory,
            classify_task_request("hello there"),
            allow_remote_fallback=True,
            policy_name="default",
            policy={"cost_cap_per_1m": 1.0, "allowlist": []},
            trace_id="sel-1",
        )
        self.assertEqual("ollama:llama3", selection["selected_model"])
        self.assertEqual("ollama", selection["provider"])

    def test_selector_does_not_treat_unhealthy_inventory_row_as_healthy_selection(self) -> None:
        inventory = [
            {
                "id": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "installed": True,
                "available": True,
                "healthy": False,
                "capabilities": ["chat"],
                "context_window": 8192,
                "local": True,
                "approved": True,
                "reason": "provider_down",
                "health_failure_kind": "provider_down",
                "quality_rank": 6,
                "cost_rank": 1,
                "price_in": None,
                "price_out": None,
            },
            {
                "id": "openrouter:cheap-chat",
                "provider": "openrouter",
                "installed": False,
                "available": True,
                "healthy": True,
                "capabilities": ["chat", "json"],
                "context_window": 65536,
                "local": False,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 4,
                "cost_rank": 2,
                "price_in": 0.1,
                "price_out": 0.2,
            },
        ]
        selection = select_model_for_task(
            inventory,
            classify_task_request("how is my pc"),
            allow_remote_fallback=True,
            policy_name="default",
            policy={"cost_cap_per_1m": 1.0, "allowlist": []},
            trace_id="sel-health",
        )
        self.assertEqual("openrouter:cheap-chat", selection["selected_model"])
        self.assertIn("healthy", str(selection["reason"]))
        self.assertNotIn("ollama:qwen2.5:3b-instruct", str(selection["reason"]))

    def test_selector_respects_remote_disable_and_policy(self) -> None:
        inventory = [
            {
                "id": "openrouter:expensive-chat",
                "provider": "openrouter",
                "installed": False,
                "available": True,
                "healthy": True,
                "capabilities": ["chat"],
                "context_window": 65536,
                "local": False,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 8,
                "cost_rank": 5,
                "price_in": 2.0,
                "price_out": 3.0,
            }
        ]
        blocked_remote = select_model_for_task(
            inventory,
            classify_task_request("hello there"),
            allow_remote_fallback=False,
            policy_name="default",
            policy={"cost_cap_per_1m": 10.0, "allowlist": []},
        )
        self.assertIsNone(blocked_remote["selected_model"])

        blocked_cost = select_model_for_task(
            inventory,
            classify_task_request("hello there"),
            allow_remote_fallback=True,
            policy_name="default",
            policy={"cost_cap_per_1m": 1.0, "allowlist": []},
        )
        self.assertIsNone(blocked_cost["selected_model"])

    def test_selector_never_offers_embedding_model_as_chat_or_coding_fallback(self) -> None:
        inventory = [
            {
                "id": "ollama:nomic-embed-text:latest",
                "provider": "ollama",
                "installed": True,
                "available": True,
                "healthy": True,
                "capabilities": ["embedding"],
                "context_window": 8192,
                "local": True,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 1,
                "cost_rank": 1,
                "price_in": None,
                "price_out": None,
            },
            {
                "id": "openrouter:coding-pro",
                "provider": "openrouter",
                "installed": False,
                "available": True,
                "healthy": True,
                "capabilities": ["chat", "json"],
                "context_window": 65536,
                "local": False,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 8,
                "cost_rank": 3,
                "price_in": 0.3,
                "price_out": 0.6,
            },
        ]
        selection = select_model_for_task(
            inventory,
            classify_task_request("debug this python traceback"),
            allow_remote_fallback=True,
            policy_name="default",
            policy={"cost_cap_per_1m": 10.0, "allowlist": []},
            trace_id="sel-code",
        )
        self.assertEqual("openrouter:coding-pro", selection["selected_model"])
        self.assertNotIn("ollama:nomic-embed-text:latest", selection["fallbacks"])

    def test_install_planner_creates_approved_local_plan_when_needed(self) -> None:
        plan = build_install_plan(
            inventory=[],
            task_request=classify_task_request("please debug this code"),
            selection_result={"selected_model": None, "provider": None, "reason": "no_suitable_model", "fallbacks": [], "trace_id": "t1"},
        )
        self.assertTrue(bool(plan["needed"]))
        self.assertTrue(bool(plan["approved"]))
        self.assertEqual("ollama pull qwen2.5-coder:7b", plan["install_command"])
        self.assertEqual("ollama:qwen2.5-coder:7b", plan["candidates"][0]["model_id"])
        first = plan["plan"][0]
        self.assertEqual("ollama.pull_model", first["action"])
        self.assertEqual("qwen2.5-coder:7b", first["model"])

    def test_install_planner_returns_concrete_vision_profile_when_needed(self) -> None:
        plan = build_install_plan(
            inventory=[
                {
                    "id": "ollama:qwen2.5:3b-instruct",
                    "provider": "ollama",
                    "installed": True,
                    "available": True,
                    "healthy": True,
                    "capabilities": ["chat"],
                    "context_window": 8192,
                    "local": True,
                    "approved": True,
                    "reason": "healthy",
                }
            ],
            task_request=classify_task_request("analyze this image"),
            selection_result={"selected_model": None, "provider": None, "reason": "no_suitable_model", "fallbacks": [], "trace_id": "t1"},
        )
        self.assertTrue(bool(plan["needed"]))
        self.assertTrue(bool(plan["approved"]))
        self.assertEqual("ollama pull llava:7b", plan["install_command"])
        self.assertEqual("ollama:llava:7b", plan["candidates"][0]["model_id"])
        self.assertEqual("Run: python -m agent llm_install --model ollama:llava:7b --approve", plan["next_action"])

    def test_install_planner_local_only_is_not_satisfied_by_remote_only_candidate(self) -> None:
        inventory = [
            {
                "id": "openrouter:cheap-chat",
                "provider": "openrouter",
                "installed": False,
                "available": True,
                "healthy": True,
                "capabilities": ["chat", "json"],
                "context_window": 65536,
                "local": False,
                "approved": True,
                "reason": "healthy",
                "quality_rank": 6,
                "cost_rank": 2,
                "price_in": 0.1,
                "price_out": 0.2,
            }
        ]
        selection = select_model_for_task(
            inventory,
            classify_task_request("hello there"),
            allow_remote_fallback=False,
            policy_name="default",
            policy={"cost_cap_per_1m": 10.0, "allowlist": []},
            trace_id="sel-local-only",
        )
        self.assertIsNone(selection["selected_model"])
        plan = build_install_plan(
            inventory=inventory,
            task_request=classify_task_request("hello there"),
            selection_result=selection,
            allow_remote_fallback=False,
            policy={"cost_cap_per_1m": 10.0, "allowlist": []},
            policy_name="default",
        )
        self.assertTrue(bool(plan["needed"]))
        self.assertTrue(bool(plan["approved"]))

    def test_install_planner_stays_honest_when_no_approved_profile_matches_requirements(self) -> None:
        plan = build_install_plan(
            inventory=[],
            task_request={"task_type": "vision", "requirements": ["chat", "vision", "json"], "preferred_local": True},
            selection_result={
                "selected_model": None,
                "provider": None,
                "reason": "no_local_model_with_required_capabilities",
                "fallbacks": [],
                "trace_id": "t-json-vision",
            },
        )
        self.assertTrue(bool(plan["needed"]))
        self.assertFalse(bool(plan["approved"]))
        self.assertEqual("no_local_model_with_required_capabilities", plan["reason"])
        self.assertEqual([], plan["candidates"])
        self.assertIn("No approved local install plan exists for this task yet.", str(plan["next_action"]))

    def test_selector_reports_capability_gap_reason_when_no_local_model_fits(self) -> None:
        inventory = [
            {
                "id": "ollama:qwen2.5:3b-instruct",
                "provider": "ollama",
                "installed": True,
                "available": True,
                "healthy": True,
                "capabilities": ["chat"],
                "context_window": 8192,
                "local": True,
                "approved": True,
                "reason": "healthy",
            }
        ]
        selection = select_model_for_task(
            inventory,
            classify_task_request("analyze this image"),
            allow_remote_fallback=False,
            policy_name="default",
            policy={"cost_cap_per_1m": 10.0, "allowlist": []},
            trace_id="sel-vision-gap",
        )
        self.assertIsNone(selection["selected_model"])
        self.assertEqual("no_local_model_with_required_capabilities", selection["reason"])


if __name__ == "__main__":
    unittest.main()
