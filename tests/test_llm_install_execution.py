from __future__ import annotations

import subprocess
import unittest

from agent.config import Config
from agent.llm.install_executor import execute_install_plan
from agent.llm.registry import DefaultsConfig, ProviderConfig, Registry


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
        default_policy={},
        premium_policy={},
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
        },
        models={},
        defaults=DefaultsConfig(
            routing_mode="prefer_local_lowest_cost_capable",
            default_provider="ollama",
            default_model=None,
            allow_remote_fallback=False,
            chat_model=None,
            embed_model=None,
            last_chat_model=None,
        ),
        fallback_chain=(),
    )


def _approved_plan() -> dict[str, object]:
    return {
        "needed": True,
        "approved": True,
        "approval_required": True,
        "reason": "no_local_model_with_required_capabilities",
        "install_command": "ollama pull llava:7b",
        "next_action": "Run: python -m agent llm_install --model ollama:llava:7b --approve",
        "candidates": [{"model_id": "ollama:llava:7b", "install_name": "llava:7b"}],
        "plan": [{"id": "01_pull_model", "action": "ollama.pull_model", "provider": "ollama", "model": "llava:7b"}],
    }


class _InventoryBuilder:
    def __init__(self, rows_by_call: list[list[dict[str, object]]]) -> None:
        self._rows_by_call = list(rows_by_call)
        self.calls = 0

    def __call__(self, **_kwargs: object) -> list[dict[str, object]]:
        index = min(self.calls, len(self._rows_by_call) - 1)
        self.calls += 1
        return list(self._rows_by_call[index])


class TestLLMInstallExecution(unittest.TestCase):
    def test_approved_local_plan_executes_with_approve(self) -> None:
        builder = _InventoryBuilder(
            [
                [],
                [
                    {
                        "id": "ollama:llava:7b",
                        "provider": "ollama",
                        "installed": True,
                        "available": True,
                        "healthy": True,
                        "reason": "healthy",
                        "capabilities": ["chat", "vision"],
                    }
                ],
            ]
        )
        commands: list[list[str]] = []

        def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="pull ok", stderr="")

        result = execute_install_plan(
            config=_config(),
            registry=_registry(),
            plan=_approved_plan(),
            approve=True,
            trace_id="install-1",
            run_fn=_run,
            inventory_builder=builder,
        )
        self.assertTrue(bool(result["ok"]))
        self.assertTrue(bool(result["executed"]))
        self.assertEqual(["ollama", "pull", "llava:7b"], commands[0])
        self.assertTrue(bool(result["verification"]["healthy"]))

    def test_same_plan_without_approve_does_not_execute(self) -> None:
        builder = _InventoryBuilder([[]])
        run_called = False

        def _run(_cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal run_called
            run_called = True
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        result = execute_install_plan(
            config=_config(),
            registry=_registry(),
            plan=_approved_plan(),
            approve=False,
            trace_id="install-2",
            run_fn=_run,
            inventory_builder=builder,
        )
        self.assertFalse(bool(result["ok"]))
        self.assertFalse(bool(result["executed"]))
        self.assertEqual("approval_required", result["error_kind"])
        self.assertFalse(run_called)

    def test_unapproved_or_non_local_plan_is_rejected(self) -> None:
        result = execute_install_plan(
            config=_config(),
            registry=_registry(),
            plan={
                "needed": True,
                "approved": True,
                "candidates": [{"model_id": "openrouter:bad", "install_name": "bad"}],
                "plan": [{"action": "ollama.pull_model", "provider": "openrouter", "model": "bad"}],
            },
            approve=True,
            trace_id="install-3",
            inventory_builder=_InventoryBuilder([[]]),
        )
        self.assertFalse(bool(result["ok"]))
        self.assertEqual("model_not_approved", result["error_kind"])

    def test_already_installed_model_returns_noop(self) -> None:
        builder = _InventoryBuilder(
            [
                [
                    {
                        "id": "ollama:llava:7b",
                        "provider": "ollama",
                        "installed": True,
                        "available": True,
                        "healthy": True,
                        "reason": "healthy",
                        "capabilities": ["chat", "vision"],
                    }
                ]
            ]
        )
        result = execute_install_plan(
            config=_config(),
            registry=_registry(),
            plan=_approved_plan(),
            approve=True,
            trace_id="install-4",
            inventory_builder=builder,
        )
        self.assertTrue(bool(result["ok"]))
        self.assertFalse(bool(result["executed"]))
        self.assertIn("already installed", str(result["message"]).lower())

    def test_post_install_verification_is_included_even_if_degraded(self) -> None:
        builder = _InventoryBuilder(
            [
                [],
                [
                    {
                        "id": "ollama:llava:7b",
                        "provider": "ollama",
                        "installed": True,
                        "available": True,
                        "healthy": False,
                        "reason": "provider_down",
                        "capabilities": ["chat", "vision"],
                    }
                ],
            ]
        )

        def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, 0, stdout="pull ok", stderr="")

        result = execute_install_plan(
            config=_config(),
            registry=_registry(),
            plan=_approved_plan(),
            approve=True,
            trace_id="install-5",
            run_fn=_run,
            inventory_builder=builder,
        )
        self.assertTrue(bool(result["ok"]))
        self.assertTrue(bool(result["executed"]))
        self.assertFalse(bool(result["verification"]["healthy"]))
        self.assertIn("degraded", str(result["message"]).lower())


if __name__ == "__main__":
    unittest.main()
