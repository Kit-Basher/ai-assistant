from __future__ import annotations

import copy
import unittest

from agent.config import Config
from agent.llm.self_heal import build_drift_report, build_self_heal_plan


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
        ollama_model="llama3",
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
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=None,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path="/tmp/usage.json",
        llm_health_state_path="/tmp/health.json",
        llm_automation_enabled=False,
        model_scout_state_path="/tmp/scout.json",
        autopilot_notify_store_path="/tmp/notify.json",
    )


def _base_registry() -> dict[str, object]:
    return {
        "providers": {
            "ollama": {
                "enabled": True,
                "local": True,
            },
            "local2": {
                "enabled": True,
                "local": True,
            },
            "openrouter": {
                "enabled": True,
                "local": False,
            },
        },
        "models": {
            "ollama:qwen2.5:3b-instruct": {
                "provider": "ollama",
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "max_context_tokens": 8192,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "local2:tiny-chat": {
                "provider": "local2",
                "model": "tiny-chat",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
                "max_context_tokens": 4096,
                "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
            },
            "openrouter:openai/gpt-4o-mini": {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "capabilities": ["chat", "json", "tools"],
                "enabled": True,
                "available": True,
                "max_context_tokens": 128000,
                "pricing": {"input_per_million_tokens": 0.15, "output_per_million_tokens": 0.6},
            },
        },
        "defaults": {
            "routing_mode": "prefer_local_lowest_cost_capable",
            "default_provider": "ollama",
            "default_model": "ollama:qwen2.5:3b-instruct",
            "allow_remote_fallback": True,
        },
    }


def _snapshot_from_registry(document: dict[str, object]) -> dict[str, object]:
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    provider_rows = []
    for provider_id, payload in sorted(providers.items()):
        if not isinstance(payload, dict):
            continue
        provider_rows.append(
            {
                "id": provider_id,
                "enabled": bool(payload.get("enabled", True)),
                "available": bool(payload.get("enabled", True)),
                "local": bool(payload.get("local", False)),
                "health": {"status": "ok"},
            }
        )
    model_rows = []
    for model_id, payload in sorted(models.items()):
        if not isinstance(payload, dict):
            continue
        model_rows.append(
            {
                "id": model_id,
                "provider": payload.get("provider"),
                "capabilities": list(payload.get("capabilities") or []),
                "enabled": bool(payload.get("enabled", True)),
                "available": bool(payload.get("available", True)),
                "routable": bool(payload.get("enabled", True) and payload.get("available", True)),
                "max_context_tokens": payload.get("max_context_tokens"),
                "input_cost_per_million_tokens": (payload.get("pricing") or {}).get("input_per_million_tokens"),
                "output_cost_per_million_tokens": (payload.get("pricing") or {}).get("output_per_million_tokens"),
                "health": {"status": "ok"},
            }
        )
    return {"providers": provider_rows, "models": model_rows}


def _health_summary_from_registry(document: dict[str, object]) -> dict[str, object]:
    providers = document.get("providers") if isinstance(document.get("providers"), dict) else {}
    models = document.get("models") if isinstance(document.get("models"), dict) else {}
    return {
        "providers": [
            {"id": provider_id, "status": "ok"}
            for provider_id, payload in sorted(providers.items())
            if isinstance(payload, dict)
        ],
        "models": [
            {"id": model_id, "status": "ok"}
            for model_id, payload in sorted(models.items())
            if isinstance(payload, dict)
        ],
    }


def _snapshot_provider(snapshot: dict[str, object], provider_id: str) -> dict[str, object]:
    for row in snapshot.get("providers") or []:
        if isinstance(row, dict) and str(row.get("id") or "").strip().lower() == provider_id:
            return row
    raise AssertionError(f"provider row not found: {provider_id}")


def _snapshot_model(snapshot: dict[str, object], model_id: str) -> dict[str, object]:
    for row in snapshot.get("models") or []:
        if isinstance(row, dict) and str(row.get("id") or "").strip() == model_id:
            return row
    raise AssertionError(f"model row not found: {model_id}")


class TestLLMSelfHeal(unittest.TestCase):
    def test_drift_report_no_drift(self) -> None:
        document = _base_registry()
        snapshot = _snapshot_from_registry(document)
        summary = _health_summary_from_registry(document)
        report = build_drift_report(document, summary, router_snapshot=snapshot)
        self.assertFalse(report["has_drift"])
        self.assertEqual([], report["reasons"])

    def test_drift_report_detects_required_cases(self) -> None:
        base = _base_registry()
        cases = [
            (
                "default_provider_missing",
                lambda doc, snap, _summary: doc["defaults"].update({"default_provider": None}),
            ),
            (
                "default_provider_disabled",
                lambda doc, snap, _summary: doc["providers"]["ollama"].update({"enabled": False}),
            ),
            (
                "default_provider_unavailable",
                lambda _doc, snap, _summary: _snapshot_provider(snap, "ollama").update({"available": False}),
            ),
            (
                "default_model_missing",
                lambda doc, snap, _summary: doc["defaults"].update({"default_model": "ollama:missing"}),
            ),
            (
                "default_model_not_in_provider_inventory",
                lambda doc, snap, _summary: doc["models"]["ollama:qwen2.5:3b-instruct"].update({"available": False}),
            ),
            (
                "default_model_not_chat_capable",
                lambda doc, snap, _summary: doc["models"]["ollama:qwen2.5:3b-instruct"].update(
                    {"capabilities": ["embedding"]}
                ),
            ),
            (
                "default_model_health_not_ok",
                lambda _doc, snap, _summary: _snapshot_model(snap, "ollama:qwen2.5:3b-instruct")["health"].update(
                    {"status": "down"}
                ),
            ),
            (
                "default_model_unroutable",
                lambda _doc, snap, _summary: _snapshot_model(snap, "ollama:qwen2.5:3b-instruct").update(
                    {"routable": False}
                ),
            ),
        ]
        for expected, mutate in cases:
            with self.subTest(expected=expected):
                document = copy.deepcopy(base)
                snapshot = _snapshot_from_registry(document)
                summary = _health_summary_from_registry(document)
                mutate(document, snapshot, summary)
                report = build_drift_report(document, summary, router_snapshot=snapshot)
                self.assertTrue(report["has_drift"])
                self.assertIn(expected, report["reasons"])

    def test_plan_prefers_stronger_local_candidate(self) -> None:
        document = _base_registry()
        document["defaults"]["default_model"] = "ollama:missing"  # type: ignore[index]
        document["models"]["ollama:alpha"] = {  # type: ignore[index]
            "provider": "ollama",
            "model": "alpha",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "max_context_tokens": 16000,
            "pricing": {"input_per_million_tokens": 0.2, "output_per_million_tokens": 0.2},
        }
        document["models"]["ollama:beta"] = {  # type: ignore[index]
            "provider": "ollama",
            "model": "beta",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "max_context_tokens": 4096,
            "pricing": {"input_per_million_tokens": 0.2, "output_per_million_tokens": 0.2},
        }
        document["models"]["local2:cheaper"] = {  # type: ignore[index]
            "provider": "local2",
            "model": "cheaper",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "max_context_tokens": 4096,
            "pricing": {"input_per_million_tokens": 0.01, "output_per_million_tokens": 0.01},
        }
        snapshot = _snapshot_from_registry(document)
        summary = _health_summary_from_registry(document)
        plan = build_self_heal_plan(document, summary, config=_config(), router_snapshot=snapshot)
        self.assertEqual("ollama:alpha", plan["proposed_defaults"]["default_model"])
        self.assertTrue(plan["impact"]["changes_count"] >= 1)

    def test_plan_prefers_larger_context_when_quality_is_equal(self) -> None:
        document = _base_registry()
        document["defaults"]["default_model"] = "ollama:missing"  # type: ignore[index]
        document["models"]["ollama:aaa"] = {  # type: ignore[index]
            "provider": "ollama",
            "model": "aaa",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "max_context_tokens": 1024,
            "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
        }
        document["models"]["ollama:zzz"] = {  # type: ignore[index]
            "provider": "ollama",
            "model": "zzz",
            "capabilities": ["chat"],
            "enabled": True,
            "available": True,
            "max_context_tokens": 32768,
            "pricing": {"input_per_million_tokens": None, "output_per_million_tokens": None},
        }
        snapshot = _snapshot_from_registry(document)
        summary = _health_summary_from_registry(document)
        plan = build_self_heal_plan(document, summary, config=_config(), router_snapshot=snapshot)
        self.assertEqual("ollama:zzz", plan["proposed_defaults"]["default_model"])

    def test_plan_is_deterministic_under_input_ordering(self) -> None:
        document = _base_registry()
        document["defaults"]["default_model"] = "ollama:missing"  # type: ignore[index]
        snapshot = _snapshot_from_registry(document)
        summary = _health_summary_from_registry(document)

        reversed_doc = {
            "providers": dict(reversed(list(document["providers"].items()))),  # type: ignore[index]
            "models": dict(reversed(list(document["models"].items()))),  # type: ignore[index]
            "defaults": dict(document["defaults"]),  # type: ignore[index]
        }
        reversed_snapshot = {
            "providers": list(reversed(snapshot["providers"])),  # type: ignore[index]
            "models": list(reversed(snapshot["models"])),  # type: ignore[index]
        }

        plan_a = build_self_heal_plan(document, summary, config=_config(), router_snapshot=snapshot)
        plan_b = build_self_heal_plan(reversed_doc, summary, config=_config(), router_snapshot=reversed_snapshot)
        self.assertEqual(plan_a["changes"], plan_b["changes"])
        self.assertEqual(plan_a["reasons"], plan_b["reasons"])
        self.assertEqual(
            (plan_a.get("selected_candidate") or {}).get("model_id"),
            (plan_b.get("selected_candidate") or {}).get("model_id"),
        )


if __name__ == "__main__":
    unittest.main()
