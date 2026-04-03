from __future__ import annotations

import tempfile
import unittest

from agent.llm.capabilities import (
    apply_capabilities_reconcile_plan,
    build_capabilities_reconcile_plan,
    capability_list_from_inference,
    infer_capabilities_from_catalog,
)
from agent.llm.health import HealthProbeSettings, HealthStateStore, LLMHealthMonitor


def _registry_document() -> dict[str, object]:
    return {
        "providers": {
            "ollama": {
                "enabled": True,
                "local": True,
            },
            "openrouter": {
                "enabled": True,
                "local": False,
            },
        },
        "models": {
            "ollama:nomic-embed-text:latest": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest",
                "capabilities": ["chat"],
                "default_for": ["chat"],
                "enabled": True,
                "available": True,
            },
            "ollama:qwen2.5:3b-instruct": {
                "provider": "ollama",
                "model": "qwen2.5:3b-instruct",
                "capabilities": ["embedding"],
                "default_for": ["embedding"],
                "enabled": True,
                "available": True,
            },
        },
        "defaults": {},
    }


def _catalog_snapshot() -> dict[str, object]:
    return {
        "providers": {
            "ollama": {
                "provider_id": "ollama",
                "models": [
                    {
                        "id": "ollama:nomic-embed-text:latest",
                        "provider_id": "ollama",
                        "model": "nomic-embed-text:latest",
                        "capabilities": ["chat"],
                    },
                    {
                        "id": "ollama:qwen2.5:3b-instruct",
                        "provider_id": "ollama",
                        "model": "qwen2.5:3b-instruct",
                        "capabilities": ["embedding"],
                    },
                ],
            }
        }
    }


class TestLLMCapabilities(unittest.TestCase):
    def test_ollama_inference_embedding_and_chat(self) -> None:
        embed = capability_list_from_inference(
            infer_capabilities_from_catalog(
                "ollama",
                {
                    "model": "nomic-embed-text:latest",
                    "capabilities": ["chat"],
                },
            )
        )
        chat = capability_list_from_inference(
            infer_capabilities_from_catalog(
                "ollama",
                {
                    "model": "qwen2.5:3b-instruct",
                    "capabilities": ["embedding"],
                },
            )
        )
        self.assertEqual(["embedding"], embed)
        self.assertEqual(["chat"], chat)

    def test_reconcile_plan_updates_mismatches_and_does_not_touch_providers(self) -> None:
        registry = _registry_document()
        plan = build_capabilities_reconcile_plan(registry, _catalog_snapshot())
        changes = plan.get("changes") if isinstance(plan.get("changes"), list) else []
        changed_ids = {str(row.get("id") or "") for row in changes if isinstance(row, dict)}
        self.assertIn("ollama:nomic-embed-text:latest", changed_ids)
        self.assertIn("ollama:qwen2.5:3b-instruct", changed_ids)

        updated = apply_capabilities_reconcile_plan(registry, plan)
        providers_before = registry.get("providers")
        providers_after = updated.get("providers")
        self.assertEqual(providers_before, providers_after)
        models_after = updated.get("models") if isinstance(updated.get("models"), dict) else {}
        self.assertEqual(["embedding"], models_after["ollama:nomic-embed-text:latest"]["capabilities"])
        self.assertEqual(["chat"], models_after["ollama:qwen2.5:3b-instruct"]["capabilities"])

    def test_health_not_applicable_probe_does_not_downgrade_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = LLMHealthMonitor(
                HealthProbeSettings(
                    interval_seconds=60,
                    max_probes_per_run=2,
                    probe_timeout_seconds=2.0,
                    initial_backoff_seconds=30,
                    max_backoff_seconds=120,
                    models_per_provider=1,
                ),
                store=HealthStateStore(path=f"{tmpdir}/health_state.json"),
                probe_fn=lambda *_args: {"ok": True, "error_kind": "not_applicable"},
                now_fn=lambda: 1_700_000_000,
            )
            summary = monitor.run_once(
                {
                    "providers": {"ollama": {"enabled": True, "local": True}},
                    "models": {
                        "ollama:misclassified-embed": {
                            "provider": "ollama",
                            "model": "misclassified-embed",
                            "capabilities": ["chat"],
                            "enabled": True,
                            "available": True,
                        }
                    },
                }
            )
            provider_rows = summary.get("providers") if isinstance(summary.get("providers"), list) else []
            self.assertEqual("unknown", provider_rows[0]["status"])
            self.assertEqual(1, int(summary.get("not_applicable") or 0))


if __name__ == "__main__":
    unittest.main()
