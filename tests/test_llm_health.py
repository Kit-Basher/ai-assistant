from __future__ import annotations

import tempfile
import unittest

from agent.llm.health import HealthProbeSettings, HealthStateStore, LLMHealthMonitor


def _registry_document() -> dict[str, object]:
    return {
        "providers": {
            "ollama": {
                "enabled": True,
                "local": True,
            }
        },
        "models": {
            "ollama:llama3": {
                "provider": "ollama",
                "model": "llama3",
                "capabilities": ["chat"],
                "enabled": True,
                "available": True,
            }
        },
        "defaults": {},
    }


class TestLLMHealthMonitor(unittest.TestCase):
    def test_probe_respects_cooldown_and_skips_until_due(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            now_holder = [1_000]
            calls = {"count": 0}

            def _probe(_provider_id: str, _model_id: str, _timeout: float) -> dict[str, object]:
                calls["count"] += 1
                if calls["count"] == 1:
                    return {"ok": False, "error_kind": "provider_unavailable", "status_code": 503}
                return {"ok": True}

            monitor = LLMHealthMonitor(
                HealthProbeSettings(
                    interval_seconds=60,
                    max_probes_per_run=3,
                    probe_timeout_seconds=2.0,
                    initial_backoff_seconds=60,
                    max_backoff_seconds=600,
                    models_per_provider=1,
                ),
                store=HealthStateStore(path=f"{tmpdir}/health_state.json"),
                probe_fn=_probe,
                now_fn=lambda: int(now_holder[0]),
            )

            first = monitor.run_once(_registry_document())
            self.assertEqual(1, len(first["probed"]))
            self.assertEqual("down", first["probed"][0]["status"])

            now_holder[0] = 1_050
            second = monitor.run_once(_registry_document())
            self.assertEqual(0, len(second["probed"]))
            self.assertEqual(1, second["skipped"])

            now_holder[0] = 1_121
            third = monitor.run_once(_registry_document())
            self.assertEqual(1, len(third["probed"]))
            self.assertEqual("ok", third["probed"][0]["status"])
            self.assertEqual(2, calls["count"])

    def test_state_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = f"{tmpdir}/health_state.json"
            now_holder = [2_000]

            monitor = LLMHealthMonitor(
                HealthProbeSettings(
                    interval_seconds=60,
                    max_probes_per_run=2,
                    probe_timeout_seconds=2.0,
                    initial_backoff_seconds=30,
                    max_backoff_seconds=300,
                    models_per_provider=1,
                ),
                store=HealthStateStore(path=state_path),
                probe_fn=lambda *_args: {"ok": False, "error_kind": "auth_error", "status_code": 401},
                now_fn=lambda: int(now_holder[0]),
            )
            first = monitor.run_once(_registry_document())
            self.assertEqual("down", first["models"][0]["status"])

            restarted = LLMHealthMonitor(
                HealthProbeSettings(),
                store=HealthStateStore(path=state_path),
                probe_fn=lambda *_args: {"ok": True},
                now_fn=lambda: int(now_holder[0]),
            )
            summary = restarted.summary(_registry_document())
            self.assertEqual("down", summary["models"][0]["status"])
            self.assertEqual("auth_error", summary["models"][0]["last_error_kind"])


if __name__ == "__main__":
    unittest.main()
