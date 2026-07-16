from __future__ import annotations

import json
from pathlib import Path
import unittest

from agent.setup_chat_flow import classify_runtime_chat_route
from agent.skills.system_health_summary import render_system_health_summary


ROOT = Path(__file__).resolve().parents[1]


def _fixture_observed() -> dict[str, object]:
    gib = 1024**3
    mib = 1024**2
    return {
        "cpu": {"load_average": {"1m": 0.4, "5m": 0.3, "15m": 0.2}, "usage_pct": 8.0, "cpu_count": 8},
        "memory": {
            "total_bytes": 64 * gib,
            "used_bytes": 12 * gib,
            "available_bytes": 52 * gib,
            "used_pct": 18.75,
        },
        "processes": {
            "available": True,
            "groups": [
                {
                    "display_name": "Ollama",
                    "description": "Local AI model service.",
                    "memory_rss_bytes": 5 * gib + 200 * mib,
                    "process_count_group": 1,
                    "cpu_percent": 2.0,
                },
                {
                    "display_name": "Firefox",
                    "description": "Browser tabs and extensions.",
                    "memory_rss_bytes": 2 * gib + 100 * mib,
                    "process_count_group": 13,
                    "cpu_percent": 3.0,
                },
                {
                    "display_name": "GNOME Shell",
                    "description": "The desktop environment.",
                    "memory_rss_bytes": 620 * mib,
                    "process_count_group": 1,
                    "cpu_percent": 1.0,
                },
            ],
            "redaction": {"command_lines_included": False, "environment_included": False},
        },
        "disk": [{"mountpoint": "/", "used_pct": 57.0, "high_usage": False}, {"mountpoint": "/data2", "used_pct": 89.7, "high_usage": True}],
        "gpu": {
            "available": True,
            "driver_version": "fixture",
            "gpus": [
                {
                    "name": "NVIDIA GeForce RTX 2060",
                    "utilization_gpu_pct": 20.0,
                    "memory_used_mb": 5161,
                    "memory_total_mb": 6144,
                    "temperature_c": 48,
                }
            ],
        },
        "services": {},
        "network": {},
        "warnings": [],
    }


class TestLocalSystemIntentAndHealth(unittest.TestCase):
    def test_intent_corpus_routes_local_pc_requests_before_web(self) -> None:
        cases = json.loads((ROOT / "tests/fixtures/local_system_intent_cases.json").read_text(encoding="utf-8"))
        for case in cases:
            with self.subTest(case=case["utterance"]):
                result = classify_runtime_chat_route(str(case["utterance"]))
                self.assertEqual(case["expected_route"], result.get("route"))
                if case["expected_capability"] == "web_search":
                    self.assertEqual("safe_web_search", result.get("kind"))
                if case["expected_capability"] == "local_system.inspect":
                    self.assertEqual("operational_observe", result.get("kind"))

    def test_reported_phrase_routes_to_local_observe(self) -> None:
        result = classify_runtime_chat_route("can you take a look at my pc and tell me what is using the most memory?")
        self.assertEqual("operational_status", result.get("route"))
        self.assertEqual("operational_observe", result.get("kind"))
        self.assertNotEqual("safe_web_search", result.get("kind"))

    def test_memory_response_answers_question_first_and_distinguishes_vram(self) -> None:
        text = render_system_health_summary(
            _fixture_observed(),
            {"status": "warn", "warnings": [], "suggestions": []},
            question="what program is using the most memory on my pc?",
        )
        first_line = text.splitlines()[0]
        self.assertIn("RAM", first_line)
        self.assertIn("memory pressure is low", first_line)
        self.assertIn("1. Ollama", text)
        self.assertIn("Local AI model service", text)
        self.assertIn("Firefox", text)
        self.assertIn("across 13 processes", text)
        self.assertIn("GPU VRAM", text)
        self.assertIn("Separate note: disk usage is high", text)
        self.assertNotIn("System health\nCPU:", text)
        self.assertNotIn("search", text.lower())

    def test_memory_response_does_not_include_command_lines_or_environment(self) -> None:
        text = render_system_health_summary(
            _fixture_observed(),
            {"status": "ok", "warnings": [], "suggestions": []},
            question="what is using my RAM?",
        )
        self.assertNotIn("--token", text)
        self.assertNotIn("Environment=", text)
        self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
