from __future__ import annotations

import unittest

from agent.skills.system_health_analyzer import analyze_system_health


def _observed_payload() -> dict[str, object]:
    return {
        "cpu": {"usage_pct": 12.0, "cpu_count": 8, "load_average": {"1m": 0.5, "5m": 0.4, "15m": 0.3}},
        "memory": {
            "total_bytes": 16 * 1024**3,
            "used_bytes": 6 * 1024**3,
            "available_bytes": 10 * 1024**3,
            "used_pct": 37.5,
        },
        "disk": [{"mountpoint": "/", "used_pct": 44.0, "high_usage": False}],
        "gpu": {"available": False, "expected": False, "driver_version": None, "gpus": [], "error_kind": "not_installed"},
        "services": {
            "ollama": {"service_state": "active", "reachable": True},
            "personal_agent": {"service_state": "active", "reachable": True},
        },
        "network": {"state": "up", "up_interfaces": ["eth0"], "default_route": True, "dns_configured": True},
        "warnings": [],
    }


class TestSystemHealthAnalyzer(unittest.TestCase):
    def test_healthy_system_is_ok(self) -> None:
        result = analyze_system_health(_observed_payload())
        self.assertEqual("ok", result["status"])
        self.assertEqual([], result["warnings"])
        self.assertEqual([], result["suggestions"])

    def test_disk_warn(self) -> None:
        observed = _observed_payload()
        observed["disk"] = [{"mountpoint": "/", "used_pct": 86.5, "high_usage": True}]
        result = analyze_system_health(observed)
        self.assertEqual("warn", result["status"])
        self.assertEqual("disk", result["warnings"][0]["component"])
        self.assertIn("high on /", result["warnings"][0]["message"])
        self.assertEqual("inspect_disk_usage", result["suggestions"][0]["id"])

    def test_disk_critical(self) -> None:
        observed = _observed_payload()
        observed["disk"] = [{"mountpoint": "/", "used_pct": 96.0, "high_usage": True}]
        result = analyze_system_health(observed)
        self.assertEqual("critical", result["status"])
        self.assertEqual("critical", result["warnings"][0]["severity"])

    def test_memory_warn(self) -> None:
        observed = _observed_payload()
        observed["memory"] = {
            "total_bytes": 100,
            "used_bytes": 90,
            "available_bytes": 10,
            "used_pct": 90.0,
        }
        result = analyze_system_health(observed)
        self.assertEqual("warn", result["status"])
        self.assertEqual("memory", result["warnings"][0]["component"])
        self.assertEqual("inspect_memory_processes", result["suggestions"][0]["id"])

    def test_service_warn(self) -> None:
        observed = _observed_payload()
        observed["services"] = {
            "ollama": {"service_state": "inactive", "reachable": False},
            "personal_agent": {"service_state": "active", "reachable": True},
        }
        result = analyze_system_health(observed)
        self.assertEqual("warn", result["status"])
        self.assertEqual("services", result["warnings"][0]["component"])
        self.assertEqual("check_ollama_service", result["suggestions"][0]["id"])

    def test_suggestions_are_deterministic(self) -> None:
        observed = _observed_payload()
        observed["disk"] = [
            {"mountpoint": "/", "used_pct": 92.0, "high_usage": True},
            {"mountpoint": "/data", "used_pct": 97.0, "high_usage": True},
        ]
        result = analyze_system_health(observed)
        suggestion_ids = [item["id"] for item in result["suggestions"]]
        self.assertEqual(["inspect_disk_usage"], suggestion_ids)


if __name__ == "__main__":
    unittest.main()
