from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.orchestrator import Orchestrator
from agent.skills.system_health import collect_system_health
from agent.skills.system_health_analyzer import analyze_system_health
from agent.skills.system_health_summary import render_system_health_summary
from agent.tool_executor import ToolExecutor
from memory.db import MemoryDB


class TestSystemHealthSkill(unittest.TestCase):
    def test_collect_system_health_returns_expected_top_level_keys(self) -> None:
        with patch("agent.skills.system_health._collect_cpu", return_value={"usage_pct": 12.5, "load_average": {"1m": 0.2, "5m": 0.1, "15m": 0.1}, "cpu_count": 8}), patch(
            "agent.skills.system_health._collect_memory",
            return_value={"total_bytes": 16, "used_bytes": 8, "available_bytes": 8, "used_pct": 50.0},
        ), patch(
            "agent.skills.system_health._collect_disk",
            return_value=[{"mountpoint": "/", "used_pct": 42.0, "high_usage": False}],
        ), patch(
            "agent.skills.system_health._collect_gpu",
            return_value={"available": False, "driver_version": None, "gpus": [], "error_kind": "not_installed"},
        ), patch(
            "agent.skills.system_health._collect_services",
            return_value={
                "ollama": {"reachable": True, "service_state": "active"},
                "personal_agent": {"reachable": True, "service_state": "active"},
            },
        ), patch(
            "agent.skills.system_health._collect_network",
            return_value={"state": "up", "up_interfaces": ["eth0"], "default_route": True, "dns_configured": True},
        ):
            data = collect_system_health(sample_seconds=0.0)
        self.assertEqual({"cpu", "memory", "disk", "gpu", "services", "network", "warnings", "collected_at"}, set(data.keys()))
        self.assertIsInstance(data["warnings"], list)
        self.assertEqual([], data["warnings"])

    def test_collect_system_health_does_not_crash_without_gpu(self) -> None:
        with patch("agent.skills.system_health.shutil.which", return_value=None):
            data = collect_system_health(sample_seconds=0.0)
        self.assertFalse(bool(data["gpu"]["available"]))
        self.assertEqual("not_installed", data["gpu"]["error_kind"])

    def test_summary_renderer_formats_expected_sections(self) -> None:
        observed = {
            "cpu": {"load_average": {"1m": 0.2, "5m": 0.1, "15m": 0.05}, "usage_pct": 9.3},
            "memory": {"total_bytes": 8 * 1024**3, "used_bytes": 3 * 1024**3, "available_bytes": 5 * 1024**3, "used_pct": 37.5},
            "disk": [{"mountpoint": "/", "used_pct": 44.2, "high_usage": False}],
            "gpu": {"available": False, "expected": False, "gpus": [], "driver_version": None},
            "services": {
                "ollama": {"service_state": "active", "reachable": True},
                "personal_agent": {"service_state": "active", "reachable": True},
            },
            "network": {"state": "up", "up_interfaces": ["eth0"], "default_route": True, "dns_configured": True},
            "warnings": [],
        }
        analysis = analyze_system_health(observed)
        text = render_system_health_summary(observed, analysis)
        self.assertIn("System health", text)
        self.assertIn("CPU:", text)
        self.assertIn("Memory:", text)
        self.assertIn("Disk:", text)
        self.assertIn("Overall: OK", text)
        self.assertIn("Services:", text)
        self.assertIn("Network:", text)

    def test_tool_executor_integrates_observe_system_health(self) -> None:
        executor = ToolExecutor(
            handlers={
                "observe_system_health": lambda _req, _user: {
                    "ok": True,
                    "user_text": "System health\nCPU: ok\nOverall: OK",
                    "data": {"system_health": {"observed": {"warnings": []}, "analysis": {"status": "ok"}}},
                }
            },
            component="test.system_health",
        )
        result = executor.execute(
            request={"tool": "observe_system_health", "args": {}, "reason": "test"},
            user_id="u1",
            surface="cli",
            runtime_mode="DEGRADED",
            enable_writes=False,
            safe_mode=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual("observe_system_health", result["tool"])
        self.assertIn("System health", str(result["user_text"]))
        self.assertIn("system_health", result["data"])
        self.assertEqual("ok", result["data"]["system_health"]["analysis"]["status"])

    def test_orchestrator_routes_pc_health_text_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = MemoryDB(os.path.join(tmpdir, "test.db"))
            schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql"))
            db.init_schema(schema_path)
            orchestrator = Orchestrator(
                db=db,
                skills_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                log_path=os.path.join(tmpdir, "events.log"),
                timezone="UTC",
                llm_client=None,
            )
            try:
                with patch("agent.orchestrator.collect_system_health", return_value={"warnings": []}), patch(
                    "agent.orchestrator.build_system_health_report",
                    return_value={"observed": {"warnings": []}, "analysis": {"status": "ok", "warnings": [], "suggestions": []}},
                ), patch(
                    "agent.orchestrator.render_system_health_summary",
                    return_value="System health\nCPU: ok\nOverall: OK",
                ):
                    response = orchestrator.handle_message("how is my pc", "user-1")
                self.assertEqual("System health\nCPU: ok\nOverall: OK", response.text)
                tool_data = response.data.get("tool_result", {}).get("data", {})
                self.assertEqual("ok", tool_data.get("system_health", {}).get("analysis", {}).get("status"))
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
