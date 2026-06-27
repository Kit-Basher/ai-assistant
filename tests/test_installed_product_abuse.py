from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.api_server import AgentRuntime
from scripts.installed_product_abuse import (
    contains_false_podman_missing,
    contains_secret,
    contains_vague_handoff,
    referenced_routes,
)
from tests.test_api_packs_endpoints import _HandlerForTest, _config


class TestInstalledProductAbuseHelpers(unittest.TestCase):
    def test_detects_false_podman_missing_text(self) -> None:
        self.assertTrue(contains_false_podman_missing("This machine is missing Podman."))
        self.assertTrue(contains_false_podman_missing("Podman prerequisite setup did not finish."))
        self.assertFalse(contains_false_podman_missing("Rootless Podman is available at /usr/bin/podman."))

    def test_detects_vague_handoff_without_command(self) -> None:
        self.assertTrue(contains_vague_handoff("Run the handoff command, then retry setup."))
        self.assertFalse(contains_vague_handoff('Run: sudo apt-get install -y podman, then retry setup.'))

    def test_detects_obvious_secrets(self) -> None:
        self.assertTrue(contains_secret("Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"))
        self.assertTrue(contains_secret("1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"))
        self.assertFalse(contains_secret("podman version 5.4.2"))

    def test_route_reference_scanner_finds_search_setup_plan_docs(self) -> None:
        refs = referenced_routes()
        files = [path for path, routes in refs.items() if "/search/setup/plan" in routes]
        self.assertTrue(files)


class TestInstalledProductEndpointClarity(unittest.TestCase):
    def test_get_search_setup_plan_returns_method_not_allowed_not_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = AgentRuntime(
                _config(
                    str(root / "registry.json"),
                    str(root / "memory.db"),
                    str(root / "skills"),
                )
            )
            handler = _HandlerForTest(runtime, "/search/setup/plan")

            handler.do_GET()

            payload = json.loads(handler.body.decode("utf-8"))
            self.assertEqual(405, handler.status_code)
            self.assertEqual("method_not_allowed", payload.get("error"))
            self.assertEqual(["POST"], payload.get("allowed_methods"))


if __name__ == "__main__":
    unittest.main()
