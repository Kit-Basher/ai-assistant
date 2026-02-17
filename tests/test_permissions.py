from __future__ import annotations

import os
import tempfile
import unittest

from agent.permissions import PermissionPolicy, PermissionRequest, PermissionStore, default_permissions_document


class TestPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "permissions.json")
        self.store = PermissionStore(self.path)
        self.policy = PermissionPolicy()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_default_is_deny(self) -> None:
        config = self.store.load()
        request = PermissionRequest(
            action="modelops.pull_ollama_model",
            params={"model": "llama3"},
            estimated_bytes=100,
        )
        decision = self.policy.evaluate(request, config)
        self.assertFalse(decision.allow)
        self.assertEqual("action_not_permitted", decision.reason)

    def test_allow_with_constraints(self) -> None:
        config = self.store.update(
            {
                "actions": {
                    "modelops.pull_ollama_model": True,
                },
                "constraints": {
                    "max_download_bytes": 1024,
                    "allowed_providers": ["ollama"],
                },
            }
        )
        request = PermissionRequest(
            action="modelops.pull_ollama_model",
            params={"model": "llama3"},
            estimated_bytes=512,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertTrue(decision.requires_confirmation)

    def test_download_limit_enforced(self) -> None:
        config = self.store.update(
            {
                "actions": {
                    "modelops.pull_ollama_model": True,
                },
                "constraints": {
                    "max_download_bytes": 100,
                },
            }
        )
        request = PermissionRequest(
            action="modelops.pull_ollama_model",
            params={"model": "llama3"},
            estimated_bytes=500,
        )
        decision = self.policy.evaluate(request, config)
        self.assertFalse(decision.allow)
        self.assertEqual("download_limit_exceeded", decision.reason)

    def test_provider_allowlist_enforced(self) -> None:
        config = self.store.update(
            {
                "actions": {
                    "modelops.set_default_model": True,
                },
                "constraints": {
                    "allowed_providers": ["ollama"],
                },
            }
        )
        request = PermissionRequest(
            action="modelops.set_default_model",
            params={"default_provider": "openrouter", "default_model": "openrouter:openai/gpt-4o-mini"},
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertFalse(decision.allow)
        self.assertEqual("provider_not_allowed", decision.reason)

    def test_auto_mode_does_not_require_confirmation(self) -> None:
        config = default_permissions_document()
        config["mode"] = "auto"
        config["actions"]["modelops.pull_ollama_model"] = True
        config["constraints"]["allowed_providers"] = ["ollama"]

        request = PermissionRequest(
            action="modelops.pull_ollama_model",
            params={"model": "llama3"},
            estimated_bytes=100,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
