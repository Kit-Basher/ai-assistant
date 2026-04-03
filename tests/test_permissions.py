from __future__ import annotations

import json
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

    def test_load_returns_defaults_when_permissions_file_is_invalid_json(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{not-json")
        loaded = self.store.load()
        self.assertEqual(default_permissions_document(), loaded)

    def test_invalid_constraint_numbers_fall_back_to_default_limit(self) -> None:
        updated = self.store.update(
            {
                "actions": {"modelops.pull_ollama_model": True},
                "constraints": {
                    "max_download_bytes": "bad-value",
                    "max_download_gb": "also-bad",
                },
            }
        )
        expected_default = default_permissions_document()["constraints"]["max_download_bytes"]
        self.assertEqual(expected_default, int(updated["constraints"]["max_download_bytes"]))
        with open(self.path, "r", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertEqual(expected_default, int(on_disk["constraints"]["max_download_bytes"]))

    def test_notifications_send_action_is_supported(self) -> None:
        config = self.store.update(
            {
                "actions": {"llm.notifications.send": True},
                "mode": "auto",
            }
        )
        request = PermissionRequest(
            action="llm.notifications.send",
            params={"trigger": "scheduler"},
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)

    def test_notifications_prune_action_is_supported(self) -> None:
        config = self.store.update(
            {
                "actions": {"llm.notifications.prune": True},
                "mode": "auto",
            }
        )
        request = PermissionRequest(
            action="llm.notifications.prune",
            params={"actor": "test"},
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)

    def test_registry_rollback_action_is_supported(self) -> None:
        config = self.store.update(
            {
                "actions": {"llm.registry.rollback": True},
                "mode": "auto",
            }
        )
        request = PermissionRequest(
            action="llm.registry.rollback",
            params={"snapshot_id": "s00000001-abc"},
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)

    def test_self_heal_apply_action_is_supported(self) -> None:
        config = self.store.update(
            {
                "actions": {"llm.self_heal.apply": True},
                "mode": "auto",
            }
        )
        request = PermissionRequest(
            action="llm.self_heal.apply",
            params={
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5:3b-instruct",
            },
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)

    def test_capabilities_reconcile_apply_action_is_supported(self) -> None:
        config = self.store.update(
            {
                "actions": {"llm.capabilities.reconcile.apply": True},
                "mode": "auto",
            }
        )
        request = PermissionRequest(
            action="llm.capabilities.reconcile.apply",
            params={},
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)

    def test_autopilot_bootstrap_apply_action_is_supported(self) -> None:
        config = self.store.update(
            {
                "actions": {"llm.autopilot.bootstrap.apply": True},
                "mode": "auto",
            }
        )
        request = PermissionRequest(
            action="llm.autopilot.bootstrap.apply",
            params={
                "default_provider": "ollama",
                "default_model": "ollama:qwen2.5:3b-instruct",
            },
            estimated_bytes=0,
        )
        decision = self.policy.evaluate(request, config)
        self.assertTrue(decision.allow)
        self.assertFalse(decision.requires_confirmation)


if __name__ == "__main__":
    unittest.main()
