from __future__ import annotations

import json
import os
import tempfile
import unittest

from agent.audit_log import AuditLog, redact


class TestAuditLog(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.audit = AuditLog(self.path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_redaction_masks_secret_fields(self) -> None:
        payload = {
            "api_key": "sk-secret-value",
            "nested": {
                "bot_token": "123456:ABCDEF1234567890xyzXYZ",
                "ok": "plain",
            },
        }
        redacted = redact(payload)
        self.assertEqual("***redacted***", redacted["api_key"])
        self.assertEqual("***redacted***", redacted["nested"]["bot_token"])
        self.assertEqual("plain", redacted["nested"]["ok"])

    def test_append_and_recent(self) -> None:
        self.audit.append(
            actor="system",
            action="modelops.pull_ollama_model",
            params={"model": "llama3", "api_key": "sk-super-secret"},
            decision="deny",
            reason="action_not_permitted",
            dry_run=True,
            outcome="blocked",
            error_kind=None,
            duration_ms=5,
        )
        self.audit.append(
            actor="user",
            action="modelops.set_default_model",
            params={"default_provider": "ollama", "default_model": "ollama:llama3"},
            decision="allow",
            reason="allowed",
            dry_run=False,
            outcome="success",
            error_kind=None,
            duration_ms=15,
        )

        entries = self.audit.recent(limit=5)
        self.assertEqual(2, len(entries))
        self.assertEqual("modelops.set_default_model", entries[0]["action"])
        self.assertEqual("modelops.pull_ollama_model", entries[1]["action"])
        self.assertEqual("***redacted***", entries[1]["params_redacted"]["api_key"])

        with open(self.path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(2, len(lines))
        first = json.loads(lines[0])
        self.assertIn("params_redacted", first)


if __name__ == "__main__":
    unittest.main()
