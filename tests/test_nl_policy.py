import os
import tempfile
import unittest

from agent.nl_policy import can_run_nl_skill
from agent.orchestrator import Orchestrator
from agent.logging_utils import redact_payload
from memory.db import MemoryDB


class TestNLPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))
        self.orch = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=os.path.join(self.tmpdir.name, "events.log"),
            timezone="UTC",
            llm_client=None,
        )

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_allows_read_only_report(self) -> None:
        allowed, reason = can_run_nl_skill(
            self.orch.skills,
            "storage_governor",
            "storage_report",
            requested_permissions=["db:read"],
        )
        self.assertTrue(allowed)
        self.assertEqual("allowed", reason)

    def test_rejects_non_read_only_function(self) -> None:
        allowed, reason = can_run_nl_skill(
            self.orch.skills,
            "storage_governor",
            "storage_snapshot",
            requested_permissions=["db:read"],
        )
        self.assertFalse(allowed)
        self.assertEqual("function_not_read_only", reason)

    def test_rejects_disallowed_permission(self) -> None:
        allowed, reason = can_run_nl_skill(
            self.orch.skills,
            "resource_governor",
            "resource_report",
            requested_permissions=["db:write"],
        )
        self.assertFalse(allowed)
        self.assertEqual("permission_not_allowed", reason)

    def test_allows_new_read_only_skills(self) -> None:
        for skill_name, function_name, perms in (
            ("network_governor", "network_report", ["db:read"]),
            ("service_health_report", "service_health_report", ["db:read"]),
            ("disk_pressure_report", "disk_pressure_report", ["db:read", "sys:read"]),
        ):
            allowed, reason = can_run_nl_skill(
                self.orch.skills,
                skill_name,
                function_name,
                requested_permissions=perms,
            )
            self.assertTrue(allowed, msg=(skill_name, reason))

    def test_rejects_new_skill_if_read_only_flag_removed(self) -> None:
        self.orch.skills["service_health_report"].functions["service_health_report"].read_only = False
        allowed, reason = can_run_nl_skill(
            self.orch.skills,
            "service_health_report",
            "service_health_report",
            requested_permissions=["db:read"],
        )
        self.assertFalse(allowed)
        self.assertEqual("function_not_read_only", reason)

    def test_redaction_masks_secret_keys(self) -> None:
        payload = redact_payload({"api_key": "secret", "nested": {"telegram_bot_token": "x"}})
        self.assertEqual("***redacted***", payload["api_key"])
        self.assertEqual("***redacted***", payload["nested"]["telegram_bot_token"])


if __name__ == "__main__":
    unittest.main()
