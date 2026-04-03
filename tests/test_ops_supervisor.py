import os
import tempfile
import unittest
from unittest import mock

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB
from ops import supervisor as ops_supervisor
from skills.ops_supervisor import handler as ops_handler


class TestOpsSupervisor(unittest.TestCase):
    def test_hmac_sign_verify(self) -> None:
        key = "test-key"
        payload = {
            "op": "status",
            "ts": "2026-02-05T00:00:00+00:00",
            "nonce": "abc123",
            "payload": {},
        }
        sig = ops_supervisor.sign_request(payload, key)
        self.assertTrue(ops_supervisor.verify_signature(payload, key, sig))

    def test_replay_rejected(self) -> None:
        key = "test-key"
        supervisor = ops_supervisor.Supervisor(
            socket_path="/tmp/test.sock",
            hmac_key=key,
            agent_unit_name="personal-agent.service",
        )
        request = {
            "op": "status",
            "ts": ops_supervisor._utc_now().isoformat(),
            "nonce": "nonce-1",
            "payload": {},
        }
        request["sig"] = ops_supervisor.sign_request(
            {"op": request["op"], "ts": request["ts"], "nonce": request["nonce"], "payload": request["payload"]},
            key,
        )
        with mock.patch.object(ops_supervisor.Supervisor, "_handle_status", return_value={"ok": True}):
            first = supervisor.handle_request(request)
            second = supervisor.handle_request(request)
        self.assertTrue(first.get("ok"))
        self.assertEqual(second.get("error"), "replayed_nonce")

    def test_unknown_operation_rejected(self) -> None:
        key = "test-key"
        supervisor = ops_supervisor.Supervisor(
            socket_path="/tmp/test.sock",
            hmac_key=key,
            agent_unit_name="personal-agent.service",
        )
        request = {
            "op": "drop",
            "ts": ops_supervisor._utc_now().isoformat(),
            "nonce": "nonce-2",
            "payload": {},
        }
        request["sig"] = ops_supervisor.sign_request(
            {"op": request["op"], "ts": request["ts"], "nonce": request["nonce"], "payload": request["payload"]},
            key,
        )
        response = supervisor.handle_request(request)
        self.assertFalse(response.get("ok"))
        self.assertEqual(response.get("error"), "unknown_operation")


class TestOpsSkills(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = MemoryDB(db_path)
        schema_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")
        )
        self.db.init_schema(schema_path)
        self.log_path = os.path.join(self.tmpdir.name, "events.log")
        self.skills_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills"))

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def test_restart_requires_confirmation(self) -> None:
        orchestrator = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=None,
        )
        called = {"count": 0}

        def fake_restart(context: dict) -> dict:
            called["count"] += 1
            return {"text": "ok"}

        orchestrator.skills["ops_supervisor"].functions["restart_agent"].handler = fake_restart
        response = orchestrator.handle_message("/restart", "user1")
        self.assertIn("/confirm", response.text)
        self.assertEqual(called["count"], 0)
        orchestrator.handle_message("/confirm", "user1")
        self.assertEqual(called["count"], 1)

    def test_logs_bounded(self) -> None:
        with mock.patch.dict(os.environ, {"SUPERVISOR_LOG_LINES_MAX": "100"}):
            with mock.patch("agent.ops.supervisor_client.send_request") as send_request:
                send_request.return_value = {"ok": True, "result": {"lines": ""}}
                ops_handler.service_logs({}, lines=500)
                send_request.assert_called_once()
                args, _ = send_request.call_args
                self.assertEqual(args[1]["lines"], 100)


if __name__ == "__main__":
    unittest.main()
