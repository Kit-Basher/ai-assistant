import os
import tempfile
import unittest
from unittest.mock import patch

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


class _NoNarrationLLM:
    def chat(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("native reports must not call LLM presentation/narration")


class _AssistantChatLLM:
    def enabled(self) -> bool:
        return True

    def chat(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("test patches route_inference before LLM calls")


class _RuntimeTruth:
    def runtime_status(self, kind: str) -> dict[str, object]:
        return {
            "summary": f"Runtime is ready with grounded {kind} status.",
            "runtime_mode": "READY",
        }


class TestNativeReportCommands(unittest.TestCase):
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
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=_NoNarrationLLM(),
            enable_writes=False,
        )

    def _assistant_orchestrator(self) -> Orchestrator:
        return Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=_AssistantChatLLM(),
            enable_writes=False,
            runtime_truth_service=_RuntimeTruth(),  # type: ignore[arg-type]
        )

    def test_direct_report_commands_are_native_without_llm(self) -> None:
        orchestrator = self._orchestrator()

        storage = orchestrator.handle_message("/storage_report", "user1")
        resource = orchestrator.handle_message("/resource_report", "user1")
        network = orchestrator.handle_message("/network_report", "user1")

        self.assertTrue(storage.text.strip())
        self.assertTrue(resource.text.strip())
        self.assertTrue(network.text.strip())
        self.assertFalse(storage.text.startswith("Narration"))
        self.assertFalse(resource.text.startswith("Narration"))
        self.assertFalse(network.text.startswith("Narration"))
        self.assertIn(
            True,
            [
                "No storage snapshots found yet" in storage.text,
                "Storage" in storage.text,
                "storage" in storage.text,
            ],
        )
        self.assertIn(
            True,
            [
                "Live memory probe taken" in resource.text,
                "No resource snapshots found yet" in resource.text,
                "couldn't get a live memory probe" in resource.text.lower(),
            ],
        )
        self.assertIn("network", network.text.lower())

    def test_legacy_narration_env_flags_do_not_change_native_reports(self) -> None:
        os.environ["ENABLE_NARRATION"] = "1"
        os.environ["LLM_NARRATION_ENABLED"] = "1"
        orchestrator = self._orchestrator()

        with patch("agent.orchestrator.route_inference") as route_inference:
            storage = orchestrator.handle_message("/storage_report", "user1")
            resource = orchestrator.handle_message("/resource_report", "user1")
            network = orchestrator.handle_message("/network_report", "user1")

        self.assertTrue(storage.text.strip())
        self.assertTrue(resource.text.strip())
        self.assertTrue(network.text.strip())
        self.assertFalse(storage.text.startswith("Narration"))
        self.assertFalse(resource.text.startswith("Narration"))
        self.assertFalse(network.text.startswith("Narration"))
        route_inference.assert_not_called()

    def test_ordinary_assistant_chat_still_uses_normal_chat_path(self) -> None:
        orchestrator = self._assistant_orchestrator()
        calls: list[dict[str, object]] = []

        def _fake_route_inference(**kwargs: object) -> dict[str, object]:
            calls.append(dict(kwargs))
            return {
                "ok": True,
                "text": "Assistant explained through the normal chat path.",
                "provider": "ollama",
                "model": "llama3",
                "duration_ms": 1,
                "attempts": [],
            }

        with patch("agent.orchestrator.route_inference", side_effect=_fake_route_inference):
            response = orchestrator.handle_message(
                "compose a brief friendly note about keeping tools tidy",
                "user1",
            )

        self.assertEqual("Assistant explained through the normal chat path.", response.text)
        self.assertEqual(1, len(calls))
        self.assertEqual("chat", calls[0].get("purpose"))
        self.assertNotEqual("narration", calls[0].get("purpose"))
        self.assertEqual("generic_chat", response.data.get("route"))
        self.assertTrue(response.data.get("used_llm"))

    def test_assistant_can_explain_runtime_info_through_runtime_path(self) -> None:
        orchestrator = self._assistant_orchestrator()

        with patch("agent.orchestrator.route_inference") as route_inference:
            response = orchestrator.handle_message("is everything working with the agent?", "user1")

        self.assertIn("Runtime is ready", response.text)
        self.assertEqual("runtime_status", response.data.get("route"))
        self.assertTrue(response.data.get("used_runtime_state"))
        self.assertFalse(response.data.get("used_llm"))
        route_inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
