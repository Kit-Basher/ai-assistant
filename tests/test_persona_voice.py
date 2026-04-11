from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.changed_report import build_changed_report_from_system_facts
from agent.error_response_ux import bad_request_next_question, compose_actionable_message
from agent.llm.model_discovery_manager import ModelDiscoveryManager
from agent.runtime_truth_service import RuntimeTruthService


class _FakeRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(llm_registry_path="")
        self.registry_document = {}
        self.secret_store = SimpleNamespace(get_secret=lambda _name: None)

    def ready_status(self) -> dict[str, object]:
        return {"ready": False, "message": "", "runtime_status": {}}


class _FakeDb:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_system_facts_snapshots(self, user_id: str, limit: int = 2):  # noqa: ANN001
        _ = user_id
        _ = limit
        return list(self._rows)


class TestPersonaVoice(unittest.TestCase):
    def test_discovery_fallback_uses_interpreted_voice(self) -> None:
        manager = ModelDiscoveryManager(runtime=_FakeRuntime())
        failure = {
            "source": "huggingface",
            "enabled": True,
            "queried": True,
            "ok": False,
            "count": 0,
            "error_kind": "down",
            "error": "offline",
        }
        with patch.object(manager, "_query_huggingface", return_value=([], dict(failure))), patch.object(
            manager,
            "_query_openrouter",
            return_value=([], dict(failure, source="openrouter")),
        ), patch.object(
            manager,
            "_query_ollama",
            return_value=([], dict(failure, source="ollama")),
        ), patch.object(
            manager,
            "_query_external_snapshots",
            return_value=([], dict(failure, source="external_snapshots")),
        ):
            result = manager.query("tiny gemma", {})

        self.assertFalse(result["ok"])
        self.assertTrue(result["message"].startswith("I checked the enabled discovery sources, but none returned usable data."))
        self.assertIn("Source errors:", result["message"])

    def test_runtime_status_fallback_uses_the_same_voice(self) -> None:
        service = RuntimeTruthService(_FakeRuntime())
        with patch.object(service, "current_chat_target_status", return_value={}), patch.object(
            service,
            "chat_target_truth",
            return_value={},
        ):
            payload = service.runtime_status()

        self.assertEqual("I can't read a clean runtime status yet.", payload["summary"])

    def test_brief_summary_reads_like_a_person_not_a_dump(self) -> None:
        latest = {
            "taken_at": "2026-04-08T00:00:00+00:00",
            "facts_json": json.dumps(
                {
                    "snapshot": {"collector": {"hostname": "workstation"}},
                    "os": {"kernel": {"release": "6.8.0"}},
                    "filesystems": {"mounts": [{"mountpoint": "/", "used_pct": 50.0}]},
                    "memory": {"ram_bytes": {"available": 8 * 1024**3}},
                    "cpu": {"load": {"load_1m": 0.5}},
                    "process_summary": {"top_mem": [{"name": "python"}]},
                }
            ),
        }
        previous = {
            "taken_at": "2026-04-07T00:00:00+00:00",
            "facts_json": json.dumps(
                {
                    "snapshot": {"collector": {"hostname": "workstation"}},
                    "os": {"kernel": {"release": "6.8.0"}},
                    "filesystems": {"mounts": [{"mountpoint": "/", "used_pct": 49.0}]},
                    "memory": {"ram_bytes": {"available": 9 * 1024**3}},
                    "cpu": {"load": {"load_1m": 0.25}},
                    "process_summary": {"top_mem": [{"name": "python"}]},
                }
            ),
        }
        report = build_changed_report_from_system_facts(_FakeDb([latest, previous]), "user1")
        self.assertTrue(report.machine_summary.startswith("This machine is workstation"))
        self.assertTrue(report.delta_lines)

    def test_recovery_messages_share_direct_voice(self) -> None:
        message = compose_actionable_message(
            what_happened="I can't do that yet",
            why="It needs a valid local path.",
            next_action="Choose a directory under the allowed roots.",
        )
        self.assertTrue(message.startswith("That won't work yet"))
        self.assertIn("Next: Choose a directory under the allowed roots.", message)
        self.assertTrue(bad_request_next_question(error_message="messages must be a non-empty list").startswith("Send it again using this JSON shape:"))
        self.assertTrue(bad_request_next_question(error_message="", json_error="invalid_json_body").startswith("Send valid JSON like this:"))


if __name__ == "__main__":
    unittest.main()
