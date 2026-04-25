from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from agent.control_plane import ControlPlaneHandler, ControlPlaneStore


def _tasks_doc(tasks: list[dict[str, object]]) -> str:
    return "# Development Tasks\n\n```json\n" + json.dumps(tasks, ensure_ascii=True, sort_keys=True, indent=2) + "\n```\n"


class TestControlPlane(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        base = Path(self.tmpdir.name)
        self.store = ControlPlaneStore(
            master_plan_path=base / "control" / "master_plan.md",
            tasks_path=base / "control" / "DEVELOPMENT_TASKS.md",
            events_path=base / "control" / "agent_events.jsonl",
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _handler(
        self,
        *,
        path: str,
        method: str = "GET",
        payload: object | None = None,
        client_host: str = "127.0.0.1",
    ) -> ControlPlaneHandler:
        class _HandlerForTest(ControlPlaneHandler):
            def __init__(self, store: ControlPlaneStore, *, path: str, method: str, payload: object | None, client_host: str) -> None:
                self.store = store
                self.path = path
                self.command = method
                content_length = "0"
                if payload is not None:
                    content_length = str(len(json.dumps(payload, ensure_ascii=True).encode("utf-8")))
                self.headers = {"Content-Length": content_length, "Content-Type": "application/json"}
                self.client_address = (client_host, 12345)
                self.status_code = 0
                self.response_payload: dict[str, object] = {}
                self._payload = payload

            def _read_json(self) -> dict[str, object]:  # type: ignore[override]
                if isinstance(self._payload, dict):
                    return dict(self._payload)
                if self._payload is None:
                    return {}
                return self._payload  # type: ignore[return-value]

            def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
                self.status_code = status
                self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))

        return _HandlerForTest(self.store, path=path, method=method, payload=payload, client_host=client_host)

    def _write_tasks(self, tasks: list[dict[str, object]]) -> None:
        self.store.write_tasks(_tasks_doc(tasks))

    def test_get_missing_files_return_empty_defaults(self) -> None:
        master_plan = self._handler(path="/control/master_plan")
        master_plan.do_GET()
        self.assertEqual(200, master_plan.status_code)
        self.assertEqual("", master_plan.response_payload.get("content"))
        self.assertIsNone(master_plan.response_payload.get("mtime"))

        tasks = self._handler(path="/control/tasks")
        tasks.do_GET()
        self.assertEqual(200, tasks.status_code)
        self.assertEqual("", tasks.response_payload.get("content"))
        self.assertIsNone(tasks.response_payload.get("mtime"))

        events = self._handler(path="/control/events")
        events.do_GET()
        self.assertEqual(200, events.status_code)
        self.assertEqual([], events.response_payload.get("events"))

    def test_get_events_fails_closed_on_malformed_jsonl(self) -> None:
        self.store.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.events_path.write_text('{"ok": true}\n{"broken": ', encoding="utf-8")
        handler = self._handler(path="/control/events")
        handler.do_GET()
        self.assertEqual(500, handler.status_code)
        self.assertEqual("invalid_jsonl", handler.response_payload.get("error"))
        self.assertIn("line", str(handler.response_payload.get("message") or "").lower())

    def test_task_index_parsing_returns_normalized_tasks(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-001",
                    "title": "Build queue",
                    "owner": "unassigned",
                    "status": "ready",
                    "kind": "feature",
                    "priority": 2,
                    "depends_on": ["T-000"],
                    "summary": "Set up the queue",
                    "files_expected": ["agent/control_plane.py"],
                    "acceptance_criteria": ["task list parses"],
                },
                {
                    "task_id": "T-000",
                    "title": "Bootstrap control plane",
                    "owner": "manager",
                    "status": "DONE",
                    "kind": "chore",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                },
            ]
        )
        handler = self._handler(path="/control/tasks/index")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        tasks = handler.response_payload.get("tasks") if isinstance(handler.response_payload.get("tasks"), list) else []
        self.assertEqual(["T-000", "T-001"], [row.get("task_id") for row in tasks])
        self.assertEqual("READY", tasks[1].get("status"))
        self.assertEqual("unassigned", tasks[1].get("owner"))

    def test_put_master_plan_writes_atomically_and_round_trips(self) -> None:
        content = "# Plan\n\n- do the thing\n"
        writer = self._handler(path="/control/master_plan", method="PUT", payload={"content": content})
        writer.do_PUT()
        self.assertEqual(200, writer.status_code)
        self.assertEqual(len(content.encode("utf-8")), writer.response_payload.get("bytes_written"))
        self.assertTrue(bool(writer.response_payload.get("mtime")))
        self.assertEqual(content, self.store.master_plan_path.read_text(encoding="utf-8"))

    def test_claim_success_sets_status_and_owner_and_appends_event(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-001",
                    "title": "Implement control plane",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "feature",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        poster = self._handler(path="/control/tasks/claim", method="POST", payload={"task_id": "T-001", "owner": "codex"})
        poster.do_POST()
        self.assertEqual(200, poster.status_code)
        task = poster.response_payload.get("task") if isinstance(poster.response_payload.get("task"), dict) else {}
        self.assertEqual("CLAIMED", task.get("status"))
        self.assertEqual("codex", task.get("owner"))
        event = poster.response_payload.get("event") if isinstance(poster.response_payload.get("event"), dict) else {}
        self.assertEqual("claim", event.get("type"))
        self.assertEqual("T-001", event.get("task_id"))
        self.assertTrue(str(event.get("ts") or "").strip())

    def test_claim_fails_on_non_ready_task(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-002",
                    "title": "Blocked task",
                    "owner": "unassigned",
                    "status": "BLOCKED",
                    "kind": "task",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        poster = self._handler(path="/control/tasks/claim", method="POST", payload={"task_id": "T-002", "owner": "codex"})
        poster.do_POST()
        self.assertEqual(400, poster.status_code)
        self.assertIn("ready", str(poster.response_payload.get("message") or "").lower())

    def test_comment_appends_timestamped_event(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-002C",
                    "title": "Commented task",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "task",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        poster = self._handler(
            path="/control/tasks/comment",
            method="POST",
            payload={"task_id": "T-002C", "owner": "codex", "comment": "Working on it"},
        )
        poster.do_POST()
        self.assertEqual(200, poster.status_code)
        event = poster.response_payload.get("event") if isinstance(poster.response_payload.get("event"), dict) else {}
        self.assertEqual("comment", event.get("type"))
        self.assertTrue(str(event.get("ts") or "").strip())
        events = self.store.read_events().get("events")
        self.assertEqual(1, len(events) if isinstance(events, list) else 0)

    def test_valid_transitions_follow_the_expected_owner_flow(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-003",
                    "title": "Workflow task",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "feature",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        self._handler(path="/control/tasks/claim", method="POST", payload={"task_id": "T-003", "owner": "codex"}).do_POST()
        step_1 = self._handler(
            path="/control/tasks/update",
            method="POST",
            payload={"task_id": "T-003", "owner": "codex", "status": "IMPLEMENTED", "summary": "done"},
        )
        step_1.do_POST()
        self.assertEqual(200, step_1.status_code)
        self.assertEqual("IMPLEMENTED", step_1.response_payload.get("task", {}).get("status"))

        step_2 = self._handler(
            path="/control/tasks/update",
            method="POST",
            payload={"task_id": "T-003", "owner": "kimi", "status": "VERIFIED"},
        )
        step_2.do_POST()
        self.assertEqual(200, step_2.status_code)
        self.assertEqual("VERIFIED", step_2.response_payload.get("task", {}).get("status"))

        step_3 = self._handler(
            path="/control/tasks/update",
            method="POST",
            payload={"task_id": "T-003", "owner": "manager", "status": "DONE"},
        )
        step_3.do_POST()
        self.assertEqual(200, step_3.status_code)
        self.assertEqual("DONE", step_3.response_payload.get("task", {}).get("status"))

    def test_invalid_transition_rejected(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-004",
                    "title": "Invalid transition",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "feature",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        self._handler(path="/control/tasks/claim", method="POST", payload={"task_id": "T-004", "owner": "codex"}).do_POST()
        bad = self._handler(
            path="/control/tasks/update",
            method="POST",
            payload={"task_id": "T-004", "owner": "codex", "status": "VERIFIED"},
        )
        bad.do_POST()
        self.assertEqual(400, bad.status_code)
        self.assertIn("invalid transition", str(bad.response_payload.get("message") or "").lower())

    def test_owner_enforcement_blocks_non_owner_updates(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-005",
                    "title": "Ownership task",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "feature",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        self._handler(path="/control/tasks/claim", method="POST", payload={"task_id": "T-005", "owner": "codex"}).do_POST()
        bad = self._handler(
            path="/control/tasks/update",
            method="POST",
            payload={"task_id": "T-005", "owner": "kimi", "status": "IMPLEMENTED"},
        )
        bad.do_POST()
        self.assertEqual(400, bad.status_code)
        self.assertIn("invalid transition", str(bad.response_payload.get("message") or "").lower())

    def test_next_skips_tasks_with_unmet_dependencies(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-010",
                    "title": "Blocked parent",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "task",
                    "priority": 1,
                    "depends_on": ["T-011"],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                },
                {
                    "task_id": "T-011",
                    "title": "Dependency",
                    "owner": "unassigned",
                    "status": "CLAIMED",
                    "kind": "task",
                    "priority": 2,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                },
                {
                    "task_id": "T-012",
                    "title": "Eligible task",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "task",
                    "priority": 3,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                },
            ]
        )
        handler = self._handler(path="/control/tasks/next?owner=codex")
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        task = handler.response_payload.get("task") if isinstance(handler.response_payload.get("task"), dict) else {}
        self.assertEqual("T-012", task.get("task_id"))

    def test_next_accepts_owner_from_json_body(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-013",
                    "title": "Body lookup task",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "task",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        handler = self._handler(path="/control/tasks/next", method="GET", payload={"owner": "codex"})
        handler.do_GET()
        self.assertEqual(200, handler.status_code)
        task = handler.response_payload.get("task") if isinstance(handler.response_payload.get("task"), dict) else {}
        self.assertEqual("T-013", task.get("task_id"))

    def test_concurrent_mutation_protection_allows_only_one_claim(self) -> None:
        self._write_tasks(
            [
                {
                    "task_id": "T-020",
                    "title": "Race task",
                    "owner": "unassigned",
                    "status": "READY",
                    "kind": "task",
                    "priority": 1,
                    "depends_on": [],
                    "summary": "",
                    "files_expected": [],
                    "acceptance_criteria": [],
                }
            ]
        )
        barrier = threading.Barrier(2)
        results: list[tuple[str, object]] = []

        def _attempt(owner: str) -> None:
            barrier.wait()
            try:
                results.append((owner, self.store.claim_task(task_id="T-020", owner=owner)))
            except Exception as exc:  # pragma: no cover - thread helper
                results.append((owner, exc))

        threads = [threading.Thread(target=_attempt, args=(owner,)) for owner in ("codex", "kimi")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        successes = [item for item in results if isinstance(item[1], dict)]
        failures = [item for item in results if isinstance(item[1], Exception)]
        self.assertEqual(1, len(successes))
        self.assertEqual(1, len(failures))
        index = self.store.read_tasks_index()
        task = index.get("tasks")[0] if isinstance(index.get("tasks"), list) else {}
        self.assertEqual("CLAIMED", task.get("status"))
        self.assertIn(task.get("owner"), {"codex", "kimi"})

    def test_bootstrap_init_creates_missing_files(self) -> None:
        result = self.store.bootstrap()
        self.assertTrue(self.store.master_plan_path.is_file())
        self.assertTrue(self.store.tasks_path.is_file())
        self.assertTrue(self.store.events_path.is_file())
        self.assertEqual([], self.store.read_tasks_index().get("tasks"))
        self.assertEqual(3, len(result.get("existing") or []))

    def test_event_append_schema_validation(self) -> None:
        poster = self._handler(
            path="/control/events",
            method="POST",
            payload={
                "actor": "codex",
                "type": "progress",
                "task_id": "T-030",
                "status_before": "CLAIMED",
                "status_after": "IMPLEMENTED",
                "message": "Started work",
                "extra": {"note": "hello"},
            },
        )
        poster.do_POST()
        self.assertEqual(200, poster.status_code)
        event = poster.response_payload.get("event") if isinstance(poster.response_payload.get("event"), dict) else {}
        self.assertEqual("codex", event.get("actor"))
        self.assertTrue(str(event.get("ts") or "").strip())
        self.assertEqual("progress", event.get("type"))

        bad = self._handler(
            path="/control/events",
            method="POST",
            payload={
                "actor": "codex",
                "type": "progress",
                "task_id": "T-030",
                "status_before": "CLAIMED",
                "status_after": "IMPLEMENTED",
                "message": "Missing extra",
            },
        )
        bad.do_POST()
        self.assertEqual(400, bad.status_code)
        self.assertIn("extra", str(bad.response_payload.get("message") or "").lower())

    def test_malformed_event_payload_is_rejected(self) -> None:
        bad = self._handler(
            path="/control/events",
            method="POST",
            payload={
                "actor": "codex",
                "type": "progress",
                "task_id": "T-030",
                "status_before": "CLAIMED",
                "status_after": "IMPLEMENTED",
                "message": "bad extra",
                "extra": "not-an-object",
            },
        )
        bad.do_POST()
        self.assertEqual(400, bad.status_code)
        self.assertIn("extra", str(bad.response_payload.get("message") or "").lower())

    def test_control_plane_is_loopback_only(self) -> None:
        forbidden = self._handler(path="/control/master_plan", client_host="10.0.0.8")
        forbidden.do_GET()
        self.assertEqual(403, forbidden.status_code)
        self.assertEqual("forbidden", forbidden.response_payload.get("error"))
        self.assertTrue(bool(forbidden.response_payload.get("operator_only")))
        self.assertIn("loopback", str(forbidden.response_payload.get("message") or "").lower())

    def test_repo_canonical_tasks_file_is_parseable(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        store = ControlPlaneStore(
            master_plan_path=repo_root / "control" / "master_plan.md",
            tasks_path=repo_root / "control" / "DEVELOPMENT_TASKS.md",
            events_path=repo_root / "control" / "agent_events.jsonl",
        )
        index = store.read_tasks_index()
        tasks = index.get("tasks") if isinstance(index.get("tasks"), list) else []
        self.assertGreaterEqual(len(tasks), 1)
        self.assertTrue(str(tasks[0].get("task_id") or "").strip())
