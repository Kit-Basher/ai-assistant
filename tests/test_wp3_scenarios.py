from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from agent.capability_registry import CapabilityHealthState
from agent.task_loop import TaskCoordinator, TaskState, TaskStore, build_missing_capability
from tests import test_general_task_loop as general
from tests import test_task_loop_properties as properties


SCENARIO_CATEGORIES = (
    "simple_fast_path",
    "multi_capability_read",
    "filesystem_search_read",
    "verified_mutation",
    "denied_approval",
    "cancellation",
    "recoverable_failure",
    "verification_failure",
    "unavailable_dependency",
    "missing_capability",
    "partial_completion",
    "direction_change",
    "unrelated_interruption",
    "refresh_restart",
    "task_inquiry_control",
    "planner_attack",
)


@pytest.mark.parametrize("category", SCENARIO_CATEGORIES)
def test_production_path_scenario_corpus(category: str) -> None:
    if category in {"simple_fast_path", "multi_capability_read"}:
        general.test_production_chat_preserves_fast_path_and_creates_verified_multi_capability_task()
    elif category == "filesystem_search_read":
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            allowed = root / "allowed"
            allowed.mkdir()
            note = allowed / "project-note.txt"
            note.write_text("verified bounded content\n", encoding="utf-8")
            runtime = general.AgentRuntime(general._config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(str(allowed),)))
            response = general._chat(runtime, f"find project-note under {allowed}; then read the matching file", user="filesystem", thread="filesystem:t")
            task = response.get("setup", {}).get("task", {})
            assert task.get("state") == TaskState.SUCCEEDED.value
            assert [step["capability_id"] for step in task["steps"]] == ["filesystem.search", "filesystem.read"]
    elif category == "verified_mutation":
        general.test_production_chat_verified_directory_mutation_requires_exact_approval()
    elif category in {"denied_approval", "unrelated_interruption", "task_inquiry_control"}:
        general.test_task_chat_status_and_unrelated_casual_turn_do_not_advance_task()
    elif category == "cancellation":
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "must-not-exist"
            runtime = general.AgentRuntime(general._config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
            waiting = general._chat(runtime, f"check system health; then create a folder at {target}", user="cancel", thread="cancel:t")
            assert waiting["setup"]["task"]["state"] == TaskState.AWAITING_APPROVAL.value
            cancelled = general._chat(runtime, "stop that task", user="cancel", thread="cancel:t")
            assert cancelled["setup"]["task"]["state"] == TaskState.CANCELLED.value
            assert not target.exists()
    elif category == "recoverable_failure":
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = general.AgentRuntime(general._config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
            registry = runtime.orchestrator()._capability_registry  # noqa: SLF001
            definition = registry.require("system.status")
            calls = []
            def flaky(inputs):  # type: ignore[no-untyped-def]
                calls.append(1)
                if len(calls) == 1:
                    raise TimeoutError("injected transient read")
                return definition.invocation_hook(inputs)
            registry._items["system.status"] = general.replace(definition, invocation_hook=flaky)  # noqa: SLF001
            response = general._chat(runtime, "check system health; then show installed local models", user="retry", thread="retry:t")
            assert response["setup"]["task"]["state"] == TaskState.SUCCEEDED.value
            assert len(calls) == 2
    elif category == "verification_failure":
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = general.AgentRuntime(general._config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
            registry = runtime.orchestrator()._capability_registry  # noqa: SLF001
            definition = registry.require("system.status")
            registry._items["system.status"] = general.replace(definition, verification_hook=lambda _result: False)  # noqa: SLF001
            handler = general._Handler(runtime, "/chat", {
                "messages": [{"role": "user", "content": "check system health; then show installed local models"}],
                "user_id": "verify", "thread_id": "verify:t", "session_id": "verify:t", "source_surface": "webui",
            })
            handler.do_POST()
            assert handler.status == 400
            response = handler.body
            task = response["setup"]["task"]
            assert task["state"] == TaskState.FAILED.value
            assert "completed and verified" not in str(response.get("message") or "").lower()
    elif category == "unavailable_dependency":
        with tempfile.TemporaryDirectory() as raw:
            registry = general._registry()
            definition = registry.require("fixture.inspect")
            registry._items["fixture.inspect"] = general.replace(definition, health_hook=lambda: (False, "fixture_provider_stopped"))  # noqa: SLF001
            coordinator = TaskCoordinator(store=TaskStore(general._db(Path(raw))), registry=registry)
            task = coordinator.create(general._proposal(registry, capabilities=("fixture.inspect",)), actor_id="actor", session_id="session", thread_id="thread")
            result = coordinator.run(task["task_id"], actor_id="actor", thread_id="thread")
            assert result["state"] == TaskState.BLOCKED.value
            assert result["failure"]["classification"] == "unavailable_dependency"
    elif category in {"missing_capability", "partial_completion"}:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = general.AgentRuntime(general._config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
            response = general._chat(runtime, "check system health; then show installed local models; then transcribe a video into captions", user="partial", thread="partial:t")
            task = response["setup"]["task"]
            missing = response["setup"]["missing_capability"]
            assert task["state"] == TaskState.PARTIALLY_COMPLETED.value
            assert task["steps"][0]["verified"] is True
            assert missing["automatic_pack_action"] is False
            assert missing["why_incomplete"]
    elif category == "direction_change":
        general.test_direction_change_revises_plan_invalidates_old_target_and_keeps_completed_evidence()
    elif category == "refresh_restart":
        general.test_restart_reconciliation_resumes_reads_and_marks_dispatched_mutation_indeterminate()
    elif category == "planner_attack":
        registry = general._registry()
        proposal = general._proposal(registry)
        proposal["steps"][0]["capability_id"] = "unknown.raw_shell"
        with pytest.raises(ValueError, match="unknown_capability_id"):
            general.validate_plan(proposal, registry, actor_id="actor", session_id="session", thread_id="thread")
    else:  # pragma: no cover - protected tuple and exhaustive dispatch above.
        raise AssertionError(f"unhandled scenario category: {category}")
