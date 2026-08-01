from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from agent.api_server import APIServerHandler, AgentRuntime
from agent.capability_registry import (
    ApprovalPolicy,
    CapabilityContract,
    CapabilityDefinition,
    CapabilityMode,
    CapabilityRegistry,
)
from test_api_server import _config


class _Handler(APIServerHandler):
    def __init__(self, runtime: AgentRuntime, payload: dict[str, object]) -> None:
        self.runtime = runtime
        self.path = "/chat"
        self.headers = {"Content-Length": "0"}
        self._payload = payload
        self.status = 0
        self.body: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status = status
        self.body = json.loads(json.dumps(payload))


def _chat(runtime: AgentRuntime, text: str, *, user: str = "unified", thread: str = "unified:t") -> dict[str, object]:
    handler = _Handler(
        runtime,
        {
            "messages": [{"role": "user", "content": text}],
            "user_id": user,
            "thread_id": thread,
            "session_id": thread,
            "source_surface": "webui",
        },
    )
    handler.do_POST()
    assert handler.status == 200, handler.body
    return handler.body


def _understanding(response: dict[str, object]) -> dict[str, object]:
    setup = response.get("setup") if isinstance(response.get("setup"), dict) else {}
    return setup.get("request_understanding") if isinstance(setup.get("request_understanding"), dict) else {}


def test_registry_rejects_unknown_ids_invalid_inputs_and_unapproved_mutation() -> None:
    registry = CapabilityRegistry()
    calls: list[dict[str, object]] = []
    registry.register(
        CapabilityDefinition(
            capability_id="test.mutate",
            description="change one test value",
            example_goals=("update the test value",),
            input_contract=CapabilityContract(properties={"value": str}, required=("value",)),
            output_contract=CapabilityContract(properties={"ok": bool}, required=("ok",)),
            mode=CapabilityMode.MUTATING,
            approval_policy=ApprovalPolicy.REQUIRED,
            invocation_hook=lambda value: calls.append(dict(value)) or {"ok": True},
            verification_hook=lambda result: result == {"ok": True},
            health_hook=lambda: (True, None),
        )
    )
    with pytest.raises(ValueError, match="unknown_capability_id"):
        registry.validate_selection("invented.capability", {})
    with pytest.raises(ValueError, match="invalid_input_type"):
        registry.validate_selection("test.mutate", {"value": 3})
    with pytest.raises(PermissionError, match="approval_required"):
        registry.invoke("test.mutate", {"value": "safe"})
    assert calls == []
    assert registry.invoke("test.mutate", {"value": "safe"}, approved=True) == {"ok": True}
    assert calls == [{"value": "safe"}]


def test_held_out_language_corpus_uses_production_chat_path() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "unified_routing_held_out.json"
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert len(cases) == 24
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        note = allowed / "heldout-note.txt"
        note.write_text("held-out routing text\n", encoding="utf-8")
        (allowed / "budget-2026.txt").write_text("budget\n", encoding="utf-8")
        runtime = AgentRuntime(
            _config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(str(allowed),))
        )
        for index, case in enumerate(cases):
            text = str(case["text"]).format(allowed=allowed, note=note)
            response = _chat(runtime, text, user=f"heldout-{index}", thread=f"heldout-{index}:t")
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            understanding = _understanding(response)
            assert meta.get("route") == case["route"], (case, response)
            assert understanding.get("selected_capability_id") == case["capability"], (case, understanding)
            assert meta.get("used_llm") is False, (case, response)


@pytest.mark.parametrize(
    "text",
    [
        "U HERE", "u...here???", "  u   here  ", "r u hre", "you tehre", "hey, are ya still around please",
        "there you are?", "yo u stil ther",
    ],
)
def test_presence_transformations_never_reach_generic_model(text: str) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("presence reached model")):
            response = _chat(runtime, text, user="presence", thread=f"presence:{text}")
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    assert meta.get("route") == "social_turn"
    assert _understanding(response).get("selected_capability_id") == "assistant.presence"
    assert "local personal agent service" in str(response.get("message") or "").lower()


def test_ambiguous_request_asks_once_without_invocation() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("ambiguity reached model")):
            response = _chat(runtime, "open or find the file from yesterday")
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    assert meta.get("route") == "assistant_clarification"
    assert str(response.get("message") or "").count("?") == 1
    assert meta.get("used_tools") == []


def test_model_mutation_approval_typo_and_punctuation_cannot_bypass_boundary() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        runtime.add_provider_model("ollama", {"model": "fixture:7b", "capabilities": ["chat"], "available": True})
        before = runtime.get_defaults().get("chat_model")
        preview = _chat(runtime, "switch the chat model to ollama:fixture:7b", user="approval", thread="approval:t")
        assert "approve" in str(preview.get("message") or "").lower() or "continue" in str(preview.get("message") or "").lower()
        for reply in ("yex", "y.e.s", "sure??? maybe"):
            guarded = _chat(runtime, reply, user="approval", thread="approval:t")
            assert runtime.get_defaults().get("chat_model") == before
            assert "yes" in str(guarded.get("message") or "").lower() or "approval" in str(guarded.get("message") or "").lower()
        cancelled = _chat(runtime, "never mind", user="approval", thread="approval:t")
        assert runtime.get_defaults().get("chat_model") == before
        assert "cancel" in str(cancelled.get("message") or "").lower() or "cleared" in str(cancelled.get("message") or "").lower()


def test_false_runtime_claim_from_model_is_replaced_by_grounded_contract() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        fake = {
            "ok": True,
            "text": "I am in a secure sandbox and I cannot access your files because I do not have a physical form.",
            "provider": "ollama",
            "model": "Gemma:latest",
            "duration_ms": 4,
            "data": {},
        }
        with patch("agent.orchestrator.route_inference", return_value=fake):
            response = _chat(runtime, "tell me something pleasant about summer")
    message = str(response.get("message") or "").lower()
    assert "secure sandbox" not in message
    assert "cannot access your files" not in message
    assert "physical form" not in message
    assert "local personal agent" in message


def test_live_capability_answer_is_registry_snapshot_not_static_prose() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(raw,)))
        response = _chat(runtime, "what can you actually do?")
    setup = response.get("setup") if isinstance(response.get("setup"), dict) else {}
    rows = setup.get("capabilities") if isinstance(setup.get("capabilities"), list) else []
    assert rows
    assert {row.get("id") for row in rows if isinstance(row, dict)} >= {
        "assistant.presence", "filesystem.search", "models.inventory", "packs.use"
    }
    assert all("available" in row and "health_reason" in row for row in rows if isinstance(row, dict))


def test_unrelated_new_goal_is_not_hijacked_by_pending_approval() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        runtime.add_provider_model("ollama", {"model": "fixture:7b", "capabilities": ["chat"], "available": True})
        _chat(runtime, "switch the chat model to ollama:fixture:7b", user="stale", thread="stale:t")
        response = _chat(runtime, "r u hre", user="stale", thread="stale:t")
        meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
        assert meta.get("route") == "social_turn"
        assert "awaiting approval" not in str(response.get("message") or "").lower()


def test_same_thread_again_reuses_last_capability_without_forging_approval() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        first = _chat(runtime, "is the local runtime healthy", user="again", thread="again:t")
        second = _chat(runtime, "check that again", user="again", thread="again:t")
    assert (first.get("meta") or {}).get("route") == "runtime_status"  # type: ignore[union-attr]
    assert (second.get("meta") or {}).get("route") == "runtime_status"  # type: ignore[union-attr]
    understanding = _understanding(second)
    assert understanding.get("selected_capability_id") == "system.status"
    assert "same_thread_capability_reference" in understanding.get("context_used", [])
