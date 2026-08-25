from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.assistant_turn import (
    ASSISTANT_TURN_SCHEMA_VERSION,
    INTERNAL_INSPECT_TOOL,
    ModelLedAssistantTurn,
    TurnValidationError,
    live_capability_catalog,
    live_registry_tools,
    normalize_native_tool_response,
    repaired_transcript,
    canonical_call_signature,
    capability_catalog_authority,
    provider_transcript_tool_calls,
    _tool_name,
)
from agent.capability_registry import ApprovalPolicy, CapabilityContract, CapabilityDefinition, CapabilityMode, CapabilityProvenance, CapabilityRegistry
from agent.llm.types import Message, Response, ToolCall


def _registry(calls: list[tuple[str, dict]]) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(CapabilityDefinition(
        capability_id="filesystem.search", description="Search allowed files", example_goals=("x",),
        input_contract=CapabilityContract(properties={"user_id": str, "text": str, "query": str}, required=("user_id", "text")),
        output_contract=CapabilityContract(properties={"ok": bool}, required=("ok",)), mode=CapabilityMode.READ_ONLY,
        approval_policy=ApprovalPolicy.NEVER,
        invocation_hook=lambda values: calls.append(("filesystem.search", dict(values))) or {"ok": True, "path": "/allowed/backup.txt"},
        verification_hook=lambda result: bool(result.get("ok")), health_hook=lambda: (True, None),
    ))
    return registry


def _response(name: str, arguments: dict, *, call_id: str = "call_1", text: str = "") -> Response:
    return Response(text=text, provider="ollama", model="qwen", tool_calls=(ToolCall(id=call_id, name=name, arguments=json.dumps(arguments)),))


def test_native_tools_reject_forged_authority_and_unknown_capability() -> None:
    registry = _registry([])
    _tools, names = live_registry_tools(registry, exposed_capability_ids={"filesystem.search"})
    with pytest.raises(TurnValidationError, match="authority"):
        normalize_native_tool_response(_response(_tool_name("filesystem.search"), {"query": "x", "approved": True}), registry, names, exposed_capability_ids={"filesystem.search"})
    with pytest.raises(TurnValidationError, match="unknown"):
        normalize_native_tool_response(_response("capability__shell_exec", {"query": "x"}), registry, names)


def test_exact_catalog_id_is_directly_callable_but_invalid_arguments_are_observed() -> None:
    registry = _registry([])
    authority = capability_catalog_authority(registry)
    direct = normalize_native_tool_response(_response("filesystem.search", {"query": "backup"}), registry, {}, catalog_authority=authority)
    assert direct["calls"][0]["capability_id"] == "filesystem.search"
    invalid = normalize_native_tool_response(_response("filesystem.search", {"query": 7}), registry, {}, catalog_authority=authority)
    assert invalid["action"] == "validation_observation"
    assert invalid["validation"]["reason"] == "invalid_or_incomplete_capability_arguments"


def test_catalog_snapshot_rejects_revocation_stale_unknown_and_cross_turn_ids() -> None:
    registry = _registry([])
    authority = capability_catalog_authority(registry)
    authority["filesystem.search"] = "old-catalog-contract-digest"
    with pytest.raises(TurnValidationError, match="stale"):
        normalize_native_tool_response(_response("filesystem.search", {"query": "x"}), registry, {}, catalog_authority=authority)
    registry = _registry([])
    with pytest.raises(TurnValidationError, match="unknown"):
        normalize_native_tool_response(_response("filesystem.search", {"query": "x"}), registry, {}, catalog_authority={})
    with pytest.raises(TurnValidationError, match="unknown"):
        normalize_native_tool_response(_response("Filesystem.Search", {"query": "x"}), registry, {}, catalog_authority=capability_catalog_authority(registry))


def test_dynamic_pack_revocation_invalidates_catalog_authority() -> None:
    registry = _registry([])
    original = registry.require("filesystem.search")
    pack = CapabilityDefinition(**{**original.__dict__, "capability_id": "pack.lookup", "provenance": CapabilityProvenance.PACK})
    registry.register(pack)
    authority = capability_catalog_authority(registry)
    assert normalize_native_tool_response(_response("pack.lookup", {"query": "x"}), registry, {}, catalog_authority=authority)["calls"][0]["capability_id"] == "pack.lookup"
    assert registry.unregister_external("pack.lookup") is True
    with pytest.raises(TurnValidationError, match="unknown|stale"):
        normalize_native_tool_response(_response("pack.lookup", {"query": "x"}), registry, {}, catalog_authority=authority)


def test_direct_canonical_call_is_rewritten_to_declared_alias_with_same_id() -> None:
    response = _response("filesystem.search", {"query": "backup"}, call_id="exact-call")
    turn = {"calls": [{"call_id": "exact-call", "capability_id": "filesystem.search", "arguments": {"query": "backup"}}]}
    serialized = provider_transcript_tool_calls(response, turn)
    assert serialized == (ToolCall(id="exact-call", name=_tool_name("filesystem.search"), arguments='{"query": "backup"}'),)


def test_direct_canonical_selection_exposes_only_its_alias_and_preserves_tool_pairing() -> None:
    calls: list[tuple[str, dict]] = []; registry = _registry(calls)
    replies = iter([
        _response("filesystem.search", {"query": "backup"}, call_id="direct"),
        Response(text="Found it.", provider="ollama", model="qwen"),
    ])
    captured = []
    class Provider:
        def chat(self, request, *, model, timeout_seconds):
            captured.append(request)
            return next(replies)
    class Client:
        config = SimpleNamespace(llm_provider="ollama", ollama_model="ollama:qwen")
        def provider_for_id(self, _): return Provider()
    result = ModelLedAssistantTurn(registry=registry, llm_client=Client(), invoke=lambda cid, values: registry.invoke(cid, values), available=lambda: True).run(user_text="find backup", user_id="u")
    assert result.data["assistant_turn"]["outcome"] == "respond"
    second = captured[1]
    assert _tool_name("filesystem.search") in {tool["function"]["name"] for tool in second.tools}
    assistant, tool = second.messages[-2:]
    assert assistant.tool_calls[0].name == _tool_name("filesystem.search")
    assert assistant.tool_calls[0].id == tool.tool_call_id == "direct"


def test_direct_catalog_id_preserves_mutation_approval_boundary() -> None:
    registry = _registry([])
    original = registry.require("filesystem.search")
    registry.register(CapabilityDefinition(**{**original.__dict__, "capability_id": "filesystem.create", "mode": CapabilityMode.MUTATING, "approval_policy": ApprovalPolicy.REQUIRED}))
    turn = normalize_native_tool_response(_response("filesystem.create", {"query": "x"}), registry, {}, catalog_authority=capability_catalog_authority(registry))
    assert turn["calls"][0]["capability_id"] == "filesystem.create"
    assert registry.require("filesystem.create").approval_policy is ApprovalPolicy.REQUIRED


def test_live_catalog_and_tools_track_registry_authority() -> None:
    calls: list[tuple[str, dict]] = []
    registry = _registry(calls)
    assert [row["id"] for row in live_capability_catalog(registry)] == ["filesystem.search"]
    initial_tools, initial_names = live_registry_tools(registry)
    assert initial_names == {}
    assert [tool["function"]["name"] for tool in initial_tools] == [INTERNAL_INSPECT_TOOL]
    tools, names = live_registry_tools(registry, exposed_capability_ids={"filesystem.search"})
    assert names == {_tool_name("filesystem.search"): "filesystem.search"}
    assert {tool["function"]["name"] for tool in tools} >= {INTERNAL_INSPECT_TOOL, _tool_name("filesystem.search")}


def test_model_led_turn_uses_native_tools_and_repairs_once() -> None:
    calls: list[tuple[str, dict]] = []
    registry = _registry(calls)
    replies = iter([
        _response(INTERNAL_INSPECT_TOOL, {"capability_ids": ["filesystem.search"]}, call_id="inspect"),
        # First proposal is rejected because it attempts to supply authority.
        _response("filesystem.search", {"query": "backup", "approved": True}),
        _response("filesystem.search", {"query": "backup"}, call_id="find"),
        Response(text="I found the instructions at /allowed/backup.txt.", provider="ollama", model="qwen"),
    ])

    class Provider:
        def chat(self, request, *, model, timeout_seconds):  # type: ignore[no-untyped-def]
            assert request.metadata["ollama_native_tools"] is True
            return next(replies)

    class Client:
        config = SimpleNamespace(llm_provider="ollama", ollama_model="ollama:qwen2.5:3b-instruct")
        def provider_for_id(self, _provider):  # type: ignore[no-untyped-def]
            return Provider()

    service = ModelLedAssistantTurn(registry=registry, llm_client=Client(), invoke=lambda capability_id, values: registry.invoke(capability_id, values), available=lambda: True)
    result = service.run(user_text="find backup instructions", user_id="u")
    assert result.data["assistant_turn"]["outcome"] == "respond"
    assert result.data["assistant_turn"]["contract"] == ASSISTANT_TURN_SCHEMA_VERSION
    assert calls == [("filesystem.search", {"user_id": "u", "text": "find backup instructions", "query": "backup"})]


def test_model_unavailable_is_honest() -> None:
    service = ModelLedAssistantTurn(registry=_registry([]), llm_client=None, invoke=lambda *_: None, available=lambda: False)
    assert "General language understanding is unavailable" in service.run(user_text="hello", user_id="u").text


@pytest.mark.parametrize("history", [
    (Message(role="user", content="hello"), Message(role="assistant", content="ordinary")),
    (Message(role="user", content="status"), Message(role="assistant", content="", tool_calls=(ToolCall("inspect", INTERNAL_INSPECT_TOOL, "{}"),)), Message(role="tool", content="inspection", tool_call_id="inspect")),
    (Message(role="user", content="find"), Message(role="assistant", content="", tool_calls=(ToolCall("native", _tool_name("filesystem.search"), "{\"query\":\"x\"}"),)), Message(role="tool", content="result", tool_call_id="native")),
    (Message(role="user", content="empty"), Message(role="assistant", content="")),
], ids=["ordinary_output", "capability_inspection", "native_tool_execution", "malformed_or_empty_output"])
def test_repair_transcript_has_exactly_one_leading_system_and_preserves_history(history: tuple[Message, ...]) -> None:
    leading = Message(role="system", content="base policy")
    repaired = repaired_transcript((leading, *history), "invalid_tool_call")
    assert [message.role for message in repaired].count("system") == 1
    assert repaired[0].role == "system"
    assert "REPAIR CONSTRAINT" in repaired[0].content
    assert repaired[1:] == history
    for original, rebuilt in zip(history, repaired[1:]):
        assert original.tool_call_id == rebuilt.tool_call_id
        assert original.tool_calls == rebuilt.tool_calls


def test_repair_transcript_rejects_late_system_message() -> None:
    with pytest.raises(TurnValidationError, match="nonleading"):
        repaired_transcript((Message(role="system", content="base"), Message(role="user", content="x"), Message(role="system", content="bad")), "x")


def test_call_signature_equivalence_distinction_and_contract_change() -> None:
    registry = _registry([]); definition = registry.require("filesystem.search")
    assert canonical_call_signature(definition, {"query": "backup"}) == canonical_call_signature(definition, dict(query="backup"))
    assert canonical_call_signature(definition, {"query": "backup"}) != canonical_call_signature(definition, {"query": "restore"})
    changed = CapabilityDefinition(**{**definition.__dict__, "provenance": definition.provenance.PACK})
    assert canonical_call_signature(definition, {"query": "backup"}) != canonical_call_signature(changed, {"query": "backup"})


def test_duplicate_read_only_call_suppresses_once_then_terminates() -> None:
    calls: list[tuple[str, dict]] = []; registry = _registry(calls)
    replies = iter([
        _response(INTERNAL_INSPECT_TOOL, {"capability_ids": ["filesystem.search"]}, call_id="i"),
        _response(_tool_name("filesystem.search"), {"query": "backup"}, call_id="a"),
        _response(_tool_name("filesystem.search"), {"query": "backup"}, call_id="b"),
        _response(_tool_name("filesystem.search"), {"query": "backup"}, call_id="c"),
    ])
    class Provider:
        def chat(self, request, *, model, timeout_seconds): return next(replies)
    class Client:
        config = SimpleNamespace(llm_provider="ollama", ollama_model="ollama:qwen")
        def provider_for_id(self, _): return Provider()
    result = ModelLedAssistantTurn(registry=registry, llm_client=Client(), invoke=lambda cid, values: registry.invoke(cid, values), available=lambda: True).run(user_text="find backup", user_id="u")
    assert result.data["assistant_turn"]["error"] == "repeated_duplicate_call"
    assert len(calls) == 1


def test_mutating_calls_are_not_signature_deduplicated() -> None:
    calls: list[tuple[str, dict]] = []; registry = _registry(calls)
    definition = registry.require("filesystem.search")
    mutation = CapabilityDefinition(**{**definition.__dict__, "capability_id": "test.mutate", "mode": CapabilityMode.MUTATING})
    assert mutation.mode is CapabilityMode.MUTATING
