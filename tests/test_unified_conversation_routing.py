from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
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
from agent.request_understanding import RequestUnderstandingService
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


HELD_OUT_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "unified_routing_held_out.json").read_text(encoding="utf-8")
)

TRANSFORMATION_FAMILIES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "presence": (
        "assistant.presence",
        (
            ("casing", "ARE YOU STILL PRESENT"),
            ("punctuation", "you there!!!"),
            ("spacing", "are   you   around"),
            ("misspelling", "are you availble"),
            ("transposition", "are you presnet"),
            ("sms_shorthand", "r u online"),
            ("dropped_words", "you present"),
            ("word_order", "present are you"),
            ("polite_filler", "could you please confirm you are present"),
            ("unseen_paraphrase", "is this assistant responding"),
        ),
    ),
    "filesystem": (
        "filesystem.search",
        (
            ("casing", "FIND THE BUDGET FILE UNDER {allowed}"),
            ("punctuation", "find budget file under {allowed}!!!"),
            ("spacing", "find   budget   file   under   {allowed}"),
            ("misspelling", "serach for the budget file under {allowed}"),
            ("transposition", "searhc for budget under {allowed}"),
            ("sms_shorthand", "pls find budget file under {allowed}"),
            ("dropped_words", "find budget {allowed}"),
            ("word_order", "under {allowed} budget file locate"),
            ("polite_filler", "could you please locate the budget file beneath {allowed}"),
            ("unseen_paraphrase", "scan {allowed} for a document named budget"),
        ),
    ),
    "system_status": (
        "system.status",
        (
            ("casing", "CHECK RUNTIME HEALTH"),
            ("punctuation", "runtime health???!!!"),
            ("spacing", "check   the   runtime   status"),
            ("misspelling", "check sytem status"),
            ("transposition", "check sytsem health"),
            ("sms_shorthand", "pls check runtime"),
            ("dropped_words", "runtime healthy"),
            ("word_order", "health runtime check"),
            ("polite_filler", "would you please inspect system health"),
            ("unseen_paraphrase", "give me the machine service health picture"),
        ),
    ),
    "model_inventory": (
        "models.inventory",
        (
            ("casing", "SHOW INSTALLED LOCAL MODELS"),
            ("punctuation", "which models are installed?!"),
            ("spacing", "show   local   model   inventory"),
            ("misspelling", "show the modle inventory"),
            ("transposition", "show the moedl status"),
            ("sms_shorthand", "pls show local models"),
            ("dropped_words", "local models installed"),
            ("word_order", "installed models local show"),
            ("polite_filler", "could you please list the available local models"),
            ("unseen_paraphrase", "enumerate language engines available locally"),
        ),
    ),
    "model_switch": (
        "models.switch",
        (
            ("casing", "SWITCH MODEL TO OLLAMA:FIXTURE-CHAT:7B"),
            ("punctuation", "switch to ollama:fixture-chat:7b!!!"),
            ("spacing", "switch   model   to   ollama:fixture-chat:7b"),
            ("misspelling", "swich model to ollama:fixture-chat:7b"),
            ("transposition", "swtich model to ollama:fixture-chat:7b"),
            ("sms_shorthand", "pls switch 2 ollama:fixture-chat:7b"),
            ("dropped_words", "switch ollama:fixture-chat:7b"),
            ("word_order", "ollama:fixture-chat:7b switch model"),
            ("polite_filler", "could you please change the chat model to ollama:fixture-chat:7b"),
            ("unseen_paraphrase", "make ollama:fixture-chat:7b handle this chat instead"),
        ),
    ),
    "model_scout": (
        "models.scout",
        (
            ("casing", "RUN MODEL SCOUT"),
            ("punctuation", "model scout now?!"),
            ("spacing", "run   model   scout   now"),
            ("misspelling", "run model scot"),
            ("transposition", "run moedl scout"),
            ("sms_shorthand", "pls run model scout"),
            ("dropped_words", "model scout now"),
            ("word_order", "now scout models compare"),
            ("polite_filler", "could you please have Model Scout assess coding choices"),
            ("unseen_paraphrase", "assess stronger coding engines worth using"),
        ),
    ),
    "packs": (
        "packs.use",
        (
            ("casing", "SHOW INSTALLED PACKS"),
            ("punctuation", "show installed packs?!"),
            ("spacing", "show   installed   packs"),
            ("misspelling", "show installed paks"),
            ("transposition", "show installed pcaks"),
            ("sms_shorthand", "pls show packs"),
            ("dropped_words", "installed packs"),
            ("word_order", "packs installed show"),
            ("polite_filler", "could you please list the skill packs available"),
            ("unseen_paraphrase", "enumerate usable guidance bundles"),
        ),
    ),
    "conversation_history": (
        "conversation.history",
        (
            ("casing", "RECAP OUR PREVIOUS CONVERSATION"),
            ("punctuation", "recap our last conversation?!"),
            ("spacing", "recap   the   previous   conversation"),
            ("misspelling", "recap the prevous conversation"),
            ("transposition", "recap the pervious conversation"),
            ("sms_shorthand", "pls recap last chat"),
            ("dropped_words", "recap previous"),
            ("word_order", "previous conversation recap"),
            ("polite_filler", "could you please remind me what we discussed earlier"),
            ("unseen_paraphrase", "resume the task context from earlier"),
        ),
    ),
}

TRANSFORMATION_CASES = [
    {"family": family, "capability": capability, "category": category, "text": text}
    for family, (capability, cases) in TRANSFORMATION_FAMILIES.items()
    for category, text in cases
]


def _ready_mock_chat_runtime(runtime: AgentRuntime) -> None:
    runtime.add_provider_model(
        "ollama",
        {"model": "fixture-chat:7b", "capabilities": ["chat"], "available": True},
    )
    runtime.set_default_chat_model("ollama:fixture-chat:7b")
    runtime._health_monitor.state = {  # type: ignore[attr-defined]
        "providers": {"ollama": {"status": "ok", "last_checked_at": 123}},
        "models": {
            "ollama:fixture-chat:7b": {
                "provider_id": "ollama",
                "status": "ok",
                "last_checked_at": 123,
            }
        },
    }
    runtime._router.set_external_health_state(runtime._health_monitor.state)  # type: ignore[attr-defined]
    runtime.startup_phase = "ready"


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


def test_registered_native_invocations_do_not_reclassify_selected_capability() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        note = allowed / "note.txt"
        note.write_text("registry direct invocation\n", encoding="utf-8")
        runtime = AgentRuntime(
            _config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(str(allowed),))
        )
        orchestrator = runtime.orchestrator()
        registry = orchestrator._capability_registry
        payloads = {
            "assistant.presence": {"user_id": "direct", "text": "present"},
            "assistant.capabilities": {"user_id": "direct", "text": "abilities"},
            "filesystem.list": {"user_id": "direct", "text": "list", "path_hint": str(allowed)},
            "filesystem.search": {
                "user_id": "direct", "text": "find note", "path_hint": str(allowed),
                "query": "note", "search_mode": "filename", "filesystem_view": "search",
            },
            "filesystem.read": {"user_id": "direct", "text": "read", "path_hint": str(note)},
            "system.status": {"user_id": "direct", "text": "health", "status_scope": "runtime"},
            "models.inventory": {"user_id": "direct", "text": "models", "model_view": "inventory"},
            "models.scout": {"user_id": "direct", "text": "scout", "scout_view": "strategy"},
            "packs.use": {"user_id": "direct", "text": "packs", "pack_operation": "list"},
            "conversation.history": {"user_id": "direct", "text": "history", "history_focus": "overview"},
        }
        with patch(
            "agent.orchestrator.classify_runtime_chat_route",
            side_effect=AssertionError("registry invocation attempted legacy reclassification"),
        ):
            results = {
                capability_id: registry.invoke(capability_id, payload)
                for capability_id, payload in payloads.items()
            }
    assert set(results) == set(payloads)
    assert all(str(result.text or "").strip() for result in results.values())


def test_held_out_language_corpus_evaluates_every_production_chat_scenario() -> None:
    """Never let pytest maxfail hide the remainder of the held-out corpus."""
    failures: list[dict[str, object]] = []
    capability_totals: Counter[str] = Counter()
    capability_passed: Counter[str] = Counter()
    category_totals: Counter[str] = Counter()
    category_passed: Counter[str] = Counter()
    for index, case in enumerate(HELD_OUT_CASES):
        capability = str(case["capability"])
        category = str(case["category"])
        capability_totals[capability] += 1
        category_totals[category] += 1
        try:
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
                text = str(case["text"]).format(allowed=allowed, note=note)
                response = _chat(runtime, text, user=f"heldout-{index}", thread=f"heldout-{index}:t")
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            understanding = _understanding(response)
            observed = {
                "route": meta.get("route"),
                "capability": understanding.get("selected_capability_id"),
                "used_llm": meta.get("used_llm"),
            }
            expected = {
                "route": case["route"],
                "capability": case["capability"],
                "used_llm": False,
            }
            if observed != expected:
                failures.append({"index": index, "text": case["text"], "expected": expected, "observed": observed})
                continue
            capability_passed[capability] += 1
            category_passed[category] += 1
        except Exception as exc:  # keep evaluating independent held-out rows
            failures.append({"index": index, "text": case["text"], "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "total": len(HELD_OUT_CASES),
        "passed": len(HELD_OUT_CASES) - len(failures),
        "failed": len(failures),
        "by_capability": {
            key: {"passed": capability_passed[key], "total": capability_totals[key]}
            for key in sorted(capability_totals)
        },
        "by_category": {
            key: {"passed": category_passed[key], "total": category_totals[key]}
            for key in sorted(category_totals)
        },
        "failures": failures,
    }
    print("WP1_HELD_OUT_REPORT=" + json.dumps(report, sort_keys=True))
    assert not failures, json.dumps(report, indent=2, sort_keys=True)


def test_held_out_manifest_reports_all_categories_independently() -> None:
    assert len(HELD_OUT_CASES) == 24
    counts = Counter(str(case["category"]) for case in HELD_OUT_CASES)
    assert counts == {
        "sms_shorthand": 1,
        "misspelling": 1,
        "dropped_words": 1,
        "word_order": 1,
        "punctuation": 1,
        "polite_filler": 1,
        "paraphrase": 1,
        "capability_inventory": 2,
        "model_inventory": 3,
        "model_scout": 2,
        "filesystem_search": 1,
        "filesystem_list": 1,
        "filesystem_read": 1,
        "packs": 2,
        "history": 2,
        "casual": 2,
        "ambiguity": 1,
    }
@pytest.mark.parametrize(
    "case",
    TRANSFORMATION_CASES,
    ids=lambda case: f"{case['family']}:{case['category']}",
)
def test_systematic_language_transformations_use_production_chat(case: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        (allowed / "budget-2026.txt").write_text("budget\n", encoding="utf-8")
        runtime = AgentRuntime(
            _config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(str(allowed),))
        )
        _ready_mock_chat_runtime(runtime)
        text = case["text"].format(allowed=allowed)
        response = _chat(
            runtime,
            text,
            user=f"transform-{case['family']}-{case['category']}",
            thread=f"transform-{case['family']}-{case['category']}:t",
        )
    understanding = _understanding(response)
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    assert understanding.get("selected_capability_id") == case["capability"], (case, understanding, response)
    assert meta.get("used_llm") is False, (case, response)


def test_transformation_matrix_is_balanced_by_family_and_category() -> None:
    family_counts = Counter(case["family"] for case in TRANSFORMATION_CASES)
    category_counts = Counter(case["category"] for case in TRANSFORMATION_CASES)
    assert family_counts == {family: 10 for family in TRANSFORMATION_FAMILIES}
    assert set(category_counts) == {
        "casing", "punctuation", "spacing", "misspelling", "transposition",
        "sms_shorthand", "dropped_words", "word_order", "polite_filler", "unseen_paraphrase",
    }
    assert all(count == len(TRANSFORMATION_FAMILIES) for count in category_counts.values())


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
        _ready_mock_chat_runtime(runtime)
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
        _ready_mock_chat_runtime(runtime)
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


@pytest.mark.parametrize(
    ("false_text", "forbidden"),
    (
        ("I am in a secure sandbox and I cannot access your files because I do not have a physical form.", "secure sandbox"),
        ("I am unable to access external information, so I cannot help.", "unable to access external information"),
        ("I do not have sensory perception and cannot discuss that.", "sensory perception"),
        ("I cannot use tools and I do not have any skills.", "cannot use tools"),
        ("I am not connected to a model and no language model is available.", "not connected to a model"),
    ),
)
def test_false_runtime_claim_from_model_is_replaced_by_grounded_contract(false_text: str, forbidden: str) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        fake = {
            "ok": True,
            "text": false_text,
            "provider": "ollama",
            "model": "Gemma:latest",
            "duration_ms": 4,
            "data": {},
        }
        with patch("agent.orchestrator.route_inference", return_value=fake) as inference:
            response = _chat(runtime, "tell me something pleasant about summer")
        assert inference.call_count in {1, 2}
        assert not inference.call_args_list[0].kwargs["metadata"].get("response_repair")
        if inference.call_count == 2:
            assert inference.call_args_list[1].kwargs["metadata"]["response_repair"] == "capability_disclaimer"
    message = str(response.get("message") or "").lower()
    assert forbidden not in message
    assert "local personal agent" in message or "general knowledge" in message


def test_literal_registry_anchor_removal_preserves_semantic_family_on_production_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        orchestrator = runtime.orchestrator()
        original_registry = orchestrator._capability_registry
        rebuilt = CapabilityRegistry()
        removed_anchor = "which language model is active"
        for definition in original_registry.definitions():
            if definition.capability_id == "models.inventory":
                assert removed_anchor in definition.example_goals
                examples = tuple(example for example in definition.example_goals if example != removed_anchor)
                rebuilt.register(replace(definition, example_goals=examples))
            else:
                rebuilt.register(definition)
        assert all(
            removed_anchor not in definition.example_goals
            for definition in rebuilt.definitions()
        )
        orchestrator._capability_registry = rebuilt
        orchestrator._request_understanding = RequestUnderstandingService(rebuilt)

        model_response = _chat(runtime, removed_anchor, user="anchor-model", thread="anchor-model:t")
        presence_response = _chat(runtime, "u here?", user="anchor-presence", thread="anchor-presence:t")

    assert _understanding(model_response).get("selected_capability_id") == "models.inventory"
    assert _understanding(presence_response).get("selected_capability_id") == "assistant.presence"


def test_ordinary_change_explanation_reaches_generic_chat_not_model_capability() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        fake = {
            "ok": True,
            "text": "Leaves change colour as chlorophyll breaks down and other pigments become visible.",
            "provider": "ollama",
            "model": "Gemma:latest",
            "duration_ms": 4,
            "data": {},
        }
        with patch("agent.orchestrator.route_inference", return_value=fake):
            response = _chat(runtime, "In one sentence, explain why leaves change colour in autumn.")
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    assert meta.get("route") == "generic_chat"
    assert meta.get("used_llm") is True
    assert _understanding(response).get("selected_capability_id") is None


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


def test_model_inventory_registry_selection_outranks_legacy_system_fast_path() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        response = _chat(runtime, "which local model is answering now?")
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    assert meta.get("route") == "model_status"
    assert meta.get("used_llm") is False
    assert _understanding(response).get("selected_capability_id") == "models.inventory"


def test_unified_selection_cannot_be_overridden_by_any_legacy_route_label() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        contradictory = {
            "route": "runtime_status",
            "kind": "product_specific_guard",
            "fallback_reason": "legacy_classifier_disagreed",
        }
        with patch.object(runtime, "chat_route_decision", return_value=contradictory):
            response = _chat(runtime, "which local model is answering now?")
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    assert meta.get("route") == "model_status"
    assert _understanding(response).get("selected_capability_id") == "models.inventory"


def test_production_chat_computes_unified_understanding_once() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        orchestrator = runtime.orchestrator()
        with patch.object(
            orchestrator,
            "preview_conversation_request",
            wraps=orchestrator.preview_conversation_request,
        ) as preview:
            response = _chat(runtime, "which local model is answering now?")
    assert preview.call_count == 1
    timings = (response.get("meta") or {}).get("chat_timing_ms")  # type: ignore[union-attr]
    assert isinstance(timings, dict)
    assert int(timings.get("request_understanding_ms") or 0) >= 0


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


@pytest.mark.parametrize(
    ("first_text", "reply"),
    (
        ("list files under {allowed}", "that file"),
        ("list files under {allowed}", "no, I meant the other drive"),
        ("which local models are installed", "the second one"),
        ("is the local runtime healthy", "do that"),
    ),
)
def test_unresolved_multiturn_references_ask_once_instead_of_guessing(first_text: str, reply: str) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        (allowed / "one.txt").write_text("one\n", encoding="utf-8")
        runtime = AgentRuntime(
            _config(str(root / "registry.json"), str(root / "agent.db"), perception_roots=(str(allowed),))
        )
        _ready_mock_chat_runtime(runtime)
        rendered_first = first_text.format(allowed=allowed)
        _chat(runtime, rendered_first, user="reference", thread="reference:t")
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("reference reached model")):
            response = _chat(runtime, reply, user="reference", thread="reference:t")
    meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
    understanding = _understanding(response)
    assert meta.get("route") == "assistant_clarification"
    assert understanding.get("selected_capability_id") is None
    assert "same_thread_unresolved_reference" in understanding.get("context_used", [])
    assert str(response.get("message") or "").count("?") == 1


def test_result_reference_never_crosses_chat_threads_for_same_user() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        model = _chat(
            runtime,
            "which local model is answering now?",
            user="thread-bound",
            thread="thread-bound:model",
        )
        assert (model.get("meta") or {}).get("route") == "model_status"  # type: ignore[union-attr]
        fake = {
            "ok": True,
            "text": "Leaves change colour as chlorophyll breaks down.",
            "provider": "ollama",
            "model": "Gemma:latest",
            "duration_ms": 4,
            "data": {},
        }
        with patch("agent.orchestrator.route_inference", return_value=fake):
            unrelated = _chat(
                runtime,
                "In one sentence, explain why leaves change colour in autumn.",
                user="thread-bound",
                thread="thread-bound:science",
            )
    assert (unrelated.get("meta") or {}).get("route") == "generic_chat"  # type: ignore[union-attr]
    understanding = _understanding(unrelated)
    assert understanding.get("selected_capability_id") is None
    assert "same_thread_capability_reference" not in understanding.get("context_used", [])
