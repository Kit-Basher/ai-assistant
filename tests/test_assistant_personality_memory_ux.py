from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from agent.assistant_ux import (
    FORBIDDEN_INTERNAL_TERMS,
    build_user_facing_capability_answer,
    classify_memory_request,
)
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


def _make_orchestrator() -> tuple[tempfile.TemporaryDirectory[str], MemoryDB, Orchestrator]:
    tmpdir = tempfile.TemporaryDirectory()
    db = MemoryDB(os.path.join(tmpdir.name, "test.db"))
    db.init_schema(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "memory", "schema.sql")))
    orchestrator = Orchestrator(
        db=db,
        skills_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
        log_path=os.path.join(tmpdir.name, "events.log"),
        timezone="UTC",
        llm_client=None,
    )
    return tmpdir, db, orchestrator


def test_capability_answer_is_user_facing_not_architecture() -> None:
    text = build_user_facing_capability_answer(search_available=True)
    lowered = text.lower()
    assert "everyday questions" in lowered
    assert "check what is using memory" in lowered
    assert "ask before doing it" in lowered
    for term in FORBIDDEN_INTERNAL_TERMS:
        assert term not in lowered


def test_live_capability_prompt_routes_to_friendly_answer() -> None:
    tmpdir, db, orchestrator = _make_orchestrator()
    try:
        response = orchestrator.handle_message(
            "im wondering what you as an agent can help me with, what are your capabilities",
            "user1",
        )
        lowered = response.text.lower()
        assert response.data["route"] == "assistant_capabilities"
        assert "everyday questions" in lowered
        assert "plan and ask before doing it" in lowered
        for term in FORBIDDEN_INTERNAL_TERMS:
            assert term not in lowered
    finally:
        db.close()
        tmpdir.cleanup()


def test_explicit_agent_layer_question_still_gets_architecture_answer() -> None:
    tmpdir, db, orchestrator = _make_orchestrator()
    try:
        response = orchestrator.handle_message("what are you and what is the agent layer supposed to do?", "user1")
        assert response.data["route"] == "assistant_capabilities"
        assert "agent layer" in response.text.lower()
        assert "assistant" in response.text.lower()
    finally:
        db.close()
        tmpdir.cleanup()


def test_memory_classification_durable_temporary_sensitive_uncertain() -> None:
    durable = classify_memory_request("remember that my main PC has an RTX 2060")
    assert durable is not None and durable.kind == "store"
    assert durable.key == "assistant_memory:main_pc_gpu"
    assert durable.value == "RTX 2060"

    temporary = classify_memory_request("remember that I like pizza today")
    assert temporary is not None and temporary.kind == "low_value"

    sensitive = classify_memory_request("remember my Telegram bot token is 123456:TEST_TOKEN_REDACT_ME")
    assert sensitive is not None and sensitive.kind == "refuse_sensitive"
    assert "123456:TEST_TOKEN_REDACT_ME" not in sensitive.message

    uncertain = classify_memory_request("remember that Bob is probably the one who broke the server")
    assert uncertain is not None and uncertain.kind == "confirm_uncertain"


def test_memory_store_recall_and_forget() -> None:
    tmpdir, db, orchestrator = _make_orchestrator()
    try:
        stored = orchestrator.handle_message("remember that my main PC has an RTX 2060", "user1")
        assert "remember" in stored.text.lower()
        recalled = orchestrator.handle_message("what GPU does my main PC have?", "user1")
        assert "rtx 2060" in recalled.text.lower()
        forgot = orchestrator.handle_message("forget my GPU", "user1")
        assert "forgot" in forgot.text.lower()
        recalled_after = orchestrator.handle_message("what GPU does my main PC have?", "user1")
        assert "do not have" in recalled_after.text.lower()
    finally:
        db.close()
        tmpdir.cleanup()


def test_temporary_memory_not_stored_as_durable_preference() -> None:
    tmpdir, db, orchestrator = _make_orchestrator()
    try:
        response = orchestrator.handle_message("remember that I like pizza today", "user1")
        assert "temporary or low-value" in response.text
        assert db.get_user_pref("assistant_memory:food_preference") is None
        food = orchestrator.handle_message("what food do I like?", "user1")
        assert "do not have a durable food preference" in food.text.lower()
    finally:
        db.close()
        tmpdir.cleanup()


def test_clarifying_questions_are_specific_and_safe_actions_do_not_overask() -> None:
    tmpdir, db, orchestrator = _make_orchestrator()
    try:
        vague = orchestrator.handle_message("I want to build something cool", "user1")
        assert vague.text.count("?") == 1
        assert "small app" in vague.text.lower()

        send = orchestrator.handle_message("send a message saying I’ll be late", "user1")
        assert "who should i send it to" in send.text.lower()
        assert "before sending anything" in send.text.lower()

        cleanup = orchestrator.handle_message("can you clean up my downloads folder?", "user1")
        assert "confirmation" in cleanup.text.lower()
        assert "read-only" in cleanup.text.lower()

        greeting = orchestrator.handle_message("you there?", "user1")
        assert "i’m here" in greeting.text.lower() or "i'm here" in greeting.text.lower()
    finally:
        db.close()
        tmpdir.cleanup()


def test_personality_fixture_cases_are_recorded() -> None:
    path = Path(__file__).resolve().parent / "fixtures" / "personality_ux_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) >= 8
    assert any(case["utterance"] == "what can you help me do?" for case in cases)
