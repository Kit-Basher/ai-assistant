#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.telegram_bridge import handle_telegram_text


SurfaceName = Literal["webui", "telegram"]
FailureCategory = Literal[
    "transport/runtime",
    "grounding/truth",
    "memory/continuity",
    "assistant-behavior",
    "confirmation/action",
]

DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("ASSISTANT_VIABILITY_TIMEOUT_SECONDS", "30"))
REQUEST_RETRY_ATTEMPTS = int(os.environ.get("ASSISTANT_VIABILITY_RETRY_ATTEMPTS", "1"))

INTERNAL_LEAK_MARKERS = (
    "trace_id:",
    "route_reason:",
    "selection_policy",
    "runtime_payload",
    "runtime_state_failure_reason",
    "source_surface:",
    "thread_id:",
    "user_id:",
    "pending_confirm_token",
    "confirm_token",
    "chat_lock",
    "in_flight_total",
)
BUSY_FALLBACK_MARKERS = (
    "still working on your last request",
    "thinking…",
    "thinking...",
    "i’m still here. what should i do next?",
    "i'm still here. what should i do next?",
    "can't reach chat right now",
    "can’t reach chat right now",
    "temporarily busy",
    "ask for status",
)


@dataclass(frozen=True)
class TurnResult:
    user_text: str
    assistant_text: str
    route: str
    ok: bool
    surface: SurfaceName
    trace_id: str
    status: int = 200
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    surface: SurfaceName
    user_turns: tuple[str, ...]
    expected_behavior: str
    allowed_variation: str
    fail_conditions: tuple[str, ...]
    requirements: tuple[str, ...]
    checker: str


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    surface: SurfaceName
    passed: bool
    failure_category: FailureCategory | None
    failure_reason: str | None
    evidence: str
    turns: tuple[TurnResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "surface": self.surface,
            "passed": self.passed,
            "failure_category": self.failure_category,
            "failure_reason": self.failure_reason,
            "evidence": self.evidence,
            "turns": [
                {
                    "user_text": turn.user_text,
                    "assistant_text": turn.assistant_text,
                    "route": turn.route,
                    "ok": turn.ok,
                    "status": turn.status,
                    "surface": turn.surface,
                    "trace_id": turn.trace_id,
                    "error": turn.error,
                }
                for turn in self.turns
            ],
        }


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _normalized(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _text_excerpt(text: str, *, limit: int = 96) -> str:
    excerpt = _first_line(text)
    if len(excerpt) <= limit:
        return excerpt
    return f"{excerpt[: limit - 1].rstrip()}…"


def _response_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    text = str(
        assistant.get("content")
        or payload.get("message")
        or payload.get("text")
        or payload.get("error")
        or meta.get("summary")
        or ""
    ).strip()
    return text


def _response_route(payload: dict[str, Any], default: str = "unknown") -> str:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    route = str(payload.get("route") or meta.get("route") or default).strip().lower()
    return route or default


def _response_ok(payload: dict[str, Any], fallback: bool = True) -> bool:
    if "ok" in payload:
        return bool(payload.get("ok"))
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if "ok" in meta:
        return bool(meta.get("ok"))
    return fallback


def _transport_error_text(exc: Exception) -> str:
    reason = getattr(exc, "reason", None)
    if reason is not None:
        return f"transport error: {reason}"
    return f"transport error: {exc}"


def _http_post_json(url: str, payload: dict[str, Any], *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> tuple[int, dict[str, Any], str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    attempts = max(1, REQUEST_RETRY_ATTEMPTS + 1)
    last_status = 0
    last_body = ""
    last_parsed: dict[str, Any] = {}
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                status = int(getattr(response, "status", 200))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            status = int(getattr(exc, "code", 500))
        except urllib.error.URLError as exc:
            body = _transport_error_text(exc)
            status = 0
        except Exception as exc:  # pragma: no cover - defensive live-run guard
            body = _transport_error_text(exc)
            status = 0
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        last_status = status
        last_body = body
        last_parsed = parsed
        candidate_text = _response_text(parsed) if parsed else body
        should_retry = body.startswith("transport error:") or _has_busy_fallback(candidate_text)
        if status != 0 and not should_retry:
            return status, parsed, body
        if attempt + 1 >= attempts or not should_retry:
            return status, parsed, body
        time.sleep(0.75 * (attempt + 1))
    return last_status, last_parsed, last_body


class WebUISurface:
    surface_name: SurfaceName = "webui"

    def __init__(self, base_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def send(self, *, conversation_id: str, user_text: str, turn_index: int) -> TurnResult:
        trace_id = f"assistant-viability-webui-{conversation_id}-{turn_index}-{int(time.time())}"
        payload = {
            "messages": [{"role": "user", "content": str(user_text or "")}],
            "purpose": "chat",
            "task_type": "chat",
            "source_surface": "webui",
            "user_id": conversation_id,
            "thread_id": f"{conversation_id}:thread",
            "trace_id": trace_id,
        }
        status, response, raw = _http_post_json(f"{self.base_url}/chat", payload, timeout=self.timeout)
        if not response and raw.startswith("transport error:"):
            return TurnResult(
                user_text=user_text,
                assistant_text=raw,
                route="error",
                ok=False,
                surface=self.surface_name,
                trace_id=trace_id,
                status=0,
                error=raw,
            )
        assistant_text = _response_text(response)
        route = _response_route(response)
        ok = _response_ok(response, fallback=status < 400)
        error = None
        if status >= 400:
            error = f"HTTP {status}"
        elif not ok and not assistant_text:
            error = f"HTTP {status}" if status else "request failed"
        return TurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            route=route,
            ok=ok,
            surface=self.surface_name,
            trace_id=trace_id,
            status=status,
            payload=response,
            error=error,
        )


class TelegramSurface:
    surface_name: SurfaceName = "telegram"

    def __init__(self, base_url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _proxy_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, response, raw = _http_post_json(f"{self.base_url}/chat", payload, timeout=self.timeout)
        if not response and raw.startswith("transport error:"):
            return {"ok": False, "error": raw, "_proxy_error": {"kind": "transport", "detail": raw}}
        if status >= 400 and "ok" not in response:
            response = dict(response)
            response.setdefault("ok", False)
        return response if isinstance(response, dict) else {}

    def send(self, *, conversation_id: str, user_text: str, turn_index: int) -> TurnResult:
        trace_id = f"assistant-viability-telegram-{conversation_id}-{turn_index}-{int(time.time())}"
        response = handle_telegram_text(
            text=str(user_text or ""),
            chat_id=str(conversation_id or ""),
            trace_id=trace_id,
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=self._proxy_chat,
        )
        assistant_text = str(response.get("text") or "").strip()
        route = str(response.get("route") or response.get("selected_route") or "unknown").strip().lower() or "unknown"
        ok = _response_ok(response)
        error = None
        if not ok and not assistant_text:
            error = str(response.get("error") or "telegram request failed").strip() or "telegram request failed"
        return TurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            route=route,
            ok=ok,
            surface=self.surface_name,
            trace_id=trace_id,
            status=0 if response.get("_proxy_error") else 200,
            payload=response if isinstance(response, dict) else {},
            error=error,
        )


def _has_internal_leak(text: str) -> bool:
    lowered = _normalized(text)
    if not lowered:
        return False
    if text.lstrip().startswith("{") or text.lstrip().startswith("["):
        return True
    return any(marker in lowered for marker in INTERNAL_LEAK_MARKERS)


def _has_busy_fallback(text: str) -> bool:
    lowered = _normalized(text)
    return any(marker in lowered for marker in BUSY_FALLBACK_MARKERS)


def _common_failure(turns: tuple[TurnResult, ...]) -> tuple[FailureCategory | None, str | None]:
    for turn in turns:
        if turn.error or turn.status >= 400 or not turn.ok or turn.route in {"error", "proxy_error"}:
            detail = turn.error or f"{turn.route or 'request'} failed"
            return "transport/runtime", detail
        if _has_internal_leak(turn.assistant_text):
            return "assistant-behavior", "internal state leaked in assistant text"
        if _has_busy_fallback(turn.assistant_text):
            return "assistant-behavior", "busy fallback or placeholder text surfaced"
    return None, None


def _check_greeting_followup(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 2:
        return False, "assistant-behavior", "expected two turns"
    first = _normalized(turns[0].assistant_text)
    second = _normalized(turns[1].assistant_text)
    if not first or not second:
        return False, "assistant-behavior", "empty assistant reply"
    if first == second:
        return False, "assistant-behavior", "follow-up repeated the same answer"
    if len(second.split()) < 4:
        return False, "assistant-behavior", "follow-up was too short to read as an answer"
    if not any(token in second for token in ("help", "assist", "can", "support", "here")):
        return False, "assistant-behavior", "follow-up did not read as a helpful reply"
    return True, None, None


def _check_status_followup(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 2:
        return False, "grounding/truth", "expected two turns"
    combined = " ".join(_normalized(turn.assistant_text) for turn in turns)
    if not any(token in combined for token in ("runtime", "status", "ready", "model", "health")):
        return False, "grounding/truth", "status reply did not mention runtime or readiness"
    if any(token in combined for token in ("not ready", "degraded", "failed")) and "ready" in combined:
        return False, "grounding/truth", "status reply contradicted itself"
    return True, None, None


def _check_hardware_inspection(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    combined = " ".join(_normalized(turn.assistant_text) for turn in turns)
    if any(token in combined for token in ("ram", "vram", "memory", "gpu", "unavailable", "not available")):
        return True, None, None
    return False, "grounding/truth", "hardware answer did not mention ram, vram, or an explicit unavailable state"


def _check_preview_confirm(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 2:
        return False, "confirmation/action", "expected preview and confirmation turns"
    first = _normalized(turns[0].assistant_text)
    second = _normalized(turns[1].assistant_text)
    if not any(token in first for token in ("confirm", "do you want", "say yes", "cancel", "continue")):
        return False, "confirmation/action", "preview did not ask for confirmation"
    if not second:
        return False, "confirmation/action", "confirmation reply was empty"
    if not any(token in second for token in ("using", "switch", "cancel", "confirm", "model", "created", "unable", "now")):
        return False, "confirmation/action", "confirmation reply did not look like a confirm/cancel outcome"
    return True, None, None


def _check_topic_shift_return(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 3:
        return False, "memory/continuity", "expected three turns"
    final = _normalized(turns[-1].assistant_text)
    if not final:
        return False, "memory/continuity", "return turn was empty"
    if not any(token in final for token in ("back", "return", "resume", "previous", "plan", "topic")):
        return False, "memory/continuity", "return turn did not restore the original topic"
    return True, None, None


def _check_resume_continuity(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 2:
        return False, "memory/continuity", "expected two turns"
    second = _normalized(turns[1].assistant_text)
    if not second:
        return False, "memory/continuity", "resume reply was empty"
    if not any(token in second for token in ("gate", "viability", "testing", "current task", "what we are doing", "working on", "resume")):
        return False, "memory/continuity", "resume reply did not reference the prior task"
    return True, None, None


def _check_memory_persistence_soak(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 5:
        return False, "memory/continuity", "expected five turns"

    topic = "assistant_viability_gate"
    resume_indexes = (1, 3, 4)

    def _turn_mentions_topic(turn: TurnResult) -> bool:
        text = _normalized(turn.assistant_text)
        if topic in text:
            return True
        payload = turn.payload if isinstance(turn.payload, dict) else {}
        setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
        current_topic = _normalized(str(setup.get("current_topic") or ""))
        last_request = _normalized(str(setup.get("last_request") or ""))
        if topic in current_topic or topic in last_request:
            return True
        if "focused on" in text and topic.replace("_", " ") in text:
            return True
        return False

    if not all(_turn_mentions_topic(turns[index]) for index in resume_indexes):
        return False, "memory/continuity", "memory topic was not preserved across the soak"
    if not all(_normalized(turns[index].assistant_text) for index in resume_indexes):
        return False, "memory/continuity", "memory replies were empty"
    return True, None, None


def _check_long_human_like_session(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> tuple[bool, FailureCategory | None, str | None]:
    if len(turns) < 10:
        return False, "assistant-behavior", "expected at least ten turns"
    combined = " ".join(_normalized(turn.assistant_text) for turn in turns)
    required_domains = {
        "runtime": ("runtime", "ready", "model", "health"),
        "memory": ("memory", "remember", "preferences"),
        "files": ("file", "directory", "repo"),
        "skill_packs": ("skill pack", "pack", "abilities", "skills"),
    }
    missing = [label for label, tokens in required_domains.items() if not any(token in combined for token in tokens)]
    if missing:
        return False, "assistant-behavior", f"long session did not cover: {', '.join(missing)}"
    if _normalized(turns[-1].assistant_text) == _normalized(turns[-2].assistant_text):
        return False, "assistant-behavior", "final turn repeated the previous answer"
    if not any(token in _normalized(turns[-1].assistant_text) for token in ("continue", "next", "summary", "work", "task")):
        return False, "assistant-behavior", "final turn did not sound like a useful next step"
    if any(token in combined for token in ("need more context", "i need more context", "can't help", "cannot help")):
        return False, "assistant-behavior", "long session fell back to a dead-end answer"
    return True, None, None


CHECKERS: dict[str, Callable[[ScenarioSpec, tuple[TurnResult, ...]], tuple[bool, FailureCategory | None, str | None]]] = {
    "greeting_followup": _check_greeting_followup,
    "status_followup": _check_status_followup,
    "hardware_inspection": _check_hardware_inspection,
    "preview_confirm": _check_preview_confirm,
    "topic_shift_return": _check_topic_shift_return,
    "resume_continuity": _check_resume_continuity,
    "memory_persistence_soak": _check_memory_persistence_soak,
    "long_human_like_session": _check_long_human_like_session,
}


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        id="greeting_followup_webui",
        surface="webui",
        user_turns=("hi", "what can you help me with right now?"),
        expected_behavior="Warm greeting, then a concrete capability summary without internal mode jargon.",
        allowed_variation="Any friendly greeting and useful capability summary.",
        fail_conditions=("raw JSON, internal-state leak, dead-end fallback, or a repeated answer.",),
        requirements=("plain conversational quality", "continuity"),
        checker="greeting_followup",
    ),
    ScenarioSpec(
        id="runtime_status_followup_webui",
        surface="webui",
        user_turns=("what is the runtime status?", "and are you ready right now?"),
        expected_behavior="Return a status answer first, then keep the follow-up consistent and not contradict readiness.",
        allowed_variation="Status wording, uptime, and model names may vary.",
        fail_conditions=("Contradictory readiness, raw dump, or internal fields in the reply.",),
        requirements=("deterministic truth", "continuity"),
        checker="status_followup",
    ),
    ScenarioSpec(
        id="local_system_inspection_webui",
        surface="webui",
        user_turns=("what do i have for ram and vram right now?",),
        expected_behavior="Inspect local hardware or say clearly that it is unavailable; do not invent a machine profile.",
        allowed_variation="Exact hardware names and numbers may vary; unavailable is acceptable if explicit.",
        fail_conditions=("Invented hardware specifics, raw dump, or internal-state leak.",),
        requirements=("deterministic truth", "tool use"),
        checker="hardware_inspection",
    ),
    ScenarioSpec(
        id="preview_confirm_flow_webui",
        surface="webui",
        user_turns=("switch temporarily to ollama:qwen2.5:7b-instruct", "yes"),
        expected_behavior="Preview the action, request confirmation, then either apply it or explain why it cannot.",
        allowed_variation="Exact model wording and confirmation phrasing may vary.",
        fail_conditions=("Action before confirmation, lost confirmation context, or operator-style fallback.",),
        requirements=("confirmation flow", "deterministic truth"),
        checker="preview_confirm",
    ),
    ScenarioSpec(
        id="interruption_topic_shift_webui",
        surface="webui",
        user_turns=("help me plan my day", "actually, give me a one-line joke", "go back to the day plan"),
        expected_behavior="Handle the topic shift without forgetting the prior thread and return to the original topic on the last turn.",
        allowed_variation="Any short joke and any reasonable transition back.",
        fail_conditions=("The assistant forgets the original task or restarts from scratch.",),
        requirements=("continuity", "plain conversational quality"),
        checker="topic_shift_return",
    ),
    ScenarioSpec(
        id="continuity_resume_webui",
        surface="webui",
        user_turns=("we are testing the assistant viability gate", "what are we doing?"),
        expected_behavior="Summarize the current task or conversation state from the prior turn.",
        allowed_variation="Any concise resume phrasing that refers to the prior task is fine.",
        fail_conditions=("Empty answer, generic restart, or no reference to the prior task.",),
        requirements=("continuity", "memory"),
        checker="resume_continuity",
    ),
    ScenarioSpec(
        id="memory_persistence_soak_webui",
        surface="webui",
        user_turns=(
            "we are testing the assistant viability gate",
            "what are we doing?",
            "what is the runtime status?",
            "what were we working on before?",
            "what are we doing right now?",
        ),
        expected_behavior="Keep the current topic across an unrelated status check and continue to remember it after several turns.",
        allowed_variation="Any concise topic recall that keeps the original topic intact is fine.",
        fail_conditions=("The assistant loses the original topic, restarts from scratch, or returns empty memory replies.",),
        requirements=("continuity", "memory"),
        checker="memory_persistence_soak",
    ),
    ScenarioSpec(
        id="long_human_like_session_webui",
        surface="webui",
        user_turns=(
            "I'm testing whether you can stay coherent through a long chat.",
            "What are we working on right now?",
            "Actually, keep the answer short.",
            "No, go back and explain the larger task.",
            "What is the runtime status?",
            "What do you remember about my preferences?",
            "List the files in this repo.",
            "Read the file /home/c/personal-agent/README.md.",
            "What skill packs can you use for extra abilities?",
            "Okay, now summarize the work in one sentence.",
            "If you had to continue from here, what would you do next?",
        ),
        expected_behavior="Hold a long, mixed-intent conversation without losing context or collapsing into generic fallback answers.",
        allowed_variation="Any coherent multi-turn replies that handle the interruptions and return to a useful next step.",
        fail_conditions=("The assistant loses the thread, repeats itself, or drops into generic fallback behavior.",),
        requirements=("continuity", "memory", "files", "runtime", "assistant role"),
        checker="long_human_like_session",
    ),
    ScenarioSpec(
        id="long_human_like_session_telegram",
        surface="telegram",
        user_turns=(
            "I'm testing whether you can stay coherent through a long chat.",
            "What are we working on right now?",
            "Actually, keep the answer short.",
            "No, go back and explain the larger task.",
            "What is the runtime status?",
            "What do you remember about my preferences?",
            "List the files in this repo.",
            "Read the file /home/c/personal-agent/README.md.",
            "What skill packs can you use for extra abilities?",
            "Okay, now summarize the work in one sentence.",
            "If you had to continue from here, what would you do next?",
        ),
        expected_behavior="Hold the same long, mixed-intent conversation through the Telegram bridge without losing thread continuity or assistant voice.",
        allowed_variation="Any coherent multi-turn replies that handle interruptions and return to a useful next step.",
        fail_conditions=("The bridge leaks proxy state, the assistant loses the thread, or it collapses into generic fallback behavior.",),
        requirements=("continuity", "memory", "files", "runtime", "assistant role"),
        checker="long_human_like_session",
    ),
    ScenarioSpec(
        id="telegram_surface_behavior",
        surface="telegram",
        user_turns=("hello", "what can you help me with right now?"),
        expected_behavior="Same natural two-turn conversation should work through the Telegram bridge without leaking proxy state.",
        allowed_variation="Any friendly greeting and helpful follow-up.",
        fail_conditions=("Bridge/internal leakage, transport failure, or a busy fallback.",),
        requirements=("plain conversational quality", "continuity"),
        checker="greeting_followup",
    ),
    ScenarioSpec(
        id="webui_surface_behavior",
        surface="webui",
        user_turns=("hello", "what can you help me with right now?"),
        expected_behavior="Same natural two-turn conversation should work through the web UI surface with the same thread preserved.",
        allowed_variation="Any friendly greeting and helpful follow-up.",
        fail_conditions=("Transport failure, internal leakage, or lost thread continuity.",),
        requirements=("plain conversational quality", "continuity"),
        checker="greeting_followup",
    ),
)

MINIMUM_VIABILITY_GATE: tuple[str, ...] = (
    "greeting_followup_webui",
    "runtime_status_followup_webui",
    "continuity_resume_webui",
)


def _select_scenarios(*, surface: str, scenario_ids: set[str] | None) -> tuple[ScenarioSpec, ...]:
    selected = []
    for spec in SCENARIOS:
        if surface != "all" and spec.surface != surface:
            continue
        if scenario_ids is not None and spec.id not in scenario_ids:
            continue
        selected.append(spec)
    return tuple(selected)


def evaluate_scenario(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> ScenarioResult:
    common_category, common_reason = _common_failure(turns)
    if common_category is not None:
        return ScenarioResult(
            scenario_id=spec.id,
            surface=spec.surface,
            passed=False,
            failure_category=common_category,
            failure_reason=common_reason,
            evidence=_format_evidence(spec, turns),
            turns=turns,
        )

    checker = CHECKERS[spec.checker]
    passed, category, reason = checker(spec, turns)
    return ScenarioResult(
        scenario_id=spec.id,
        surface=spec.surface,
        passed=bool(passed),
        failure_category=None if passed else (category or "assistant-behavior"),
        failure_reason=None if passed else (reason or "scenario failed"),
        evidence=_format_evidence(spec, turns),
        turns=turns,
    )


def run_scenario(spec: ScenarioSpec, driver: Any) -> ScenarioResult:
    conversation_id = f"assistant-viability-{spec.id}"
    if spec.id.startswith("long_human_like_session_"):
        warmup_conversation_id = f"{conversation_id}:warmup"
        warmup_ready = False
        for warmup_index in range(10):
            try:
                warmup_turn = driver.send(
                    conversation_id=warmup_conversation_id,
                    user_text="hello",
                    turn_index=-1,
                )
                warmup_ready = bool(warmup_turn.ok) and not _has_busy_fallback(warmup_turn.assistant_text)
                if warmup_ready:
                    break
            except Exception:
                warmup_ready = False
            time.sleep(3)
        if not warmup_ready:
            time.sleep(5)
    turn_results: list[TurnResult] = []
    for index, user_text in enumerate(spec.user_turns):
        turn = driver.send(conversation_id=conversation_id, user_text=user_text, turn_index=index)
        turn_results.append(turn)
    return evaluate_scenario(spec, tuple(turn_results))


def _format_evidence(spec: ScenarioSpec, turns: tuple[TurnResult, ...]) -> str:
    parts = []
    for index, turn in enumerate(turns, start=1):
        parts.append(
            f"{index}:{_text_excerpt(turn.user_text)} -> {_text_excerpt(turn.assistant_text)} "
            f"[route={turn.route or 'unknown'} ok={turn.ok} status={turn.status}"
            f"{' error=' + _text_excerpt(turn.error, limit=64) if turn.error else ''}]"
        )
    return " | ".join(parts)


def _print_catalog() -> None:
    print("Assistant viability catalog", flush=True)
    print(f"Minimum gate: {', '.join(MINIMUM_VIABILITY_GATE)}", flush=True)
    for spec in SCENARIOS:
        print(
            f"- {spec.id} [{spec.surface}] turns={len(spec.user_turns)} requirements={', '.join(spec.requirements)}",
            flush=True,
        )
        print(f"  turns: {' -> '.join(spec.user_turns)}", flush=True)
        print(f"  expected: {spec.expected_behavior}", flush=True)
        print(f"  allowed variation: {spec.allowed_variation}", flush=True)
        print(f"  fail conditions: {' '.join(spec.fail_conditions)}", flush=True)


def _build_driver(surface: SurfaceName, base_url: str, *, timeout: float) -> Any:
    if surface == "telegram":
        return TelegramSurface(base_url, timeout=timeout)
    return WebUISurface(base_url, timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    global REQUEST_RETRY_ATTEMPTS
    parser = argparse.ArgumentParser(
        description="Run a focused assistant-viability smoke suite against real conversation surfaces."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument(
        "--surface",
        choices=["all", "webui", "telegram"],
        default="all",
        help="Limit execution to one surface family.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run only the selected scenario id(s). May be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report instead of human-readable lines.")
    parser.add_argument("--list", action="store_true", help="Print the scenario catalog and exit.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=REQUEST_RETRY_ATTEMPTS,
        help="Number of transport retries per turn.",
    )
    args = parser.parse_args(argv)
    REQUEST_RETRY_ATTEMPTS = max(0, int(args.retry_attempts))

    if bool(args.list):
        _print_catalog()
        return 0

    selected_ids = set(str(item).strip() for item in args.scenario if str(item).strip()) or None
    selected = _select_scenarios(surface=str(args.surface), scenario_ids=selected_ids)
    if not selected:
        print("No scenarios selected.", flush=True)
        return 1

    results: list[ScenarioResult] = []
    for spec in selected:
        driver = _build_driver(spec.surface, str(args.base_url), timeout=float(args.timeout))
        result = run_scenario(spec, driver)
        results.append(result)

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    if bool(args.json):
        print(
            json.dumps(
                {
                    "summary": {"passed": passed, "failed": failed, "total": len(results)},
                    "results": [result.to_dict() for result in results],
                },
                ensure_ascii=True,
                indent=2,
            ),
            flush=True,
        )
    else:
        print(f"assistant_viability summary: passed={passed} failed={failed} total={len(results)}", flush=True)
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            category = f" category={result.failure_category}" if result.failure_category else ""
            reason = f" reason={result.failure_reason}" if result.failure_reason else ""
            print(f"{status} {result.scenario_id} [{result.surface}]{category}{reason}", flush=True)
            print(f"  evidence: {result.evidence}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
