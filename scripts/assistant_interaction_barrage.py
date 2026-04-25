#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.telegram_bridge import handle_telegram_text


SurfaceName = Literal["webui", "telegram"]
DEFAULT_BASE_URL = os.environ.get("AGENT_API_BASE_URL") or "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("ASSISTANT_BARRAGE_TIMEOUT_SECONDS", "30"))

INTERNAL_LEAK_MARKERS = (
    "trace_id:",
    "route_reason:",
    "selection_policy",
    "runtime_payload",
    "runtime_state_failure_reason",
    "source_surface:",
    "thread_id:",
    "user_id:",
    "confirm_token",
    "in_flight_total",
)
BAD_PERSONA_MARKERS = (
    "i am an ai language model",
    "as an ai language model",
    "i do not have physical form",
    "i don't have physical form",
    "i do not have sensory",
    "i don't have sensory",
    "unable to access external information",
    "cannot access external information",
    "temporarily busy",
    "what do you want me to do right now",
    "do you mean:",
)


@dataclass(frozen=True)
class TurnResult:
    user_text: str
    assistant_text: str
    route: str
    ok: bool
    surface: SurfaceName
    status: int
    error: str | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    surface: SurfaceName
    turns: tuple[str, ...]
    checker: str
    notes: str


def _normalized(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _first_line(text: str | None) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _response_text(payload: dict[str, Any]) -> str:
    assistant = payload.get("assistant") if isinstance(payload.get("assistant"), dict) else {}
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return str(
        assistant.get("content")
        or payload.get("message")
        or payload.get("text")
        or payload.get("error")
        or meta.get("summary")
        or ""
    ).strip()


def _response_route(payload: dict[str, Any]) -> str:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return str(payload.get("route") or meta.get("route") or "unknown").strip().lower() or "unknown"


def _transport_error_text(exc: Exception) -> str:
    reason = getattr(exc, "reason", None)
    if reason is not None:
        return f"transport error: {reason}"
    return f"transport error: {exc}"


def _http_post_json(url: str, payload: dict[str, Any], *, timeout: float) -> tuple[int, dict[str, Any], str]:
    try:
        completed = subprocess.run(
            [
                "curl",
                "-sS",
                "-X",
                "POST",
                url,
                "-H",
                "Content-Type: application/json",
                "-H",
                "Accept: application/json",
                "--data",
                json.dumps(payload, ensure_ascii=True),
                "--max-time",
                str(max(1, int(timeout))),
                "-w",
                "\n%{http_code}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return 0, {}, _transport_error_text(exc)
    if completed.returncode != 0:
        error_text = (completed.stderr or completed.stdout or "").strip()
        return 0, {}, error_text or "transport error: curl failed"
    stdout = str(completed.stdout or "")
    if "\n" in stdout:
        raw, status_text = stdout.rsplit("\n", 1)
    else:
        raw, status_text = stdout, "0"
    try:
        status = int(status_text.strip() or "0")
    except ValueError:
        status = 0
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return status, parsed, raw


class WebUISurface:
    surface_name: SurfaceName = "webui"

    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def send(self, *, conversation_id: str, turn_index: int, user_text: str) -> TurnResult:
        payload = {
            "messages": [{"role": "user", "content": str(user_text or "")}],
            "purpose": "chat",
            "task_type": "chat",
            "source_surface": "webui",
            "user_id": conversation_id,
            "thread_id": f"{conversation_id}:thread",
            "trace_id": f"assistant-barrage-{conversation_id}-{turn_index}-{int(time.time())}",
        }
        status, response, raw = _http_post_json(f"{self.base_url}/chat", payload, timeout=self.timeout)
        if not response and raw.startswith("transport error:"):
            return TurnResult(user_text=user_text, assistant_text=raw, route="error", ok=False, surface=self.surface_name, status=0, error=raw)
        text = _response_text(response)
        return TurnResult(
            user_text=user_text,
            assistant_text=text,
            route=_response_route(response),
            ok=bool(response.get("ok", status < 400)),
            surface=self.surface_name,
            status=status,
            error=None if status < 400 else f"HTTP {status}",
        )


class TelegramSurface:
    surface_name: SurfaceName = "telegram"

    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _proxy_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, response, raw = _http_post_json(f"{self.base_url}/chat", payload, timeout=self.timeout)
        if not response and raw.startswith("transport error:"):
            return {"ok": False, "error": raw, "_proxy_error": {"kind": "transport", "detail": raw}}
        if status >= 400 and "ok" not in response:
            response = dict(response)
            response["ok"] = False
        return response

    def send(self, *, conversation_id: str, turn_index: int, user_text: str) -> TurnResult:
        response = handle_telegram_text(
            text=str(user_text or ""),
            chat_id=conversation_id,
            trace_id=f"assistant-barrage-{conversation_id}-{turn_index}-{int(time.time())}",
            runtime=None,
            orchestrator=None,
            fetch_local_api_chat_json=self._proxy_chat,
        )
        text = str(response.get("text") or "").strip()
        return TurnResult(
            user_text=user_text,
            assistant_text=text,
            route=str(response.get("route") or response.get("selected_route") or "unknown").strip().lower() or "unknown",
            ok=bool(response.get("ok", True)),
            surface=self.surface_name,
            status=200 if not response.get("_proxy_error") else 0,
            error=str(response.get("error") or "").strip() or None,
        )


def _common_warnings(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings: list[str] = []
    for turn in turns:
        lowered = _normalized(turn.assistant_text)
        if not turn.ok or turn.status >= 400 or turn.route in {"error", "chat_proxy_error"}:
            warnings.append(f"transport/runtime failure on `{turn.user_text}`")
        if not lowered:
            warnings.append(f"empty assistant reply on `{turn.user_text}`")
        if turn.assistant_text.lstrip().startswith("{") or turn.assistant_text.lstrip().startswith("["):
            warnings.append(f"raw structured output on `{turn.user_text}`")
        if any(marker in lowered for marker in INTERNAL_LEAK_MARKERS):
            warnings.append(f"internal leak on `{turn.user_text}`")
        if any(marker in lowered for marker in BAD_PERSONA_MARKERS):
            warnings.append(f"bad persona marker on `{turn.user_text}`")
    return warnings


def _check_presence_helpful(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(token in text for token in ("help", "ready", "here", "can", "got it", "what should i do next", "you’re welcome", "you're welcome")):
        warnings.append("presence/greeting reply did not sound helpful")
    return warnings


def _check_general_knowledge(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if "general knowledge" in text and "source-backed check" in text:
        return warnings
    if not any(token in text for token in ("blue", "jay", "bird")):
        warnings.append("general knowledge reply did not answer the subject directly")
    return warnings


def _check_runtime_truth(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = " ".join(_normalized(turn.assistant_text) for turn in turns)
    if not any(token in text for token in ("runtime", "ready", "model", "health", "status")):
        warnings.append("runtime reply did not mention runtime state")
    return warnings


def _check_model_truth(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(token in text for token in ("model", "using", "provider", "ollama", "openrouter", "openai")):
        warnings.append("model reply did not mention the active model or provider")
    return warnings


def _check_hardware_truth(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(token in text for token in ("ram", "vram", "memory", "gpu", "unavailable")):
        warnings.append("hardware reply did not mention ram/vram or an unavailable state")
    return warnings


def _check_memory_or_resume(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(
        token in text
        for token in (
            "remember",
            "memory",
            "preferences",
            "working on",
            "resume",
            "focused",
            "last concrete request",
            "last thing you asked me to do",
            "pick up from that context",
            "pick up from there",
            "i can pick up",
            "we were working on",
        )
    ):
        warnings.append("memory/resume reply did not reference memory or prior task")
    return warnings


def _check_capability(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(token in text for token in ("voice", "pack", "install", "skill", "coding", "camera", "avatar", "text")):
        warnings.append("capability reply did not mention a concrete capability or fallback")
    if text.count("i can help with the text side") > 1 or text.count("i can help in text") > 1:
        warnings.append("capability reply repeated its partial-help intro")
    if "install preview" in text or "best fit for this machine" in text:
        warnings.append("capability reply still uses stiff install-preview wording")
    return warnings


def _check_thinking_help(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(token in text for token in ("goal", "messy", "constraint", "break it down", "simply")):
        warnings.append("thinking-help reply did not offer a simple next-step framing")
    if any(token in text for token in ("1.", "2.", "3.", "questions to get us started", "structured and straightforward")):
        warnings.append("thinking-help reply drifted into canned coaching structure")
    return warnings


def _check_custom_helper(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not any(token in text for token in ("ready made helper", "ready made", "studio", "helper")):
        warnings.append("custom-helper reply did not explain the missing helper clearly")
    if "simplest way to add it" not in text:
        warnings.append("custom-helper reply did not use the simpler proposal framing")
    if "sketch that with you" not in text:
        warnings.append("custom-helper reply did not offer a collaborative next step")
    if "make an assistant that" in text or "help me sketch an assistant that" in text:
        warnings.append("custom-helper reply echoed prompt scaffolding instead of a clean task summary")
    return warnings


def _check_assistant_agent_boundary(turns: tuple[TurnResult, ...]) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if not all(token in text for token in ("assistant", "agent layer")):
        warnings.append("assistant/agent boundary reply did not name both layers")
    if not any(token in text for token in ("grounded work", "bounded facts", "action results", "runtime status")):
        warnings.append("assistant/agent boundary reply did not describe bounded agent work")
    if any(token in text for token in ("can't read a clean runtime status", "runtime state unavailable", "do you mean")):
        warnings.append("assistant/agent boundary reply fell into runtime fallback or chooser UX")
    return warnings


def _check_mixed_session_soak(turns: tuple[TurnResult, ...], *, expected_file_token: str = "") -> list[str]:
    warnings = _common_warnings(turns)
    if len(turns) < 8:
        warnings.append("mixed-session soak expected at least eight turns")
        return warnings
    combined = " ".join(_normalized(turn.assistant_text) for turn in turns)
    required_domains = {
        "runtime": ("runtime", "ready", "model", "health", "status"),
        "memory": ("remember", "preferences", "working on", "pick up from there", "we were working on"),
        "files": ("text from", "file", _normalized(expected_file_token)),
        "capability": ("voice", "install", "helper", "skill", "pack"),
        "continuity": ("plan", "day plan", "back", "next"),
    }
    missing = [
        label
        for label, tokens in required_domains.items()
        if not any(token and token in combined for token in tokens)
    ]
    if missing:
        warnings.append(f"mixed-session soak missed domains: {', '.join(missing)}")
    if _normalized(turns[-1].assistant_text) == _normalized(turns[-2].assistant_text):
        warnings.append("mixed-session soak ended with a repeated answer")
    final = _normalized(turns[-1].assistant_text)
    if not any(token in final for token in ("next", "continue", "pick up", "plan", "work")):
        warnings.append("mixed-session soak final turn did not sound like a useful next step")
    if any(token in combined for token in ("need more context", "i need more context", "can't help", "cannot help")):
        warnings.append("mixed-session soak fell back to a dead-end answer")
    return warnings


def _check_file_answer(turns: tuple[TurnResult, ...], expected_token: str) -> list[str]:
    warnings = _common_warnings(turns)
    text = _normalized(turns[-1].assistant_text)
    if _normalized(expected_token) not in text:
        warnings.append(f"file reply missed expected token `{expected_token}`")
    return warnings


CHECKERS: dict[str, Callable[..., list[str]]] = {
    "presence_helpful": _check_presence_helpful,
    "general_knowledge": _check_general_knowledge,
    "runtime_truth": _check_runtime_truth,
    "model_truth": _check_model_truth,
    "hardware_truth": _check_hardware_truth,
    "memory_or_resume": _check_memory_or_resume,
    "capability": _check_capability,
    "thinking_help": _check_thinking_help,
    "custom_helper": _check_custom_helper,
    "assistant_agent_boundary": _check_assistant_agent_boundary,
    "mixed_session_soak": _check_mixed_session_soak,
}


def _base_scenarios() -> list[Scenario]:
    return [
        Scenario("air-001-webui-greeting", "webui", ("hi", "what can you help me with right now?"), "presence_helpful", "Warm greeting and useful follow-up."),
        Scenario("air-002-telegram-greeting", "telegram", ("hi", "what can you help me with right now?"), "presence_helpful", "Warm greeting and useful follow-up."),
        Scenario("air-003-webui-typo-greeting", "webui", ("herllo",), "presence_helpful", "Typo greeting should still read as assistant small talk."),
        Scenario("air-004-telegram-presence", "telegram", ("Are you really there?",), "presence_helpful", "Presence check should stay helpful."),
        Scenario("air-005-webui-say-hi", "webui", ("say hi",), "presence_helpful", "Terse social ask should not trigger chooser UX."),
        Scenario("air-006-telegram-say-hi", "telegram", ("say hi",), "presence_helpful", "Terse social ask should not trigger chooser UX."),
        Scenario("air-007-webui-bluejay", "webui", ("What colour is a bluejay?",), "general_knowledge", "Ordinary factual question should not leak model disclaimers."),
        Scenario("air-008-telegram-bluejay", "telegram", ("What colour is a bluejay?",), "general_knowledge", "Telegram parity for ordinary factual question."),
        Scenario("air-009-webui-runtime", "webui", ("what is the runtime status?", "and are you ready right now?"), "runtime_truth", "Runtime truth stays coherent across follow-up."),
        Scenario("air-010-telegram-runtime", "telegram", ("what is the runtime status?", "and are you ready right now?"), "runtime_truth", "Telegram runtime truth parity."),
        Scenario("air-011-webui-hardware", "webui", ("what do i have for ram and vram right now?",), "hardware_truth", "Local hardware truth grounded in runtime/tool state."),
        Scenario("air-012-telegram-hardware", "telegram", ("what do i have for ram and vram right now?",), "hardware_truth", "Telegram hardware truth parity."),
        Scenario("air-013-webui-memory", "webui", ("what do you remember about my preferences?",), "memory_or_resume", "Memory answer should be useful and not fabricated."),
        Scenario("air-014-webui-resume", "webui", ("we are testing the barrage hardening task", "what are we doing?"), "memory_or_resume", "Resume should reflect current task."),
        Scenario("air-015-telegram-resume", "telegram", ("help me plan my day", "continue from here"), "memory_or_resume", "Telegram resume continuity through the proxy-safe chat path."),
        Scenario("air-016-webui-topic-shift", "webui", ("help me plan my day", "actually tell me a one-line joke", "go back to the day plan"), "memory_or_resume", "Return to original topic after interruption."),
        Scenario("air-017-telegram-topic-shift", "telegram", ("help me plan my day", "actually tell me a one-line joke", "go back to the day plan"), "memory_or_resume", "Telegram return to original topic after interruption."),
        Scenario("air-018-webui-capability-voice", "webui", ("Talk to me out loud",), "capability", "Capability routing should respond sanely."),
        Scenario("air-019-telegram-capability-voice", "telegram", ("Talk to me out loud",), "capability", "Telegram capability routing should respond sanely."),
        Scenario("air-020-webui-capability-code", "webui", ("Help me code",), "capability", "Coding capability ask should not be dropped."),
        Scenario("air-021-telegram-capability-code", "telegram", ("Help me code",), "capability", "Telegram coding capability ask should not be dropped."),
        Scenario("air-022-webui-politeness", "webui", ("thanks", "ok"), "presence_helpful", "Acknowledgements should stay polite and simple."),
        Scenario("air-023-telegram-politeness", "telegram", ("thanks", "ok"), "presence_helpful", "Telegram acknowledgements should stay polite and simple."),
        Scenario("air-024-webui-rewind", "webui", ("help me plan my day", "what are we doing?"), "memory_or_resume", "Working-context rewind should sound natural."),
        Scenario("air-025-telegram-correction", "telegram", ("help me plan my day", "no, go back and explain the larger task"), "memory_or_resume", "Correction-style rewind should keep the same assistant voice."),
        Scenario("air-026-webui-typo-model", "webui", ("what model are you uding",), "model_truth", "Typo-heavy model query should still route to grounded model status."),
        Scenario("air-027-telegram-typo-runtime", "telegram", ("r u heathy right now",), "runtime_truth", "Typo-heavy runtime query should not leak generic model boilerplate."),
        Scenario("air-028-webui-typo-memory", "webui", ("help me plan my day", "go bak to the day plan", "what shoud we do next"), "memory_or_resume", "Typo-heavy continuity prompts should stay on the working-context path."),
        Scenario("air-049-webui-provider-status", "webui", ("What providers are set up right now?",), "model_truth", "Natural provider-status wording should route to runtime truth without LLM fallback."),
        Scenario("air-050-telegram-provider-status", "telegram", ("What providers are set up right now?",), "model_truth", "Telegram natural provider-status wording should not hit the busy fallback."),
        Scenario("air-042-webui-summary-recap", "webui", ("help me plan my day", "summarize where we left this in one sentence"), "memory_or_resume", "Summary recap should stay natural and not sound like a log replay."),
        Scenario("air-043-telegram-next-step", "telegram", ("help me plan my day", "if you had to continue from here, what would you do next?"), "memory_or_resume", "Next-step recap should sound like one assistant, not a control-plane summary."),
        Scenario("air-044-webui-natural-role-summary", "webui", ("say what you do in one sentence, but keep it natural",), "capability", "Natural role-summary ask should not fall into confused generic chat."),
        Scenario("air-045-webui-thinking-help", "webui", ("i need help thinking through something messy, but keep it simple",), "thinking_help", "Open-ended collaboration asks should sound natural and brief."),
        Scenario("air-046-webui-custom-helper-proposal", "webui", ("make an assistant that coordinates my studio light cues with my music cues during live shows",), "custom_helper", "Custom helper proposals should sound collaborative instead of template-like."),
        Scenario("air-047-webui-assistant-agent-boundary", "webui", ("what are you and what is the agent layer supposed to do?",), "assistant_agent_boundary", "Assistant/agent-layer boundary should be explained by the assistant, not runtime fallback."),
        Scenario("air-048-telegram-assistant-agent-boundary", "telegram", ("what are you and what is the agent layer supposed to do?",), "assistant_agent_boundary", "Telegram parity for assistant/agent-layer boundary explanation."),
    ]


def _file_scenarios() -> tuple[list[Scenario], str]:
    tmpdir = tempfile.mkdtemp(prefix="assistant-barrage-", dir=str(REPO_ROOT))
    root = Path(tmpdir)
    note = root / "note.txt"
    token = f"assistant-barrage-token-{int(time.time())}"
    note.write_text(f"{token}\nThis canary file exists for barrage checks.\n", encoding="utf-8")
    scenarios = [
        Scenario("air-029-webui-file-read", "webui", (f"read the file {note}",), "file_read", "WebUI should read a real canary file."),
        Scenario("air-030-telegram-file-read", "telegram", (f"read the file {note}",), "file_read", "Telegram should proxy a real canary file read."),
        Scenario(
            "air-031-webui-mixed-session",
            "webui",
            (
                "help me plan my day",
                "what is the runtime status?",
                "what do you remember about my preferences?",
                f"read the file {note}",
                "talk to me out loud",
                "go back to the day plan",
                "what are we doing?",
                "what should we do next?",
            ),
            "mixed_session_soak",
            "WebUI mixed session should stay coherent across runtime, memory, files, capability, and continuity turns.",
        ),
        Scenario(
            "air-032-telegram-mixed-session",
            "telegram",
            (
                "help me plan my day",
                "what is the runtime status?",
                "what do you remember about my preferences?",
                f"read the file {note}",
                "talk to me out loud",
                "go back to the day plan",
                "what are we doing?",
                "what should we do next?",
            ),
            "mixed_session_soak",
            "Telegram mixed session should keep the same assistant voice across the proxy path.",
        ),
    ]
    return scenarios, token


def _run_scenario(
    scenario: Scenario,
    *,
    webui: WebUISurface,
    telegram: TelegramSurface,
    file_token: str | None = None,
) -> dict[str, Any]:
    surface = webui if scenario.surface == "webui" else telegram
    conversation_id = f"assistant-barrage-{scenario.id}"
    turns: list[TurnResult] = []
    for index, user_text in enumerate(scenario.turns, start=1):
        turns.append(surface.send(conversation_id=conversation_id, turn_index=index, user_text=user_text))
    turn_tuple = tuple(turns)
    if scenario.checker == "file_read":
        warnings = _check_file_answer(turn_tuple, expected_token=file_token or "")
    elif scenario.checker == "mixed_session_soak":
        warnings = _check_mixed_session_soak(turn_tuple, expected_file_token=file_token or "")
    else:
        warnings = CHECKERS[scenario.checker](turn_tuple)
    return {
        "id": scenario.id,
        "surface": scenario.surface,
        "passed": not warnings,
        "warnings": warnings,
        "notes": scenario.notes,
        "turns": [
            {
                "user_text": turn.user_text,
                "assistant_text": turn.assistant_text,
                "first_line": _first_line(turn.assistant_text),
                "route": turn.route,
                "ok": turn.ok,
                "status": turn.status,
                "error": turn.error,
            }
            for turn in turn_tuple
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a broad user-facing assistant interaction barrage against the live runtime.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the live API server.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout.")
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    args = parser.parse_args(argv)

    webui = WebUISurface(str(args.base_url), timeout=float(args.timeout))
    telegram = TelegramSurface(str(args.base_url), timeout=float(args.timeout))
    scenarios = _base_scenarios()
    file_scenarios, token = _file_scenarios()
    scenarios.extend(file_scenarios)

    results = [_run_scenario(s, webui=webui, telegram=telegram, file_token=token) for s in scenarios]
    failures = [row for row in results if not bool(row.get("passed"))]

    print("Assistant interaction barrage", flush=True)
    for row in results:
        status = "PASS" if bool(row.get("passed")) else "FAIL"
        first_line = str(((row.get("turns") or [{}])[-1] or {}).get("first_line") or "")
        warnings = ", ".join(str(item) for item in (row.get("warnings") or [])) or "none"
        print(f"[{status}] {row['id']} ({row['surface']})", flush=True)
        print(f"  first_line: {first_line}", flush=True)
        print(f"  warnings: {warnings}", flush=True)

    print(f"assistant_interaction_barrage summary: passed={len(results) - len(failures)} failed={len(failures)} total={len(results)}", flush=True)
    if bool(args.json):
        print(json.dumps({"results": results}, ensure_ascii=True, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
