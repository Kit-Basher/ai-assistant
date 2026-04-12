from __future__ import annotations

import re
from typing import Any

from agent.cards import normalize_card

_INTENTS = {
    "OBSERVE_PC",
    "EXPLAIN_PREVIOUS",
    "MEMORY_WRITE_REQUEST",
    "PLAN_DAY",
    "SHOW_PREFERENCES",
    "MEMORY_INSPECT",
    "OPEN_LOOP_ADD",
    "OPEN_LOOP_DONE",
    "OPEN_LOOPS_LIST",
    "DAILY_BRIEF_STATUS",
    "CHITCHAT",
    "UNKNOWN",
}

_OBSERVE_DISK = re.compile(r"\b(disk|storage|ssd|space|filesystem|mount|directory|folder)\b", re.IGNORECASE)
_OBSERVE_RESOURCE = re.compile(r"\b(cpu|load|memory|ram|process|proc|performance)\b", re.IGNORECASE)
_OBSERVE_MACHINE = re.compile(
    r"\b(gpu|graphics|video card|hardware|pc stats|pc specs|pc info|computer stats|computer specs|computer info|machine stats|machine specs|machine info|system stats|system info|system details)\b",
    re.IGNORECASE,
)
_OBSERVE_DEEPER = re.compile(
    r"\b(show me more system info|tell me more about (?:this )?(?:machine|computer|system)|"
    r"inspect the system more deeply|inspect the machine more deeply|inspect the computer more deeply|"
    r"what else can you find|dig deeper|dig deeper into (?:my|the) system|"
    r"run a check and see if you can learn more|run a check and see what else you can find|"
    r"run a system check|check the system|scan the system|inspect the machine more|learn more)\b",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(r"\b(explain|why|what changed|change[d]?|previous|last time|since)\b", re.IGNORECASE)
_MEMORY_WRITE = re.compile(
    r"^(remember that|from now on|remember this|set max cards to|turn confidence (on|off)|default compare (on|off)|daily brief (on|off)|set disk delta threshold to|only send if service unhealthy|set open loops due window to)\b",
    re.IGNORECASE,
)
_PLAN_DAY = re.compile(
    r"\b(plan my day|today plan|help me plan my day|help me plan today|what should i do today|what should i work on today|show quick wins|show top 3 priorities)\b",
    re.IGNORECASE,
)
_SHOW_PREFS = re.compile(r"\b(show my preferences|what are my preferences|preferences)\b", re.IGNORECASE)
_MEMORY_INSPECT = re.compile(
    r"\b("
    r"what do you remember about me|"
    r"what do you remember|"
    r"what are we working on|"
    r"what were we working on|"
    r"what were we doing before|"
    r"can you help with the thing we were doing before|"
    r"what do you know about my system|"
    r"what do you know about this system|"
    r"what do you know about my machine|"
    r"what do you know about this machine|"
    r"what is currently in your memory files|"
    r"what is in your memory files|"
    r"what s in your memory files"
    r")\b",
    re.IGNORECASE,
)
_OPEN_LOOP_ADD = re.compile(r"^remember that (?:i need to )?.+ by .+$", re.IGNORECASE)
_OPEN_LOOP_DONE = re.compile(r"^mark .+ done$", re.IGNORECASE)
_OPEN_LOOPS_LIST = re.compile(r"\b(open loops|show open loops|list open loops)\b", re.IGNORECASE)
_DAILY_BRIEF_STATUS = re.compile(
    r"\b(daily brief status|why didn.?t .*daily brief|why did(n't| not) .*daily brief)\b",
    re.IGNORECASE,
)
_CHITCHAT = re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.IGNORECASE)
_HARDWARE_INVENTORY_PHRASES = (
    "what do i have for ram and vram right now",
    "how much ram and vram do i have",
    "what cpu do i have",
    "what gpu do i have",
    "what cpu and gpu do i have",
    "can you tell what cpu and gpu i have",
    "can you see the gpu",
    "what hardware am i running",
    "what are my pc specs",
    "what are my computer specs",
    "what specs do i have",
    "what hardware do i have",
)
_RAM_VRAM_REQUEST_RE = re.compile(r"\bram\b.*\bvram\b|\bvram\b.*\bram\b", re.IGNORECASE)
_MACHINE_BROAD_PHRASES = (
    "what other pc stats can you find",
    "other pc stats",
    "show me more system info",
    "tell me more about this machine",
    "tell me more about this computer",
    "tell me more about this system",
    "inspect the system more deeply",
    "inspect the machine more deeply",
    "inspect the computer more deeply",
    "what else can you find",
    "dig deeper",
    "can you dig deeper into my system",
    "dig deeper into my system",
    "run a check and see if you can learn more",
    "run a check and see what else you can find",
    "run a system check",
    "check the system",
    "scan the system",
    "learn more about my computer",
    "learn more about this machine",
    "learn more about this system",
)


def _contains_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _hardware_inventory_requested(text: str) -> bool:
    return _contains_any_phrase(text, _HARDWARE_INVENTORY_PHRASES) or bool(_RAM_VRAM_REQUEST_RE.search(text))


def _machine_broad_requested(text: str) -> bool:
    return _contains_any_phrase(text, _MACHINE_BROAD_PHRASES) or bool(_OBSERVE_DEEPER.search(text))


def _append_skill(selected: list[dict[str, str]], skill: str, function: str) -> None:
    entry = {"skill": skill, "function": function}
    if entry not in selected:
        selected.append(entry)


def classify_free_text(text: str) -> str:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    if not cleaned:
        return "UNKNOWN"
    if _OPEN_LOOP_ADD.search(lowered):
        return "OPEN_LOOP_ADD"
    if _OPEN_LOOP_DONE.search(lowered):
        return "OPEN_LOOP_DONE"
    if _OPEN_LOOPS_LIST.search(lowered):
        return "OPEN_LOOPS_LIST"
    if _DAILY_BRIEF_STATUS.search(lowered):
        return "DAILY_BRIEF_STATUS"
    if _MEMORY_WRITE.search(lowered):
        return "MEMORY_WRITE_REQUEST"
    if _PLAN_DAY.search(lowered):
        return "PLAN_DAY"
    if _SHOW_PREFS.search(lowered):
        return "SHOW_PREFERENCES"
    if _MEMORY_INSPECT.search(lowered):
        return "MEMORY_INSPECT"
    if _CHITCHAT.search(lowered):
        return "CHITCHAT"
    if _EXPLAIN.search(lowered) and (
        _OBSERVE_DISK.search(lowered)
        or _OBSERVE_RESOURCE.search(lowered)
        or _OBSERVE_MACHINE.search(lowered)
        or _hardware_inventory_requested(lowered)
        or _machine_broad_requested(lowered)
    ):
        return "EXPLAIN_PREVIOUS"
    if (
        _OBSERVE_DISK.search(lowered)
        or _OBSERVE_RESOURCE.search(lowered)
        or _OBSERVE_MACHINE.search(lowered)
        or _OBSERVE_DEEPER.search(lowered)
        or _hardware_inventory_requested(lowered)
        or _machine_broad_requested(lowered)
        or "my pc" in lowered
        or "my computer" in lowered
    ):
        return "OBSERVE_PC"
    return "UNKNOWN"


def select_observe_skills(text: str) -> list[dict[str, str]]:
    lowered = (text or "").lower()
    selected: list[dict[str, str]] = []
    hardware_requested = _hardware_inventory_requested(lowered)
    machine_broad = _machine_broad_requested(lowered)
    machine_requested = bool(
        _OBSERVE_MACHINE.search(lowered)
        or hardware_requested
        or machine_broad
        or "my pc" in lowered
        or "my computer" in lowered
        or "this machine" in lowered
        or "this system" in lowered
    )
    if machine_broad:
        _append_skill(selected, "hardware_report", "hardware_report")
        _append_skill(selected, "resource_governor", "resource_report")
        _append_skill(selected, "storage_governor", "storage_report")
    elif hardware_requested:
        _append_skill(selected, "hardware_report", "hardware_report")
        if bool(_OBSERVE_RESOURCE.search(lowered)):
            _append_skill(selected, "resource_governor", "resource_report")
    else:
        if machine_requested:
            _append_skill(selected, "hardware_report", "hardware_report")
        if _OBSERVE_DISK.search(lowered):
            _append_skill(selected, "storage_governor", "storage_report")
            if "pressure" in lowered or "largest files" in lowered:
                selected = [{"skill": "disk_pressure_report", "function": "disk_pressure_report"}]
        if _OBSERVE_RESOURCE.search(lowered):
            _append_skill(selected, "resource_governor", "resource_report")
    if re.search(r"\bnetwork|dns|gateway|latency|ping\b", lowered):
        _append_skill(selected, "network_governor", "network_report")
    if re.search(r"\bservice health|service status|agent service|personal-agent service\b", lowered):
        _append_skill(selected, "service_health_report", "service_health_report")
    if not selected:
        if machine_requested:
            if machine_broad:
                _append_skill(selected, "hardware_report", "hardware_report")
                _append_skill(selected, "resource_governor", "resource_report")
                _append_skill(selected, "storage_governor", "storage_report")
            else:
                _append_skill(selected, "hardware_report", "hardware_report")
        else:
            _append_skill(selected, "storage_governor", "storage_report")
            _append_skill(selected, "resource_governor", "resource_report")
    # Deterministic order.
    selected.sort(key=lambda item: (item["skill"], item["function"]))
    return selected


def build_cards_payload(
    cards: list[dict[str, Any]],
    raw_available: bool,
    summary: str,
    confidence: float,
    next_questions: list[str],
) -> dict[str, Any]:
    normalized = [normalize_card(card, idx) for idx, card in enumerate(cards)]
    return {
        "cards": normalized,
        "raw_available": bool(raw_available),
        "summary": summary,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "next_questions": [str(item) for item in next_questions],
    }


def nl_route(text: str) -> dict[str, Any]:
    intent = classify_free_text(text)
    if intent not in _INTENTS:
        intent = "UNKNOWN"
    response: dict[str, Any] = {"intent": intent, "skills": [], "cards_payload": None}

    if intent in {"OBSERVE_PC", "EXPLAIN_PREVIOUS"}:
        response["skills"] = select_observe_skills(text)
    return response
