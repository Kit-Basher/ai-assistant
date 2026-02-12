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
_EXPLAIN = re.compile(r"\b(explain|why|what changed|change[d]?|previous|last time|since)\b", re.IGNORECASE)
_MEMORY_WRITE = re.compile(
    r"^(remember that|from now on|remember this|set max cards to|turn confidence (on|off)|default compare (on|off)|daily brief (on|off)|set disk delta threshold to|only send if service unhealthy|set open loops due window to)\b",
    re.IGNORECASE,
)
_PLAN_DAY = re.compile(
    r"\b(plan my day|today plan|what should i do today|show quick wins|show top 3 priorities)\b",
    re.IGNORECASE,
)
_SHOW_PREFS = re.compile(r"\b(show my preferences|what are my preferences|preferences)\b", re.IGNORECASE)
_MEMORY_INSPECT = re.compile(r"\b(what do you remember about me|what do you remember)\b", re.IGNORECASE)
_OPEN_LOOP_ADD = re.compile(r"^remember that (?:i need to )?.+ by .+$", re.IGNORECASE)
_OPEN_LOOP_DONE = re.compile(r"^mark .+ done$", re.IGNORECASE)
_OPEN_LOOPS_LIST = re.compile(r"\b(open loops|show open loops|list open loops)\b", re.IGNORECASE)
_DAILY_BRIEF_STATUS = re.compile(
    r"\b(daily brief status|why didn.?t .*daily brief|why did(n't| not) .*daily brief)\b",
    re.IGNORECASE,
)
_CHITCHAT = re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.IGNORECASE)


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
    if _EXPLAIN.search(lowered) and (_OBSERVE_DISK.search(lowered) or _OBSERVE_RESOURCE.search(lowered)):
        return "EXPLAIN_PREVIOUS"
    if _OBSERVE_DISK.search(lowered) or _OBSERVE_RESOURCE.search(lowered) or "my pc" in lowered:
        return "OBSERVE_PC"
    return "UNKNOWN"


def select_observe_skills(text: str) -> list[dict[str, str]]:
    lowered = (text or "").lower()
    selected: list[dict[str, str]] = []
    if _OBSERVE_DISK.search(lowered):
        selected.append({"skill": "storage_governor", "function": "storage_report"})
        if "pressure" in lowered or "largest files" in lowered:
            selected = [{"skill": "disk_pressure_report", "function": "disk_pressure_report"}]
    if _OBSERVE_RESOURCE.search(lowered):
        selected.append({"skill": "resource_governor", "function": "resource_report"})
    if re.search(r"\bnetwork|dns|gateway|latency|ping\b", lowered):
        selected.append({"skill": "network_governor", "function": "network_report"})
    if re.search(r"\bservice health|service status|agent service|personal-agent service\b", lowered):
        selected.append({"skill": "service_health_report", "function": "service_health_report"})
    if not selected:
        selected.append({"skill": "storage_governor", "function": "storage_report"})
        selected.append({"skill": "resource_governor", "function": "resource_report"})
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
