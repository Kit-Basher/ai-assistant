from __future__ import annotations

import re

from agent.epistemics.types import CandidateContract, ContextPack


_PRIORITY = (
    "UNRESOLVED_REFERENCE",
    "MISSING_REQUIRED_SLOT",
    "MULTI_INTENT",
    "CROSS_THREAD_RISK",
    "MEMORY_MISS",
    "TOOL_FAILURE_OR_UNAVAILABLE",
    "UNSUPPORTED_CLAIM",
)

_INTENT_TOKENS = (
    ("schedule", ("schedule",)),
    ("remind", ("remind", "/remind")),
    ("task", ("task", "/task_add", "/done")),
    ("delete", ("delete", "remove")),
    ("move", ("move",)),
    ("rename", ("rename",)),
    ("copy", ("copy",)),
    ("plan", ("plan",)),
)
_TIMEZONE_HINTS = {"utc", "gmt", "pst", "pdt", "est", "edt", "cst", "cdt", "mst", "mdt"}
_FILE_OP_WORDS = ("delete", "remove", "move", "rename", "copy")


def _normalize_one_question(value: str) -> str:
    question = " ".join((value or "").replace("\n", " ").split())
    if not question:
        question = "What should I clarify"
    question = question.replace("?", "").strip().rstrip(".! ")
    if not question:
        question = "What should I clarify"
    normalized = f"{question}?"
    if len(normalized) > 120:
        truncated = question[:119].rstrip()
        if not truncated:
            truncated = "What should I clarify"[:119]
        normalized = f"{truncated}?"
    normalized = normalized.replace("\n", " ")
    normalized = " ".join(normalized.split())
    if normalized.endswith("?"):
        normalized = normalized[:-1]
    normalized = normalized.replace("?", "").strip()
    if not normalized:
        normalized = "What should I clarify"
    normalized = f"{normalized}?"
    if len(normalized) > 120:
        normalized = f"{normalized[:-1][:119].rstrip()}?"
    return normalized


def _has_date_hint(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return True
    if any(token in lowered for token in ("today", "tomorrow", "yesterday")):
        return True
    return False


def _has_time_hint(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b\d{1,2}:\d{2}\b", text):
        return True
    if re.search(r"\b\d{1,2}\s?(am|pm)\b", lowered):
        return True
    return False


def _has_timezone_hint(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\butc[+-]\d{1,2}(:\d{2})?\b", lowered):
        return True
    words = set(re.findall(r"[a-zA-Z]+", lowered))
    return any(token in words for token in _TIMEZONE_HINTS)


def _has_path_hint(text: str) -> bool:
    if re.search(r"\s/[\w./-]+", text):
        return True
    if re.search(r"\b[\w.-]+\.[A-Za-z0-9]{1,8}\b", text):
        return True
    if re.search(r"\b[A-Za-z]:\\\\", text):
        return True
    return False


def _extract_intents(user_text: str) -> tuple[str, ...]:
    lowered = user_text.lower()
    intents: list[str] = []
    for label, tokens in _INTENT_TOKENS:
        if any(token in lowered for token in tokens):
            intents.append(label)
    return tuple(intents[:2])


def _missing_slot_question(user_text: str) -> str:
    lowered = user_text.lower()
    if any(word in lowered for word in _FILE_OP_WORDS):
        if not _has_path_hint(user_text):
            return "What file or path should I use?"
    if "schedule" in lowered or "remind" in lowered:
        if not _has_date_hint(user_text):
            return "What date should I use?"
        if not _has_time_hint(user_text):
            return "What time should I use?"
        if not _has_timezone_hint(user_text):
            return "What timezone should I use?"
    if lowered.startswith("/task_add"):
        return "What title should I use?"
    return "What detail is missing?"


def select_one_question(
    user_text: str,
    ctx: ContextPack,
    candidate: CandidateContract,
    reason_codes: tuple[str, ...],
) -> str:
    chosen = None
    reason_set = set(reason_codes)
    for code in _PRIORITY:
        if code in reason_set:
            chosen = code
            break
    if chosen is None and candidate.clarifying_question:
        return _normalize_one_question(candidate.clarifying_question)

    if chosen == "UNRESOLVED_REFERENCE":
        if len(ctx.referents) >= 2:
            first = ctx.referents[0]
            second = ctx.referents[1]
            return _normalize_one_question(f"Which do you mean: {first} or {second}")
        return _normalize_one_question("What exactly are you referring to")

    if chosen == "MISSING_REQUIRED_SLOT":
        return _normalize_one_question(_missing_slot_question(user_text))

    if chosen == "MULTI_INTENT":
        intents = _extract_intents(user_text)
        if len(intents) >= 2:
            return _normalize_one_question(f"Do you want {intents[0]} or {intents[1]}")
        return _normalize_one_question("Do you want one request first")

    if chosen == "CROSS_THREAD_RISK":
        return _normalize_one_question("Do you want me to use information from another conversation")

    if chosen == "MEMORY_MISS":
        return _normalize_one_question("What should I base this on")

    if chosen == "TOOL_FAILURE_OR_UNAVAILABLE":
        return _normalize_one_question("Do you want me to try again, or do you want to change the inputs")

    if chosen == "UNSUPPORTED_CLAIM":
        return _normalize_one_question("What source should I use")

    return _normalize_one_question("What should I clarify")
