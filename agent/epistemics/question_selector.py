from __future__ import annotations

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


def _normalize_one_question(value: str) -> str:
    question = " ".join((value or "").replace("\n", " ").split())
    if not question:
        question = "What exact outcome do you want me to produce"
    if "?" in question:
        # Drop all question suffixes beyond the first.
        question = question.split("?", 1)[0]
    question = question.replace("?", "").strip().rstrip(".! ")
    if not question:
        question = "What exact outcome do you want me to produce"
    return f"{question}?"


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

    lowered = user_text.lower()

    if chosen == "UNRESOLVED_REFERENCE":
        if len(ctx.referents) >= 2:
            first = ctx.referents[0]
            second = ctx.referents[1]
            return _normalize_one_question(f'Do you mean "{first}" or "{second}"')
        return _normalize_one_question("What does that refer to")

    if chosen == "MISSING_REQUIRED_SLOT":
        if "schedule" in lowered or "remind" in lowered:
            return _normalize_one_question("What exact date and time should I use")
        if lowered.startswith("/task_add"):
            return _normalize_one_question("What title should I use for the task")
        if any(word in lowered for word in ("delete", "remove", "move", "rename", "copy")) and "file" in lowered:
            return _normalize_one_question("Which exact file path should I use")
        return _normalize_one_question("What exact detail is missing")

    if chosen == "MULTI_INTENT":
        return _normalize_one_question("Should I handle one request first before the other")

    if chosen == "CROSS_THREAD_RISK":
        if candidate.thread_refs:
            target = sorted(ref.target_thread_id for ref in candidate.thread_refs if ref.needs_confirmation)
            if target:
                return _normalize_one_question(f'Do you want me to use details from thread "{target[0]}"')
        return _normalize_one_question("Do you want me to stay on the current thread only")

    if chosen == "MEMORY_MISS":
        return _normalize_one_question("Can you paste the exact detail you want me to use")

    if chosen == "TOOL_FAILURE_OR_UNAVAILABLE":
        return _normalize_one_question("Should I retry now or wait for the tool to recover")

    if chosen == "UNSUPPORTED_CLAIM":
        return _normalize_one_question("Which source should I rely on for that claim")

    return _normalize_one_question("What exact outcome do you want me to produce")
