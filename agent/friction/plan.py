from __future__ import annotations

import re

from agent.epistemics.types import CandidateContract, Claim


_PLANNING_PHRASES = (
    "how do i",
    "how should i",
    "what's the best way",
    "where do i start",
    "plan",
    "steps",
    "break this down",
)
_ALLOWED_VERBS = (
    "Create",
    "Add",
    "Run",
    "Write",
    "Define",
    "Set",
    "Commit",
    "Tag",
    "Test",
    "Refactor",
    "Implement",
    "Inspect",
    "Verify",
)
_ALLOWED_VERB_SET = set(_ALLOWED_VERBS)
_MAX_STEPS = 5
_MAX_STEP_LEN = 100


def _normalize_text(value: str) -> str:
    lowered = (value or "").lower()
    lowered = re.sub(r"[^a-z0-9 ./_\\-]+", " ", lowered)
    return " ".join(lowered.split())


def _claim_has_supported_provenance(claim: Claim) -> bool:
    if claim.support == "tool":
        return bool(claim.tool_event_id)
    if claim.support == "memory":
        return claim.memory_id is not None
    if claim.support == "user":
        return bool(claim.user_turn_id)
    return False


def _supported_claim_texts(candidate: CandidateContract) -> tuple[str, ...]:
    out: list[str] = []
    for claim in candidate.claims:
        if claim.support == "none":
            continue
        if not _claim_has_supported_provenance(claim):
            continue
        text = " ".join((claim.text or "").replace("\n", " ").split())
        if text:
            out.append(text)
    return tuple(out)


def _iter_sentence_chunks(text: str) -> tuple[str, ...]:
    chunks: list[str] = []
    for line in (text or "").splitlines():
        for piece in line.split(". "):
            value = piece.strip()
            if value:
                chunks.append(value)
    return tuple(chunks)


def _sanitize_step(chunk: str) -> str | None:
    value = " ".join((chunk or "").replace("\n", " ").split())
    if not value:
        return None
    value = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", value).strip()
    value = value.strip("`").strip()
    value = value.replace("?", "").strip()
    value = value.rstrip(" .!;:")
    if not value:
        return None
    match = re.match(r"^([A-Za-z]+)\b(.*)$", value)
    if not match:
        return None
    verb = match.group(1).capitalize()
    if verb not in _ALLOWED_VERB_SET:
        return None
    rest = match.group(2).strip()
    step = verb if not rest else f"{verb} {rest}"
    step = " ".join(step.split())
    if not step:
        return None
    if len(step) > _MAX_STEP_LEN:
        step = step[:_MAX_STEP_LEN].rstrip()
    if not step:
        return None
    return step


def _extract_imperative_steps(text: str) -> tuple[str, ...]:
    steps: list[str] = []
    seen: set[str] = set()
    for chunk in _iter_sentence_chunks(text):
        step = _sanitize_step(chunk)
        if not step:
            continue
        key = step.lower()
        if key in seen:
            continue
        seen.add(key)
        steps.append(step)
    return tuple(steps)


def _planning_phrase_trigger(user_text: str) -> bool:
    lowered = (user_text or "").lower()
    return any(phrase in lowered for phrase in _PLANNING_PHRASES)


def _answer_imperative_trigger(candidate: CandidateContract) -> bool:
    return len(_extract_imperative_steps(candidate.final_answer or "")) >= 2


def _step_supported(step: str, supported_claims: tuple[str, ...]) -> bool:
    step_norm = _normalize_text(step)
    if not step_norm:
        return False
    for claim_text in supported_claims:
        claim_norm = _normalize_text(claim_text)
        if not claim_norm:
            continue
        if step_norm in claim_norm or claim_norm in step_norm:
            return True
    return False


def compute_plan(user_text: str, candidate: CandidateContract, rendered_answer: str) -> list[str] | None:
    if candidate.kind != "answer":
        return None

    supported_claims = _supported_claim_texts(candidate)
    if not supported_claims:
        return None

    planning_oriented = _planning_phrase_trigger(user_text) or _answer_imperative_trigger(candidate)
    if not planning_oriented:
        return None

    steps: list[str] = []
    for step in _extract_imperative_steps(rendered_answer):
        if _step_supported(step, supported_claims):
            steps.append(step)
        if len(steps) >= _MAX_STEPS:
            break

    if len(steps) < 2:
        fallback: list[str] = []
        for step in _extract_imperative_steps("\n".join(supported_claims)):
            fallback.append(step)
            if len(fallback) >= _MAX_STEPS:
                break
        steps = fallback

    if len(steps) < 2:
        return None
    return steps[:_MAX_STEPS]
