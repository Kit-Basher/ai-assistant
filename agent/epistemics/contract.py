from __future__ import annotations

import json
import re
from typing import Any

from agent.epistemics.types import CandidateContract, Claim, ThreadRef


_REQUIRED_KEYS = (
    "kind",
    "final_answer",
    "clarifying_question",
    "claims",
    "assumptions",
    "unresolved_refs",
    "thread_refs",
)


def is_trivially_definitional(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return False
    if re.fullmatch(r"[0-9+\-*/= ().]+", cleaned):
        return True
    if " is defined as " in cleaned:
        return True
    if cleaned.startswith("definition:"):
        return True
    return False


def parse_candidate_json(raw: str) -> tuple[CandidateContract | None, tuple[str, ...]]:
    errors: list[str] = []
    try:
        payload = json.loads(raw)
    except Exception:
        return None, ("NON_JSON",)

    if not isinstance(payload, dict):
        return None, ("ROOT_NOT_OBJECT",)

    for key in _REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"MISSING_KEY:{key}")
    if errors:
        return None, tuple(errors)

    kind = payload.get("kind")
    if kind not in {"answer", "clarify"}:
        errors.append("INVALID_KIND")

    final_answer = payload.get("final_answer")
    if not isinstance(final_answer, str):
        errors.append("INVALID_TYPE:final_answer")

    clarifying_question = payload.get("clarifying_question")
    if clarifying_question is not None and not isinstance(clarifying_question, str):
        errors.append("INVALID_TYPE:clarifying_question")

    raw_claims = payload.get("claims")
    claims: list[Claim] = []
    if not isinstance(raw_claims, list):
        errors.append("INVALID_TYPE:claims")
    else:
        for idx, item in enumerate(raw_claims):
            if not isinstance(item, dict):
                errors.append(f"INVALID_CLAIM:{idx}")
                continue
            text = item.get("text")
            support = item.get("support")
            ref = item.get("ref")
            if not isinstance(text, str):
                errors.append(f"INVALID_CLAIM_TEXT:{idx}")
                continue
            if support not in {"memory", "tool", "user", "none"}:
                errors.append(f"INVALID_CLAIM_SUPPORT:{idx}")
                continue
            if ref is not None and not isinstance(ref, str):
                errors.append(f"INVALID_CLAIM_REF:{idx}")
                continue
            claims.append(Claim(text=text, support=support, ref=ref))

    raw_assumptions = payload.get("assumptions")
    assumptions: list[str] = []
    if not isinstance(raw_assumptions, list) or any(not isinstance(v, str) for v in raw_assumptions):
        errors.append("INVALID_TYPE:assumptions")
    else:
        assumptions = [v for v in raw_assumptions]

    raw_unresolved = payload.get("unresolved_refs")
    unresolved_refs: list[str] = []
    if not isinstance(raw_unresolved, list) or any(not isinstance(v, str) for v in raw_unresolved):
        errors.append("INVALID_TYPE:unresolved_refs")
    else:
        unresolved_refs = [v for v in raw_unresolved]

    raw_thread_refs = payload.get("thread_refs")
    thread_refs: list[ThreadRef] = []
    if not isinstance(raw_thread_refs, list):
        errors.append("INVALID_TYPE:thread_refs")
    else:
        for idx, item in enumerate(raw_thread_refs):
            if not isinstance(item, dict):
                errors.append(f"INVALID_THREAD_REF:{idx}")
                continue
            target_thread_id = item.get("target_thread_id")
            needs_confirmation = item.get("needs_confirmation")
            if not isinstance(target_thread_id, str):
                errors.append(f"INVALID_THREAD_REF_TARGET:{idx}")
                continue
            if not isinstance(needs_confirmation, bool):
                errors.append(f"INVALID_THREAD_REF_CONFIRM:{idx}")
                continue
            thread_refs.append(
                ThreadRef(target_thread_id=target_thread_id, needs_confirmation=needs_confirmation)
            )

    if errors:
        return None, tuple(sorted(errors))

    candidate = CandidateContract(
        kind=kind,
        final_answer=final_answer,
        clarifying_question=clarifying_question,
        claims=tuple(claims),
        assumptions=tuple(assumptions),
        unresolved_refs=tuple(unresolved_refs),
        thread_refs=tuple(thread_refs),
        raw_json=raw,
    )
    return candidate, tuple()


def validate_candidate(candidate: CandidateContract) -> tuple[str, ...]:
    errors: list[str] = []
    is_clarify = candidate.kind == "clarify"

    if candidate.kind == "answer" and not candidate.final_answer.strip():
        errors.append("ANSWER_REQUIRES_FINAL_ANSWER")

    if candidate.assumptions:
        if not is_clarify:
            errors.append("ASSUMPTIONS_REQUIRE_CLARIFY")
        if not (candidate.clarifying_question or "").strip():
            errors.append("ASSUMPTIONS_REQUIRE_QUESTION")

    if candidate.unresolved_refs and not is_clarify:
        errors.append("UNRESOLVED_REFS_REQUIRE_CLARIFY")

    unsupported_claim = any(
        claim.support == "none" and not is_trivially_definitional(claim.text)
        for claim in candidate.claims
    )
    if unsupported_claim and not is_clarify:
        errors.append("UNSUPPORTED_CLAIM_REQUIRES_CLARIFY")

    if any(ref.needs_confirmation for ref in candidate.thread_refs) and not is_clarify:
        errors.append("THREAD_REF_REQUIRES_CLARIFY")

    if is_clarify:
        question = (candidate.clarifying_question or "").strip()
        if not question:
            errors.append("CLARIFY_REQUIRES_QUESTION")
        if candidate.final_answer.strip():
            errors.append("CLARIFY_FINAL_ANSWER_MUST_BE_EMPTY")
        if question.count("?") != 1:
            errors.append("CLARIFY_SINGLE_QUESTION")

    return tuple(sorted(errors))


def build_plain_answer_candidate(text: str) -> CandidateContract:
    return CandidateContract(
        kind="answer",
        final_answer=text,
        clarifying_question=None,
        claims=tuple(),
        assumptions=tuple(),
        unresolved_refs=tuple(),
        thread_refs=tuple(),
        raw_json=None,
    )

