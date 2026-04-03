from __future__ import annotations

import os
import re

from agent.epistemics.contract import is_trivially_definitional
from agent.epistemics.types import CandidateContract, ContextPack, DetectorReason, DetectorResult


_PRONOUNS = {"it", "that", "those", "them"}
_SCHEDULE_WORDS = ("schedule", "remind", "/remind")
_FILE_OP_WORDS = ("delete", "remove", "move", "rename", "copy")
_SOFT_CROSS_THREAD_PHRASES = (
    "as we discussed",
    "earlier you said",
    "previously",
    "last time",
)
_SUMMARY_DRIFT_PHRASES = (
    "we've been working on",
    "we have been working on",
    "your project",
    "as usual",
)
_EXPLICIT_CROSS_THREAD_REQUESTS = (
    "other thread",
    "previous thread",
    "previous conversation",
    "earlier conversation",
    "cross-thread",
)


def _soft_cross_thread_phrases() -> tuple[str, ...]:
    raw = (os.getenv("SOFT_CROSS_THREAD_PHRASES", "") or "").strip()
    if not raw:
        return _SOFT_CROSS_THREAD_PHRASES
    parsed = [
        token.strip().lower()
        for token in re.split(r"[|,]", raw)
        if token and token.strip()
    ]
    if not parsed:
        return _SOFT_CROSS_THREAD_PHRASES
    return tuple(parsed)


def _has_datetime_hint(text: str) -> bool:
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", text):
        return True
    return False


def _has_path_hint(text: str) -> bool:
    if re.search(r"\s/[\w./-]+", text):
        return True
    if re.search(r"\b[\w.-]+\.[A-Za-z0-9]{1,8}\b", text):
        return True
    if re.search(r"\b[A-Za-z]:\\\\", text):
        return True
    return False


def detect_unresolved_reference(user_text: str, ctx: ContextPack) -> tuple[DetectorReason, ...]:
    lowered = user_text.lower()
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", user_text)}
    has_pronoun = bool(_PRONOUNS & words)
    if not has_pronoun:
        return tuple()
    if len(ctx.referents) >= 2:
        return (
            DetectorReason(
                code="UNRESOLVED_REFERENCE",
                detail="Pronoun reference is ambiguous.",
                evidence=" | ".join(ctx.referents[:2]),
                hard=True,
            ),
        )
    action_like = any(term in lowered for term in ("do ", "run ", "repeat", "again", "same"))
    if not ctx.referents and action_like:
        return (
            DetectorReason(
                code="UNRESOLVED_REFERENCE",
                detail="Pronoun reference has no resolvable target.",
                evidence=None,
                hard=True,
            ),
        )
    return tuple()


def detect_missing_required_slot(user_text: str) -> tuple[DetectorReason, ...]:
    lowered = user_text.strip().lower()
    reasons: list[DetectorReason] = []
    if any(word in lowered for word in _SCHEDULE_WORDS) and not _has_datetime_hint(user_text):
        reasons.append(
            DetectorReason(
                code="MISSING_REQUIRED_SLOT",
                detail="Scheduling request is missing date/time.",
                evidence=user_text.strip(),
                hard=True,
            )
        )
    if lowered.startswith("/task_add") and not (user_text.split(maxsplit=1)[1:] and user_text.split(maxsplit=1)[1].strip()):
        reasons.append(
            DetectorReason(
                code="MISSING_REQUIRED_SLOT",
                detail="Task creation request is missing title.",
                evidence=user_text.strip(),
                hard=True,
            )
        )
    if any(word in lowered for word in _FILE_OP_WORDS) and "file" in lowered and not _has_path_hint(user_text):
        reasons.append(
            DetectorReason(
                code="MISSING_REQUIRED_SLOT",
                detail="File operation is missing a path.",
                evidence=user_text.strip(),
                hard=True,
            )
        )
    return tuple(reasons)


def detect_multi_intent(user_text: str) -> tuple[DetectorReason, ...]:
    lowered = user_text.lower()
    if " and " not in lowered:
        return tuple()
    intents = [
        "schedule",
        "remind",
        "task",
        "delete",
        "move",
        "rename",
        "copy",
        "plan",
    ]
    hits = [word for word in intents if word in lowered]
    if len(hits) < 2:
        return tuple()
    return (
        DetectorReason(
            code="MULTI_INTENT",
            detail="Message appears to contain multiple intents.",
            evidence=", ".join(sorted(set(hits))),
            hard=False,
        ),
    )


def detect_cross_thread_risk(user_text: str, ctx: ContextPack, candidate: CandidateContract) -> tuple[DetectorReason, ...]:
    reasons: list[DetectorReason] = []
    needs_confirm = [ref.target_thread_id for ref in candidate.thread_refs if ref.needs_confirmation]
    if needs_confirm:
        reasons.append(
            DetectorReason(
                code="CROSS_THREAD_RISK",
                detail="Candidate references a non-active thread without confirmation.",
                evidence=" | ".join(sorted(needs_confirm)),
                hard=True,
            )
        )
    out_of_scope_set = set(ctx.out_of_scope_memory)
    memory_claim_refs = tuple(
        (str(claim.memory_id).strip() if claim.memory_id is not None else claim.ref.strip())
        for claim in candidate.claims
        if claim.support == "memory"
        and (
            claim.memory_id is not None
            or (isinstance(claim.ref, str) and claim.ref.strip())
        )
    )
    if out_of_scope_set and any(ref in out_of_scope_set for ref in memory_claim_refs):
        reasons.append(
            DetectorReason(
                code="CROSS_THREAD_RISK",
                detail="Candidate references memory from a different thread.",
                evidence=" | ".join(sorted(set(ref for ref in memory_claim_refs if ref in out_of_scope_set))),
                hard=True,
            )
        )
    elif ctx.out_of_scope_relevant_memory and any(claim.support == "memory" for claim in candidate.claims):
        reasons.append(
            DetectorReason(
                code="CROSS_THREAD_RISK",
                detail="Candidate may rely on out-of-scope memory.",
                evidence=" | ".join(ctx.out_of_scope_memory[:2]) if ctx.out_of_scope_memory else None,
                hard=True,
            )
        )
    lowered = user_text.lower()
    if ctx.active_thread_id and ("other thread" in lowered or "previous thread" in lowered):
        reasons.append(
            DetectorReason(
                code="CROSS_THREAD_RISK",
                detail="User text requests non-active thread context.",
                evidence=ctx.active_thread_id,
                hard=True,
            )
        )
    if ctx.active_thread_id:
        explicit_cross_thread_request = any(token in lowered for token in _EXPLICIT_CROSS_THREAD_REQUESTS)
        candidate_text = " ".join(
            [
                candidate.final_answer,
                candidate.clarifying_question or "",
                *[claim.text for claim in candidate.claims],
            ]
        ).lower()
        if not explicit_cross_thread_request:
            for phrase in _soft_cross_thread_phrases():
                if phrase in candidate_text:
                    reasons.append(
                        DetectorReason(
                            code="CROSS_THREAD_RISK",
                            detail="Soft cross-thread reference requires confirmation.",
                            evidence=phrase,
                            hard=True,
                        )
                    )
                    break
        if ctx.thread_turn_count <= 0:
            for phrase in _SUMMARY_DRIFT_PHRASES:
                if phrase in candidate_text:
                    reasons.append(
                        DetectorReason(
                            code="CROSS_THREAD_RISK",
                            detail="Summary-style reference in a new thread requires confirmation.",
                            evidence=phrase,
                            hard=True,
                        )
                    )
                    break
    return tuple(reasons)


def detect_memory_miss(ctx: ContextPack) -> tuple[DetectorReason, ...]:
    if ctx.memory_miss:
        return (
            DetectorReason(
                code="MEMORY_MISS",
                detail="Required memory lookup was a miss.",
                evidence=None,
                hard=True,
            ),
        )
    if ctx.memory_ambiguous:
        return (
            DetectorReason(
                code="MEMORY_MISS",
                detail="Memory lookup returned ambiguous results.",
                evidence=" | ".join(ctx.memory_ambiguous),
                hard=True,
            ),
        )
    return tuple()


def detect_tool_failure(ctx: ContextPack) -> tuple[DetectorReason, ...]:
    if not ctx.tool_failures:
        return tuple()
    return (
        DetectorReason(
            code="TOOL_FAILURE_OR_UNAVAILABLE",
            detail="Tool failure/unavailable signal present.",
            evidence=" | ".join(ctx.tool_failures),
            hard=True,
        ),
    )


def detect_unsupported_claims(candidate: CandidateContract) -> tuple[DetectorReason, ...]:
    unsupported = [
        claim.text
        for claim in candidate.claims
        if claim.support == "none" and not is_trivially_definitional(claim.text)
    ]
    if not unsupported:
        return tuple()
    return (
        DetectorReason(
            code="UNSUPPORTED_CLAIM",
            detail="Claim has no support source.",
            evidence=" | ".join(unsupported),
            hard=True,
        ),
    )


def run_detectors(user_text: str, ctx: ContextPack, candidate: CandidateContract) -> DetectorResult:
    reasons: list[DetectorReason] = []
    reasons.extend(detect_unresolved_reference(user_text, ctx))
    reasons.extend(detect_missing_required_slot(user_text))
    reasons.extend(detect_multi_intent(user_text))
    reasons.extend(detect_cross_thread_risk(user_text, ctx, candidate))
    reasons.extend(detect_memory_miss(ctx))
    reasons.extend(detect_tool_failure(ctx))
    reasons.extend(detect_unsupported_claims(candidate))

    ordered = tuple(sorted(reasons, key=lambda r: (r.code, r.detail, r.evidence or "")))
    weight_map = {
        "UNRESOLVED_REFERENCE": 1.0,
        "MISSING_REQUIRED_SLOT": 0.9,
        "MULTI_INTENT": 0.4,
        "CROSS_THREAD_RISK": 0.9,
        "MEMORY_MISS": 0.8,
        "TOOL_FAILURE_OR_UNAVAILABLE": 0.8,
        "UNSUPPORTED_CLAIM": 0.8,
    }
    score = min(1.0, sum(weight_map.get(reason.code, 0.2) for reason in ordered))
    hard_reasons = tuple(sorted({reason.code for reason in ordered if reason.hard}))
    return DetectorResult(score=score, reasons=ordered, hard_reasons=hard_reasons)
