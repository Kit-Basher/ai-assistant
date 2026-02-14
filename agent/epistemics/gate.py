from __future__ import annotations

import os

from agent.epistemics.contract import (
    build_plain_answer_candidate,
    parse_candidate_json,
    validate_candidate,
)
from agent.epistemics.detectors import run_detectors
from agent.epistemics.question_selector import select_one_question
from agent.epistemics.types import CandidateContract, ContextPack, GateDecision


_INTERCEPT_TEXT_PREFIX = "I’m not sure."


def _safe_env_float(name: str, default: float, low: float, high: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return max(low, min(high, value))


def _pass_score_threshold() -> float:
    return _safe_env_float("PASS_SCORE_THRESHOLD", 0.55, 0.0, 1.0)


def _sanitize_intercept_question(question: str | None) -> str:
    line = " ".join((question or "").replace("\n", " ").split())
    if not line:
        line = "What exact detail should I use"
    if "?" in line:
        line = line.split("?", 1)[0]
    line = line.replace("?", "").strip().rstrip(".! ")
    if not line:
        line = "What exact detail should I use"
    return f"{line}?"


def _render_intercept_text(question: str | None) -> str:
    normalized = _sanitize_intercept_question(question)
    if normalized.count("?") != 1:
        normalized = _sanitize_intercept_question(normalized)
    # Spec lock: exactly 3 lines with blank middle line.
    return "\n".join([_INTERCEPT_TEXT_PREFIX.rstrip(), "", normalized.rstrip()])


def apply_epistemic_gate(
    user_text: str,
    ctx: ContextPack,
    candidate: CandidateContract | str,
) -> GateDecision:
    contract_errors: tuple[str, ...] = tuple()
    parsed: CandidateContract

    if isinstance(candidate, str):
        parsed_candidate, parse_errors = parse_candidate_json(candidate)
        contract_errors = parse_errors
        if parsed_candidate is None:
            parsed = build_plain_answer_candidate("")
        else:
            parsed = parsed_candidate
    else:
        parsed = candidate

    validation_errors = validate_candidate(parsed)
    detector_result = run_detectors(user_text, ctx, parsed)
    reason_set = {reason.code for reason in detector_result.reasons}

    if contract_errors or validation_errors:
        reason_set.add("CONTRACT_INVALID")
    if parsed.kind == "clarify":
        reason_set.add("CLARIFY_REQUESTED")

    hard_reasons = set(detector_result.hard_reasons)
    if contract_errors or validation_errors:
        hard_reasons.add("CONTRACT_INVALID")
    if parsed.kind == "clarify":
        hard_reasons.add("CLARIFY_REQUESTED")

    should_intercept = bool(hard_reasons)
    if detector_result.score >= _pass_score_threshold():
        should_intercept = True

    if should_intercept:
        reason_codes = tuple(sorted(reason_set))
        question = select_one_question(user_text, ctx, parsed, reason_codes)
        user_visible = _render_intercept_text(question)
        return GateDecision(
            intercepted=True,
            user_text=user_visible,
            reasons=reason_codes,
            hard_reasons=tuple(sorted(hard_reasons)),
            score=detector_result.score,
            question=question,
            contract_errors=tuple(sorted(set(contract_errors + validation_errors))),
            candidate_kind=parsed.kind,
        )

    return GateDecision(
        intercepted=False,
        user_text=parsed.final_answer,
        reasons=tuple(sorted(reason_set)),
        hard_reasons=tuple(sorted(hard_reasons)),
        score=detector_result.score,
        question=None,
        contract_errors=tuple(sorted(set(contract_errors + validation_errors))),
        candidate_kind=parsed.kind,
    )
