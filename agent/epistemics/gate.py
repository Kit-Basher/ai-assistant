from __future__ import annotations

from agent.epistemics.contract import (
    build_plain_answer_candidate,
    parse_candidate_json,
    validate_candidate,
)
from agent.epistemics.detectors import run_detectors
from agent.epistemics.question_selector import select_one_question
from agent.epistemics.types import CandidateContract, ContextPack, GateDecision


_INTERCEPT_TEXT_PREFIX = "I’m not sure."
_SCORE_THRESHOLD = 0.55


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
    if detector_result.score >= _SCORE_THRESHOLD:
        should_intercept = True

    if should_intercept:
        reason_codes = tuple(sorted(reason_set))
        question = select_one_question(user_text, ctx, parsed, reason_codes)
        user_visible = f"{_INTERCEPT_TEXT_PREFIX}\n\n{question}"
        return GateDecision(
            intercepted=True,
            user_text=user_visible,
            reasons=reason_codes,
            hard_reasons=tuple(sorted(hard_reasons)),
            score=detector_result.score,
            question=question,
            contract_errors=tuple(sorted(set(contract_errors + validation_errors))),
        )

    return GateDecision(
        intercepted=False,
        user_text=parsed.final_answer,
        reasons=tuple(sorted(reason_set)),
        hard_reasons=tuple(sorted(hard_reasons)),
        score=detector_result.score,
        question=None,
        contract_errors=tuple(sorted(set(contract_errors + validation_errors))),
    )

