from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.epistemics.canary_cases import MUST_INTERCEPT, MUST_PASS
from agent.epistemics.contract import build_plain_answer_candidate, parse_candidate_json
from agent.epistemics.gate import apply_epistemic_gate
from agent.epistemics.types import CandidateContract, Claim, ContextPack, GateDecision, MessageTurn


@dataclass(frozen=True)
class CanaryResult:
    name: str
    must_intercept: bool
    passed: bool
    decision: GateDecision
    candidate: CandidateContract
    ctx: ContextPack


def _stable_unique_strings(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        return tuple()
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _build_recent_messages(raw_messages: Any) -> tuple[MessageTurn, ...]:
    if not isinstance(raw_messages, (list, tuple)):
        return tuple()
    result: list[MessageTurn] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("text")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(text, str):
            continue
        turn_id = item.get("turn_id")
        if turn_id is not None and not isinstance(turn_id, str):
            turn_id = None
        result.append(MessageTurn(role=role, text=text, turn_id=turn_id))
    return tuple(result)


def build_case_context(case: dict[str, Any], user_id: str = "canary-user") -> ContextPack:
    knobs = case.get("ctx") if isinstance(case.get("ctx"), dict) else {}
    assert isinstance(knobs, dict)
    recent_messages = _build_recent_messages(knobs.get("recent_messages"))

    recent_turn_ids = _stable_unique_strings(knobs.get("recent_turn_ids"))
    if not recent_turn_ids:
        recent_turn_ids = tuple(
            turn.turn_id
            for turn in recent_messages
            if isinstance(turn.turn_id, str) and turn.turn_id.strip()
        )

    in_scope_memory = _stable_unique_strings(knobs.get("in_scope_memory"))
    in_scope_memory_ids = _stable_unique_strings(knobs.get("in_scope_memory_ids"))
    if not in_scope_memory_ids:
        in_scope_memory_ids = in_scope_memory

    return ContextPack(
        user_id=user_id,
        active_thread_id=str(knobs.get("active_thread_id") or "thread-1"),
        thread_created_at=(str(knobs.get("thread_created_at")) if knobs.get("thread_created_at") else None),
        thread_label=(str(knobs.get("thread_label")) if knobs.get("thread_label") else None),
        recent_messages=recent_messages,
        recent_turn_ids=recent_turn_ids,
        memory_hits=_stable_unique_strings(knobs.get("memory_hits")),
        memory_ambiguous=_stable_unique_strings(knobs.get("memory_ambiguous")),
        memory_miss=bool(knobs.get("memory_miss")),
        in_scope_memory=in_scope_memory,
        in_scope_memory_ids=in_scope_memory_ids,
        out_of_scope_memory=_stable_unique_strings(knobs.get("out_of_scope_memory")),
        out_of_scope_relevant_memory=bool(knobs.get("out_of_scope_relevant_memory")),
        thread_turn_count=(
            int(knobs.get("thread_turn_count"))
            if isinstance(knobs.get("thread_turn_count"), int)
            else len(recent_messages)
        ),
        tools_available=("core",),
        tool_event_ids=_stable_unique_strings(knobs.get("tool_event_ids")),
        tool_failures=_stable_unique_strings(knobs.get("tool_failures")),
        referents=_stable_unique_strings(knobs.get("referents")),
    )


def _hydrate_claim_provenance(candidate: CandidateContract, ctx: ContextPack) -> CandidateContract:
    if not candidate.claims:
        return candidate
    default_user_turn_id = ctx.recent_turn_ids[-1] if ctx.recent_turn_ids else None
    default_memory_id = ctx.in_scope_memory_ids[0] if len(ctx.in_scope_memory_ids) == 1 else None
    default_tool_event_id = ctx.tool_event_ids[0] if len(ctx.tool_event_ids) == 1 else None
    in_scope_set = set(ctx.in_scope_memory_ids)
    hydrated: list[Claim] = []
    for claim in candidate.claims:
        if claim.support == "none":
            hydrated.append(Claim(text=claim.text, support="none"))
            continue
        if claim.support == "user":
            user_turn_id = claim.user_turn_id or default_user_turn_id
            if not user_turn_id:
                hydrated.append(Claim(text=claim.text, support="none"))
                continue
            hydrated.append(
                Claim(
                    text=claim.text,
                    support="user",
                    ref=claim.ref,
                    user_turn_id=user_turn_id,
                )
            )
            continue
        if claim.support == "memory":
            memory_id = claim.memory_id
            if memory_id is None and isinstance(claim.ref, str) and claim.ref.strip():
                memory_id_candidate = claim.ref.strip()
                if memory_id_candidate in in_scope_set:
                    memory_id = memory_id_candidate
            if memory_id is None:
                memory_id = default_memory_id
            if memory_id is None:
                hydrated.append(Claim(text=claim.text, support="none"))
                continue
            hydrated.append(
                Claim(
                    text=claim.text,
                    support="memory",
                    ref=claim.ref,
                    memory_id=memory_id,
                )
            )
            continue
        if claim.support == "tool":
            tool_event_id = claim.tool_event_id or default_tool_event_id
            if not tool_event_id:
                hydrated.append(Claim(text=claim.text, support="none"))
                continue
            hydrated.append(
                Claim(
                    text=claim.text,
                    support="tool",
                    ref=claim.ref,
                    tool_event_id=tool_event_id,
                )
            )
            continue
        hydrated.append(claim)
    return CandidateContract(
        kind=candidate.kind,
        final_answer=candidate.final_answer,
        clarifying_question=candidate.clarifying_question,
        claims=tuple(hydrated),
        assumptions=candidate.assumptions,
        unresolved_refs=candidate.unresolved_refs,
        thread_refs=candidate.thread_refs,
        raw_json=candidate.raw_json,
    )


def build_case_candidate(case: dict[str, Any], ctx: ContextPack) -> CandidateContract:
    candidate_cfg = case.get("candidate") if isinstance(case.get("candidate"), dict) else {}
    assert isinstance(candidate_cfg, dict)
    hydrate_provenance = bool(candidate_cfg.get("hydrate_provenance", True))
    payload = {
        "kind": candidate_cfg.get("kind", "answer"),
        "final_answer": candidate_cfg.get("final_answer", "OK."),
        "clarifying_question": candidate_cfg.get("clarifying_question"),
        "claims": candidate_cfg.get("claims", []),
        "assumptions": candidate_cfg.get("assumptions", []),
        "unresolved_refs": candidate_cfg.get("unresolved_refs", []),
        "thread_refs": candidate_cfg.get("thread_refs", []),
    }
    parsed, errors = parse_candidate_json(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    candidate = parsed if parsed is not None and not errors else build_plain_answer_candidate(str(payload["final_answer"]))
    if not hydrate_provenance:
        return candidate
    return _hydrate_claim_provenance(candidate, ctx)


def evaluate_case(case: dict[str, Any], must_intercept: bool) -> CanaryResult:
    name = str(case.get("name") or "unnamed")
    user_text = str(case.get("user_text") or "")
    ctx = build_case_context(case)
    candidate = build_case_candidate(case, ctx)
    decision = apply_epistemic_gate(user_text, ctx, candidate)
    passed = bool(decision.intercepted) if must_intercept else not bool(decision.intercepted)
    return CanaryResult(
        name=name,
        must_intercept=must_intercept,
        passed=passed,
        decision=decision,
        candidate=candidate,
        ctx=ctx,
    )


def run_canary_suite() -> list[CanaryResult]:
    results: list[CanaryResult] = []
    for case in MUST_INTERCEPT:
        results.append(evaluate_case(case, must_intercept=True))
    for case in MUST_PASS:
        results.append(evaluate_case(case, must_intercept=False))
    return results


def main() -> int:
    results = run_canary_suite()
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    failed_names = [result.name for result in results if not result.passed]
    failed = len(failed_names)

    print("Epistemics canary")
    print(f"total: {total}")
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print("failed_cases:")
    if failed_names:
        for name in failed_names:
            print(f"- {name}")
    else:
        print("- (none)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

