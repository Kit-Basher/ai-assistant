from __future__ import annotations

from typing import Any

from agent.epistemics.types import CandidateContract, Claim, ContextPack
from agent.friction.canary_cases import MUST_HAVE, MUST_NOT_APPEAR
from agent.friction.next_action import compute_next_action
from agent.friction.summary import compute_summary


def _stable_unique(values: Any) -> tuple[str, ...]:
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


def _build_context(case: dict[str, Any]) -> ContextPack:
    ctx = case.get("ctx") if isinstance(case.get("ctx"), dict) else {}
    assert isinstance(ctx, dict)
    return ContextPack(
        user_id="__test_canary__",
        active_thread_id="thread-1",
        recent_turn_ids=_stable_unique(ctx.get("recent_turn_ids")),
        in_scope_memory_ids=_stable_unique(ctx.get("in_scope_memory_ids")),
        tool_event_ids=_stable_unique(ctx.get("tool_event_ids")),
    )


def _build_candidate(case: dict[str, Any]) -> CandidateContract:
    claims_cfg = case.get("claims") if isinstance(case.get("claims"), list) else []
    claims: list[Claim] = []
    for item in claims_cfg:
        if not isinstance(item, dict):
            continue
        support = item.get("support")
        if support not in {"user", "memory", "tool", "none"}:
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        claims.append(
            Claim(
                text=text,
                support=support,
                user_turn_id=(item.get("user_turn_id") if isinstance(item.get("user_turn_id"), str) else None),
                memory_id=(
                    item.get("memory_id")
                    if isinstance(item.get("memory_id"), (str, int)) and not isinstance(item.get("memory_id"), bool)
                    else None
                ),
                tool_event_id=(item.get("tool_event_id") if isinstance(item.get("tool_event_id"), str) else None),
            )
        )
    return CandidateContract(
        kind="answer",
        final_answer=str(case.get("rendered_answer") or ""),
        clarifying_question=None,
        claims=tuple(claims),
    )


def _compose_pass_reply(user_text: str, candidate: CandidateContract, ctx: ContextPack) -> tuple[str, str | None, str | None]:
    body = candidate.final_answer
    summary = compute_summary(candidate, body)
    rendered = body
    if summary:
        rendered = f"{summary}\n\n{rendered}"
    next_step = compute_next_action(user_text, ctx, candidate)
    if next_step:
        rendered = f"{rendered}\n\nNext: {next_step}"
    return rendered, summary, next_step


def _validate_have_case(case: dict[str, Any], failures: list[str]) -> None:
    name = str(case.get("name") or "unnamed")
    user_text = str(case.get("user_text") or "")
    ctx = _build_context(case)
    candidate = _build_candidate(case)
    rendered, summary, next_step = _compose_pass_reply(user_text, candidate, ctx)

    expect_summary = bool(case.get("expect_summary"))
    expect_next = bool(case.get("expect_next"))
    if expect_summary != bool(summary):
        failures.append(name)
        return
    if expect_next != bool(next_step):
        failures.append(name)
        return

    if summary:
        if "?" in summary or "\n" in summary or len(summary) > 140:
            failures.append(name)
            return
        if not rendered.startswith(summary + "\n\n"):
            failures.append(name)
            return

    if next_step:
        expected_prefix = case.get("expected_next_prefix")
        if isinstance(expected_prefix, str) and not rendered.splitlines()[-1].startswith(expected_prefix):
            failures.append(name)
            return
        if "?" in next_step or "\n" in next_step or len(next_step) > 120:
            failures.append(name)
            return
        if not rendered.splitlines()[-1].startswith("Next: "):
            failures.append(name)
            return

    if summary and next_step:
        summary_index = rendered.find(summary)
        body_index = rendered.find(candidate.final_answer)
        next_index = rendered.rfind("\n\nNext: ")
        if not (summary_index == 0 and body_index > summary_index and next_index > body_index):
            failures.append(name)
            return


def _validate_not_appear_case(case: dict[str, Any], failures: list[str]) -> None:
    name = str(case.get("name") or "unnamed")
    gate_passed = bool(case.get("gate_passed"))
    if not gate_passed:
        intercepted = str(case.get("intercepted_reply_text") or "")
        rendered = intercepted
        if rendered != intercepted:
            failures.append(name)
            return
    else:
        user_text = str(case.get("user_text") or "")
        ctx = _build_context(case)
        candidate = _build_candidate(case)
        rendered, _, _ = _compose_pass_reply(user_text, candidate, ctx)

    if bool(case.get("expect_no_summary")) and "In short:" in rendered:
        failures.append(name)
        return
    if bool(case.get("expect_no_next")) and "\n\nNext: " in rendered:
        failures.append(name)
        return

    if not gate_passed:
        expected = str(case.get("intercepted_reply_text") or "")
        if rendered != expected:
            failures.append(name)
            return


def run_friction_canaries() -> dict[str, Any]:
    failures: list[str] = []
    for case in MUST_HAVE:
        _validate_have_case(case, failures)
    for case in MUST_NOT_APPEAR:
        _validate_not_appear_case(case, failures)

    failed_names = tuple(sorted(set(failures)))
    total = len(MUST_HAVE) + len(MUST_NOT_APPEAR)
    failed = len(failed_names)
    passed = total - failed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "failed_names": failed_names,
    }


def main() -> int:
    result = run_friction_canaries()
    print("Friction canary")
    print(f"total: {result['total']}")
    print(f"passed: {result['passed']}")
    print(f"failed: {result['failed']}")
    print("failed_cases:")
    names = result.get("failed_names") or tuple()
    if names:
        for name in names:
            print(f"- {name}")
    else:
        print("- (none)")
    return 0 if int(result["failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

