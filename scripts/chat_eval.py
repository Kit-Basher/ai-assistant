from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.chat_eval_harness import (  # noqa: E402
    all_route_cases,
    evaluate_route_cases,
    run_conversation_evals,
    summarize_results,
)


def _route_debug(result) -> str:  # noqa: ANN001
    decision = result.decision if isinstance(result.decision, dict) else {}
    evidence = decision.get("evidence")
    if not isinstance(evidence, list):
        semantic = decision.get("semantic") if isinstance(decision.get("semantic"), dict) else {}
        evidence = list(semantic.get("evidence") or []) if isinstance(semantic.get("evidence"), (list, tuple)) else []
    return (
        f"seed={result.case.seed if result.case.seed is not None else 'none'} "
        f"semantic_intent={decision.get('semantic_intent')} "
        f"route={decision.get('route')} "
        f"kind={decision.get('kind')} "
        f"confidence={decision.get('confidence')} "
        f"evidence={evidence}"
    )


def _print_failure_group(title: str, failures: dict[str, list]) -> None:  # noqa: ANN001
    if not failures:
        print(f"{title}: none")
        return
    print(f"{title}:")
    for invariant, rows in sorted(failures.items(), key=lambda item: item[0]):
        print(f"- {invariant}: {len(rows)}")
        for result in rows[:5]:
            print(f"  - {result.case.case_id}: {result.case.message!r}")
            print(f"    {_route_debug(result)}")
            for failure in result.failures:
                print(f"    {failure}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Personal Agent adversarial chat routing evaluations.")
    parser.add_argument("--no-generated", action="store_true", help="Skip deterministic generated/fuzz cases.")
    parser.add_argument("--no-fixtures", action="store_true", help="Skip live-regression fixture JSON cases.")
    args = parser.parse_args(argv)

    route_cases = all_route_cases(
        include_generated=not bool(args.no_generated),
        include_fixtures=not bool(args.no_fixtures),
    )
    route_results = evaluate_route_cases(route_cases)
    conversation_results = run_conversation_evals()
    results = route_results + conversation_results
    summary = summarize_results(results)

    fixed_count = sum(1 for case in route_cases if not case.generated and case.source == "fixed")
    generated_count = sum(1 for case in route_cases if case.generated)
    fixture_count = sum(1 for case in route_cases if case.source != "fixed" and not case.generated)

    print("# Personal Agent Chat Eval")
    print(f"Total cases: {summary['total']}")
    print(f"Pass: {summary['passed']}")
    print(f"Fail: {summary['failed']}")
    print(f"Fixed cases: {fixed_count}")
    print(f"Generated/fuzz cases: {generated_count}")
    print(f"Fixture regression cases: {fixture_count}")
    print(f"Conversation simulator cases: {len(conversation_results)}")
    print("")
    print("Category distribution:")
    category_counts: dict[str, int] = {}
    for result in results:
        category_counts[result.case.category] = category_counts.get(result.case.category, 0) + 1
    for category, count in sorted(category_counts.items()):
        print(f"- {category}: {count}")
    print("")
    print("Route distribution:")
    for route, count in sorted(summary["route_distribution"].items()):
        print(f"- {route}: {count}")
    print("")

    invariant_failures = summary["invariant_failures"]
    _print_failure_group("Failures grouped by invariant", invariant_failures)

    banned_text = [
        result
        for result in results
        if any("banned text found" in failure for failure in result.failures)
    ]
    mutation_safety = [
        result
        for result in results
        if any("mutation_preview" in failure or "stale install confirmation executed" in failure for failure in result.failures)
    ]
    stale_context = [
        result
        for result in results
        if any("stale" in failure or "clarification" in failure or "pending" in failure for failure in result.failures)
    ]
    print("")
    print(f"Banned text findings: {len(banned_text)}")
    print(f"Mutation-safety violations: {len(mutation_safety)}")
    print(f"Stale-context violations: {len(stale_context)}")

    if summary["failed"]:
        print("")
        print("FAIL chat_eval")
        return 1
    print("")
    print("PASS chat_eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
