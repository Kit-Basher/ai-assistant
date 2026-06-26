#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _contains_all(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    doc = ROOT / "docs/operator/RELEASE_GATE_MATRIX.md"
    ci = ROOT / ".github/workflows/ci.yml"
    doc_text = doc.read_text(encoding="utf-8") if doc.is_file() else ""
    ci_text = ci.read_text(encoding="utf-8") if ci.is_file() else ""

    checks.append(("release gate matrix doc exists", doc.is_file(), str(doc.relative_to(ROOT))))
    checks.append(("github ci workflow exists", ci.is_file(), str(ci.relative_to(ROOT))))
    checks.append(
        (
            "ci-safe gates documented",
            _contains_all(doc_text, ("CI-safe", "release_smoke.py", "chat_eval.py", "external_pack_safety_smoke.py")),
            "CI-safe command group names deterministic gates",
        )
    )
    checks.append(
        (
            "live-runtime gates documented",
            _contains_all(doc_text, ("live-runtime", "daily_driver_smoke.py", "prove_ready.py", "prove_core_workflows.py")),
            "live-runtime command group names local service gates",
        )
    )
    checks.append(
        (
            "optional integration gates documented",
            _contains_all(doc_text, ("optional", "SearXNG", "Telegram", "local model")),
            "optional gate group names service-dependent checks",
        )
    )
    checks.append(
        (
            "ci does not require live local services",
            all(token not in ci_text for token in ("daily_driver_smoke.py", "prove_ready.py", "prove_core_workflows.py", "perf_smoke.py")),
            "GitHub Actions workflow avoids live runtime gates",
        )
    )
    checks.append(
        (
            "future github actions decision documented",
            _contains_all(doc_text, ("GitHub Actions", "later", "fresh Debian VM proof")),
            "doc says broader Actions coverage is deferred until after local/VM proof",
        )
    )

    print("# Personal Agent Release Gate Matrix Smoke")
    failed = 0
    for name, ok, evidence in checks:
        print(f"## {name}: {'PASS' if ok else 'FAIL'}")
        print(f"- evidence: {evidence}")
        if not ok:
            failed += 1

    if failed:
        print(f"\nFAIL release_gate_matrix_smoke failures={failed}")
        return 1
    print("\nPASS release_gate_matrix_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
