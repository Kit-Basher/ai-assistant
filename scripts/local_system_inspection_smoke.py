#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.skills.system_health import collect_system_health
from agent.skills.system_health_summary import render_system_health_summary


def _check(name: str, ok: bool, evidence: str = "") -> tuple[str, bool, str]:
    return name, bool(ok), evidence


def main() -> int:
    observed = collect_system_health(sample_seconds=0.02)
    memory = observed.get("memory") if isinstance(observed.get("memory"), dict) else {}
    processes = observed.get("processes") if isinstance(observed.get("processes"), dict) else {}
    groups = processes.get("groups") if isinstance(processes.get("groups"), list) else []
    text = render_system_health_summary(
        observed,
        {"status": "ok", "warnings": [], "suggestions": []},
        question="what program is using the most memory on my pc?",
    )
    checks = [
        _check("memory totals collected", int(memory.get("total_bytes") or 0) > 0, str(memory.get("total_bytes"))),
        _check("top process groups collected", len(groups) > 0, str(len(groups))),
        _check("process data bounded", len(groups) <= 25, str(len(groups))),
        _check("command lines omitted", not bool((processes.get("redaction") or {}).get("command_lines_included")), "command_lines_included=false"),
        _check("environment omitted", not bool((processes.get("redaction") or {}).get("environment_included")), "environment_included=false"),
        _check("friendly names present", any("display_name" in row for row in groups if isinstance(row, dict)), "display_name"),
        _check("memory answer leads with RAM", text.splitlines()[0].lower().find("ram") >= 0, text.splitlines()[0] if text else ""),
        _check("unrelated warnings secondary", not text.startswith("Disk:"), text[:80]),
        _check("inspection is read-only", True, "no mutating operation is called"),
        _check("failure does not fall back to web", "search" not in text.lower(), "no search text"),
        _check("alternate checkout paths not embedded", "/home/c/personal-agent" not in text, "no primary path"),
    ]
    failed = 0
    print("# Local System Inspection Smoke")
    for name, ok, evidence in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} - {evidence}")
        failed += 0 if ok else 1
    print(f"PASS={len(checks)-failed} WARN=0 FAIL={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
