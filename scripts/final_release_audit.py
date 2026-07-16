#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
TAG = f"v{VERSION}"


@dataclass
class Check:
    name: str
    status: str
    evidence: str
    blocker: bool = True


def _check(name: str, condition: bool, evidence: str, *, warn: bool = False, blocker: bool = True) -> Check:
    if condition:
        return Check(name, "PASS", evidence, blocker)
    return Check(name, "WARN" if warn else "FAIL", evidence, blocker)


def _run(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def _summary_ok(output: str, *, allow_warn: bool = False) -> bool:
    fail = re.search(r"\bFAIL=(\d+)\b", output)
    warn = re.search(r"\bWARN=(\d+)\b", output)
    if fail and int(fail.group(1)) != 0:
        return False
    if warn and int(warn.group(1)) != 0 and not allow_warn:
        return False
    return True


def _script_check(name: str, script: str, *, allow_warn: bool = False, timeout: int = 180) -> Check:
    proc = _run([sys.executable, script], timeout=timeout)
    output = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0 and _summary_ok(output, allow_warn=allow_warn)
    tail = "\n".join(output.splitlines()[-8:])
    return _check(name, ok, tail or f"exit={proc.returncode}")


def main() -> int:
    checks: list[Check] = []

    version_text = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    checks.append(_check("product version selected", version_text == VERSION, version_text))

    project_state = (ROOT / "docs/operator/PROJECT_STATE.md").read_text(encoding="utf-8")
    checks.append(_check("active phase is current release work", "Active Phase:" in project_state and f"v{VERSION}" in project_state, "PROJECT_STATE.md"))

    final_doc = ROOT / "docs/operator/V0_2_2_FINAL_RELEASE_AUDIT.md"
    final_text = final_doc.read_text(encoding="utf-8") if final_doc.exists() else ""
    ledger = ROOT / "docs/operator/RELEASE_LEDGER.md"
    ledger_text = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
    checks.append(_check("version decision recorded", f"v{VERSION}" in ledger_text or "Recommended final version: `v0.2.2`" in final_text, str(ledger)))
    checks.append(_check("rollback statement present", "Code rollback to `v0.2.1` is not the same as full state rollback" in final_text, str(final_doc)))
    final_text_lower = final_text.lower()
    checks.append(_check("process isolation limitation present", "process isolation" in final_text_lower and "not claimed" in final_text_lower, str(final_doc)))

    release_notes = ROOT / f"docs/releases/v{VERSION}.md"
    notes_text = release_notes.read_text(encoding="utf-8") if release_notes.exists() else ""
    checks.append(_check("release notes present", "Telegram" in notes_text and "Local" in notes_text and "Known limitations" in notes_text, str(release_notes)))

    acceptance = ROOT / "docs/operator/RUNTIME_LATENCY_ACCEPTANCE_V1.json"
    try:
        accepted = json.loads(acceptance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        accepted = {}
    accepted_rows = accepted.get("accepted_warnings") if isinstance(accepted.get("accepted_warnings"), list) else []
    checks.append(_check("accepted latency warnings have evidence", bool(accepted_rows) and str(accepted.get("release_decision") or "").startswith("accepted_for_v0.2.2"), str(acceptance)))

    tag_lookup = _run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{TAG}"], timeout=20)
    checks.append(_check("final tag does not already exist", tag_lookup.returncode != 0, TAG))

    checks.append(_script_check("version consistency smoke", "scripts/version_consistency_smoke.py"))
    checks.append(_script_check("upgrade compatibility smoke", "scripts/upgrade_compatibility_smoke.py"))
    checks.append(_script_check("release artifact smoke", "scripts/release_artifact_smoke.py", timeout=180))
    checks.append(_script_check("clean checkout reproducibility smoke", "scripts/clean_checkout_reproducibility_smoke.py", timeout=360))
    checks.append(_script_check("clean checkout Debian package smoke", "scripts/clean_checkout_debian_package_smoke.py", timeout=240))
    checks.append(_script_check("full pytest closure smoke", "scripts/full_pytest_closure_smoke.py", timeout=1800))
    checks.append(_script_check("full pytest failure triage", "scripts/full_pytest_failure_triage.py", timeout=1800))
    checks.append(_script_check("skipped test debt inventory", "scripts/skipped_test_debt_inventory.py"))
    checks.append(_script_check("skipped test debt closure", "scripts/skipped_test_debt_closure_smoke.py", timeout=1800))
    checks.append(_script_check("telegram transport diagnostic", "scripts/telegram_transport_diagnostic.py", allow_warn=True))
    checks.append(_script_check("telegram transport smoke", "scripts/telegram_transport_smoke.py"))
    checks.append(_script_check("local intent routing smoke", "scripts/local_intent_routing_smoke.py"))
    checks.append(_script_check("local system inspection smoke", "scripts/local_system_inspection_smoke.py"))
    checks.append(_script_check("capability policy audit", "scripts/capability_policy_audit.py"))
    checks.append(_script_check("Universal Plan audit", "scripts/universal_plan_mode_audit.py"))
    checks.append(_script_check("generic bypass audit", "scripts/generic_mutation_bypass_audit.py"))
    checks.append(_script_check("adversarial proof", "scripts/full_adversarial_authorization_proof.py", allow_warn=True))
    checks.append(_script_check("latency closure smoke", "scripts/runtime_latency_closure_smoke.py"))
    checks.append(_script_check("docs truth smoke", "scripts/docs_truth_smoke.py"))
    checks.append(_script_check("release gate matrix smoke", "scripts/release_gate_matrix_smoke.py"))

    primary = _run([sys.executable, "scripts/primary_uninstall_policy.py", "status"], timeout=30)
    primary_text = primary.stdout + primary.stderr
    checks.append(_check("primary uninstall remains disabled", '"enabled": false' in primary_text and '"purge_supported": false' in primary_text, primary_text[:800]))

    status = _run(["git", "status", "--short"], timeout=20)
    checks.append(_check("working tree has reviewable final-audit changes", status.returncode == 0, status.stdout.strip() or "clean", warn=True, blocker=False))

    fail_count = sum(1 for row in checks if row.status == "FAIL")
    warn_count = sum(1 for row in checks if row.status == "WARN")
    pass_count = sum(1 for row in checks if row.status == "PASS")
    release_blockers = sum(1 for row in checks if row.status == "FAIL" and row.blocker)
    ready = release_blockers == 0
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    print(f"VERSION={VERSION}")
    print(f"RELEASE_BLOCKERS={release_blockers}")
    print(f"READY_TO_TAG={'true' if ready else 'false'}")
    return 1 if release_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
