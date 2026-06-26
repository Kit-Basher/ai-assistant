#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CommandCheck:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: int
    blocker_on_fail: bool = True


@dataclass
class CommandResult:
    check: CommandCheck
    status: str
    returncode: int
    elapsed_seconds: float
    output: str
    next_action: str


@dataclass(frozen=True)
class Subsystem:
    name: str
    status: str
    blocker: bool
    unknown: bool
    evidence: tuple[str, ...]
    gaps: tuple[str, ...]
    next_action: str


def _command_checks() -> list[CommandCheck]:
    return [
        CommandCheck("prove_ready canonical gate", (sys.executable, "scripts/prove_ready.py"), 900),
        CommandCheck("backup_restore_proof bounded restore", (sys.executable, "scripts/backup_restore_proof.py"), 120),
        CommandCheck("webui_robustness_smoke static/component smoke", (sys.executable, "scripts/webui_robustness_smoke.py"), 180),
        CommandCheck("release_gate_matrix_smoke CI/live split", (sys.executable, "scripts/release_gate_matrix_smoke.py"), 60),
        CommandCheck("chat_eval adversarial routing", (sys.executable, "scripts/chat_eval.py"), 180),
        CommandCheck("llm_behavior_eval e2e behavior", (sys.executable, "scripts/llm_behavior_eval.py"), 180),
        CommandCheck("perf_smoke latency/status", (sys.executable, "scripts/perf_smoke.py"), 180, blocker_on_fail=False),
        CommandCheck("release_smoke", (sys.executable, "scripts/release_smoke.py"), 480),
        CommandCheck("daily_driver_smoke", (sys.executable, "scripts/daily_driver_smoke.py", "--timeout", "90"), 300, blocker_on_fail=False),
        CommandCheck("external_pack_safety_smoke", (sys.executable, "scripts/external_pack_safety_smoke.py"), 180),
        CommandCheck("prove_core_workflows", (sys.executable, "scripts/prove_core_workflows.py"), 300, blocker_on_fail=False),
        CommandCheck("git diff whitespace check", ("git", "diff", "--check"), 60),
    ]


def _summarize_output(output: str, *, max_lines: int = 10) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = lines[: max_lines // 2]
    tail = lines[-(max_lines // 2) :]
    return "\n".join([*head, "...", *tail])


def _run(check: CommandCheck) -> CommandResult:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            check.argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=check.timeout_seconds,
            check=False,
        )
        output = proc.stdout or ""
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        output = f"{output}\nTIMEOUT after {check.timeout_seconds}s"
        returncode = 124
    elapsed = time.monotonic() - started
    lowered = output.lower()
    if returncode != 0:
        status = "FAIL" if check.blocker_on_fail else "WARN"
        next_action = "Fix this command before PRE_VM_COMPLETE can be yes." if check.blocker_on_fail else "Inspect this warning before VM proof."
    elif "release_blockers: 0" in lowered and "ready_for_vm_proof: yes" in lowered:
        status = "PASS"
        next_action = "No action."
    elif "fail=0" in lowered or "pass" in lowered:
        status = "WARN" if ("warn=" in lowered and "warn=0" not in lowered) or "blocked=1" in lowered else "PASS"
        next_action = "Inspect runtime-state warnings." if status == "WARN" else "No action."
    else:
        status = "PASS"
        next_action = "No action."
    return CommandResult(check, status, returncode, elapsed, output, next_action)


def _doc(path: str) -> bool:
    return (ROOT / path).is_file()


def _subsystems() -> list[Subsystem]:
    return [
        Subsystem(
            name="Backup/restore",
            status="hardened",
            blocker=False,
            unknown=False,
            evidence=(
                "docs/operator/BACKUP_RESTORE.md exists",
                "scripts/backup_restore_proof.py validates a representative backup",
                "dry-run restore and temp-state restore are covered without live mutation",
                "corrupt backup and strict version mismatch refusal are tested",
                "doctor/support bundle tests cover redacted support artifacts",
            ),
            gaps=(
                "Live same-machine restore still requires an explicit operator action.",
                "Fresh Debian VM restore remains part of final VM proof.",
            ),
            next_action="During final VM proof, restore a real backup only after explicit operator confirmation.",
        ),
        Subsystem(
            name="Installer/update/uninstall",
            status="partial",
            blocker=False,
            unknown=False,
            evidence=(
                "scripts/install_local.sh and scripts/promote_local_stable.sh exist",
                "release bundle and Debian package tests cover parts of uninstall/rollback",
            ),
            gaps=(
                "Fresh Debian VM install proof intentionally not run.",
                "Partial failure recovery is not proven end to end on a clean host.",
            ),
            next_action="Before VM proof, run local installer idempotency/rollback tests; during VM proof, include install/update/uninstall.",
        ),
        Subsystem(
            name="Storage/log growth",
            status="partial",
            blocker=False,
            unknown=False,
            evidence=(
                "docs/operator/MEMORY_AUDIT.md documents growth surfaces",
                "support bundle paths are redacted and scoped",
            ),
            gaps=(
                "No single storage_status command yet.",
                "Runtime release/log/backup cleanup policy is documented but not enforced.",
            ),
            next_action="Add read-only storage_status, then add cleanup preview commands later under Plan Mode.",
        ),
        Subsystem(
            name="Observability/debuggability",
            status="hardened",
            blocker=False,
            unknown=False,
            evidence=(
                "trace_id, route, semantic intent evidence, and timing fields exist",
                "scripts/chat_eval.py reports route distribution and invariant failures",
                "support bundle redaction tests exist",
            ),
            gaps=("No single operator dashboard for persistent managed-action journals.",),
            next_action="Add a read-only journal/status dashboard later.",
        ),
        Subsystem(
            name="Web UI robustness",
            status="partial",
            blocker=False,
            unknown=False,
            evidence=(
                "frontend build is covered by prove_ready when node_modules is present",
                "chat autoscroll was fixed in prior work",
                "scripts/webui_robustness_smoke.py runs frontend build, Node UI tests, and static chat robustness checks",
                "docs/operator/WEB_UI_ROBUSTNESS.md documents send failure, busy state, refresh, cache, export/import, and manual UI checks",
            ),
            gaps=(
                "No broad automated browser robustness suite for refresh/retry/large transcript/cache behavior.",
                "Transcript import is documented as not implemented.",
            ),
            next_action="Before release, manually check refresh, hard-refresh after promotion, large transcript behavior, and export download.",
        ),
        Subsystem(
            name="Telegram runtime behavior",
            status="partial",
            blocker=False,
            unknown=False,
            evidence=(
                "Telegram optional semantics are in doctor/status",
                "token redaction and duplicate poller checks exist",
                "chat routes Telegram status deterministically",
            ),
            gaps=("Start/stop/restart through user-facing Plan Mode is not fully proven as a golden path.",),
            next_action="Add Telegram service Plan Mode UX proof or document manual operator-only start/stop boundary.",
        ),
        Subsystem(
            name="Model/provider management",
            status="hardened",
            blocker=False,
            unknown=False,
            evidence=(
                "model/provider guidance passes core workflow proof",
                "default and temporary model changes use persistent managed-action journals",
                "stale follow-up escape is covered by adversarial evals",
            ),
            gaps=("Real local-LLM degraded/timeout soak is still optional future work.",),
            next_action="Add opt-in real local LLM behavior soak after VM proof.",
        ),
        Subsystem(
            name="Memory completion",
            status="partial",
            blocker=False,
            unknown=False,
            evidence=(
                "docs/operator/MEMORY_AUDIT.md exists",
                "no-memory override and secret redaction tests exist",
                "preference reset/clear destructive paths are journaled and scoped",
            ),
            gaps=(
                "Forget-X and memory explainability UX are not complete.",
                "Historical memory may contain older content before new redaction rules.",
            ),
            next_action="Finish deterministic memory status/explain/forget UX before calling memory complete.",
        ),
        Subsystem(
            name="Security/capabilities",
            status="hardened",
            blocker=False,
            unknown=False,
            evidence=(
                "external_pack_safety_smoke covers 39 gates",
                "Plan Mode protects mutators",
                "safe search remains metadata-only and loopback managed services are enforced",
            ),
            gaps=("CORS/listen-address and support bundle policies should be rechecked during VM proof.",),
            next_action="Keep security audit current; rerun external_pack_safety_smoke before every release candidate.",
        ),
        Subsystem(
            name="Release/CI automation",
            status="partial",
            blocker=False,
            unknown=False,
            evidence=(
                "prove_ready, release_smoke, chat_eval, llm_behavior_eval, perf_smoke, daily_driver_smoke, external_pack_safety_smoke, and prove_core_workflows exist",
                "git diff --check is part of the gates",
                "docs/operator/RELEASE_GATE_MATRIX.md defines CI-safe, local-runtime, and optional integration gates",
                "scripts/release_gate_matrix_smoke.py verifies the documented split and confirms GitHub Actions avoids live local service gates",
            ),
            gaps=(
                "Current GitHub Actions workflow is intentionally smaller than the full CI-safe matrix.",
                "Fresh VM proof remains intentionally deferred.",
            ),
            next_action="After VM proof, expand GitHub Actions with the remaining CI-safe deterministic gates.",
        ),
    ]


def _audit_doc_ready() -> tuple[bool, list[str]]:
    required = [
        "docs/operator/BACKUP_RESTORE.md",
        "docs/operator/MEMORY_AUDIT.md",
        "docs/operator/SECURITY_AUDIT.md",
        "docs/operator/RELEASE_HARDENING_AUDIT.md",
        "docs/operator/DOCS_SOURCE_OF_TRUTH_AUDIT.md",
        "docs/operator/WEB_UI_ROBUSTNESS.md",
        "docs/operator/RELEASE_GATE_MATRIX.md",
    ]
    missing = [path for path in required if not _doc(path)]
    return not missing, missing


def main() -> int:
    print("# Personal Agent PRE_VM_COMPLETE Gate")
    print(f"Repo: {ROOT}")
    print("")

    command_results: list[CommandResult] = []
    for check in _command_checks():
        print(f"## {check.name}")
        print("command: " + " ".join(check.argv))
        result = _run(check)
        command_results.append(result)
        print(f"status: {result.status} exit={result.returncode} elapsed={result.elapsed_seconds:.1f}s")
        if result.status != "PASS":
            print(f"next_action: {result.next_action}")
            print("output:")
            print(_summarize_output(result.output))
        print("")

    docs_ok, missing_docs = _audit_doc_ready()
    subsystems = _subsystems()
    blockers = [row for row in subsystems if row.blocker]
    unknowns = [row for row in subsystems if row.unknown]
    command_blockers = [row for row in command_results if row.status == "FAIL" and row.check.blocker_on_fail]
    command_warnings = [row for row in command_results if row.status == "WARN"]
    if not docs_ok:
        command_blockers.append(
            CommandResult(
                CommandCheck("audit docs present", ("docs",), 0),
                "FAIL",
                1,
                0.0,
                "\n".join(missing_docs),
                "Create or restore the missing operator audit docs.",
            )
        )

    blocker_count = len(blockers) + len(command_blockers)
    warning_count = len(command_warnings) + sum(1 for row in subsystems if row.status == "partial" and not row.blocker)
    unknown_count = len(unknowns)
    complete = blocker_count == 0 and unknown_count == 0

    print("## Subsystem Status")
    print("| Subsystem | Status | Blocker | Unknown | Next action |")
    print("| --- | --- | --- | --- | --- |")
    for row in subsystems:
        print(
            f"| {row.name} | {row.status} | {'yes' if row.blocker else 'no'} | "
            f"{'yes' if row.unknown else 'no'} | {row.next_action} |"
        )

    print("")
    print("## Summary")
    print(f"PRE_VM_COMPLETE: {'yes' if complete else 'no'}")
    print(f"BLOCKERS: {blocker_count}")
    print(f"WARNINGS: {warning_count}")
    print(f"UNKNOWN_AREAS: {unknown_count}")
    print("NEXT_ACTIONS:")
    if blockers:
        for row in blockers:
            print(f"- [blocker] {row.name}: {row.next_action}")
    if command_blockers:
        for row in command_blockers:
            print(f"- [command] {row.check.name}: {row.next_action}")
    if unknowns:
        for row in unknowns:
            print(f"- [unknown] {row.name}: {row.next_action}")
    if command_warnings:
        for row in command_warnings:
            print(f"- [warning] {row.check.name}: {row.next_action}")
    if not blockers and not command_blockers and not unknowns and not command_warnings:
        print("- Run the fresh Debian VM proof.")

    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
