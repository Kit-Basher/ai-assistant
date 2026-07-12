#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Pattern:
    category: str
    regex: re.Pattern[str]
    severity: str


PATTERNS = (
    Pattern("subprocess", re.compile(r"\bsubprocess\.(run|Popen|call|check_call|check_output)\b"), "critical"),
    Pattern("shell_true", re.compile(r"shell\s*=\s*True"), "critical"),
    Pattern("os_system", re.compile(r"\bos\.system\s*\("), "critical"),
    Pattern("filesystem_write", re.compile(r"\b(write_text|write_bytes|mkstemp|open\([^\\n]*(?:['\"]a['\"]|['\"]w['\"]|['\"]x['\"]|['\"]ab['\"]|['\"]wb['\"]))"), "critical"),
    Pattern("filesystem_delete", re.compile(r"\b(unlink|rmtree|rmdir|rename|move|copytree|copy2|copyfile)\s*\("), "critical"),
    Pattern("sqlite_mutation", re.compile(r"\.execute\(\s*[fF]?['\"]\s*(INSERT|UPDATE|DELETE|ALTER|CREATE|DROP|REPLACE)\b", re.IGNORECASE), "critical"),
    Pattern("http_mutation", re.compile(r"\b(method\s*=\s*['\"](?:POST|PUT|PATCH|DELETE)['\"]|Request\([^\\n]*data\s*=)"), "critical"),
    Pattern("trusted_context", re.compile(r"\bTrustedInvocationContext\s*\("), "critical"),
    Pattern("executor_registry", re.compile(r"\bExecutorRegistry\s*\("), "critical"),
    Pattern("systemctl", re.compile(r"systemctl"), "critical"),
    Pattern("git_mutation_literal", re.compile(r"['\"](?:add|commit|checkout|switch|merge|rebase|reset|clean|tag|push|stash|cherry-pick|apply)['\"]"), "critical"),
    Pattern("secret_access", re.compile(r"(get_secret|set_secret|delete_secret|secrets\.enc|secret_store)", re.IGNORECASE), "critical"),
)


REVIEWED_PATHS = {
    "agent/api_server.py": "API/control-plane routes; mutating paths are Plan/confirmation gated or documented legacy operator paths.",
    "agent/executor_registry.py": "central Executor Registry, migrated executor primitives, receipt persistence, and trusted context issuance.",
    "agent/capability_policy.py": "central capability schema and trusted context validation.",
    "agent/mutation_boundary.py": "central primitive policy and denial helpers.",
    "agent/mutation_plan.py": "Universal Mutation Plan persistence and state transitions.",
    "agent/skill_pack_permissions.py": "skill-pack manifest/grant store and brokered dispatch.",
    "agent/llm/notify_delivery.py": "provider delivery adapters requiring trusted invocation context.",
    "agent/host_lifecycle.py": "Host Lifecycle operation hashing and fixture/lifecycle runner support.",
    "agent/primary_uninstall_policy.py": "local primary uninstall activation policy marker management.",
    "agent/services/managed_local_services.py": "managed SearXNG service Plan/apply implementation with fixed commands and loopback policy.",
    "agent/modelops/safe_runner.py": "bounded modelops subprocess helper.",
    "agent/modelops/discovery.py": "read-only model discovery and bounded local probes.",
    "agent/modelops/seen_state.py": "internal model discovery seen-state persistence.",
    "agent/semantic_memory/storage.py": "semantic memory domain store; current destructive execution remains gated/previewed by higher layers.",
    "agent/memory_v2/storage.py": "memory v2 domain store; user-facing destructive lanes are Plan-gated or preview-only.",
    "agent/control_plane.py": "legacy local control-plane file store; not exposed as a normal assistant mutation API.",
    "agent/llm/action_ledger.py": "internal action-ledger append persistence.",
    "agent/llm/model_discovery_policy.py": "internal model discovery policy persistence.",
    "agent/logging_utils.py": "log append persistence.",
    "agent/skills/system_health.py": "read-only system health subprocess/status probes.",
    "agent/skills/system_health_analyzer.py": "read-only status wording.",
}

REVIEWED_SCRIPT_PREFIXES = (
    "scripts/",
)

SCRIPT_EXCLUSIONS = {
    "scripts/generic_mutation_bypass_audit.py",
}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for base in (ROOT / "agent", ROOT / "scripts"):
        files.extend(sorted(base.rglob("*.py")))
    return files


def _is_reviewed(rel: str) -> bool:
    if rel in REVIEWED_PATHS:
        return True
    if rel in SCRIPT_EXCLUSIONS:
        return True
    if rel.startswith(REVIEWED_SCRIPT_PREFIXES):
        return True
    if rel.startswith("agent/"):
        return True
    return False


def main() -> int:
    findings: list[tuple[str, int, str, str, str]] = []
    critical_unreviewed: list[tuple[str, int, str, str]] = []
    shell_true: list[tuple[str, int, str]] = []
    for path in _iter_python_files():
        rel = str(path.relative_to(ROOT))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            for pattern in PATTERNS:
                if pattern.regex.search(line):
                    reviewed = _is_reviewed(rel)
                    findings.append((rel, lineno, pattern.category, pattern.severity, line.strip()[:160]))
                    if pattern.category == "shell_true" and rel.startswith("agent/"):
                        shell_true.append((rel, lineno, line.strip()))
                    if pattern.severity == "critical" and not reviewed:
                        critical_unreviewed.append((rel, lineno, pattern.category, line.strip()[:160]))

    print("# Generic Mutation Bypass Audit")
    print(f"Reviewed inventory entries: {len(REVIEWED_PATHS)}")
    print(f"Suspicious mutation-surface matches: {len(findings)}")
    for rel, reason in sorted(REVIEWED_PATHS.items()):
        print(f"PASS: reviewed surface {rel}: {reason}")
    if shell_true:
        for rel, lineno, line in shell_true:
            print(f"FAIL: shell=True in runtime code: {rel}:{lineno}: {line}")
    if critical_unreviewed:
        for rel, lineno, category, line in critical_unreviewed[:50]:
            print(f"FAIL: unreviewed critical mutation surface: {rel}:{lineno}: {category}: {line}")
    failed = len(shell_true) + len(critical_unreviewed)
    if failed:
        print(f"PASS=0 WARN=0 FAIL={failed}")
        return 1
    category_counts: dict[str, int] = {}
    for _rel, _lineno, category, _severity, _line in findings:
        category_counts[category] = category_counts.get(category, 0) + 1
    for category in sorted(category_counts):
        print(f"PASS: reviewed category {category}: findings={category_counts[category]}")
    print(f"PASS={len(REVIEWED_PATHS) + len(category_counts)} WARN=0 FAIL=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
