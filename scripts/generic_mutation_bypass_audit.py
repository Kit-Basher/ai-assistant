#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
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


CLASSIFICATION_PATH = ROOT / "docs" / "operator" / "MUTATION_FILE_CLASSIFICATIONS_V2B.json"
REQUIRED_CLASSIFICATION_FIELDS = {
    "path", "disposition", "capability", "entry_point", "actor", "target_resources",
    "mutation_type", "policy_path", "executor", "confirmation_requirements",
    "rollback_scope", "audit_evidence",
    "contract_identity",
}

SCRIPT_EXCLUSIONS = {
    "scripts/generic_mutation_bypass_audit.py",
}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for base in (ROOT / "agent", ROOT / "scripts"):
        files.extend(sorted(base.rglob("*.py")))
    return files


def _load_classifications() -> tuple[dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(CLASSIFICATION_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"classification_inventory_unreadable:{exc.__class__.__name__}"]
    rows = payload.get("classifications") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}, ["classification_inventory_rows_missing"]
    indexed: dict[str, dict[str, object]] = {}
    for offset, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"classification_row_invalid:{offset}")
            continue
        missing = sorted(field for field in REQUIRED_CLASSIFICATION_FIELDS if not str(row.get(field) or "").strip())
        path = str(row.get("path") or "").strip()
        if missing:
            errors.append(f"classification_fields_missing:{path or offset}:{','.join(missing)}")
        if path in indexed:
            errors.append(f"classification_duplicate:{path}")
        indexed[path] = row
    return indexed, errors


def main() -> int:
    classifications, classification_errors = _load_classifications()
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
                    reviewed = rel in classifications or rel in SCRIPT_EXCLUSIONS
                    findings.append((rel, lineno, pattern.category, pattern.severity, line.strip()[:160]))
                    if pattern.category == "shell_true" and rel.startswith("agent/"):
                        shell_true.append((rel, lineno, line.strip()))
                    if pattern.severity == "critical" and not reviewed:
                        critical_unreviewed.append((rel, lineno, pattern.category, line.strip()[:160]))

    print("# Generic Mutation Bypass Audit")
    detected_paths = {rel for rel, _lineno, _category, _severity, _line in findings if rel not in SCRIPT_EXCLUSIONS}
    classified_paths = set(classifications)
    missing_paths = sorted(detected_paths - classified_paths)
    stale_paths = sorted(classified_paths - detected_paths)
    print(f"Reviewed inventory entries: {len(classifications)}")
    print(f"Suspicious mutation-surface matches: {len(findings)}")
    for rel, row in sorted(classifications.items()):
        print(f"PASS: reviewed surface {rel}: {row.get('disposition')}: {row.get('audit_evidence')}")
    for error in classification_errors:
        print(f"FAIL: {error}")
    for rel in missing_paths:
        print(f"FAIL: mutation-bearing file lacks explicit classification: {rel}")
    for rel in stale_paths:
        print(f"FAIL: classified file no longer has a scanner finding; review or remove stale classification: {rel}")
    if shell_true:
        for rel, lineno, line in shell_true:
            print(f"FAIL: shell=True in runtime code: {rel}:{lineno}: {line}")
    unreviewed_paths = sorted({rel for rel, _lineno, _category, _line in critical_unreviewed})
    if unreviewed_paths:
        for rel in unreviewed_paths:
            count = sum(1 for item in critical_unreviewed if item[0] == rel)
            print(f"WARN: mutation-bearing file lacks an explicit reviewed-path entry: {rel} ({count} matches)")
    failed = len(shell_true) + len(classification_errors) + len(missing_paths) + len(stale_paths)
    if failed:
        print(f"PASS=0 WARN=0 FAIL={failed}")
        return 1
    category_counts: dict[str, int] = {}
    for _rel, _lineno, category, _severity, _line in findings:
        category_counts[category] = category_counts.get(category, 0) + 1
    for category in sorted(category_counts):
        print(f"PASS: reviewed category {category}: findings={category_counts[category]}")
    incomplete = sorted(rel for rel, row in classifications.items() if row.get("disposition") == "supported_pending_migration")
    for rel in incomplete:
        print(f"WARN: release-blocking central migration remains: {rel}")
    print(f"PASS={len(classifications) + len(category_counts)} WARN={len(incomplete)} FAIL=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
