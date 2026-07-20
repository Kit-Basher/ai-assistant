#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _sqlite_checks(db_path: Path) -> list[Check]:
    if not db_path.is_file():
        return [Check("canonical database", "FAIL", f"missing: {db_path}")]
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return [Check("canonical database", "FAIL", f"read failed: {exc.__class__.__name__}")]
    schema = str(row[0]) if row else "missing"
    return [
        Check("database integrity", "PASS" if integrity == "ok" else "FAIL", integrity),
        Check("database schema", "PASS" if schema == "2" else "FAIL", f"schema_version={schema}"),
    ]


def audit(*, repo_root: Path, state_root: Path, config_root: Path, expected_artifacts: list[Path]) -> list[Check]:
    checks = [
        Check("state root", "PASS" if state_root.is_dir() else "FAIL", str(state_root)),
        Check("config root", "PASS" if config_root.is_dir() else "WARN", str(config_root)),
    ]
    checks.extend(_sqlite_checks(state_root / "agent.db"))
    registry = state_root / "llm_registry.json"
    checks.append(Check("canonical model registry", "PASS" if registry.is_file() else "FAIL", str(registry)))
    for relative in (Path("memory/agent.db"), Path("llm_registry.json")):
        path = repo_root / relative
        checks.append(
            Check(
                f"repo-local mutable state: {relative}",
                "WARN" if path.exists() else "PASS",
                f"preserve and migrate; do not use as canonical: {path}" if path.exists() else "absent",
            )
        )
    for path in expected_artifacts:
        checks.append(Check(f"preserved artifact: {path.name}", "PASS" if path.exists() else "FAIL", str(path)))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Personal Agent recovery and installation audit.")
    parser.add_argument("--repo-root", default=str(Path.home() / "personal-agent"))
    parser.add_argument("--state-root", default=str(Path.home() / ".local/share/personal-agent"))
    parser.add_argument("--config-root", default=str(Path.home() / ".config/personal-agent"))
    parser.add_argument("--expect-artifact", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = audit(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        state_root=Path(args.state_root).expanduser().resolve(),
        config_root=Path(args.config_root).expanduser().resolve(),
        expected_artifacts=[Path(item).expanduser().resolve() for item in args.expect_artifact],
    )
    if args.json:
        print(json.dumps({"checks": [asdict(check) for check in checks]}, indent=2, sort_keys=True))
    else:
        for check in checks:
            print(f"{check.status}: {check.name}: {check.detail}")
        print(
            f"PASS={sum(check.status == 'PASS' for check in checks)} "
            f"WARN={sum(check.status == 'WARN' for check in checks)} "
            f"FAIL={sum(check.status == 'FAIL' for check in checks)}"
        )
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
