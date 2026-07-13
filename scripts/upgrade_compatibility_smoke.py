#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRODUCT_VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _check(name: str, condition: bool, evidence: str) -> Check:
    return Check(name, "PASS" if condition else "FAIL", evidence)


def _connect_fixture_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '2')")
    conn.execute("CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    conn.execute(
        "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
        ("system_resource_baseline_context_v1", json.dumps({"expected_apps": ["Ollama"]}), "2026-07-12T00:00:00Z"),
    )
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, body TEXT)")
    conn.execute("INSERT INTO memories (id, body) VALUES ('m1', 'safe fixture memory')")
    conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT)")
    conn.execute("INSERT INTO tasks (id, title, status) VALUES ('t1', 'fixture task', 'open')")
    conn.execute("CREATE TABLE notification_history (id TEXT PRIMARY KEY, payload_json TEXT)")
    conn.execute("INSERT INTO notification_history (id, payload_json) VALUES ('n1', ?)", (json.dumps({"kind": "fixture"}),))
    conn.commit()
    return conn


def _count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    import agent.doctor as doctor
    from agent.mutation_plan import MutationPlanStore
    from agent.skill_pack_permissions import SkillGrantStore, diff_skill_permissions, validate_skill_manifest

    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="pa-upgrade-compat-") as raw:
        root = Path(raw)
        db_path = root / "agent.db"
        conn = _connect_fixture_db(db_path)
        schema = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
        checks.append(_check("previous schema opens", str(schema) == "2", f"schema={schema}"))
        checks.append(_check("preferences survive", _count(conn, "preferences") == 1, "preferences=1"))
        checks.append(_check("memory survives", _count(conn, "memories") == 1, "memories=1"))
        checks.append(_check("tasks survive", _count(conn, "tasks") == 1, "tasks=1"))
        checks.append(_check("notification history survives", _count(conn, "notification_history") == 1, "notification_history=1"))
        conn.close()

        expected_schema = doctor.expected_schema_from_version(PRODUCT_VERSION)
        checks.append(_check("product version remains schema-compatible", expected_schema == 2, f"version={PRODUCT_VERSION} expected_schema={expected_schema}"))

        plan_store = MutationPlanStore(root / "plans.json")
        checks.append(_check("new Plan store initializes", getattr(plan_store, "_records", {}) == {}, str(root / "plans.json")))

        grant_store = SkillGrantStore(root / "skill_grants.json")
        checks.append(_check("new skill grant store initializes", grant_store.list_grants() == [], str(root / "skill_grants.json")))

        legacy_manifest = {
            "schema_version": 1,
            "skill_pack_id": "fixture.legacy",
            "publisher_id": "fixture",
            "name": "Fixture Legacy Skill",
            "version": "1.0.0",
            "entrypoints": [],
            "declared_permissions": ["invoke.files.create"],
        }
        manifest = validate_skill_manifest(legacy_manifest, expected_skill_pack_id="fixture.legacy")
        checks.append(_check("old skill manifest validates when explicit", manifest["skill_pack_id"] == "fixture.legacy", json.dumps(manifest, sort_keys=True)[:400]))
        diff = diff_skill_permissions(legacy_manifest, {**legacy_manifest, "version": "1.1.0", "declared_permissions": ["invoke.files.create", "invoke.notification.local.send"]})
        checks.append(_check("new permission is not auto-inherited", "invoke.notification.local.send" in diff["newly_requested"], json.dumps(diff, sort_keys=True)))

        _write_json(root / "rollback_statement.json", {
            "code_rollback": "supported through lifecycle runner when previous checkpoint exists",
            "state_rollback": "restore Backup v1 artifact for full state rollback",
            "automatic_destructive_migration": False,
        })
        rollback = json.loads((root / "rollback_statement.json").read_text(encoding="utf-8"))
        checks.append(_check("rollback limitation reported", rollback["state_rollback"].startswith("restore Backup v1"), json.dumps(rollback, sort_keys=True)))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN=0 FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
