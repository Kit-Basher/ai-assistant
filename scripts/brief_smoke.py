#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


def _first_line(text: str) -> str:
    stripped = str(text or "").strip()
    return stripped.splitlines()[0] if stripped else ""


def _insert_system_facts(db: MemoryDB, snapshot_id: str, taken_at: str, load_1m: float, mem_used: int, disk_used: int) -> None:
    facts = {
        "schema": {"name": "system_facts", "version": 1},
        "snapshot": {
            "snapshot_id": snapshot_id,
            "taken_at": taken_at,
            "timezone": "UTC",
            "collector": {
                "agent_version": "0.6.0",
                "hostname": "host",
                "boot_id": "boot",
                "uptime_s": 1,
                "collection_duration_ms": 1,
                "partial": False,
                "errors": [],
            },
            "provenance": {"sources": []},
        },
        "os": {"kernel": {"release": "6.0.0", "arch": "x86_64"}},
        "cpu": {"load": {"load_1m": load_1m, "load_5m": load_1m, "load_15m": load_1m}},
        "memory": {
            "ram_bytes": {
                "total": 16 * 1024**3,
                "used": mem_used,
                "free": 0,
                "available": (16 * 1024**3) - mem_used,
                "buffers": 0,
                "cached": 0,
            },
            "swap_bytes": {"total": 0, "free": 0, "used": 0},
            "pressure": {
                "psi_supported": False,
                "memory_some_avg10": None,
                "io_some_avg10": None,
                "cpu_some_avg10": None,
            },
        },
        "filesystems": {
            "mounts": [
                {
                    "mountpoint": "/",
                    "device": "/dev/sda1",
                    "fstype": "ext4",
                    "total_bytes": 100 * 1024**3,
                    "used_bytes": disk_used,
                    "avail_bytes": 100 * 1024**3 - disk_used,
                    "used_pct": (float(disk_used) / float(100 * 1024**3)) * 100.0,
                    "inodes": {"total": None, "used": None, "avail": None, "used_pct": None},
                }
            ]
        },
        "process_summary": {
            "top_cpu": [],
            "top_mem": [{"pid": 1, "name": "proc", "cpu_pct": None, "rss_bytes": mem_used // 4}],
        },
        "integrity": {"content_hash_sha256": "0" * 64, "signed": False, "signature": None},
    }
    facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    db.insert_system_facts_snapshot(
        id=snapshot_id,
        user_id="user1",
        taken_at=taken_at,
        boot_id="boot",
        schema_version=1,
        facts_json=facts_json,
        content_hash_sha256="0" * 64,
        partial=False,
        errors_json="[]",
    )


def _warnings(text: str, first_line: str) -> list[str]:
    warnings: list[str] = []
    lowered = f"{first_line}\n{text}".lower()
    if not first_line:
        warnings.append("empty first line")
    if first_line.startswith("{") or first_line.startswith("["):
        warnings.append("raw dump")
    if first_line.endswith("?") or first_line.lower().startswith(("what do you", "how can i", "what would you", "what should i")):
        warnings.append("question-style first line")
    if len(first_line) > 220:
        warnings.append("not concise")
    if any(token in lowered for token in ("debug", "traceback", "stack trace")):
        warnings.append("debug wording")
    if any(token in lowered for token in ("need more context", "i need more context", "couldn't read", "could not read")):
        warnings.append("dead-end wording")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a brief transcript smoke against the orchestrator.")
    _ = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "brief.db")
        db = MemoryDB(db_path)
        schema_path = REPO_ROOT / "memory" / "schema.sql"
        db.init_schema(str(schema_path))
        log_path = os.path.join(tmpdir, "events.log")
        skills_path = str(REPO_ROOT / "skills")
        env_backup = dict(os.environ)
        os.environ["UI_MODE"] = "conversational"
        try:
            orch = Orchestrator(
                db=db,
                skills_path=skills_path,
                log_path=log_path,
                timezone="UTC",
                llm_client=None,
            )

            _insert_system_facts(db, "snap-brief-1", "2026-02-06T00:00:00+00:00", 0.1, 2 * 1024**3, 60 * 1024**3)

            def observe_handler(_ctx: object, user_id: str | None = None) -> dict[str, object]:
                _insert_system_facts(db, "snap-brief-2", "2026-02-07T00:00:00+00:00", 0.6, 4 * 1024**3, 70 * 1024**3)
                return {"text": "Snapshot taken: 2026-02-07T00:00:00+00:00 (UTC)", "payload": {}}

            orch.skills["observe_now"].functions["observe_now"].handler = observe_handler  # type: ignore[assignment]
            response = orch.handle_message("/brief", "user1")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
            db.close()

    text = str(getattr(response, "text", ""))
    first_line = _first_line(text)
    warnings = _warnings(text, first_line)

    print("[brief] route: brief")
    print("[brief] status: ok")
    print(f"[brief] first_line: {first_line}")
    print(f"[brief] dead_end_warnings: {', '.join(warnings) if warnings else 'none'}")

    if warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
