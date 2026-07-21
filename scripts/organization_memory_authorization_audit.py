#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.organization_memory_authorization import ASSISTANT_SUBOPERATIONS, MUTATING_ASSISTANT_COMMANDS, SPECS  # noqa: E402


OUTPUT = ROOT / "docs" / "operator" / "ORGANIZATION_MEMORY_AUTHORIZATION_INVENTORY_V2E.json"

ALIASES = {
    "memory.reset": ["POST /memory/reset"],
    "semantic.ingest": ["POST /semantic/documents/ingest"],
    "semantic.rebuild": ["POST /semantic/rebuild"],
    "semantic.repair": ["POST /semantic/repair"],
    "notification.test": ["POST /llm/notifications/test"],
    "notification.mark_read": ["POST /llm/notifications/mark_read"],
    "notification.prune": ["POST /llm/notifications/prune"],
}

READ_ONLY = [
    "GET /memory/status", "POST /semantic/doctor", "GET /llm/notifications",
    "GET /llm/notifications/status", "GET /llm/notifications/policy",
    "assistant list/search/status/recall commands",
]

INTERNAL = [
    {"writer_id": "llm_notifications", "operation": "record_delivery", "scope": "delivery metadata only"},
    {"writer_id": "model_watch", "operation": "persist_watch_state", "scope": "scheduler state only"},
    {"writer_id": "model_watch_hf", "operation": "persist_scan_state", "scope": "scheduler scan state only"},
    {"writer_id": "llm_model_discovery_policy", "operation": "persist_effective_policy", "scope": "effective-policy cache only"},
]

DENIED = [
    "notification.retry_or_resend (no public surface; denied)",
    "notification.delivery_policy_write (no public surface; denied)",
    "semantic.remote_url_ingest (not implemented)",
    "semantic.executable_or_macro_ingest (denied)",
    "memory plaintext in Plan/receipt/journal (denied)",
]


def build() -> dict[str, object]:
    central = []
    for operation in sorted(SPECS):
        spec = SPECS[operation]
        central.append({
            "operation": operation,
            "disposition": "central_authorized",
            "capability_id": spec.capability_id,
            "executor_id": spec.executor_id,
            "action_type": spec.action_type,
            "rollback_available": spec.rollback_available,
            "surfaces": ALIASES.get(operation, []),
        })
    for command in sorted(ASSISTANT_SUBOPERATIONS):
        spec = ASSISTANT_SUBOPERATIONS[command]
        surfaces = [
            f"assistant command {command}",
            f"Telegram thin-transport alias {command}",
            f"CLI/native command alias {command}",
        ]
        if command == "done":
            surfaces.append("POST /done")
        central.append({
            "operation": f"assistant.{command}",
            "command": command,
            "disposition": "central_authorized",
            "capability_id": spec.capability_id,
            "executor_id": spec.executor_id,
            "action_type": spec.action_type,
            "argument_schema": spec.argument_schema,
            "resource_types": list(spec.resource_types),
            "target_tables": list(spec.target_tables),
            "rollback_available": spec.rollback_available,
            "rollback_hint": spec.rollback_hint,
            "audit_description": spec.audit_description,
            "create_operation": spec.create_operation,
            "surfaces": surfaces,
        })
    return {
        "schema": "personal-agent.organization-memory-authorization-inventory.v2e",
        "source_checkpoint": "35d9a91eb2bac763cbeda4bb49ba81875a3c873f",
        "central_authorized": central,
        "assistant_commands": sorted(MUTATING_ASSISTANT_COMMANDS),
        "read_only": READ_ONLY,
        "bounded_internal_writers": INTERNAL,
        "unimplemented_denied": DENIED,
        "deferred_v2f": ["pack sources", "pack lifecycle", "pack permissions", "search setup compatibility cleanup"],
        "validation": {
            "missing_aliases": sorted(set(SPECS) - set(ALIASES)),
            "stale_aliases": sorted(set(ALIASES) - set(SPECS)),
        },
        "summary": {
            "central_operations": len(central),
            "public_aliases": sum(len(row["surfaces"]) for row in central),
            "assistant_commands": len(MUTATING_ASSISTANT_COMMANDS),
            "read_only_groups": len(READ_ONLY),
            "internal_writers": len(INTERNAL),
            "denied_groups": len(DENIED),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build()
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write:
        OUTPUT.write_text(canonical, encoding="utf-8")
    if args.check and (not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != canonical):
        print("v2E authorization inventory differs from regeneration", file=sys.stderr)
        return 1
    print(canonical, end="")
    validation = payload["validation"]
    return 0 if not validation["missing_aliases"] and not validation["stale_aliases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
