#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.pack_search_authorization import SPECS  # noqa: E402


OUTPUT = ROOT / "docs" / "operator" / "PACK_SEARCH_AUTHORIZATION_INVENTORY_V2F.json"

ROUTES = {
    "pack_source.catalog.create": ["POST /pack_sources/catalog"],
    "pack_source.catalog.update": ["PUT /pack_sources/catalog/{source_id}"],
    "pack_source.catalog.delete": ["DELETE /pack_sources/catalog/{source_id}"],
    "pack_source.policy.update": ["PUT /pack_sources/policy"],
    "pack_source.scoped_policy.update": ["PUT /pack_sources/{source_id}/policy"],
    "permission.policy.update": ["PUT /permissions"],
    "external_pack.install": ["POST /packs/install", "POST /packs/install/plan", "POST /packs/install/apply"],
    "external_pack.approve": ["POST /packs/approve", "POST /packs/approve/plan", "POST /packs/approve/apply"],
    "external_pack.enable": ["POST /packs/enable", "POST /packs/enable/plan", "POST /packs/enable/apply"],
    "external_pack.grant": ["POST /packs/grant", "POST /packs/grant/plan", "POST /packs/grant/apply"],
    "external_pack.remove": ["POST /packs/remove", "POST /packs/remove/plan", "POST /packs/remove/apply", "DELETE /packs/{pack_id}"],
    "search.setup": ["POST /search/setup/plan", "POST /search/setup/apply"],
    "search.prerequisite": ["POST /search/setup/prerequisite/plan", "POST /search/setup/prerequisite/apply"],
}


def inventory() -> dict[str, object]:
    rows = []
    for operation, spec in sorted(SPECS.items()):
        rows.append({
            "operation": operation,
            "routes": ROUTES[operation],
            "disposition": "central_authorized",
            "capability_id": spec.capability_id,
            "executor_id": spec.executor_id,
            "actor": "scope-bound actor/thread/session",
            "resources": "operation-specific source/pack/policy/search target set",
            "confirmation": "durable single-use Universal Mutation Plan confirmation",
            "target_binding": "request + current resource state + RuntimeTruthService activation fingerprint",
            "rollback": spec.rollback,
            "receipt": "redacted Executor Registry receipt",
        })
    rows.extend([
        {
            "operation": "external_pack.remote_fetch",
            "routes": [
                "remote form of POST /packs/install",
                "legacy assistant source approval/fetch follow-ups",
                "SourceFetchController product calls",
                "AgentRuntime.packs_install remote inputs",
            ],
            "disposition": "unimplemented_denied",
            "reason": "all product and compatibility callers deny before URL I/O; separate digest-bound quarantine fetch is not implemented",
        },
        {
            "operation": "external_pack.foreign_code",
            "routes": ["all external pack paths"],
            "disposition": "unimplemented_denied",
            "reason": "foreign executable/plugin packs never execute",
        },
        {
            "operation": "notification.scheduled_delivery",
            "routes": ["trusted scheduler only"],
            "disposition": "bounded_internal_writer",
            "reason": "durable identity and state ledger; interrupted send becomes indeterminate and is never automatically resent",
        },
    ])
    return {
        "schema": "personal-agent.pack-search-authorization-inventory.v2f",
        "starting_commit": "724de3cbbbd25b2396d0f660fb0062b84d339944",
        "summary": {"central_authorized": len(SPECS), "bounded_internal_writer": 1, "unimplemented_denied": 2, "legacy": 0},
        "surfaces": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(inventory(), indent=2, sort_keys=True) + "\n"
    if args.write:
        OUTPUT.write_text(rendered, encoding="utf-8")
    if args.check and (not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered):
        print("v2F inventory differs from deterministic regeneration", file=sys.stderr)
        return 1
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
