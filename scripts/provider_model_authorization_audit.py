#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.provider_model_authorization import SPECS  # noqa: E402


OUTPUT = ROOT / "docs" / "operator" / "PROVIDER_MODEL_AUTHORIZATION_INVENTORY_V2D.json"

ALIASES = {
    "provider.add": ["POST /providers"],
    "provider.update": ["PUT /providers/{provider_id}"],
    "provider.delete": ["DELETE /providers/{provider_id}"],
    "provider.model.add": ["POST /providers/{provider_id}/models"],
    "provider.secret.set": ["POST /providers/{provider_id}/secret", "CLI agent.secrets provider:*:api_key"],
    "telegram.secret.set": ["POST /telegram/secret", "CLI agent.secrets telegram:bot_token"],
    "config.update": ["PUT /config"],
    "defaults.update": ["PUT /defaults"],
    "defaults.rollback": ["POST /defaults/rollback"],
    "model.switch": ["POST /llm/models/switch"],
    "model.switch_temporary": ["POST /llm/models/switch_temporary"],
    "model.acquire": ["POST /providers/ollama/pull", "CLI agent llm_install --apply-plan --confirmation"],
    "model.policy": ["POST /llm/models/policy"],
    "runtime.control_mode": ["POST /llm/control_mode"],
    "model.refresh": ["POST /models/refresh", "POST /providers/{provider_id}/models/refresh"],
    "model_watch.run": ["POST /model_watch/run"],
    "model_watch.refresh": ["POST /model_watch/refresh"],
    "model_watch.hf_scan": ["POST /model_watch/hf/scan"],
    "llm.fix": ["POST /llm/fixit", "CLI agent doctor --fix --apply-plan --confirmation"],
    "setup.bootstrap": ["POST /bootstrap/run"],
    "llm.autoconfig": ["POST /llm/autoconfig/apply"],
    "llm.reconcile": ["POST /llm/capabilities/reconcile/apply"],
    "llm.hygiene": ["POST /llm/hygiene/apply"],
    "llm.cleanup": ["POST /llm/cleanup/apply"],
    "llm.self_heal": ["POST /llm/self_heal/apply"],
    "llm.support.remediate": ["POST /llm/support/remediate/execute"],
    "llm.registry.rollback": ["POST /llm/registry/rollback"],
    "llm.autopilot.undo": ["POST /llm/autopilot/undo"],
    "llm.autopilot.unpause": ["POST /llm/autopilot/unpause"],
    "llm.autopilot.bootstrap": ["POST /llm/autopilot/bootstrap"],
    "modelops.execute": ["POST /modelops/execute"],
}

# Chat and Telegram are transport aliases into the same canonical operations,
# not additional mutation surfaces. They are recorded separately so the
# 36-surface HTTP/CLI count remains stable and auditable.
TRANSPORT_ALIASES = {
    "model.switch": ["assistant confirmed default/local-model switch"],
    "model.switch_temporary": [
        "assistant confirmed temporary switch/switch-back",
        "Telegram ordinary chat via the assistant front door",
    ],
    "model.acquire": ["assistant confirmed model acquisition"],
    "runtime.control_mode": ["assistant confirmed SAFE/Controlled Mode change"],
    "llm.fix": [
        "assistant confirmed fix/recovery intent",
        "Telegram recovery confirmation callback",
    ],
}

READ_ONLY = [
    "GET /providers", "GET /models", "GET /model", "GET /llm/model",
    "POST /providers/test", "POST /providers/{provider_id}/test",
    "POST /llm/models/check", "POST /llm/models/recommend",
    "POST /llm/models/proposals", "POST /llm/health/run",
    "POST /llm/catalog/run", "GET /model_watch/latest", "GET /model_watch/hf/status",
    "CLI agent setup --dry-run", "CLI agent doctor (without --fix)",
]

UNIMPLEMENTED_DENIED = [
    {"operation": "provider.secret.delete", "surface": None, "disposition": "unimplemented_denied", "reason": "No public delete surface exists; replacement is supported and plaintext rollback is forbidden."},
    {"operation": "provider.secret.import", "surface": None, "disposition": "unimplemented_denied", "reason": "Bulk secret import is not exposed."},
    {"operation": "provider.secret.repair", "surface": None, "disposition": "unimplemented_denied", "reason": "Repair is advisory; replacement requires provider.secret.set authorization."},
    {"operation": "configuration.import", "surface": None, "disposition": "unimplemented_denied", "reason": "Bulk configuration import is not exposed."},
    {"operation": "configuration.reset", "surface": None, "disposition": "unimplemented_denied", "reason": "Bulk configuration reset is not exposed; bounded field changes use config.update."},
]

MIXED_WRITERS = [
    {"writer": "llm_model_discovery_policy", "public": "model.policy central_authorized", "internal": "persist_effective_policy", "disposition": "blocked_internal_leaf_contract", "blocker": "mixed module does not yet require InternalWriterAuthority at its persistence leaf; internal authority remains denied"},
    {"writer": "llm_notifications", "public": "notification routes assigned to Audit v2E", "internal": "enqueue/delivery bookkeeping", "disposition": "deferred_v2e", "blocker": "notification public migration is explicitly outside v2D"},
    {"writer": "model_scout", "public": "proposal inspection read_only; adoption uses model.policy", "internal": "scheduled_model_scout wrapper", "disposition": "bounded_by_scheduled_parent", "blocker": None},
    {"writer": "model_watch", "public": "manual run/refresh central_authorized", "internal": "scheduler watch persistence", "disposition": "blocked_internal_leaf_contract", "blocker": "runtime scheduler still reaches the mixed persistence helper without leaf authority validation"},
    {"writer": "model_watch_hf", "public": "manual scan central_authorized", "internal": "nested scheduled scan persistence", "disposition": "blocked_internal_leaf_contract", "blocker": "nested scheduled scan lacks its own exact internal-writer leaf contract"},
]


def build() -> dict[str, object]:
    missing_alias = sorted(set(SPECS) - set(ALIASES))
    stale_alias = sorted(set(ALIASES) - set(SPECS))
    central = []
    for operation in sorted(SPECS):
        spec = SPECS[operation]
        central.append({
            "operation": operation,
            "disposition": "central_authorized",
            "capability_id": spec.capability_id,
            "executor_id": spec.executor_id,
            "action_type": spec.action_type,
            "safe_mode_blocked": spec.safe_mode_blocked,
            "secret_bearing": bool(spec.secret_field),
            "surfaces": ALIASES.get(operation, []),
            "transport_aliases": TRANSPORT_ALIASES.get(operation, []),
        })
    return {
        "schema": "personal-agent.provider-model-authorization-inventory.v2d",
        "source_checkpoint": "1521878e729aea5669a7adb67db711c2513ad718",
        "central_authorized": central,
        "read_only": READ_ONLY,
        "removed": ["POST /model", "POST /llm/model"],
        "unimplemented_denied": UNIMPLEMENTED_DENIED,
        "mixed_writers": MIXED_WRITERS,
        "validation": {"missing_aliases": missing_alias, "stale_aliases": stale_alias},
        "summary": {
            "central_operations": len(central),
            "central_surfaces": sum(len(row["surfaces"]) for row in central),
            "transport_aliases": sum(len(row["transport_aliases"]) for row in central),
            "read_only_surfaces": len(READ_ONLY),
            "removed_surfaces": 2,
            "unimplemented_denied_operations": len(UNIMPLEMENTED_DENIED),
            "mixed_writer_blockers": sum(1 for row in MIXED_WRITERS if row["blocker"]),
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
        print("provider/model authorization inventory differs from regeneration", file=sys.stderr)
        return 1
    print(canonical, end="")
    validation = payload["validation"]
    return 0 if not validation["missing_aliases"] and not validation["stale_aliases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
