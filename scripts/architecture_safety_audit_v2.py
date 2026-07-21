#!/usr/bin/env python3
"""Machine-readable public mutation inventory and architecture invariants."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "agent" / "api_server.py"

READ_ONLY_POST = {
    "/ask",
    "/chat",
    "/llm/catalog/run",
    "/llm/health/run",
    "/llm/models/check",
    "/llm/models/recommend",
    "/llm/models/proposals",
    "/providers/test",
    "/search/query",
    "/semantic/doctor",
    "/telegram/test",
}

DOMAIN_CENTRAL_AUTHORIZED = {
    "/providers",
    "/bootstrap/run",
    "/telegram/secret",
    "/models/refresh",
    "/defaults/rollback",
    "/model_watch/run",
    "/model_watch/refresh",
    "/model_watch/hf/scan",
    "/llm/models/policy",
    "/llm/models/switch",
    "/llm/models/switch_temporary",
    "/llm/fixit",
    "/llm/control_mode",
    "/llm/autoconfig/apply",
    "/llm/capabilities/reconcile/apply",
    "/llm/cleanup/apply",
    "/llm/hygiene/apply",
    "/llm/self_heal/apply",
    "/llm/support/remediate/execute",
    "/llm/registry/rollback",
    "/llm/autopilot/bootstrap",
    "/llm/autopilot/undo",
    "/llm/autopilot/unpause",
    "/modelops/execute",
    "/authorized/provider-model/preview",
    "/authorized/provider-model/apply",
    "/done",
    "/memory/reset",
    "/semantic/documents/ingest",
    "/semantic/rebuild",
    "/semantic/repair",
    "/llm/notifications/test",
    "/llm/notifications/mark_read",
    "/llm/notifications/prune",
    "/pack_sources/catalog",
    "/pack_sources/policy",
    "/packs/install",
    "/packs/approve",
    "/packs/enable",
    "/packs/grant",
    "/packs/remove",
    "/packs/install/apply",
    "/packs/approve/apply",
    "/packs/enable/apply",
    "/packs/grant/apply",
    "/packs/remove/apply",
    "/search/setup/apply",
    "/search/setup/prerequisite/apply",
    "/permissions",
}

CENTRAL_PLAN_APPLY: set[str] = set()

PLAN_GATED_LEGACY = {
    "/llm/autoconfig/apply",
    "/llm/capabilities/reconcile/apply",
    "/llm/cleanup/apply",
    "/llm/hygiene/apply",
    "/llm/self_heal/apply",
    "/llm/support/remediate/execute",
    "/modelops/execute",
}

DYNAMIC_ROUTES = [
    ("POST", "/providers/{provider_id}/secret", "central_authorized"),
    ("POST", "/providers/{provider_id}/models", "central_authorized"),
    ("POST", "/providers/{provider_id}/test", "read_only"),
    ("POST", "/providers/ollama/pull", "central_authorized"),
    ("POST", "/providers/{provider_id}/models/refresh", "central_authorized"),
    ("PUT", "/providers/{provider_id}", "central_authorized"),
    ("PUT", "/pack_sources/{source_id}/policy", "central_authorized"),
    ("PUT", "/pack_sources/catalog/{catalog_id}", "central_authorized"),
    ("DELETE", "/pack_sources/catalog/{catalog_id}", "central_authorized"),
    ("DELETE", "/packs/{pack_id}", "central_authorized"),
]

NON_API_SURFACES = [
    {
        "surface": "assistant.native_mutation",
        "kind": "assistant_intent",
        "status": "central_authorized",
        "path": "AgentOrchestrator._confirmation_preview_response -> _execute_confirmed_native_mutation -> ExecutorRegistry",
    },
    {
        "surface": "telegram.message",
        "kind": "transport",
        "status": "canonical_front_door",
        "path": "telegram_bridge -> AgentOrchestrator.handle_message",
    },
    {
        "surface": "cli.legacy_builtin_skill_writes",
        "kind": "cli_command",
        "status": "central_authorized",
        "path": "AgentOrchestrator._call_skill -> v2E preview -> durable confirmation -> organization executor",
    },
    {
        "surface": "cli.secret_set",
        "kind": "cli_command",
        "status": "central_authorized",
        "path": "agent.secrets -> loopback authorized provider/model preview/apply API; no direct SecretStore write",
    },
    {
        "surface": "skill_pack.managed_adapter_invocation",
        "kind": "skill_pack",
        "status": "central_authorized",
        "path": "SkillPackInvocationBroker.request_action preview -> persisted scoped plan -> confirm_action -> ExecutorRegistry",
    },
    {
        "surface": "executor_registry.migrated_capabilities",
        "kind": "executor",
        "status": "central_authorized",
        "path": "Universal Mutation Plan + scope-bound confirmation + capability policy + trusted invocation context",
    },
    {
        "surface": "background.model_watch_bookkeeping",
        "kind": "background_job",
        "status": "internal_state_write",
        "path": "scheduled model-watch/scout bookkeeping only; public triggers are centrally authorized",
    },
    {
        "surface": "external_pack.foreign_code",
        "kind": "pack_runtime",
        "status": "unimplemented_denied",
        "path": "external packs are normalized data/instructions only; executable fields and executable archives denied",
    },
    {
        "surface": "external_pack.remote_fetch",
        "kind": "pack_network_effect",
        "status": "unimplemented_denied",
        "path": "combined remote fetch/install is denied until separate digest-bound quarantine fetch authorization exists",
    },
    {
        "surface": "notification.scheduled_delivery",
        "kind": "background_job",
        "status": "internal_state_write",
        "path": "durable operation identity -> reserved -> executing -> terminal/indeterminate; indeterminate never auto-resends",
    },
]


def _literal_strings(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return [value for item in node.elts for value in _literal_strings(item)]
    return []


def api_routes() -> list[tuple[str, str]]:
    tree = ast.parse(API.read_text(encoding="utf-8"))
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in {"do_POST", "do_PUT", "do_DELETE"}:
            continue
        method = node.name.removeprefix("do_")
        for child in ast.walk(node):
            if not isinstance(child, ast.Compare) or len(child.ops) != 1 or len(child.comparators) != 1:
                continue
            if not isinstance(child.left, ast.Name) or child.left.id != "path":
                continue
            for route in _literal_strings(child.comparators[0]):
                if route.startswith("/"):
                    routes.add((method, route))
    routes.update((method, route) for method, route, _status in DYNAMIC_ROUTES)
    return sorted(routes)


def classify(method: str, route: str) -> tuple[str, str, str]:
    dynamic = {(m, r): status for m, r, status in DYNAMIC_ROUTES}
    if (method, route) in dynamic:
        status = dynamic[(method, route)]
    elif route in READ_ONLY_POST:
        status = "canonical_front_door" if route == "/chat" else "read_only"
    elif route in DOMAIN_CENTRAL_AUTHORIZED:
        status = "central_authorized"
    elif (method, route) in {("PUT", "/config"), ("PUT", "/defaults")}:
        status = "central_authorized"
    elif route in {"/model", "/llm/model"} and method == "POST":
        status = "removed"
    elif route.endswith("/plan"):
        status = "read_only_preview"
    elif route in CENTRAL_PLAN_APPLY:
        status = "plan_confirm_gated"
    elif route in PLAN_GATED_LEGACY:
        status = "plan_gated_legacy"
    elif method in {"PUT", "DELETE"}:
        status = "legacy_unmigrated"
    else:
        status = "legacy_unmigrated"
    effect = "read_only" if status in {"read_only", "read_only_preview", "canonical_front_door"} else "mutating"
    authority = {
        "plan_confirm_gated": "domain Plan/apply controller",
        "central_authorized": "capability policy + Executor Registry",
        "canonical_front_door": "AgentOrchestrator",
        "read_only": "runtime service",
        "read_only_preview": "domain preview controller",
        "plan_gated_legacy": "domain-specific Plan/apply controller; not central capability registry",
        "internal_state_write": "runtime internal state writer",
        "removed": "removed compatibility mutation; method is denied",
        "legacy_unmigrated": "legacy runtime/controller mutation",
    }.get(status, status)
    return status, effect, authority


def build_inventory() -> dict[str, Any]:
    surfaces: list[dict[str, Any]] = []
    for method, route in api_routes():
        status, effect, authority = classify(method, route)
        surfaces.append(
            {
                "surface": f"api.{method.lower()}:{route}",
                "kind": "api_route",
                "method": method,
                "route": route,
                "effect": effect,
                "status": status,
                "authority": authority,
            }
        )
    surfaces.extend(NON_API_SURFACES)
    counts: dict[str, int] = {}
    for item in surfaces:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "schema": "personal-agent.mutation-surface-inventory.v2",
        "source_commit": "audit-v2f-working-tree-on-724de3cbbbd25b2396d0f660fb0062b84d339944",
        "scope": "public API mutations plus assistant, Telegram, CLI, skill-pack, executor, and background entry points",
        "status_definitions": {
            "central_authorized": "central capability schema, policy, Universal Mutation Plan, confirmation, and Executor Registry",
            "plan_confirm_gated": "bounded domain Plan/apply confirmation, not yet central Executor Registry",
            "plan_gated_legacy": "domain Plan/apply exists but remains outside the central capability/executor authority",
            "legacy_unmigrated": "public or reachable mutation not fully migrated to the canonical authorization stack",
            "unimplemented_denied": "intentionally unavailable and fail-closed",
            "internal_state_write": "bounded scheduled/runtime bookkeeping, not public authority",
            "removed": "obsolete compatibility mutation denied or method-not-allowed",
        },
        "summary": counts,
        "surfaces": surfaces,
    }


def architecture_checks() -> list[dict[str, Any]]:
    orchestrator = (ROOT / "agent" / "orchestrator.py").read_text(encoding="utf-8")
    telegram = (ROOT / "agent" / "telegram_bridge.py").read_text(encoding="utf-8")
    api = API.read_text(encoding="utf-8")
    inference_defs = []
    executor_registry = (ROOT / "agent" / "executor_registry.py").read_text(encoding="utf-8")
    internal_authority = (ROOT / "agent" / "internal_writer_authority.py").read_text(encoding="utf-8")
    for path in (ROOT / "agent").rglob("*.py"):
        for match in re.finditer(r"^def route_inference\s*\(", path.read_text(encoding="utf-8"), re.MULTILINE):
            inference_defs.append(str(path.relative_to(ROOT)))
    return [
        {"check": "single_route_inference_definition", "ok": inference_defs == ["agent/llm/inference_router.py"], "evidence": inference_defs},
        {"check": "telegram_forwards_to_orchestrator", "ok": "orchestrator.handle_message(" in telegram, "evidence": "agent/telegram_bridge.py"},
        {"check": "api_does_not_call_route_inference", "ok": "route_inference(" not in api, "evidence": "agent/api_server.py"},
        {"check": "orchestrator_calls_route_inference", "ok": "route_inference(" in orchestrator, "evidence": "agent/orchestrator.py"},
        {"check": "durable_confirmation_transaction_boundary", "ok": "ConfirmationTransactionStore" in executor_registry, "evidence": "agent/confirmation_transactions.py"},
        {"check": "internal_writer_registry_enforced", "ok": "internal_writer_unregistered" in internal_authority, "evidence": "agent/internal_writer_authority.py"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-inventory", action="store_true")
    parser.add_argument("--check-inventory", action="store_true")
    args = parser.parse_args()
    inventory = build_inventory()
    checks = architecture_checks()
    payload = {**inventory, "architecture_checks": checks}
    canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    inventory_path = ROOT / "docs" / "operator" / "MUTATION_SURFACE_INVENTORY_V2.json"
    if args.write_inventory:
        inventory_path.write_text(canonical, encoding="utf-8")
    if args.check_inventory and inventory_path.read_text(encoding="utf-8") != canonical:
        print("mutation inventory differs from deterministic regeneration", file=sys.stderr)
        return 1
    if args.json:
        print(canonical, end="")
    else:
        print("# Architecture and Safety Audit v2")
        for check in checks:
            print(f"{'PASS' if check['ok'] else 'FAIL'}: {check['check']}: {check['evidence']}")
        print("Mutation surface status:")
        for status, count in sorted(inventory["summary"].items()):
            print(f"- {status}: {count}")
        print(f"TOTAL={len(inventory['surfaces'])}")
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
