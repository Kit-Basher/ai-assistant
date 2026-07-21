#!/usr/bin/env python3
"""Generate or verify the deterministic Audit 3 product-claim/journey inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/operator/END_USER_UX_INVENTORY_AUDIT_3.json"


def _claim(claim_id: str, surface: str, audience: str, truth: str, wording: str, behavior: str, disposition: str, *, original: str = "accurate") -> dict[str, str]:
    return {
        "id": claim_id,
        "surface": surface,
        "audience": audience,
        "canonical_truth": truth,
        "current_wording": wording,
        "actual_supported_behavior": behavior,
        "original_disposition": original,
        "disposition": disposition,
    }


CLAIMS = [
    _claim("assistant_front_door", "README/Web chat", "ordinary user", "route_inference + RuntimeTruthService", "Ask for help naturally.", "Web and Telegram intents share the assistant orchestration path.", "accurate"),
    _claim("remote_pack_acquisition", "README/assistant/Web pack rescue", "user/operator", "v2F pack authorization inventory", "Catalog discovery is metadata-only; arbitrary remote pack acquisition is unavailable.", "URL and remote archive acquisition fail closed before network access.", "resolved", original="contradictory"),
    _claim("pack_removal", "README pack lifecycle", "user/operator", "v2F pack authorization inventory", "Registered pack removal is confirmation-gated; arbitrary filesystem delete is denied.", "Exact registered pack/version removal is centrally authorized.", "resolved", original="contradictory"),
    _claim("source_allowlist", "assistant/pack docs", "user/operator", "pack source policy", "Allowlisted means queryable metadata, not trusted or installable.", "Source policy grants no content trust or permission.", "resolved", original="misleading"),
    _claim("pack_stage_separation", "Web/pack docs", "ordinary user", "pack lifecycle state machine", "Install, review approval, enablement, and grants are separate.", "A pack begins with zero permissions and cannot approve itself.", "accurate"),
    _claim("foreign_pack_code", "README/pack docs", "ordinary user", "pack executor policy", "Foreign executable and plugin packs are unsupported.", "Pack text remains untrusted and foreign code does not execute.", "accurate"),
    _claim("safe_mode", "assistant/Web/README", "ordinary user", "RuntimeTruthService policy", "SAFE MODE is the normal default and blocks specified higher-risk actions.", "Read/status and policy-allowed local behavior remain available.", "resolved", original="overly_technical"),
    _claim("controlled_mode", "assistant/README", "ordinary user", "RuntimeTruthService policy", "Controlled Mode changes what may be proposed; it never executes automatically.", "Every supported mutation still requires preview and approval.", "resolved", original="misleading"),
    _claim("confirmation", "assistant/Web", "ordinary user", "canonical mutation Plan", "Nothing changes until approval; cancel is always offered.", "Approval is scoped, durable, single-use, and target-bound.", "resolved", original="overly_technical"),
    _claim("indeterminate", "failure UX", "ordinary user/operator", "transaction ledger", "An unknown outcome is neither success nor failure and must be reconciled before retry.", "Indeterminate external/destructive actions are not automatically retried.", "resolved", original="missing"),
    _claim("notification_delivery", "notification docs/status", "ordinary user/operator", "delivery transaction ledger", "Exactly-once external delivery is not promised.", "Executing/uncertain restores indeterminate and is never auto-resent.", "accurate"),
    _claim("searxng_image", "search docs/setup", "operator", "v2F search executor", "The configured image tag and local runtime state are bound; immutable digest pinning remains release hardening.", "Unexpected local drift invalidates approval; upstream tag movement remains unresolved.", "accurate"),
    _claim("process_isolation", "security docs", "operator", "security boundary", "Universal authorization does not isolate malicious Python already running in the trusted process.", "Process/OS isolation remains out of scope.", "accurate"),
    _claim("telegram_disabled", "assistant/Web/Telegram docs", "ordinary user", "RuntimeTruthService transport status", "Telegram is optional and deliberately disabled.", "Disabled is neutral, creates no recovery warning, and no live contact occurs.", "accurate"),
    _claim("memory_context", "assistant/memory docs", "ordinary user", "memory authority", "Conversation context and durable saved memory are distinct.", "User-directed durable memory changes require preview and approval.", "accurate"),
    _claim("memory_reset", "assistant/memory docs", "ordinary user", "memory lifecycle Plan", "Reset is destructive, scoped, single-use, and may not be reversible.", "Private content is excluded from approval and transaction artifacts.", "accurate"),
    _claim("task_done", "assistant/task behavior", "ordinary user", "organization authorization", "Done binds one exact task and version or asks for clarification.", "A changed or ambiguous target cannot be silently completed.", "accurate"),
    _claim("search_setup", "assistant/setup docs", "ordinary user", "search service authorization", "Status is immediate; setup is blocked in SAFE MODE and approval-gated in Controlled Mode.", "The executor fixes service identity, loopback exposure, image tag, paths, and ports.", "accurate"),
    _claim("diagnostics", "Web/error responses", "ordinary user/operator", "public response serializer", "Plain-language failure and next step come first; trace data is diagnostic detail.", "Raw internal payloads are filtered from assistant messages.", "resolved", original="overly_technical"),
    _claim("capability_overview", "assistant", "ordinary user", "RuntimeTruthService + capability registry", "Capabilities are summarized by useful outcomes and current availability.", "Commands and internal capability IDs are not the primary discovery experience.", "resolved", original="stale"),
]


JOURNEYS = [
    {"id": "new_user_no_model", "entry": "public chat", "expected": "honest setup help without invented model availability", "state": "covered"},
    {"id": "healthy_local_first", "entry": "public chat", "expected": "useful local assistant and grounded target status", "state": "covered"},
    {"id": "natural_capability_discovery", "entry": "public chat", "expected": "short outcome-led suggestions tailored to runtime", "state": "covered"},
    {"id": "task_reminder_continuity", "entry": "public assistant", "expected": "exact target binding, clarification, timezone and restart continuity", "state": "covered"},
    {"id": "memory_correction_reset", "entry": "public assistant", "expected": "saved/context distinction and strong reset warning", "state": "covered"},
    {"id": "model_recommend_switch", "entry": "public assistant", "expected": "runtime-derived recommendation and bounded switch preview", "state": "covered"},
    {"id": "safe_mode_setup_block", "entry": "Web/API/CLI/Telegram mock", "expected": "blocked reason and safe next action", "state": "covered"},
    {"id": "controlled_preview_cancel", "entry": "public assistant/Web", "expected": "no automatic change and reliable cancellation", "state": "covered"},
    {"id": "local_pack_lifecycle", "entry": "public pack routes", "expected": "distinct install/review/enable/grant/remove stages", "state": "covered"},
    {"id": "remote_pack_unavailable", "entry": "assistant/Web/API", "expected": "metadata-only discovery and no fetch continuation", "state": "covered"},
    {"id": "stale_confirmation", "entry": "public assistant/Web", "expected": "nothing changed and fresh preview advice", "state": "covered"},
    {"id": "failed_indeterminate", "entry": "public status/recovery", "expected": "failure distinguished from unknown outcome; no blind retry", "state": "covered"},
    {"id": "disabled_telegram", "entry": "Web/API/CLI/Telegram mock", "expected": "neutral optional-disabled state", "state": "covered"},
    {"id": "restart_recovery", "entry": "durable public assistant state", "expected": "pending/receipt truth survives restart or fails closed", "state": "covered"},
    {"id": "notification_restore", "entry": "notification status", "expected": "indeterminate delivery never automatically resent", "state": "covered"},
]


def document() -> dict[str, object]:
    return {
        "schema_version": "audit3.ux.v1",
        "claim_count": len(CLAIMS),
        "journey_count": len(JOURNEYS),
        "claims": sorted(CLAIMS, key=lambda row: row["id"]),
        "journeys": sorted(JOURNEYS, key=lambda row: row["id"]),
    }


def rendered() -> str:
    return json.dumps(document(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def claim_source_errors() -> list[str]:
    checks = {
        "README.md": {
            "must": ("arbitrary remote archive acquisition is denied", "Registered pack removal is"),
            "must_not": ("a supported remote archive source over `https`", "- `rm`/delete/remove flows."),
        },
        "agent/assistant_ux.py": {
            "must": ("arbitrary remote pack", "SAFE MODE is the normal default"),
            "must_not": ("external skill acquisition suggestions with source approval",),
        },
        "agent/packs/registry_discovery.py": {
            "must": ("Remote pack acquisition is unavailable", "install_handoff: dict[str, Any] | None = None"),
            "must_not": ("If you install it, I will fetch it into quarantine",),
        },
        "desktop/src/components/ChatExperience.jsx": {
            "must": ("remote pack download is unavailable", 'aria-live="polite"', 'role="log"'),
            "must_not": ("Preview is required before any import", 'role="status"'),
        },
    }
    errors: list[str] = []
    for relative, rules in checks.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for expected in rules["must"]:
            if expected not in text:
                errors.append(f"{relative}: missing required product truth: {expected}")
        for stale in rules["must_not"]:
            if stale in text:
                errors.append(f"{relative}: stale product claim remains: {stale}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = rendered()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            print(f"OUT OF DATE: {OUTPUT.relative_to(ROOT)}")
            return 1
        errors = claim_source_errors()
        if errors:
            print("\n".join(f"FAIL: {error}" for error in errors))
            return 1
        print(f"PASS: {OUTPUT.relative_to(ROOT)} is deterministic ({len(CLAIMS)} claims, {len(JOURNEYS)} journeys)")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"WROTE: {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
