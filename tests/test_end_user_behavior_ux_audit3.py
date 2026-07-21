from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent.api_server import AgentRuntime
from agent.assistant_ux import build_user_facing_capability_answer
from agent.error_response_ux import deterministic_error_message
from agent.failure_ux import build_failure_recovery
from agent.orchestrator import Orchestrator


ROOT = Path(__file__).resolve().parents[1]


def test_claim_and_journey_inventory_is_deterministic_and_complete() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/end_user_ux_audit.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((ROOT / "docs/operator/END_USER_UX_INVENTORY_AUDIT_3.json").read_text(encoding="utf-8"))
    assert payload["claim_count"] >= 20
    assert payload["journey_count"] >= 15
    assert all(row["disposition"] in {"accurate", "resolved"} for row in payload["claims"])
    assert {row["id"] for row in payload["journeys"]} >= {
        "new_user_no_model",
        "task_reminder_continuity",
        "remote_pack_unavailable",
        "failed_indeterminate",
        "disabled_telegram",
        "restart_recovery",
    }


def test_capability_discovery_is_outcome_led_and_does_not_advertise_remote_pack_install() -> None:
    text = build_user_facing_capability_answer(search_available=False, safe_mode=True)
    assert "everyday questions" in text
    assert "arbitrary remote pack" in text
    assert "cannot download" in text
    assert "normal default" in text
    assert "need your approval" in text
    assert "capability_id" not in text
    assert "executor" not in text.lower()
    assert "quarantine review" not in text.lower()


def test_indeterminate_recovery_never_advises_blind_retry() -> None:
    recovery = build_failure_recovery("execution_indeterminate")
    assert recovery["status"] == "indeterminate"
    assert recovery["retryable"] is False
    assert recovery["recoverability"] == "reconciliation_required"
    assert "do not repeat" in recovery["next_step"].lower()
    mapped = AgentRuntime._failure_recovery_for_error(
        error="delivery_indeterminate",
        error_kind="indeterminate",
        message="The transport result was not recorded.",
    )
    assert mapped["kind"] == "execution_indeterminate"
    assert mapped["retryable"] is False


def test_primary_error_copy_keeps_diagnostics_secondary() -> None:
    text = deterministic_error_message(
        title="I couldn't finish that setup",
        trace_id="synthetic-trace",
        component="synthetic-component",
        next_action="check current setup status",
    )
    lines = text.splitlines()
    assert lines[:3] == [
        "I couldn't finish that setup",
        "I did not assume that the requested change succeeded.",
        "Next: check current setup status",
    ]
    assert lines.index("Diagnostic details (for support):") < lines.index("trace_id: synthetic-trace")


def test_remote_pack_error_offers_only_local_or_metadata_next_steps() -> None:
    recovery = AgentRuntime._failure_recovery_for_error(
        error="pack_not_found",
        error_kind="not_found",
        message="No local pack is installed.",
    )
    combined = " ".join(str(recovery.get(key) or "") for key in ("summary", "reason", "next_step")).lower()
    assert "remote acquisition" in combined
    assert "local text-pack" in combined
    assert "then install it" not in combined


def test_public_plan_summary_hides_private_targets_and_internal_ids() -> None:
    plan = {
        "schema_version": 1,
        "capability_id": "memory.forget",
        "executor_id": "memory.forget.v1",
        "action_type": "memory.reset",
        "risk_level": "critical",
        "mutation_inventory": [
            {
                "resources": [
                    "user_id:private-user",
                    "thread_id:private-thread",
                    "path:/private/home/secret-note.txt",
                    "record_id:private-record",
                ]
            }
        ],
        "recovery": {"rollback_available": False, "scope": "No automatic rollback."},
        "confirmation_requirement": {"required": True},
    }
    public = Orchestrator._plan_mode_public_payload(plan)
    rendered = json.dumps(public, sort_keys=True)
    assert public["resources"]["deleted"]
    assert "private-user" not in rendered
    assert "private-thread" not in rendered
    assert "secret-note" not in rendered
    assert "private-record" not in rendered
    assert "executor_id" not in rendered
    assert "capability_id" not in rendered
