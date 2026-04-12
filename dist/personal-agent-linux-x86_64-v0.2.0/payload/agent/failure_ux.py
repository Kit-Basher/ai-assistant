from __future__ import annotations

from typing import Any

from agent.persona import normalize_persona_text


_FAILURE_TEMPLATES: dict[str, dict[str, Any]] = {
    "runtime_initializing": {
        "category": "runtime",
        "status": "initializing",
        "state_label": "Initializing",
        "summary": "System is still initializing.",
        "reason": "Startup is still in progress.",
        "next_step": "Wait for startup to finish, then try again.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "runtime_degraded": {
        "category": "runtime",
        "status": "degraded",
        "state_label": "Degraded",
        "summary": "System is degraded.",
        "reason": "A dependency is unhealthy or missing.",
        "next_step": "Open /state or /ready to see what is degraded.",
        "retryable": True,
        "recoverability": "operator_fixable",
    },
    "runtime_blocked": {
        "category": "runtime",
        "status": "blocked",
        "state_label": "Blocked",
        "summary": "System is blocked.",
        "reason": "A required dependency or configuration is missing.",
        "next_step": "Fix the blocker, then try again.",
        "retryable": False,
        "recoverability": "operator_fixable",
    },
    "runtime_not_ready": {
        "category": "runtime",
        "status": "not_ready",
        "state_label": "Not ready",
        "summary": "System is not ready yet.",
        "reason": "Startup has not finished.",
        "next_step": "Wait for readiness, then retry.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "dependency_unavailable": {
        "category": "runtime",
        "status": "degraded",
        "state_label": "Dependency unavailable",
        "summary": "A required dependency is unavailable.",
        "reason": "The service or provider is not responding.",
        "next_step": "Retry later or restart the dependent service.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "pack_not_installed": {
        "category": "pack",
        "status": "missing",
        "state_label": "Available",
        "summary": "Pack is not installed.",
        "reason": "The capability exists as a pack, but it is not installed yet.",
        "next_step": "Open the preview, then install it if you want to use it.",
        "retryable": False,
        "recoverability": "user_fixable",
    },
    "pack_available_previewable": {
        "category": "pack",
        "status": "available",
        "state_label": "Available",
        "summary": "Pack is available to preview.",
        "reason": "The capability exists as a pack, but it is not installed yet.",
        "next_step": "Open the preview, then install it if you want to use it.",
        "retryable": False,
        "recoverability": "user_fixable",
    },
    "pack_task_unconfirmed": {
        "category": "pack",
        "status": "limited",
        "state_label": "Installed · Healthy",
        "summary": "Pack is installed and healthy, but task usability is not confirmed.",
        "reason": "The pack is installed, but I cannot prove it fits this task yet.",
        "next_step": "Open the pack preview before relying on it.",
        "retryable": False,
        "recoverability": "user_fixable",
    },
    "pack_disabled": {
        "category": "pack",
        "status": "disabled",
        "state_label": "Installed · Disabled",
        "summary": "Pack is installed but disabled.",
        "reason": "It is not enabled as a live capability.",
        "next_step": "Enable it before using it.",
        "retryable": False,
        "recoverability": "user_fixable",
    },
    "pack_blocked": {
        "category": "pack",
        "status": "blocked",
        "state_label": "Installed · Blocked",
        "summary": "Pack is installed but blocked.",
        "reason": "It is blocked by policy, missing files, or another unmet requirement.",
        "next_step": "Review the blocker, then reinstall or choose another pack.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "pack_missing_files": {
        "category": "pack",
        "status": "blocked",
        "state_label": "Installed · Blocked",
        "summary": "Pack is installed, but files are missing.",
        "reason": "The stored pack state points at files that are no longer there.",
        "next_step": "Reinstall the pack.",
        "retryable": False,
        "recoverability": "user_fixable",
    },
    "pack_invalid_metadata": {
        "category": "pack",
        "status": "blocked",
        "state_label": "Blocked",
        "summary": "Pack metadata is invalid or incomplete.",
        "reason": "The manifest or stored metadata cannot be trusted as-is.",
        "next_step": "Recreate the pack from a valid manifest or rebuild the metadata.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "pack_enable_missing_install": {
        "category": "pack",
        "status": "missing",
        "state_label": "Available",
        "summary": "Pack is not installed, so it cannot be enabled.",
        "reason": "There is no installed pack record yet.",
        "next_step": "Install the pack first.",
        "retryable": False,
        "recoverability": "user_fixable",
    },
    "pack_remove_missing": {
        "category": "pack",
        "status": "already_removed",
        "state_label": "Removed",
        "summary": "Pack is already removed.",
        "reason": "There is no installed pack record left to remove.",
        "next_step": "No action needed.",
        "retryable": False,
        "recoverability": "no_action_needed",
    },
    "pack_install_invalid": {
        "category": "pack",
        "status": "blocked",
        "state_label": "Blocked",
        "summary": "Pack cannot be installed from its current metadata.",
        "reason": "The source content is invalid or incomplete for safe import.",
        "next_step": "Fix the source manifest or use a valid pack source.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "pack_install_noop": {
        "category": "pack",
        "status": "already_installed",
        "state_label": "Installed",
        "summary": "Pack is already installed.",
        "reason": "The same pack state is already present.",
        "next_step": "No action needed.",
        "retryable": False,
        "recoverability": "no_action_needed",
    },
    "confirm_token_stale": {
        "category": "approval",
        "status": "stale",
        "state_label": "Confirmation stale",
        "summary": "Confirmation token is stale.",
        "reason": "The preview or pending plan changed before confirmation.",
        "next_step": "Request a new preview, then confirm again.",
        "retryable": True,
        "recoverability": "user_fixable",
    },
    "confirm_token_consumed": {
        "category": "approval",
        "status": "consumed",
        "state_label": "Confirmation consumed",
        "summary": "Confirmation token was already used.",
        "reason": "That confirm has already been applied.",
        "next_step": "No action needed unless you want a new preview.",
        "retryable": False,
        "recoverability": "no_action_needed",
    },
    "confirm_token_expired": {
        "category": "approval",
        "status": "expired",
        "state_label": "Confirmation expired",
        "summary": "Confirmation token expired.",
        "reason": "The pending plan timed out before confirmation.",
        "next_step": "Request a new preview, then confirm it.",
        "retryable": True,
        "recoverability": "user_fixable",
    },
    "confirm_plan_missing": {
        "category": "approval",
        "status": "missing",
        "state_label": "No pending plan",
        "summary": "No pending plan is available to confirm.",
        "reason": "There is no preview waiting for approval.",
        "next_step": "Ask me to generate a new preview.",
        "retryable": True,
        "recoverability": "user_fixable",
    },
    "confirm_token_mismatch": {
        "category": "approval",
        "status": "mismatch",
        "state_label": "Token mismatch",
        "summary": "Confirmation token did not match the pending plan.",
        "reason": "The preview changed or the token was copied from the wrong plan.",
        "next_step": "Confirm the latest pending plan.",
        "retryable": True,
        "recoverability": "user_fixable",
    },
    "confirm_downstream_failed": {
        "category": "approval",
        "status": "failed",
        "state_label": "Confirmation failed",
        "summary": "Confirmation was accepted, but the downstream step failed.",
        "reason": "The plan started, then an execution step failed.",
        "next_step": "Fix the downstream blocker, then retry the plan.",
        "retryable": True,
        "recoverability": "operator_fixable",
    },
    "discovery_unavailable": {
        "category": "discovery",
        "status": "unavailable",
        "state_label": "Discovery unavailable",
        "summary": "Discovery is unavailable right now.",
        "reason": "The pack source list could not be read.",
        "next_step": "Retry later or continue without recommendations.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "discovery_degraded": {
        "category": "discovery",
        "status": "degraded",
        "state_label": "Discovery degraded",
        "summary": "Discovery is degraded right now.",
        "reason": "Some pack sources are unavailable or returned errors.",
        "next_step": "Retry later or use the installed pack list.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "recommendation_no_source": {
        "category": "discovery",
        "status": "unavailable",
        "state_label": "Recommendations unavailable",
        "summary": "I cannot make a pack recommendation right now.",
        "reason": "The discovery source needed for this recommendation is unavailable.",
        "next_step": "Retry later or keep using text-only help.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "db_busy": {
        "category": "persistence",
        "status": "busy",
        "state_label": "Persistence busy",
        "summary": "State storage is temporarily busy.",
        "reason": "The database is locked or another write is in progress.",
        "next_step": "Retry the request once the current write finishes.",
        "retryable": True,
        "recoverability": "retryable",
    },
    "missing_persisted_state": {
        "category": "persistence",
        "status": "missing",
        "state_label": "State missing",
        "summary": "Persisted state is missing.",
        "reason": "The record or file that should hold the state could not be found.",
        "next_step": "Recreate or reinstall the missing state.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "partial_persisted_state": {
        "category": "persistence",
        "status": "partial",
        "state_label": "Partial state",
        "summary": "Persisted state is partial.",
        "reason": "Some required fields or files are missing.",
        "next_step": "Rebuild the state from a clean source.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "corrupted_metadata": {
        "category": "persistence",
        "status": "corrupted",
        "state_label": "Corrupted metadata",
        "summary": "Persisted metadata is corrupted.",
        "reason": "The stored JSON or manifest could not be parsed reliably.",
        "next_step": "Rebuild the metadata from a valid source.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "file_missing_after_state": {
        "category": "persistence",
        "status": "blocked",
        "state_label": "File missing",
        "summary": "A file is missing even though state says it should exist.",
        "reason": "The state and filesystem are out of sync.",
        "next_step": "Reinstall or rescan the pack.",
        "retryable": False,
        "recoverability": "user_or_operator_fixable",
    },
    "unknown": {
        "category": "unknown",
        "status": "unknown",
        "state_label": "Unknown",
        "summary": "Something failed, but I cannot prove the exact reason.",
        "reason": "The system did not return enough truth to classify the failure precisely.",
        "next_step": "Run the request again or inspect the relevant state surface.",
        "retryable": True,
        "recoverability": "retryable",
    },
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _compose_message(summary: str, reason: str | None, next_step: str | None) -> str:
    parts = [_text(summary)]
    if reason:
        parts.append(_text(reason))
    message = " ".join(part for part in parts if part).strip()
    if next_step:
        message = f"{message} Next: {_text(next_step).rstrip('.')}."
    return normalize_persona_text(message)


def build_failure_recovery(
    kind: str,
    *,
    subject: str | None = None,
    state_label: str | None = None,
    status: str | None = None,
    blocker: str | None = None,
    reason: str | None = None,
    next_step: str | None = None,
    retryable: bool | None = None,
    current_state: str | None = None,
    details: str | None = None,
) -> dict[str, Any]:
    key = _norm(kind) or "unknown"
    spec = dict(_FAILURE_TEMPLATES.get(key) or _FAILURE_TEMPLATES["unknown"])
    state_label_value = _text(state_label or spec.get("state_label")) or "Unknown"
    summary_value = _text(spec.get("summary"))
    reason_value = _text(reason or blocker or spec.get("reason"))
    next_step_value = _text(next_step or spec.get("next_step"))
    status_value = _norm(status or spec.get("status") or key) or "unknown"
    retryable_value = bool(spec.get("retryable", True) if retryable is None else retryable)
    recovery = {
        "kind": key,
        "category": _norm(spec.get("category")) or "unknown",
        "status": status_value,
        "state_label": state_label_value,
        "summary": normalize_persona_text(summary_value),
        "reason": normalize_persona_text(reason_value) if reason_value else None,
        "next_step": normalize_persona_text(next_step_value) if next_step_value else None,
        "retryable": retryable_value,
        "recoverability": _norm(spec.get("recoverability")) or ("retryable" if retryable_value else "user_fixable"),
        "blocker": _text(blocker) or None,
        "subject": _text(subject) or None,
        "current_state": _text(current_state) or None,
        "details": _text(details) or None,
    }
    recovery["message"] = _compose_message(
        recovery["summary"],
        recovery.get("reason"),
        recovery.get("next_step"),
    )
    return recovery


def failure_recovery_message(kind: str, **context: Any) -> str:
    return str(build_failure_recovery(kind, **context).get("message") or "").strip()


__all__ = [
    "build_failure_recovery",
    "failure_recovery_message",
]
