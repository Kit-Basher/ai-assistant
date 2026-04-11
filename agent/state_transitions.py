from __future__ import annotations

import json
from typing import Any, Mapping

from agent.packs.manifest import normalize_permissions
from agent.runtime_lifecycle import RuntimeLifecyclePhase

_STARTUP_PHASE_TRANSITIONS: dict[str, set[str]] = {
    "starting": {"starting", "listening", "warming", "ready", "degraded"},
    "listening": {"listening", "warming", "ready", "degraded"},
    "warming": {"warming", "ready", "degraded"},
    "ready": {"ready", "degraded"},
    "degraded": {"degraded", "ready"},
}

_RUNTIME_PHASE_TRANSITIONS: dict[str, set[str]] = {
    "boot": {"boot", "warmup", "ready", "degraded", "recovering"},
    "warmup": {"warmup", "ready", "degraded", "recovering"},
    "ready": {"ready", "degraded", "recovering"},
    "degraded": {"degraded", "ready", "recovering"},
    "recovering": {"recovering", "ready", "degraded"},
}


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _norm_state(value: Any) -> str:
    return _text(value).lower() or ""


def _json_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(_text(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _manifest_path_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def startup_phase_transition_allowed(previous: str | None, next_phase: str | None) -> bool:
    current = _norm_state(previous)
    desired = _norm_state(next_phase)
    if not current or not desired:
        return False
    return desired in _STARTUP_PHASE_TRANSITIONS.get(current, {current})


def runtime_phase_transition_allowed(
    previous: RuntimeLifecyclePhase | str | None,
    next_phase: RuntimeLifecyclePhase | str | None,
) -> bool:
    current = _norm_state(previous.value if isinstance(previous, RuntimeLifecyclePhase) else previous)
    desired = _norm_state(next_phase.value if isinstance(next_phase, RuntimeLifecyclePhase) else next_phase)
    if not current or not desired:
        return False
    return desired in _RUNTIME_PHASE_TRANSITIONS.get(current, {current})


def install_pack_write_is_noop(
    existing: Mapping[str, Any] | None,
    *,
    version: str,
    trust: str,
    manifest_path: str | None,
    permissions_json: str,
    permissions_hash: str,
    approved_permissions_hash: str | None,
    enabled_value: bool,
) -> bool:
    if not isinstance(existing, Mapping):
        return False
    existing_manifest = _manifest_path_text(existing.get("manifest_path"))
    desired_manifest = _manifest_path_text(manifest_path)
    existing_permissions_json = _json_payload(normalize_permissions(existing.get("permissions")))
    return (
        _text(existing.get("version")) == _text(version)
        and _text(existing.get("trust")) == _text(trust)
        and existing_manifest == desired_manifest
        and existing_permissions_json == _text(permissions_json)
        and _text(existing.get("permissions_hash")) == _text(permissions_hash)
        and _text(existing.get("approved_permissions_hash")) == _text(approved_permissions_hash)
        and bool(existing.get("enabled", False)) == bool(enabled_value)
    )


def pack_enabled_write_is_noop(existing: Mapping[str, Any] | None, *, enabled: bool) -> bool:
    if not isinstance(existing, Mapping):
        return False
    return bool(existing.get("enabled", False)) == bool(enabled)


def pack_approval_hash_write_is_noop(existing: Mapping[str, Any] | None, *, approval_hash: str | None) -> bool:
    if not isinstance(existing, Mapping):
        return False
    return _text(existing.get("approved_permissions_hash") or None) == _text(approval_hash)


def external_pack_record_is_noop(
    existing: Mapping[str, Any] | None,
    *,
    canonical_pack: Mapping[str, Any],
    classification: str,
    status: str,
    risk_report: Mapping[str, Any],
    review_envelope: Mapping[str, Any],
    quarantine_path: str | None,
    normalized_path: str | None,
) -> bool:
    if not isinstance(existing, Mapping):
        return False
    current_canonical = existing.get("canonical_pack") if isinstance(existing.get("canonical_pack"), Mapping) else {}
    current_risk = existing.get("risk_report") if isinstance(existing.get("risk_report"), Mapping) else {}
    current_review = existing.get("review_envelope") if isinstance(existing.get("review_envelope"), Mapping) else {}
    return (
        _text(existing.get("classification")) == _text(classification)
        and _text(existing.get("status")) == _text(status)
        and _text(existing.get("quarantine_path") or None) == _text(quarantine_path)
        and _text(existing.get("normalized_path") or None) == _text(normalized_path)
        and _json_payload(current_canonical) == _json_payload(canonical_pack)
        and _json_payload(current_risk) == _json_payload(risk_report)
        and _json_payload(current_review) == _json_payload(review_envelope)
    )


def confirmation_transition_state(
    state: Mapping[str, Any] | None,
    *,
    provided_token: str | None,
    recomputed_token: str | None,
    now_epoch: int,
) -> dict[str, Any]:
    payload = dict(state or {})
    step = _norm_state(payload.get("step"))
    pending_plan = payload.get("pending_plan") if isinstance(payload.get("pending_plan"), list) else []
    expected_token = _text(payload.get("pending_confirm_token") or payload.get("confirm_token") or None)
    consumed_token = _text(payload.get("last_confirm_token") or None)
    pending_expires_ts = 0
    try:
        pending_expires_ts = max(0, int(payload.get("pending_expires_ts") or 0))
    except (TypeError, ValueError):
        pending_expires_ts = 0
    if step == "completed" and consumed_token:
        return {
            "state": "already_consumed",
            "allowed": False,
            "error_kind": "needs_clarification",
            "message": "That confirmation was already used.",
            "next_question": "Ask me to fix it again.",
            "current_state": "completed",
            "current_step": step,
            "expected_token": None,
        }
    if not pending_plan or not expected_token:
        if consumed_token:
            return {
                "state": "already_consumed",
                "allowed": False,
                "error_kind": "needs_clarification",
                "message": "That confirmation was already used.",
                "next_question": "Ask me to fix it again.",
                "current_state": step or "idle",
                "current_step": step,
                "expected_token": None,
            }
        return {
            "state": "missing_pending_plan",
            "allowed": False,
            "error_kind": "needs_clarification",
            "message": "No pending plan to confirm.",
            "next_question": "Ask me to run fix-it again.",
            "current_state": step or "idle",
            "current_step": step,
            "expected_token": None,
        }
    if pending_expires_ts > 0 and now_epoch > pending_expires_ts:
        return {
            "state": "expired",
            "allowed": False,
            "error_kind": "needs_clarification",
            "message": "That confirmation expired. Ask me to fix it again.",
            "next_question": "Run fix-it again now?",
            "current_state": step,
            "current_step": step,
            "expected_token": expected_token,
        }
    if recomputed_token and recomputed_token != expected_token:
        return {
            "state": "plan_changed",
            "allowed": False,
            "error_kind": "needs_clarification",
            "message": "Pending plan changed. Ask me to fix it again.",
            "next_question": "Run fix-it again now?",
            "current_state": step,
            "current_step": step,
            "expected_token": expected_token,
        }
    if provided_token and provided_token != expected_token:
        return {
            "state": "token_mismatch",
            "allowed": False,
            "error_kind": "needs_clarification",
            "message": "Confirmation token did not match the pending plan.",
            "next_question": "Confirm the latest pending plan?",
            "current_state": step,
            "current_step": step,
            "expected_token": expected_token,
        }
    return {
        "state": "ok",
        "allowed": True,
        "error_kind": None,
        "message": "Confirmation accepted.",
        "next_question": None,
        "current_state": step,
        "current_step": step,
        "expected_token": expected_token,
    }
