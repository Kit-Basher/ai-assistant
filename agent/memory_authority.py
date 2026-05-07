from __future__ import annotations

import hashlib
import time
from typing import Any


AUTHORITY_CURRENT_USER_INPUT = "current_user_input"
AUTHORITY_SYSTEM_POLICY = "system_policy"
AUTHORITY_FRESH_RUNTIME_TRUTH = "fresh_runtime_truth"
AUTHORITY_DETERMINISTIC_APP_STATE = "deterministic_app_state"
AUTHORITY_CONTINUITY_STATE = "continuity_state"
AUTHORITY_WORKING_MEMORY_HOT = "working_memory_hot"
AUTHORITY_WORKING_MEMORY_SUMMARY = "working_memory_summary"
AUTHORITY_MEMORY_V2_DETERMINISTIC_FACT = "memory_v2_deterministic_fact"
AUTHORITY_SEMANTIC_CANDIDATE_EVIDENCE = "semantic_candidate_evidence"
AUTHORITY_GRAPH_CANDIDATE_EVIDENCE = "graph_candidate_evidence"
AUTHORITY_STALE_OR_LOW_CONFIDENCE = "stale_or_low_confidence"

MEMORY_AUTHORITY_LABELS = (
    AUTHORITY_CURRENT_USER_INPUT,
    AUTHORITY_SYSTEM_POLICY,
    AUTHORITY_FRESH_RUNTIME_TRUTH,
    AUTHORITY_DETERMINISTIC_APP_STATE,
    AUTHORITY_CONTINUITY_STATE,
    AUTHORITY_WORKING_MEMORY_HOT,
    AUTHORITY_WORKING_MEMORY_SUMMARY,
    AUTHORITY_MEMORY_V2_DETERMINISTIC_FACT,
    AUTHORITY_SEMANTIC_CANDIDATE_EVIDENCE,
    AUTHORITY_GRAPH_CANDIDATE_EVIDENCE,
    AUTHORITY_STALE_OR_LOW_CONFIDENCE,
)


def stable_redacted_id(value: Any, *, prefix: str = "mem") -> str | None:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def memory_v2_authority_label(*, level: str, is_current: bool = True) -> str:
    normalized = str(level or "").strip().lower()
    if not bool(is_current):
        return AUTHORITY_STALE_OR_LOW_CONFIDENCE
    if normalized == "semantic":
        return AUTHORITY_MEMORY_V2_DETERMINISTIC_FACT
    if normalized == "episodic":
        return AUTHORITY_CONTINUITY_STATE
    if normalized == "procedural":
        return AUTHORITY_DETERMINISTIC_APP_STATE
    return AUTHORITY_STALE_OR_LOW_CONFIDENCE


def semantic_authority_label(*, score: float | None = None, status: str | None = None) -> str:
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in {"", "ok"}:
        return AUTHORITY_STALE_OR_LOW_CONFIDENCE
    if score is not None and float(score) < 0.15:
        return AUTHORITY_STALE_OR_LOW_CONFIDENCE
    return AUTHORITY_SEMANTIC_CANDIDATE_EVIDENCE


def age_seconds(updated_at: Any, *, now_ts: int | None = None) -> int | None:
    try:
        updated = int(updated_at)
    except (TypeError, ValueError):
        return None
    if updated <= 0:
        return None
    now = int(now_ts if now_ts is not None else time.time())
    return max(0, now - updated)


def build_memory_injection_diagnostics(
    memory_context_payload: dict[str, Any] | None,
    *,
    now_ts: int | None = None,
) -> dict[str, Any]:
    payload = dict(memory_context_payload or {})
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    components = debug.get("components") if isinstance(debug.get("components"), dict) else {}
    semantic_payload = payload.get("semantic") if isinstance(payload.get("semantic"), dict) else {}
    semantic_debug = semantic_payload.get("debug") if isinstance(semantic_payload.get("debug"), dict) else {}
    memory_v2_enabled = bool((components.get("memory_v2") or {}).get("enabled")) if isinstance(components.get("memory_v2"), dict) else bool(payload.get("levels"))
    semantic_enabled = bool((components.get("semantic") or {}).get("enabled")) if isinstance(components.get("semantic"), dict) else bool(semantic_payload)
    context_text = str(payload.get("merged_context_text") or "").strip()

    entries: list[dict[str, Any]] = []
    selected_rows = debug.get("selected") if isinstance(debug.get("selected"), list) else []
    for row in selected_rows:
        if not isinstance(row, dict):
            continue
        level = str(row.get("level") or "").strip().lower()
        source_ref = str(row.get("source_ref") or "").strip()
        is_current = bool(row.get("is_current", True))
        label = memory_v2_authority_label(level=level, is_current=is_current)
        entries.append(
            {
                "layer": "memory_v2",
                "id": str(row.get("id") or "").strip(),
                "source_id": str(row.get("id") or "").strip(),
                "source_ref_hash": stable_redacted_id(source_ref, prefix="ref") if source_ref else None,
                "source_kind": str(row.get("source_kind") or "").strip() or None,
                "level": level or None,
                "authority_label": label,
                "score": row.get("score") if isinstance(row.get("score"), (int, float)) else None,
                "age_seconds": age_seconds(row.get("updated_at") or row.get("created_at"), now_ts=now_ts),
                "currentness": "current" if is_current else "stale_or_superseded",
                "reason_selected": row.get("why") if isinstance(row.get("why"), dict) else {},
            }
        )

    semantic_status = str(semantic_debug.get("status") or "").strip().lower()
    semantic_rows = semantic_debug.get("selected") if isinstance(semantic_debug.get("selected"), list) else []
    for row in semantic_rows:
        if not isinstance(row, dict):
            continue
        score = row.get("score") if isinstance(row.get("score"), (int, float)) else None
        source_ref = str(row.get("source_ref") or "").strip()
        entries.append(
            {
                "layer": "semantic",
                "id": str(row.get("id") or "").strip(),
                "source_id": str(row.get("source_id") or "").strip() or None,
                "source_ref_hash": stable_redacted_id(source_ref, prefix="ref") if source_ref else None,
                "source_kind": str(row.get("source_kind") or "").strip() or None,
                "authority_label": semantic_authority_label(score=float(score) if score is not None else None, status=semantic_status),
                "score": score,
                "confidence": row.get("similarity") if isinstance(row.get("similarity"), (int, float)) else None,
                "age_seconds": age_seconds(row.get("updated_at") or row.get("created_at"), now_ts=now_ts),
                "currentness": "candidate_evidence",
                "reason_selected": row.get("why") if isinstance(row.get("why"), dict) else {},
            }
        )

    failure_reasons: list[dict[str, Any]] = []
    for name, component in sorted(components.items(), key=lambda item: str(item[0])):
        if not isinstance(component, dict):
            continue
        reason = str(component.get("reason") or "").strip()
        if reason:
            failure_reasons.append({"layer": str(name), "reason": reason})
    for layer_name, layer_debug in (("memory_v2", debug), ("semantic", semantic_debug)):
        reason = str(layer_debug.get("reason") or "").strip() if isinstance(layer_debug, dict) else ""
        status = str(layer_debug.get("status") or "").strip() if isinstance(layer_debug, dict) else ""
        if reason:
            failure_reasons.append({"layer": layer_name, "reason": reason, "status": status or None})

    selected_layer_counts = {
        "memory_v2": len([entry for entry in entries if entry.get("layer") == "memory_v2"]),
        "semantic": len([entry for entry in entries if entry.get("layer") == "semantic"]),
    }
    return {
        "enabled_layers": {
            "continuity": True,
            "working_memory": True,
            "memory_v2": memory_v2_enabled,
            "semantic": semantic_enabled,
            "graph": False,
        },
        "selected_layer_counts": selected_layer_counts,
        "selected": entries,
        "failure_reasons": failure_reasons,
        "omitted": not bool(context_text),
        "omitted_reason": None if context_text else "no_relevant_memory_selected",
        "contents_exposed": False,
        "authority_labels": list(MEMORY_AUTHORITY_LABELS),
    }
