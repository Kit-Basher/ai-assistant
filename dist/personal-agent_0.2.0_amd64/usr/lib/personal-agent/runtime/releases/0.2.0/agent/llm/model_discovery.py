from __future__ import annotations

from typing import Any, Mapping

_PROPOSAL_KINDS = {"candidate_good", "candidate_stale", "insufficient_metadata"}
_CONFIDENCE_LEVELS = {"high", "medium", "low"}
_PROPOSED_ROLES = {"coding", "research", "cheap_cloud", "local_best"}
_PROPOSAL_SOURCES = {"registry", "runtime_inventory", "external_openrouter_snapshot"}


def allowed_model_discovery_proposal_kinds() -> list[str]:
    return sorted(_PROPOSAL_KINDS)


def allowed_model_discovery_proposal_sources() -> list[str]:
    return sorted(_PROPOSAL_SOURCES)


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _normalized_string_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {
        str(item).strip().lower()
        for item in values
        if str(item).strip()
    }


def _policy_entry_by_model_id(policy_entries: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in policy_entries or []:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("model_id") or "").strip()
        if model_id:
            rows[model_id] = dict(row)
    return rows


def _proposal_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    kind = str(row.get("proposal_kind") or "").strip().lower()
    confidence = str(row.get("confidence") or "").strip().lower()
    kind_rank = {
        "candidate_good": 0,
        "candidate_stale": 1,
        "insufficient_metadata": 2,
    }.get(kind, 9)
    confidence_rank = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }.get(confidence, 9)
    return (
        kind_rank,
        confidence_rank,
        str(row.get("model_id") or ""),
    )


def _review_suggestion_for_proposal(payload: Mapping[str, Any]) -> dict[str, Any]:
    proposal_kind = str(payload.get("proposal_kind") or "").strip().lower()
    model_id = str(payload.get("model_id") or "").strip()
    proposed_roles = [
        item
        for item in sorted(_normalized_string_set(payload.get("proposed_roles")))
        if item in _PROPOSED_ROLES
    ]
    reason_codes = _normalized_string_set(payload.get("reason_codes"))
    policy_status = str(payload.get("policy_status") or "").strip().lower() or None
    if proposal_kind == "candidate_good":
        notes = "Proposal reviewed from structured discovery metadata."
        return {
            "available": True,
            "write_surface": "/llm/models/policy",
            "suggested_status": "known_good",
            "suggested_role_hints": proposed_roles,
            "suggested_notes": notes,
            "payload_template": {
                "model_id": model_id,
                "status": "known_good",
                "role_hints": proposed_roles,
                "notes": notes,
                "source": "operator_review",
                "justification": "Accepted from discovery proposal",
                "reviewed_at": None,
            },
        }
    if proposal_kind == "candidate_stale":
        suggested_status = "avoid" if policy_status == "avoid" or "policy_avoid" in reason_codes else "known_stale"
        notes = "Proposal reviewed as stale from curated policy or discovery evidence."
        return {
            "available": True,
            "write_surface": "/llm/models/policy",
            "suggested_status": suggested_status,
            "suggested_role_hints": [],
            "suggested_notes": notes,
            "payload_template": {
                "model_id": model_id,
                "status": suggested_status,
                "role_hints": [],
                "notes": notes,
                "source": "operator_review",
                "justification": "Reviewed from discovery proposal",
                "reviewed_at": None,
            },
        }
    return {
        "available": False,
        "reason_code": (
            "insufficient_metadata"
            if proposal_kind == "insufficient_metadata"
            else "no_supported_review_action"
        ),
    }


def _finalize_proposal(payload: dict[str, Any]) -> dict[str, Any]:
    proposal = dict(payload)
    proposal["review_suggestion"] = _review_suggestion_for_proposal(proposal)
    return proposal


def _proposal_for_inventory_row(
    row: Mapping[str, Any],
    *,
    policy_entry: Mapping[str, Any] | None,
    cheap_remote_cap_per_1m: float,
) -> dict[str, Any]:
    model_id = str(row.get("model_id") or row.get("id") or "").strip()
    provider_id = str(row.get("provider_id") or row.get("provider") or "").strip().lower()
    model_name = str(row.get("model_name") or row.get("model") or model_id).strip()
    capabilities = sorted(_normalized_string_set(row.get("capabilities")))
    task_types = sorted(_normalized_string_set(row.get("task_types")))
    modalities = sorted(_normalized_string_set(row.get("modalities")))
    input_modalities = sorted(_normalized_string_set(row.get("input_modalities")))
    output_modalities = sorted(_normalized_string_set(row.get("output_modalities")))
    architecture_modality = str(row.get("architecture_modality") or "").strip().lower() or None
    row_source = str(row.get("source") or "runtime_inventory").strip() or "runtime_inventory"
    context_window = int(row.get("context_window") or 0) or None
    price_in = _safe_float(row.get("price_in"))
    price_out = _safe_float(row.get("price_out"))
    expected_cost_per_1m = (
        round(float(price_in) + (2.0 * float(price_out)), 6)
        if price_in is not None and price_out is not None
        else None
    )
    local = bool(row.get("local", False))
    available = bool(row.get("available", False))
    policy_payload = dict(policy_entry) if isinstance(policy_entry, Mapping) else {}
    policy_status = str(policy_payload.get("status") or "").strip().lower() or None
    policy_role_hints = [
        item
        for item in sorted(_normalized_string_set(policy_payload.get("role_hints")))
        if item in _PROPOSED_ROLES
    ]

    proposed_roles: list[str] = []
    reason_codes: list[str] = []
    notes: list[str] = []

    if policy_status == "known_good":
        reason_codes.append("policy_known_good")
        proposed_roles.extend(policy_role_hints)
        if str(policy_payload.get("notes") or "").strip():
            notes.append(str(policy_payload.get("notes") or "").strip())
    elif policy_status in {"known_stale", "avoid"}:
        reason_codes.append("policy_avoid" if policy_status == "avoid" else "policy_known_stale")
        if str(policy_payload.get("notes") or "").strip():
            notes.append(str(policy_payload.get("notes") or "").strip())
        return _finalize_proposal({
            "model_id": model_id,
            "provider_id": provider_id,
            "model_name": model_name,
            "proposal_kind": "candidate_stale",
            "proposal_state": "proposed",
            "proposed_roles": policy_role_hints,
            "evidence": {
                "task_types": task_types,
                "context_window": context_window,
                "price_in": price_in,
                "price_out": price_out,
                "expected_cost_per_1m": expected_cost_per_1m,
                "capabilities": capabilities,
                "modalities": modalities,
                "architecture_modality": architecture_modality,
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "source_metadata_fields": [
                    name
                    for name, present in (
                        ("task_types", bool(task_types)),
                        ("context_window", context_window is not None),
                        ("pricing", expected_cost_per_1m is not None),
                        ("modality", bool(modalities or architecture_modality or input_modalities or output_modalities)),
                    )
                    if present
                ],
            },
            "confidence": "high",
            "reason_codes": reason_codes,
            "review_required": True,
            "non_canonical": True,
            "canonical_status": "not_adopted",
            "policy_status": policy_status,
            "notes": notes,
            "available": available,
            "local": local,
            "source": row_source,
        })

    if "coding" in task_types:
        proposed_roles.append("coding")
        reason_codes.append("explicit_coding_task_type")
    if ("reasoning" in task_types or "research" in task_types) and int(context_window or 0) >= 131072:
        proposed_roles.append("research")
        reason_codes.extend(["explicit_reasoning_task_type", "explicit_large_context"])
    if (
        not local
        and "chat" in capabilities
        and expected_cost_per_1m is not None
        and float(expected_cost_per_1m) <= float(max(0.0, cheap_remote_cap_per_1m))
    ):
        proposed_roles.append("cheap_cloud")
        reason_codes.extend(["explicit_general_chat_capability", "explicit_low_cost_remote"])

    if policy_status == "known_good":
        proposed_roles.extend(policy_role_hints)

    proposed_roles = sorted({item for item in proposed_roles if item in _PROPOSED_ROLES})
    source_metadata_fields = [
        name
        for name, present in (
            ("task_types", bool(task_types)),
            ("context_window", context_window is not None),
            ("pricing", expected_cost_per_1m is not None),
            ("modality", bool(modalities or architecture_modality or input_modalities or output_modalities)),
        )
        if present
    ]

    if proposed_roles or policy_status == "known_good":
        confidence = "high" if {"coding", "research"} & set(proposed_roles) or policy_status == "known_good" else "medium"
        return _finalize_proposal({
            "model_id": model_id,
            "provider_id": provider_id,
            "model_name": model_name,
            "proposal_kind": "candidate_good",
            "proposal_state": "proposed",
            "proposed_roles": proposed_roles,
            "evidence": {
                "task_types": task_types,
                "context_window": context_window,
                "price_in": price_in,
                "price_out": price_out,
                "expected_cost_per_1m": expected_cost_per_1m,
                "capabilities": capabilities,
                "modalities": modalities,
                "architecture_modality": architecture_modality,
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
                "source_metadata_fields": source_metadata_fields,
            },
            "confidence": confidence,
            "reason_codes": sorted(set(reason_codes)),
            "review_required": True,
            "non_canonical": True,
            "canonical_status": "not_adopted",
            "policy_status": policy_status,
            "notes": notes,
            "available": available,
            "local": local,
            "source": row_source,
        })

    return _finalize_proposal({
        "model_id": model_id,
        "provider_id": provider_id,
        "model_name": model_name,
        "proposal_kind": "insufficient_metadata",
        "proposal_state": "proposed",
        "proposed_roles": [],
        "evidence": {
            "task_types": task_types,
            "context_window": context_window,
            "price_in": price_in,
            "price_out": price_out,
            "expected_cost_per_1m": expected_cost_per_1m,
            "capabilities": capabilities,
            "modalities": modalities,
            "architecture_modality": architecture_modality,
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "source_metadata_fields": source_metadata_fields,
        },
        "confidence": "low",
        "reason_codes": ["insufficient_structured_metadata"],
        "review_required": True,
        "non_canonical": True,
        "canonical_status": "not_adopted",
        "policy_status": policy_status,
        "notes": notes,
        "available": available,
        "local": local,
        "source": row_source,
    })


def build_model_discovery_proposals(
    *,
    inventory_rows: list[dict[str, Any]] | None,
    policy_entries: list[dict[str, Any]] | None,
    cheap_remote_cap_per_1m: float,
) -> list[dict[str, Any]]:
    policy_by_model = _policy_entry_by_model_id(policy_entries)
    proposals: list[dict[str, Any]] = []
    for row in inventory_rows or []:
        if not isinstance(row, Mapping):
            continue
        model_id = str(row.get("model_id") or row.get("id") or "").strip()
        provider_id = str(row.get("provider_id") or row.get("provider") or "").strip().lower()
        if not model_id or not provider_id:
            continue
        proposals.append(
            _proposal_for_inventory_row(
                row,
                policy_entry=policy_by_model.get(model_id),
                cheap_remote_cap_per_1m=cheap_remote_cap_per_1m,
            )
        )
    proposals.sort(key=_proposal_sort_key)
    return proposals


__all__ = [
    "allowed_model_discovery_proposal_kinds",
    "allowed_model_discovery_proposal_sources",
    "build_model_discovery_proposals",
]
