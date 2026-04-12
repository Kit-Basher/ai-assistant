from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

from agent.failure_ux import build_failure_recovery


def _clean_text(value: Any, default: str | None = None) -> str | None:
    text = " ".join(str(value or "").strip().replace("_", " ").split())
    return text or default


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _path_exists(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        return Path(text).exists()
    except OSError:
        return False


def _state_severity(state: str) -> str:
    normalized = str(state or "").strip().lower()
    if normalized in {"installed_healthy", "available"}:
        return "ready"
    if normalized in {"installed_disabled", "installed_limited", "installed_unknown", "previewable"}:
        return "degraded"
    if normalized in {"blocked", "incompatible", "installed_blocked"}:
        return "blocked"
    return "degraded"


def _installed_pack_metadata_blocker(canonical: dict[str, Any] | None) -> str | None:
    pack = canonical if isinstance(canonical, dict) else {}
    if not pack:
        return "pack metadata is incomplete"
    pack_identity = pack.get("pack_identity") if isinstance(pack.get("pack_identity"), dict) else {}
    if not str(pack_identity.get("canonical_id") or "").strip():
        return "pack identity is missing"
    if not isinstance(pack.get("source"), dict):
        return "pack source is missing"
    if not str(pack.get("name") or pack.get("display_name") or "").strip():
        return "pack name is missing"
    return None


def _base_normalized_state() -> dict[str, Any]:
    return {
        "discovery_state": "unknown",
        "install_state": "unknown",
        "activation_state": "unknown",
        "health_state": "unknown",
        "compatibility_state": "unknown",
        "usability_state": "unknown",
        "installed": False,
        "enabled": None,
        "healthy": None,
        "machine_usable": False,
        "task_usable": False,
        "previewable": False,
        "installable": False,
        "blocked": False,
        "unknown": True,
    }


def normalize_installed_pack_truth(row: dict[str, Any] | None) -> dict[str, Any]:
    pack_row = row if isinstance(row, dict) else {}
    canonical = pack_row.get("canonical_pack") if isinstance(pack_row.get("canonical_pack"), dict) else {}
    review = pack_row.get("review_envelope") if isinstance(pack_row.get("review_envelope"), dict) else {}
    risk = pack_row.get("risk_report") if isinstance(pack_row.get("risk_report"), dict) else {}
    status = str(pack_row.get("status") or "").strip().lower()
    enabled_raw = pack_row.get("enabled") if "enabled" in pack_row else None
    enabled_known = enabled_raw is True or enabled_raw is False
    normalized_path = str(pack_row.get("normalized_path") or "").strip()
    normalized_path_known = bool(normalized_path)
    normalized_path_exists = _path_exists(normalized_path)
    metadata_blocker = _installed_pack_metadata_blocker(canonical)

    normalized = _base_normalized_state()
    normalized.update(
        {
            "discovery_state": "discovered",
            "install_state": "installed",
            "installed": True,
            "enabled": enabled_raw if enabled_known else None,
            "previewable": False,
            "installable": False,
            "unknown": False,
        }
    )

    if metadata_blocker is not None:
        normalized.update(
            {
                "health_state": "failing",
                "compatibility_state": "blocked",
                "usability_state": "unusable",
                "healthy": False,
                "machine_usable": False,
                "task_usable": False,
                "blocked": True,
                "state_key": "installed_blocked",
                "state_label": "Installed · Blocked",
                "status_note": "Installed, but the pack metadata is incomplete.",
                "blocker": metadata_blocker,
                "next_action": "Rebuild the manifest or reinstall the pack.",
            }
        )
    elif status == "blocked":
        why_risk = review.get("why_risk") if isinstance(review.get("why_risk"), list) else []
        blocker_source = risk.get("blocked_reason") or (why_risk[0] if why_risk else None)
        blocker = _clean_text(blocker_source)
        if blocker is None:
            blocker = "blocked during import"
        normalized.update(
            {
                "health_state": "failing",
                "compatibility_state": "blocked",
                "usability_state": "unusable",
                "healthy": False,
                "machine_usable": False,
                "task_usable": False,
                "blocked": True,
                "state_key": "installed_blocked",
                "state_label": "Installed · Blocked",
                "status_note": "Installed, but blocked during import.",
                "blocker": blocker,
                "next_action": "Review the blocked import details.",
            }
        )
    elif status == "partial_safe_import":
        if normalized_path_known and not normalized_path_exists:
            normalized.update(
                {
                    "health_state": "failing",
                    "compatibility_state": "blocked",
                    "usability_state": "unusable",
                    "healthy": False,
                    "machine_usable": False,
                    "task_usable": False,
                    "blocked": True,
                    "state_key": "installed_blocked",
                    "state_label": "Installed · Blocked",
                    "status_note": "Installed, but the normalized files are missing.",
                    "blocker": "normalized files are missing",
                    "next_action": "Reinstall the pack.",
                }
            )
        elif not normalized_path_known:
            normalized.update(
                {
                    "health_state": "unknown",
                    "compatibility_state": "unconfirmed",
                    "usability_state": "unknown",
                    "healthy": None,
                    "machine_usable": False,
                    "task_usable": False,
                    "state_key": "installed_unknown",
                    "state_label": "Installed · Unknown",
                    "status_note": "Installed, but the normalized files are not recorded.",
                    "blocker": "normalized path not recorded",
                    "next_action": "Reinstall or rescan the pack.",
                }
            )
        else:
            normalized.update(
                {
                    "health_state": "degraded",
                    "compatibility_state": "unconfirmed",
                    "usability_state": "unusable",
                    "healthy": False,
                    "machine_usable": False,
                    "task_usable": False,
                    "state_key": "installed_limited",
                    "state_label": "Installed · Limited",
                    "status_note": "Installed, but compatibility is not fully confirmed.",
                    "blocker": "compatibility not confirmed",
                    "next_action": "Open the pack preview before relying on it.",
                }
            )
    elif status == "normalized":
        if normalized_path_known and not normalized_path_exists:
            normalized.update(
                {
                    "health_state": "failing",
                    "compatibility_state": "blocked",
                    "usability_state": "unusable",
                    "healthy": False,
                    "machine_usable": False,
                    "task_usable": False,
                    "blocked": True,
                    "state_key": "installed_blocked",
                    "state_label": "Installed · Blocked",
                    "status_note": "Installed, but the normalized files are missing.",
                    "blocker": "normalized files are missing",
                    "next_action": "Reinstall the pack.",
                }
            )
        elif not normalized_path_known:
            normalized.update(
                {
                    "health_state": "unknown",
                    "compatibility_state": "unconfirmed",
                    "usability_state": "unknown",
                    "healthy": None,
                    "machine_usable": False,
                    "task_usable": False,
                    "state_key": "installed_unknown",
                    "state_label": "Installed · Unknown",
                    "status_note": "Installed, but the normalized files are not recorded.",
                    "blocker": "normalized path not recorded",
                    "next_action": "Reinstall or rescan the pack.",
                }
            )
        if str(normalized.get("state_key") or "").strip().lower() not in {"installed_blocked", "installed_unknown"}:
            if enabled_raw is True:
                normalized.update(
                    {
                        "activation_state": "enabled",
                        "health_state": "healthy",
                        "compatibility_state": "compatible",
                        "usability_state": "task_unconfirmed",
                        "healthy": True,
                        "machine_usable": True,
                        "task_usable": False,
                        "state_key": "installed_healthy",
                        "state_label": "Installed · Healthy",
                        "status_note": "Installed and healthy, but task usability is not confirmed.",
                        "blocker": "task compatibility not confirmed",
                        "next_action": "Open the pack preview before relying on it.",
                    }
                )
            elif enabled_known and enabled_raw is False:
                normalized.update(
                    {
                        "activation_state": "disabled",
                        "health_state": "healthy",
                        "compatibility_state": "compatible",
                        "usability_state": "unusable",
                        "healthy": True,
                        "machine_usable": False,
                        "task_usable": False,
                        "state_key": "installed_disabled",
                        "state_label": "Installed · Disabled",
                        "status_note": "Installed, but disabled.",
                        "blocker": "not enabled as a live capability",
                        "next_action": "Enable it before using it.",
                    }
                )
            else:
                normalized.update(
                    {
                        "activation_state": "unknown",
                        "health_state": "healthy",
                        "compatibility_state": "compatible",
                        "usability_state": "task_unconfirmed",
                        "healthy": True,
                        "machine_usable": True,
                        "task_usable": False,
                        "state_key": "installed_healthy",
                        "state_label": "Installed · Healthy",
                        "status_note": "Installed and healthy, but task usability is not confirmed.",
                        "blocker": "task compatibility not confirmed",
                        "next_action": "Open the pack preview before relying on it.",
                    }
                )
    else:
        normalized.update(
            {
                "health_state": "unknown",
                "compatibility_state": "unconfirmed",
                "usability_state": "unknown",
                "healthy": None,
                "machine_usable": False,
                "task_usable": False,
                "state_key": "installed_unknown",
                "state_label": "Installed · Unknown",
                "status_note": "Installed, but compatibility is not confirmed yet.",
                "blocker": "compatibility not confirmed",
                "next_action": "Open the pack details to confirm compatibility.",
            }
        )

    normalized["severity"] = _state_severity(str(normalized.get("state_key") or "unknown"))
    normalized["summary_label"] = str(normalized.get("state_label") or "Unknown")
    normalized["compatibility_note"] = str(normalized.get("status_note") or "")
    normalized["capabilities"] = _clean_list((canonical.get("capabilities") if isinstance(canonical.get("capabilities"), dict) else {}).get("declared"))
    normalized["review_required"] = bool(review.get("review_required", True))
    normalized["metadata_blocker"] = metadata_blocker
    return normalized


def normalize_available_pack_truth(source: dict[str, Any] | None, listing: dict[str, Any] | None) -> dict[str, Any]:
    source_row = source if isinstance(source, dict) else {}
    listing_row = listing if isinstance(listing, dict) else {}
    installable = bool(listing_row.get("installable_by_current_policy", False))
    previewable = bool(listing_row.get("preview_available", True))
    artifact_type = str(listing_row.get("artifact_type_hint") or "").strip() or "unknown"
    tags = _clean_list(listing_row.get("tags"))
    badges = _clean_list(listing_row.get("badges"))
    name = _clean_text(listing_row.get("name"), "Imported pack")
    normalized = _base_normalized_state()
    normalized.update(
        {
            "discovery_state": "previewable" if previewable else "discovered",
            "install_state": "installable" if installable else "not_installed",
            "activation_state": "unknown",
            "health_state": "unknown" if installable else "failing",
            "compatibility_state": "unconfirmed" if installable else "blocked",
            "usability_state": "unknown" if installable else "unusable",
            "installed": False,
            "enabled": False,
            "healthy": None if installable else False,
            "machine_usable": False,
            "task_usable": False,
            "previewable": previewable,
            "installable": installable,
            "blocked": not installable,
            "unknown": False,
            "state_key": "available" if installable else "blocked",
            "state_label": "Available" if installable else "Blocked",
            "status_note": "Available to preview." if installable else "Blocked by current policy.",
            "blocker": None if installable else _clean_text(
                listing_row.get("install_block_reason_if_known")
                or listing_row.get("blocked_reason")
                or listing_row.get("policy_hint")
                or "current policy blocks it"
            ),
            "next_action": "Open the preview before installing." if installable else "Review the blocker before installing.",
            "badges": badges,
            "capabilities": tags or ([artifact_type] if artifact_type and artifact_type != "unknown" else []),
            "summary_label": "Available" if installable else "Blocked",
            "source_label": _clean_text(source_row.get("name") or source_row.get("kind") or source_row.get("id"), "unknown"),
            "source_kind": _clean_text(source_row.get("kind"), None),
            "type": artifact_type,
        }
    )
    normalized["severity"] = _state_severity(str(normalized.get("state_key") or "unknown"))
    normalized["compatibility_note"] = str(normalized.get("status_note") or "")
    return normalized


def _source_label(source: dict[str, Any] | None, *, fallback: dict[str, Any] | None = None) -> str:
    source_row = source if isinstance(source, dict) else {}
    fallback_row = fallback if isinstance(fallback, dict) else {}
    for value in (
        source_row.get("name"),
        source_row.get("kind"),
        fallback_row.get("origin"),
        fallback_row.get("name"),
        fallback_row.get("kind"),
    ):
        cleaned = " ".join(str(value or "").strip().replace("_", " ").split())
        if cleaned:
            return cleaned
    return "unknown"


def _pack_name(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return "Imported pack"
    review = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    for value in (
        review.get("pack_name"),
        canonical.get("display_name"),
        canonical.get("name"),
        source.get("display_name"),
        source.get("name"),
        row.get("name"),
    ):
        cleaned = " ".join(str(value or "").strip().replace("_", "-").split())
        if cleaned:
            return cleaned
    return "Imported pack"


def _pack_state_installed_row(row: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    normalized = normalize_installed_pack_truth(row)
    canonical = row.get("canonical_pack") if isinstance(row.get("canonical_pack"), dict) else {}
    review = row.get("review_envelope") if isinstance(row.get("review_envelope"), dict) else {}
    source = canonical.get("source") if isinstance(canonical.get("source"), dict) else {}
    capabilities = canonical.get("capabilities") if isinstance(canonical.get("capabilities"), dict) else {}
    declared = _clean_list(capabilities.get("declared"))
    inferred = _clean_list(capabilities.get("inferred"))
    summary = str(capabilities.get("summary") or row.get("review_summary") or review.get("summary") or "").strip()
    if not declared and not inferred and summary:
        declared = [summary]
    cap_values = declared or inferred or ([summary] if summary else [])
    pack_id = str(row.get("pack_id") or row.get("canonical_id") or "").strip() or None
    installed_keys = {
        str(key).strip().lower()
        for key in (
            pack_id,
            _clean_text(row.get("name")),
            _clean_text((canonical.get("pack_identity") if isinstance(canonical.get("pack_identity"), dict) else {}).get("canonical_id") or ""),
        )
        if str(key or "").strip()
    }
    state_key = str(normalized.get("state_key") or "installed_unknown")
    if state_key == "installed_disabled":
        recovery = build_failure_recovery(
            "pack_disabled",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            state_label=str(normalized.get("state_label") or "Installed · Disabled"),
        )
    elif state_key == "installed_blocked" and str(normalized.get("blocker") or "").strip().lower() in {
        "normalized files are missing",
        "files are missing",
        "missing normalized files",
    }:
        recovery = build_failure_recovery(
            "pack_missing_files",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            reason="The pack record points at missing files on disk.",
            next_step="Reinstall the pack.",
            state_label=str(normalized.get("state_label") or "Installed · Blocked"),
            status="blocked",
        )
    elif state_key == "installed_blocked" and str(normalized.get("blocker") or "").strip().lower() in {
        "pack metadata is incomplete",
        "pack identity is missing",
        "pack source is missing",
        "pack name is missing",
    }:
        recovery = build_failure_recovery(
            "pack_invalid_metadata",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            reason="The stored pack metadata is incomplete or cannot be trusted as-is.",
            next_step="Rebuild the manifest or reinstall the pack.",
            state_label=str(normalized.get("state_label") or "Blocked"),
            status="blocked",
        )
    elif state_key == "installed_blocked":
        recovery = build_failure_recovery(
            "pack_blocked",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            state_label=str(normalized.get("state_label") or "Installed · Blocked"),
        )
    elif state_key == "installed_limited":
        recovery = build_failure_recovery(
            "pack_task_unconfirmed",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            reason="Compatibility is not fully confirmed.",
            next_step="Open the pack preview before relying on it.",
            state_label=str(normalized.get("state_label") or "Installed · Limited"),
            status="limited",
        )
    elif state_key == "installed_unknown":
        recovery = build_failure_recovery(
            "partial_persisted_state",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            state_label=str(normalized.get("state_label") or "Installed · Unknown"),
        )
    else:
        recovery = build_failure_recovery(
            "pack_task_unconfirmed",
            subject=_pack_name(row),
            blocker=str(normalized.get("blocker") or ""),
            next_step="Open the pack preview before relying on it.",
            state_label=str(normalized.get("state_label") or "Installed · Healthy"),
            status="healthy",
        )
    return (
        {
            "id": pack_id,
            "name": _pack_name(row),
            "capabilities": cap_values,
            "installed": True,
            "enabled": normalized.get("enabled"),
            "healthy": normalized.get("healthy"),
            "machine_usable": bool(normalized.get("machine_usable", False)),
            "usable": bool(normalized.get("task_usable", False)),
            "state": str(normalized.get("state_key") or "installed_unknown"),
            "state_label": str(normalized.get("state_label") or "Installed · Unknown"),
            "status_note": str(normalized.get("status_note") or "Compatibility not confirmed yet."),
            "blocker": normalized.get("blocker"),
            "next_action": normalized.get("next_action"),
            "source": {
                "id": str(source.get("origin") or "").strip() or None,
                "name": str(source.get("display_name") or source.get("name") or "").strip() or None,
                "kind": str(source.get("kind") or "").strip() or None,
                "url": str(source.get("url") or "").strip() or None,
                "ref": str(source.get("ref") or "").strip() or None,
            },
            "source_label": _source_label(source, fallback=row),
            "type": str(row.get("pack_type") or "skill").strip() or "skill",
            "review_required": bool(row.get("review_required", True)),
            "severity": str(normalized.get("severity") or "degraded"),
            "normalized_state": normalized,
            "recovery": recovery,
        },
        installed_keys,
    )


def _pack_state_available_row(source: dict[str, Any], listing: dict[str, Any], *, from_cache: bool, stale: bool) -> dict[str, Any]:
    normalized = normalize_available_pack_truth(source, listing)
    source_id = str(source.get("id") or "").strip() or None
    name = str(listing.get("name") or "").strip() or "Imported pack"
    remote_id = str(listing.get("remote_id") or "").strip() or None
    artifact_type = str(listing.get("artifact_type_hint") or "").strip() or "unknown"
    tags = _clean_list(listing.get("tags"))
    badges = _clean_list(listing.get("badges"))
    cap_values = tags or ([artifact_type] if artifact_type and artifact_type != "unknown" else [])
    return {
        "id": remote_id or name,
        "name": name,
        "capabilities": cap_values,
        "installed": False,
        "enabled": False,
        "healthy": normalized.get("healthy"),
        "machine_usable": bool(normalized.get("machine_usable", False)),
        "usable": bool(normalized.get("task_usable", False)),
        "state": str(normalized.get("state_key") or "blocked"),
        "state_label": str(normalized.get("state_label") or "Blocked"),
        "status_note": str(normalized.get("status_note") or "Blocked by current policy."),
        "blocker": normalized.get("blocker"),
        "next_action": normalized.get("next_action"),
        "source": {
            "id": source_id,
            "name": str(source.get("name") or "").strip() or None,
            "kind": str(source.get("kind") or "").strip() or None,
            "url": str(source.get("base_url") or "").strip() or None,
        },
        "source_label": _source_label(source),
        "type": artifact_type,
        "review_required": False,
        "severity": str(normalized.get("severity") or "blocked"),
        "preview_state": {
            "from_cache": bool(from_cache),
            "stale": bool(stale),
        },
        "badges": badges,
        "normalized_state": normalized,
        "recovery": build_failure_recovery(
            "pack_available_previewable" if bool(normalized.get("installable", False)) else "pack_blocked",
            subject=name,
            blocker=str(normalized.get("blocker") or ""),
            state_label=str(normalized.get("state_label") or "Available"),
            reason=str(normalized.get("status_note") or ""),
            next_step=str(normalized.get("next_action") or ""),
        ),
    }


def build_pack_state_snapshot(*, pack_store: Any, discovery: Any) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_warnings: list[dict[str, Any]] = []
    try:
        installed_rows = pack_store.list_external_packs() if callable(getattr(pack_store, "list_external_packs", None)) else []
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "locked" not in message and "busy" not in message:
            raise
        source_warnings.append(
            {
                "source_id": "installed",
                "kind": "pack_store_busy",
                "error": "pack store temporarily busy during startup",
            }
        )
        installed_rows = []
    try:
        discovery_sources = discovery.list_sources() if callable(getattr(discovery, "list_sources", None)) else []
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "locked" not in message and "busy" not in message:
            raise
        source_warnings.append(
            {
                "source_id": "discovery",
                "kind": "pack_sources_busy",
                "error": "pack sources temporarily busy during startup",
            }
        )
        discovery_sources = []
    except Exception as exc:  # pragma: no cover - best effort live pack discovery
        source_warnings.append(
            {
                "source_id": "discovery",
                "kind": "pack_sources_error",
                "error": str(exc),
            }
        )
        discovery_sources = []

    installed_cards: list[dict[str, Any]] = []
    installed_keys: set[str] = set()
    for row in installed_rows:
        if not isinstance(row, dict):
            continue
        card, keys = _pack_state_installed_row(row)
        installed_cards.append(card)
        installed_keys.update(keys)

    available_cards: list[dict[str, Any]] = []
    for source in discovery_sources:
        if not isinstance(source, dict):
            continue
        if not bool(source.get("enabled", True)) or not bool(source.get("allowed_by_policy", True)):
            continue
        source_id = str(source.get("id") or "").strip()
        if not source_id:
            continue
        try:
            payload = discovery.list_packs(source_id) if callable(getattr(discovery, "list_packs", None)) else None
        except Exception as exc:  # pragma: no cover - best effort live pack discovery
            source_warnings.append(
                {
                    "source_id": source_id,
                    "error": str(exc),
                }
            )
            continue
        packs = payload.get("packs") if isinstance(payload, dict) and isinstance(payload.get("packs"), list) else []
        from_cache = bool(payload.get("from_cache", False)) if isinstance(payload, dict) else False
        stale = bool(payload.get("stale", False)) if isinstance(payload, dict) else False
        for listing in packs:
            if not isinstance(listing, dict):
                continue
            key = str(listing.get("remote_id") or listing.get("name") or "").strip().lower()
            if key and key in installed_keys:
                continue
            available_cards.append(_pack_state_available_row(source, listing, from_cache=from_cache, stale=stale))

    installed_cards = sorted(
        installed_cards,
        key=lambda item: (
            0
            if str(item.get("state") or "") == "installed_healthy"
            else 1
            if str(item.get("state") or "") == "installed_disabled"
            else 2
            if str(item.get("state") or "") == "installed_limited"
            else 3
            if str(item.get("state") or "") == "installed_blocked"
            else 4,
            str(item.get("name") or "").lower(),
        ),
    )
    available_cards = sorted(
        available_cards,
        key=lambda item: (
            0 if str(item.get("state") or "") == "available" else 1,
            str(item.get("source_label") or "").lower(),
            str(item.get("name") or "").lower(),
        ),
    )

    summary = {
        "total": len(installed_cards) + len(available_cards),
        "installed": len(installed_cards),
        "enabled": sum(1 for row in installed_cards if bool(row.get("enabled", False))),
        "healthy": sum(1 for row in installed_cards if row.get("healthy") is True),
        "machine_usable": sum(1 for row in installed_cards if bool(row.get("machine_usable", False))),
        "task_unconfirmed": sum(
            1 for row in installed_cards if str((row.get("normalized_state") or {}).get("usability_state") or "").strip().lower() == "task_unconfirmed"
        ),
        "usable": sum(1 for row in installed_cards if bool(row.get("usable", False))),
        "blocked": sum(
            1
            for row in installed_cards + available_cards
            if str(row.get("state") or "").strip().lower() in {"blocked", "installed_blocked"}
        ),
        "available": sum(1 for row in available_cards if str(row.get("state") or "").strip().lower() == "available"),
    }

    top_level_recovery: dict[str, Any] | None = None
    top_level_state_label = "Ready"
    top_level_reason: str | None = None
    top_level_next_step: str | None = None
    if source_warnings:
        warning_kinds = {
            str(item.get("kind") or "").strip().lower()
            for item in source_warnings
            if isinstance(item, dict)
        }
        if {"pack_store_busy", "pack_sources_busy"} & warning_kinds:
            top_level_recovery = build_failure_recovery(
                "db_busy",
                current_state="packs_state",
                details=";".join(sorted(warning_kinds)) or None,
            )
        else:
            top_level_recovery = build_failure_recovery(
                "discovery_degraded",
                current_state="packs_state",
                details=";".join(sorted(warning_kinds)) or None,
            )
        top_level_state_label = str(top_level_recovery.get("state_label") or "Degraded")
        top_level_reason = str(top_level_recovery.get("reason") or "").strip() or None
        top_level_next_step = str(top_level_recovery.get("next_step") or "").strip() or None

    return {
        "ok": True,
        "updated_at": now_iso,
        "state_label": top_level_state_label,
        "reason": top_level_reason,
        "next_step": top_level_next_step,
        "recovery": top_level_recovery,
        "summary": summary,
        "packs": installed_cards,
        "available_packs": available_cards,
        "source_warnings": source_warnings,
        "read_only": True,
    }
