from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agent.bootstrap.snapshot import BootstrapSnapshot, snapshot_to_dict
from agent.memory_v2.storage import SQLiteMemoryStore
from agent.memory_v2.types import MemoryItem, MemoryLevel


_ALLOWLIST_KEYS = (
    "os.name",
    "hardware.gpu.available",
    "providers.enabled_ids",
    "capsules.installed",
    "interfaces.available",
)

_SECTION_ORDER = (
    "os",
    "hardware",
    "interfaces",
    "providers",
    "capsules",
    "routes",
    "notes",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)


def _semantic_id(key: str) -> str:
    slug = key.replace(".", "-").replace("_", "-")
    return f"S-bootstrap-{slug}"


def _semantic_versioned_id(*, key: str, text: str) -> str:
    base = _semantic_id(key)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{base}-{digest}"


def _normalize_semantic_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    normalized = re.sub(r"\s+", " ", str(reason)).strip()
    if not normalized:
        return None
    return normalized[:160]


def _semantic_text_for_allowlist(
    *,
    key: str,
    snapshot_dict: dict[str, Any],
    section_payload_text: dict[str, str],
) -> str | None:
    if key == "os.name":
        name = str(((snapshot_dict.get("os") or {}) if isinstance(snapshot_dict.get("os"), dict) else {}).get("name") or "unknown")
        quote = _stable_json({"name": name})
        return f"OS name: {name}. Quote: {quote}"

    if key == "hardware.gpu.available":
        gpu = (snapshot_dict.get("hardware") or {}).get("gpu") if isinstance(snapshot_dict.get("hardware"), dict) else {}
        available = bool((gpu or {}).get("available"))
        quote = _stable_json({"available": available})
        return f"GPU present: {str(available).lower()}. Quote: {quote}"

    if key == "providers.enabled_ids":
        providers = snapshot_dict.get("providers") if isinstance(snapshot_dict.get("providers"), dict) else {}
        enabled_ids = providers.get("enabled_ids") if isinstance(providers.get("enabled_ids"), list) else []
        enabled = [str(item) for item in enabled_ids if str(item).strip()]
        quote = _stable_json({"enabled_ids": enabled})
        return f"Enabled providers: {enabled}. Quote: {quote}"

    if key == "capsules.installed":
        capsules = snapshot_dict.get("capsules") if isinstance(snapshot_dict.get("capsules"), dict) else {}
        installed_raw = capsules.get("installed") if isinstance(capsules.get("installed"), list) else []
        installed = [str(item) for item in installed_raw if str(item).strip()]
        quote = _stable_json({"installed": installed})
        return f"Native capsules installed: {installed}. Quote: {quote}"

    if key == "interfaces.available":
        interfaces = snapshot_dict.get("interfaces") if isinstance(snapshot_dict.get("interfaces"), dict) else {}
        available = [
            name
            for name, enabled in sorted(
                {
                    "memory_v2": bool(interfaces.get("memory_v2_enabled")),
                    "model_watch": bool(interfaces.get("model_watch_enabled")),
                    "llm_automation": bool(interfaces.get("llm_automation_enabled")),
                    "telegram": bool(interfaces.get("telegram_configured")),
                    "webui_dev_proxy": bool(interfaces.get("webui_dev_proxy")),
                }.items(),
                key=lambda item: item[0],
            )
            if enabled
        ]
        quote = _stable_json({"available": available})
        return f"Available interfaces: {available}. Quote: {quote}"

    return None


def ingest_bootstrap_snapshot(
    *,
    store: SQLiteMemoryStore,
    snapshot: BootstrapSnapshot,
    source_ref: str,
    promote_semantic: bool = True,
    now_ts: int | None = None,
    trigger_reason: str | None = None,
) -> dict[str, Any]:
    snapshot_dict = snapshot_to_dict(snapshot)
    created_at = int(now_ts if now_ts is not None else int(snapshot.created_at_ts))
    section_to_event_id: dict[str, str] = {}
    section_payload_text: dict[str, str] = {}
    episodic_ids: list[str] = []
    reason_tag = _normalize_reason(trigger_reason)

    for section in _SECTION_ORDER:
        section_payload = snapshot_dict.get(section)
        section_text = _stable_json(section_payload)
        section_payload_text[section] = section_text
        event_seed = f"bootstrap:{source_ref}:{section}:{section_text}"
        event_id = "E-bootstrap-" + hashlib.sha256(event_seed.encode("utf-8")).hexdigest()[:12]
        item = store.append_episodic_event(
            event_id=event_id,
            text=section_text,
            created_at=created_at,
            tags={
                **(
                    {
                        "trigger_reason": reason_tag,
                    }
                    if reason_tag
                    else {}
                ),
                "project": "personal-agent",
                "kind": "bootstrap_snapshot",
                "section": section,
            },
            source_kind="bootstrap",
            source_ref=str(source_ref),
            pinned=False,
        )
        section_to_event_id[section] = item.id
        episodic_ids.append(item.id)

    semantic_ids: list[str] = []
    promoted_keys: list[str] = []
    semantic_to_provenance: dict[str, list[str]] = {}
    semantic_inserted: list[dict[str, str]] = []
    semantic_unchanged: list[dict[str, str]] = []
    semantic_superseded: list[dict[str, str]] = []

    key_to_section = {
        "os.name": ["os"],
        "hardware.gpu.available": ["hardware"],
        "providers.enabled_ids": ["providers"],
        "capsules.installed": ["capsules"],
        "interfaces.available": ["interfaces"],
    }

    if bool(promote_semantic):
        for allowlist_key in _ALLOWLIST_KEYS:
            semantic_text = _semantic_text_for_allowlist(
                key=allowlist_key,
                snapshot_dict=snapshot_dict,
                section_payload_text=section_payload_text,
            )
            if not semantic_text:
                continue
            source_event_ids = [
                section_to_event_id[section]
                for section in key_to_section.get(allowlist_key, [])
                if section in section_to_event_id
            ]
            source_ref_value = ",".join(sorted(source_event_ids))
            current_item = store.get_current_semantic(
                fact_group="bootstrap",
                fact_key=allowlist_key,
            )
            if current_item is not None:
                current_text = _normalize_semantic_text(current_item.text)
                next_text = _normalize_semantic_text(semantic_text)
                if current_text == next_text:
                    semantic_unchanged.append({"fact_key": allowlist_key, "id": current_item.id})
                    promoted_keys.append(allowlist_key)
                    semantic_to_provenance[current_item.id] = sorted(
                        [part for part in str(current_item.source_ref or "").split(",") if part]
                    )
                    continue
                store.supersede_semantic(item_id=current_item.id, superseded_at=created_at)
                semantic_superseded.append({"fact_key": allowlist_key, "id": current_item.id})

            semantic_id = _semantic_versioned_id(key=allowlist_key, text=semantic_text)
            inserted = store.upsert_memory_item(
                MemoryItem(
                    id=semantic_id,
                    level=MemoryLevel.SEMANTIC,
                    text=semantic_text,
                    created_at=created_at,
                    updated_at=created_at,
                    tags={
                        "project": "personal-agent",
                        "kind": "bootstrap_semantic",
                        "allowlist_key": allowlist_key,
                    },
                    source_kind="bootstrap",
                    source_ref=source_ref_value,
                    pinned=True,
                    fact_key=allowlist_key,
                    fact_group="bootstrap",
                    superseded_at=None,
                    is_current=True,
                )
            )
            semantic_ids.append(inserted.id)
            promoted_keys.append(allowlist_key)
            semantic_inserted.append({"fact_key": allowlist_key, "id": inserted.id})
            semantic_to_provenance[inserted.id] = sorted(source_event_ids)

    return {
        "created_at_ts": created_at,
        "episodic_ids": sorted(episodic_ids),
        "semantic_ids": sorted(semantic_ids),
        "promoted_keys": sorted(promoted_keys),
        "semantic_updates": {
            "inserted": sorted(semantic_inserted, key=lambda row: (row["fact_key"], row["id"])),
            "unchanged": sorted(semantic_unchanged, key=lambda row: (row["fact_key"], row["id"])),
            "superseded": sorted(semantic_superseded, key=lambda row: (row["fact_key"], row["id"])),
        },
        "section_to_episodic_id": dict(sorted(section_to_event_id.items(), key=lambda item: item[0])),
        "semantic_provenance": {
            key: semantic_to_provenance[key]
            for key in sorted(semantic_to_provenance.keys())
        },
    }
