from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.packs.external_ingestion import (
    INSTALL_PATTERNS,
    NETWORK_PATTERNS,
    PROMPT_INJECTION_PATTERNS,
    SAFE_TEXT_EXTENSIONS,
    SHELL_REQUIREMENT_PATTERNS,
)


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_path_text(root: str | None, rel_path: str) -> str | None:
    base = Path(str(root or "").strip())
    if not base:
        return None
    try:
        path = (base / rel_path).resolve()
        if not path.is_file():
            return None
        raw = path.read_bytes()[:256 * 1024]
    except OSError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return None


def _text_signal_names(text: str) -> set[str]:
    lowered = str(text or "").lower()
    flags: set[str] = set()
    if any(re.search(pattern, lowered) for pattern in INSTALL_PATTERNS) or any(
        re.search(pattern, lowered) for pattern in SHELL_REQUIREMENT_PATTERNS
    ):
        flags.add("install_instructions_added")
    if any(re.search(pattern, lowered) for pattern in NETWORK_PATTERNS):
        flags.add("network_behavior_added")
    if any(re.search(pattern, lowered) for pattern in PROMPT_INJECTION_PATTERNS):
        flags.add("prompt_injection_patterns_added")
    return flags


def _build_text_signal_map(pack_row: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    by_path: dict[str, set[str]] = {}
    overall: set[str] = set()
    normalized_path = str(pack_row.get("normalized_path") or "").strip() or None
    for component in _sequence(pack_row.get("components")):
        if not isinstance(component, dict) or not bool(component.get("included", False)):
            continue
        rel_path = str(component.get("path") or "")
        suffix = Path(rel_path).suffix.lower()
        if rel_path.lower() != "skill.md" and suffix not in SAFE_TEXT_EXTENSIONS:
            continue
        text = _safe_path_text(normalized_path, rel_path)
        if not text:
            continue
        flags = _text_signal_names(text)
        if not flags:
            continue
        by_path[rel_path] = flags
        overall.update(flags)
    return overall, by_path


def _component_items(pack_row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for component in _sequence(pack_row.get("components")):
        if not isinstance(component, dict):
            continue
        path = str(component.get("path") or "").strip()
        if not path:
            continue
        items[path] = {
            "path": path,
            "component_type": str(component.get("component_type") or "component"),
            "included": bool(component.get("included", False)),
            "executable": bool(component.get("executable", False)),
            "sha256": str(component.get("sha256") or "").strip() or None,
        }
    for asset in _sequence(pack_row.get("assets")):
        if not isinstance(asset, dict):
            continue
        path = str(asset.get("path") or "").strip()
        if not path:
            continue
        items[path] = {
            "path": path,
            "component_type": "asset",
            "included": bool(asset.get("included", False)),
            "executable": bool(asset.get("executable", False)),
            "sha256": str(asset.get("sha256") or "").strip() or None,
        }
    return items


def _permissions_payload(pack_row: dict[str, Any]) -> dict[str, list[str]]:
    permissions = pack_row.get("permissions") if isinstance(pack_row.get("permissions"), dict) else {}
    requested = [
        str(item).strip()
        for item in _sequence(permissions.get("requested"))
        if str(item).strip()
    ]
    granted = [
        str(item).strip()
        for item in _sequence(permissions.get("granted"))
        if str(item).strip()
    ]
    return {
        "requested": sorted(dict.fromkeys(requested)),
        "granted": sorted(dict.fromkeys(granted)),
    }


@dataclass(frozen=True)
class PackVersionRef:
    canonical_id: str
    content_hash: str | None
    name: str
    version: str
    status: str
    source: dict[str, Any]
    updated_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackDiffEntry:
    previous_canonical_id: str
    current_canonical_id: str
    previous_content_hash: str | None
    current_content_hash: str | None
    change_kind: str
    component_type: str
    path: str
    summary: str
    risk_relevant: bool
    plain_language_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PackChangeSummary:
    files_added_count: int
    files_removed_count: int
    files_modified_count: int
    instructions_changed: bool
    assets_changed: bool
    executable_content_added: bool
    permissions_implication_changed: bool
    risk_delta: float
    notable_changes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_delta"] = round(float(self.risk_delta), 4)
        return payload


@dataclass(frozen=True)
class PackDiff:
    previous: PackVersionRef
    current: PackVersionRef
    entries: tuple[dict[str, Any], ...]
    change_summary: dict[str, Any]
    flags: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous": self.previous.to_dict(),
            "current": self.current.to_dict(),
            "entries": list(self.entries),
            "change_summary": dict(self.change_summary),
            "flags": list(self.flags),
            "summary": self.summary,
        }


def build_pack_version_ref(pack_row: dict[str, Any]) -> PackVersionRef:
    source = pack_row.get("source") if isinstance(pack_row.get("source"), dict) else {}
    return PackVersionRef(
        canonical_id=str(pack_row.get("canonical_id") or pack_row.get("pack_id") or ""),
        content_hash=str(pack_row.get("content_hash") or "").strip() or None,
        name=str(pack_row.get("name") or pack_row.get("pack_name") or ""),
        version=str(pack_row.get("version") or "0.0.0"),
        status=str(pack_row.get("status") or "unknown"),
        source=source,
        updated_at=int(pack_row.get("updated_at") or 0) or None,
    )


def _entry_messages(
    *,
    change_kind: str,
    component_type: str,
    path: str,
    risk_relevant: bool,
) -> tuple[str, str]:
    noun = {
        "instruction": "instruction file",
        "reference": "reference file",
        "secondary_reference": "secondary reference",
        "asset": "asset",
        "disallowed": "unsafe component",
        "permissions": "permission implications",
    }.get(component_type, component_type.replace("_", " "))
    if component_type == "permissions":
        summary = "Permission implications changed."
        note = "The requested or granted permissions changed."
    elif change_kind == "added":
        summary = f"{noun.capitalize()} added."
        note = f"A new {noun} was added at {path}."
    elif change_kind == "removed":
        summary = f"{noun.capitalize()} removed."
        note = f"The {noun} at {path} was removed."
    elif change_kind == "unchanged":
        summary = "Normalized content unchanged."
        note = "The safe imported content is unchanged."
    else:
        summary = f"{noun.capitalize()} modified."
        note = f"The {noun} at {path} changed."
    if risk_relevant and component_type == "disallowed" and change_kind == "added":
        note = f"A new executable or unsafe component was added at {path}."
    return summary, note


def build_pack_diff(previous_row: dict[str, Any], current_row: dict[str, Any]) -> PackDiff:
    previous = build_pack_version_ref(previous_row)
    current = build_pack_version_ref(current_row)
    previous_items = _component_items(previous_row)
    current_items = _component_items(current_row)
    previous_text_flags, previous_text_flags_by_path = _build_text_signal_map(previous_row)
    current_text_flags, current_text_flags_by_path = _build_text_signal_map(current_row)
    added_signal_flags = sorted(current_text_flags - previous_text_flags)

    entries: list[dict[str, Any]] = []
    added_count = 0
    removed_count = 0
    modified_count = 0
    instructions_changed = False
    assets_changed = False
    executable_content_added = False
    executable_content_removed = False
    permissions_implication_changed = _permissions_payload(previous_row) != _permissions_payload(current_row)

    for path in sorted(set(previous_items) | set(current_items)):
        previous_item = previous_items.get(path)
        current_item = current_items.get(path)
        if previous_item is None:
            change_kind = "added"
            component_type = str(current_item.get("component_type") or "component")
            added_count += 1
        elif current_item is None:
            change_kind = "removed"
            component_type = str(previous_item.get("component_type") or "component")
            removed_count += 1
        elif (
            str(previous_item.get("sha256") or "") == str(current_item.get("sha256") or "")
            and bool(previous_item.get("included", False)) == bool(current_item.get("included", False))
            and bool(previous_item.get("executable", False)) == bool(current_item.get("executable", False))
            and str(previous_item.get("component_type") or "") == str(current_item.get("component_type") or "")
        ):
            continue
        else:
            change_kind = "modified"
            component_type = str(current_item.get("component_type") or previous_item.get("component_type") or "component")
            modified_count += 1

        is_instruction = component_type in {"instruction", "reference", "secondary_reference"}
        is_asset = component_type == "asset"
        is_executable_added = bool(current_item and current_item.get("executable", False) and change_kind in {"added", "modified"})
        is_executable_removed = bool(previous_item and previous_item.get("executable", False) and change_kind in {"removed", "modified"} and not is_executable_added)
        risk_signals_for_path = set()
        risk_signals_for_path.update(current_text_flags_by_path.get(path, set()) - previous_text_flags_by_path.get(path, set()))
        risk_relevant = bool(is_executable_added or risk_signals_for_path or component_type == "permissions")

        if is_instruction:
            instructions_changed = True
        if is_asset:
            assets_changed = True
        if is_executable_added:
            executable_content_added = True
        if is_executable_removed:
            executable_content_removed = True

        summary, note = _entry_messages(
            change_kind=change_kind,
            component_type=component_type,
            path=path,
            risk_relevant=risk_relevant,
        )
        entries.append(
            PackDiffEntry(
                previous_canonical_id=previous.canonical_id,
                current_canonical_id=current.canonical_id,
                previous_content_hash=previous.content_hash,
                current_content_hash=current.content_hash,
                change_kind=change_kind,
                component_type=component_type,
                path=path,
                summary=summary,
                risk_relevant=risk_relevant,
                plain_language_note=note,
            ).to_dict()
        )

    if permissions_implication_changed:
        summary, note = _entry_messages(
            change_kind="modified",
            component_type="permissions",
            path="permissions",
            risk_relevant=True,
        )
        entries.append(
            PackDiffEntry(
                previous_canonical_id=previous.canonical_id,
                current_canonical_id=current.canonical_id,
                previous_content_hash=previous.content_hash,
                current_content_hash=current.content_hash,
                change_kind="modified",
                component_type="permissions",
                path="permissions",
                summary=summary,
                risk_relevant=True,
                plain_language_note=note,
            ).to_dict()
        )

    flags: list[str] = []
    normalized_content_unchanged = added_count == 0 and removed_count == 0 and modified_count == 0 and not permissions_implication_changed
    raw_source_only_changed = False
    if normalized_content_unchanged:
        flags.append("normalized_content_unchanged")
        if previous.canonical_id == current.canonical_id and len(_sequence(current_row.get("source_history"))) > 1:
            raw_source_only_changed = True
            flags.append("raw_source_only_changed")
    if executable_content_added:
        flags.append("executable_content_added")
    if "install_instructions_added" in added_signal_flags:
        flags.append("install_instructions_added")
    if "network_behavior_added" in added_signal_flags:
        flags.append("network_behavior_added")
    if "prompt_injection_patterns_added" in added_signal_flags:
        flags.append("prompt_injection_patterns_added")
    if permissions_implication_changed:
        flags.append("permissions_implication_changed")

    risk_delta = round(
        float(current_row.get("risk_score") or 0.0) - float(previous_row.get("risk_score") or 0.0),
        4,
    )
    notable_changes: list[str] = []
    if raw_source_only_changed:
        notable_changes.append("The normalized content is unchanged; only the upstream source snapshot changed.")
    if executable_content_added:
        notable_changes.append("A new executable file was added and stripped from the safe import.")
    if "install_instructions_added" in added_signal_flags:
        notable_changes.append("This version adds install steps, so it is treated as higher risk.")
    if "network_behavior_added" in added_signal_flags:
        notable_changes.append("New network-related instructions were added.")
    if "prompt_injection_patterns_added" in added_signal_flags:
        notable_changes.append("New prompt-injection-like text was added.")
    if executable_content_removed and risk_delta < 0:
        notable_changes.append("This update removed unsafe executable files and is safer than the previous source.")
    if instructions_changed and not assets_changed and not executable_content_added and not added_signal_flags:
        notable_changes.append("The instructions changed.")
    if assets_changed and not instructions_changed and not executable_content_added and added_count + removed_count + modified_count == sum(
        1 for entry in entries if str(entry.get("component_type") or "") == "asset"
    ):
        notable_changes.append("Only images or other static assets changed.")

    if normalized_content_unchanged:
        summary = "The normalized content is unchanged."
        if raw_source_only_changed:
            summary = "The normalized content is unchanged. Only the upstream source snapshot changed."
    elif executable_content_added and risk_delta > 0:
        summary = "A new executable file was added, so risk increased."
    elif "install_instructions_added" in added_signal_flags:
        summary = "This version adds install steps, so I treat it as higher risk."
    elif instructions_changed and not assets_changed:
        summary = "The instructions changed."
    elif assets_changed and not instructions_changed:
        summary = "Only images or other static assets changed."
    elif executable_content_removed and risk_delta < 0:
        summary = "This update removed unsafe executable files and is safer than the previous source."
    else:
        summary = (
            f"{added_count} file(s) added, {removed_count} removed, and {modified_count} modified "
            "in the normalized safe pack."
        )

    change_summary = PackChangeSummary(
        files_added_count=added_count,
        files_removed_count=removed_count,
        files_modified_count=modified_count,
        instructions_changed=instructions_changed,
        assets_changed=assets_changed,
        executable_content_added=executable_content_added,
        permissions_implication_changed=permissions_implication_changed,
        risk_delta=risk_delta,
        notable_changes=tuple(notable_changes),
    ).to_dict()

    if normalized_content_unchanged:
        entries = [
            PackDiffEntry(
                previous_canonical_id=previous.canonical_id,
                current_canonical_id=current.canonical_id,
                previous_content_hash=previous.content_hash,
                current_content_hash=current.content_hash,
                change_kind="unchanged",
                component_type="normalized_pack",
                path="*",
                summary="Normalized content unchanged.",
                risk_relevant=False,
                plain_language_note="The safe imported content is unchanged.",
            ).to_dict()
        ]

    return PackDiff(
        previous=previous,
        current=current,
        entries=tuple(entries),
        change_summary=change_summary,
        flags=tuple(sorted(dict.fromkeys(flags))),
        summary=summary,
    )
