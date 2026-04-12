from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json

from agent.memory_v2.types import MemoryItem, MemoryLevel, MemorySelection


_LEVEL_ORDER = (
    MemoryLevel.EPISODIC,
    MemoryLevel.SEMANTIC,
    MemoryLevel.PROCEDURAL,
)


def _format_created_at(ts: int) -> str:
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        dt = datetime.fromtimestamp(0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _format_tags(tags: dict[str, str]) -> str:
    if not tags:
        return "none"
    parts = [f"{key}={value}" for key, value in sorted(tags.items(), key=lambda item: str(item[0]))]
    return ", ".join(parts)


def _format_item(item: MemoryItem) -> list[str]:
    source_ref = str(item.source_ref or "").strip()
    source_value = str(item.source_kind or "unknown")
    if source_ref:
        source_value = f"{source_value}:{source_ref}"
    header = (
        f"MEMORY[{item.id}] ({item.level.value}, {_format_created_at(item.created_at)}, "
        f"tags: {_format_tags(item.tags)}, source: {source_value}):"
    )
    quote = json.dumps(str(item.text), ensure_ascii=True)
    return [header, quote]


def build_memory_context(selection: MemorySelection) -> str:
    lines = ["MEMORY_CONTEXT"]
    for level in _LEVEL_ORDER:
        items = selection.items_by_level.get(level) if isinstance(selection.items_by_level, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            lines.extend(_format_item(item))
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def with_built_context(selection: MemorySelection) -> MemorySelection:
    context = build_memory_context(selection)
    return replace(selection, merged_context_text=context)
