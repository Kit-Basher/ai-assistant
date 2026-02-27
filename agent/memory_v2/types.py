from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryLevel(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class MemoryItem:
    id: str
    level: MemoryLevel
    text: str
    created_at: int
    updated_at: int
    tags: dict[str, str] = field(default_factory=dict)
    source_kind: str = "unknown"
    source_ref: str = ""
    pinned: bool = False


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    tags: dict[str, str] = field(default_factory=dict)
    limit_per_level: dict[str, int] = field(
        default_factory=lambda: {
            MemoryLevel.EPISODIC.value: 3,
            MemoryLevel.SEMANTIC.value: 3,
            MemoryLevel.PROCEDURAL.value: 3,
        }
    )
    now_ts: int | None = None


@dataclass(frozen=True)
class MemorySelection:
    items_by_level: dict[MemoryLevel, list[MemoryItem]]
    merged_context_text: str
    debug: dict[str, Any]
