from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SemanticSourceKind(str, Enum):
    CONVERSATION = "conversation"
    NOTE = "note"
    DOCUMENT = "document"


@dataclass(frozen=True)
class SemanticSourceRecord:
    id: str
    source_kind: SemanticSourceKind
    source_ref: str
    scope: str
    content_hash: str
    title: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    pinned: bool = False
    status: str = "ready"
    created_at: int = 0
    updated_at: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticChunkRecord:
    id: str
    source_id: str
    chunk_index: int
    text: str
    chunk_hash: str
    created_at: int
    updated_at: int
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticIndexState:
    scope: str
    embed_provider: str | None
    embed_model: str | None
    embedding_dim: int | None
    status: str
    source_count: int
    chunk_count: int
    vector_count: int
    last_indexed_at: int | None = None
    stale_since: int | None = None
    last_error_kind: str | None = None
    last_error_message: str | None = None
    updated_at: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingTarget:
    provider_id: str
    model_id: str
    model_name: str
    local: bool = False
    dimensions: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class SemanticCandidate:
    id: str
    source_id: str
    chunk_id: str
    source_kind: SemanticSourceKind
    source_ref: str
    scope: str
    text: str
    score: float
    similarity: float
    pin_bonus: float
    scope_bonus: float
    recency_bonus: float
    overlap_bonus: float
    embed_provider: str
    embed_model: str
    embedding_dim: int
    created_at: int
    updated_at: int
    title: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticSelection:
    candidates: list[SemanticCandidate]
    merged_context_text: str
    debug: dict[str, Any]
