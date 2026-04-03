from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.config import Config
from agent.llm.capabilities import is_embedding_model_name
from agent.llm.registry import Registry
from agent.llm.providers.base import Provider
from agent.llm.types import EmbeddingResponse
from agent.semantic_memory.storage import SQLiteSemanticStore
from agent.semantic_memory.types import (
    EmbeddingTarget,
    SemanticCandidate,
    SemanticChunkRecord,
    SemanticIndexState,
    SemanticSelection,
    SemanticSourceKind,
)


_CHUNK_CHAR_LIMIT = 1200
_DEFAULT_MAX_CANDIDATES = 6
_DEFAULT_MAX_PROMPT_CHARS = 4000
_DEFAULT_QUERY_LIMIT = 200
_SCOPE_GLOBAL = "global"


@dataclass(frozen=True)
class SemanticMemoryStatus:
    enabled: bool
    configured: bool
    healthy: bool
    reason: str | None
    target: dict[str, Any]
    index_state: dict[str, Any] | None


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_scope(scope: str | None) -> str:
    value = str(scope or "").strip()
    return value or _SCOPE_GLOBAL


def _split_chunks(text: str, *, chunk_size: int = _CHUNK_CHAR_LIMIT) -> list[tuple[int, int, str]]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    limit = max(200, int(chunk_size))
    if len(normalized) <= limit:
        return [(0, len(normalized), normalized)]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        sentences = [normalized]
    chunks: list[tuple[int, int, str]] = []
    start = 0
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > limit and buffer:
            end = start + len(buffer)
            chunks.append((start, end, buffer))
            start = end
            buffer = sentence
            continue
        if len(candidate) > limit:
            pieces = [candidate[i : i + limit] for i in range(0, len(candidate), limit)]
            cursor = start
            for piece in pieces:
                chunks.append((cursor, cursor + len(piece), piece))
                cursor += len(piece)
            start = cursor
            buffer = ""
            continue
        buffer = candidate
    if buffer:
        chunks.append((start, start + len(buffer), buffer))
    return chunks


def _content_hash(kind: str, source_ref: str, scope: str, text: str, metadata: dict[str, Any]) -> str:
    payload = {
        "kind": kind,
        "source_ref": source_ref,
        "scope": scope,
        "text": _normalize_text(text),
        "metadata": metadata,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _chunk_id(source_id: str, chunk_index: int, chunk_text: str) -> str:
    seed = f"{source_id}:{chunk_index}:{chunk_text}"
    return "SC-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _source_id(kind: str, source_ref: str, scope: str, content_hash: str) -> str:
    seed = f"{kind}:{scope}:{source_ref}:{content_hash}"
    return "SS-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _vector_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if not left_norm or not right_norm:
        return 0.0
    dot = sum(float(l) * float(r) for l, r in zip(left, right))
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9._:/-]*", _normalize_text(text).lower()) if token}


def _overlap_bonus(query: str, candidate: str) -> float:
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0
    candidate_tokens = _token_set(candidate)
    if not candidate_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(candidate_tokens))
    if overlap <= 0:
        return 0.0
    return min(0.1, (overlap / max(1, len(query_tokens))) * 0.1)


def _trunc(text: str, limit: int) -> str:
    cleaned = _normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


class SemanticMemoryService:
    def __init__(
        self,
        *,
        config: Config,
        registry: Registry,
        provider_resolver: Callable[[str], Provider | None] | None,
        db_path: str,
    ) -> None:
        self.config = config
        self.registry = registry
        self._provider_resolver = provider_resolver
        self.store = SQLiteSemanticStore(db_path)
        self._max_candidates = max(1, int(getattr(config, "semantic_memory_max_candidates", _DEFAULT_MAX_CANDIDATES)))
        self._max_prompt_chars = max(200, int(getattr(config, "semantic_memory_max_prompt_chars", _DEFAULT_MAX_PROMPT_CHARS)))
        self._chunk_size = max(200, int(getattr(config, "semantic_memory_chunk_size", _CHUNK_CHAR_LIMIT)))
        self._enabled = bool(getattr(config, "semantic_memory_enabled", False))
        self._conversation_enabled = bool(getattr(config, "semantic_memory_conversations_enabled", True))
        self._note_enabled = bool(getattr(config, "semantic_memory_notes_enabled", True))
        self._document_enabled = bool(getattr(config, "semantic_memory_documents_enabled", True))
        self._query_limit = max(self._max_candidates, int(getattr(config, "semantic_memory_query_limit", _DEFAULT_QUERY_LIMIT)))

    def enabled(self) -> bool:
        return self._enabled

    def _source_kind_enabled(self, kind: SemanticSourceKind) -> bool:
        if kind is SemanticSourceKind.CONVERSATION:
            return self._conversation_enabled
        if kind is SemanticSourceKind.NOTE:
            return self._note_enabled
        if kind is SemanticSourceKind.DOCUMENT:
            return self._document_enabled
        return False

    def _enabled_source_kinds(self) -> list[SemanticSourceKind]:
        return [kind for kind in SemanticSourceKind if self._source_kind_enabled(kind)]

    def _resolve_target(self) -> EmbeddingTarget | None:
        embed_model = str(getattr(self.registry.defaults, "embed_model", None) or "").strip() or None
        if embed_model is None:
            return None
        model = self.registry.models.get(embed_model)
        if model is None:
            return EmbeddingTarget(provider_id="", model_id=embed_model, model_name=embed_model, reason="embed_model_unknown")
        if "embedding" not in {str(item).strip().lower() for item in model.capabilities if str(item).strip()} and not is_embedding_model_name(model.model):
            return EmbeddingTarget(
                provider_id=model.provider,
                model_id=model.id,
                model_name=model.model,
                local=bool(self.registry.providers.get(model.provider).local) if self.registry.providers.get(model.provider) else False,
                reason="embed_model_not_embedding_capable",
            )
        provider_cfg = self.registry.providers.get(model.provider)
        if provider_cfg is None:
            return EmbeddingTarget(
                provider_id=model.provider,
                model_id=model.id,
                model_name=model.model,
                reason="embed_model_provider_unknown",
            )
        provider = self._provider_resolver(model.provider) if callable(self._provider_resolver) else None
        if provider is None or not provider.available():
            return EmbeddingTarget(
                provider_id=model.provider,
                model_id=model.id,
                model_name=model.model,
                local=bool(provider_cfg.local),
                reason="embedding_provider_unavailable",
            )
        return EmbeddingTarget(
            provider_id=model.provider,
            model_id=model.id,
            model_name=model.model,
            local=bool(provider_cfg.local),
        )

    def _provider_for_target(self, target: EmbeddingTarget) -> Provider | None:
        if not target.provider_id:
            return None
        if not callable(self._provider_resolver):
            return None
        provider = self._provider_resolver(target.provider_id)
        if provider is None or not provider.available():
            return None
        return provider

    def _index_state(self) -> SemanticIndexState | None:
        return self.store.get_index_state(_SCOPE_GLOBAL)

    def _state_payload(self, state: SemanticIndexState | None) -> dict[str, Any] | None:
        if state is None:
            return None
        return {
            "scope": state.scope,
            "status": state.status,
            "embed_provider": state.embed_provider,
            "embed_model": state.embed_model,
            "embedding_dim": state.embedding_dim,
            "source_count": state.source_count,
            "chunk_count": state.chunk_count,
            "vector_count": state.vector_count,
            "last_indexed_at": state.last_indexed_at,
            "stale_since": state.stale_since,
            "last_error_kind": state.last_error_kind,
            "last_error_message": state.last_error_message,
            "updated_at": state.updated_at,
            "details": dict(state.details),
        }

    def _counts_payload(
        self,
        *,
        scope: str | None = None,
        source_kinds: list[SemanticSourceKind] | None = None,
    ) -> dict[str, int]:
        return self.store.count_contents(scope=scope, source_kinds=source_kinds)

    def status(self) -> SemanticMemoryStatus:
        if not self.enabled():
            return SemanticMemoryStatus(False, False, False, "semantic_memory_disabled", self._target_payload(None), self._state_payload(self._index_state()))
        target = self._resolve_target()
        state = self._index_state()
        counts = self._counts_payload(scope=None)
        if target is None:
            return SemanticMemoryStatus(True, False, False, "embed_model_not_configured", self._target_payload(target), self._state_payload(state))
        if target.reason:
            return SemanticMemoryStatus(True, False, False, target.reason, self._target_payload(target), self._state_payload(state))
        if state is None:
            return SemanticMemoryStatus(True, True, False, "index_missing", self._target_payload(target), None)
        if state.embed_model != target.model_id or state.embed_provider != target.provider_id:
            return SemanticMemoryStatus(True, True, False, "stale_embed_model", self._target_payload(target), self._state_payload(state))
        if state.embedding_dim is not None and state.embedding_dim <= 0:
            return SemanticMemoryStatus(True, True, False, "dimension_invalid", self._target_payload(target), self._state_payload(state))
        if (
            counts["source_count"] != state.source_count
            or counts["chunk_count"] != state.chunk_count
            or counts["vector_count"] != state.vector_count
        ):
            return SemanticMemoryStatus(True, True, False, "index_state_mismatch", self._target_payload(target), self._state_payload(state))
        if counts["chunk_count"] > counts["vector_count"]:
            return SemanticMemoryStatus(True, True, False, "partial_index", self._target_payload(target), self._state_payload(state))
        if state.status not in {"ready", "warm"}:
            return SemanticMemoryStatus(True, True, False, state.status, self._target_payload(target), self._state_payload(state))
        return SemanticMemoryStatus(True, True, True, None, self._target_payload(target), self._state_payload(state))

    def report(self, *, scope: str = _SCOPE_GLOBAL) -> dict[str, Any]:
        status = self.status()
        state = self._index_state()
        counts = self._counts_payload(scope=None)
        all_counts = self._counts_payload(scope=None, source_kinds=self._enabled_source_kinds())
        recovery_state = "healthy" if status.healthy else str(status.reason or (state.status if state is not None else "unknown"))
        recoverable = bool(status.enabled and status.configured and recovery_state not in {"healthy", "semantic_memory_disabled"})
        if recovery_state in {"semantic_memory_disabled"}:
            recoverable = False
        recommended_action: str | None = None
        if recovery_state == "embed_model_not_configured":
            recommended_action = "configure AGENT_SEMANTIC_MEMORY_* and a valid embed_model"
        elif recovery_state in {"embed_model_unknown", "embed_model_not_embedding_capable", "embed_model_provider_unknown"}:
            recommended_action = "select an embedding-capable model/provider pair"
        elif recovery_state == "embedding_provider_unavailable":
            recommended_action = "restore embedding provider availability"
        elif recovery_state in {"index_missing", "partial_index", "index_state_mismatch", "stale_embed_model", "dimension_invalid"}:
            recommended_action = "run a semantic rebuild"
        elif recovery_state not in {"healthy", "semantic_memory_disabled"}:
            recommended_action = "inspect semantic index details and rebuild if needed"
        embed_provider = str(status.target.get("provider_id") or "").strip() or None
        embed_model = str(status.target.get("model_id") or "").strip() or None
        summary_bits = [
            "enabled" if status.enabled else "disabled",
            "healthy" if status.healthy else f"unhealthy:{recovery_state}",
        ]
        if embed_provider or embed_model:
            summary_bits.append(f"embed={embed_provider or 'unknown'}:{embed_model or 'unknown'}")
        summary_bits.append(f"counts={counts['source_count']}/{counts['chunk_count']}/{counts['vector_count']}")
        if recommended_action:
            summary_bits.append(f"action={recommended_action}")
        return {
            "enabled": status.enabled,
            "configured": status.configured,
            "healthy": status.healthy,
            "reason": status.reason,
            "target": status.target,
            "index_state": status.index_state,
            "counts": counts,
            "all_counts": all_counts,
            "source_kinds_enabled": [kind.value for kind in self._enabled_source_kinds()],
            "recovery": {
                "state": recovery_state,
                "recoverable": recoverable,
                "recommended_action": recommended_action,
                "needs_reindex": not status.healthy and recovery_state not in {"semantic_memory_disabled"},
            },
            "summary": "; ".join(summary_bits),
        }

    def _target_payload(self, target: EmbeddingTarget | None) -> dict[str, Any]:
        if target is None:
            return {"provider_id": None, "model_id": None, "model_name": None, "reason": "embed_model_not_configured"}
        return {
            "provider_id": target.provider_id,
            "model_id": target.model_id,
            "model_name": target.model_name,
            "local": bool(target.local),
            "reason": target.reason,
            "dimensions": target.dimensions,
        }

    def _current_target_and_state(self) -> tuple[EmbeddingTarget | None, SemanticIndexState | None]:
        return self._resolve_target(), self._index_state()

    def _rebuild_state(
        self,
        *,
        target: EmbeddingTarget,
        status: str,
        source_count: int,
        chunk_count: int,
        vector_count: int,
        embedding_dim: int | None,
        last_indexed_at: int | None = None,
        stale_since: int | None = None,
        last_error_kind: str | None = None,
        last_error_message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> SemanticIndexState:
        return self.store.set_index_state(
            scope=_SCOPE_GLOBAL,
            embed_provider=target.provider_id,
            embed_model=target.model_id,
            embedding_dim=embedding_dim,
            status=status,
            source_count=source_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            last_indexed_at=last_indexed_at,
            stale_since=stale_since,
            last_error_kind=last_error_kind,
            last_error_message=last_error_message,
            details=details or {},
        )

    def _embed_texts(self, provider: Provider, model_id: str, texts: tuple[str, ...]) -> EmbeddingResponse:
        return provider.embed_texts(texts, model=model_id, timeout_seconds=float(getattr(self.config, "llm_timeout_seconds", 15)))

    def ingest_conversation_text(
        self,
        *,
        source_ref: str,
        text: str,
        scope: str | None = None,
        thread_id: str | None = None,
        project_id: str | None = None,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
        now_ts: int | None = None,
    ) -> dict[str, Any]:
        return self.ingest_text(
            source_kind=SemanticSourceKind.CONVERSATION,
            source_ref=source_ref,
            text=text,
            scope=scope,
            thread_id=thread_id,
            project_id=project_id,
            pinned=pinned,
            metadata=metadata,
            now_ts=now_ts,
        )

    def ingest_note_text(
        self,
        *,
        source_ref: str,
        text: str,
        scope: str | None = None,
        project_id: str | None = None,
        pinned: bool = True,
        metadata: dict[str, Any] | None = None,
        now_ts: int | None = None,
    ) -> dict[str, Any]:
        return self.ingest_text(
            source_kind=SemanticSourceKind.NOTE,
            source_ref=source_ref,
            text=text,
            scope=scope,
            project_id=project_id,
            pinned=pinned,
            metadata=metadata,
            now_ts=now_ts,
        )

    def ingest_document_text(
        self,
        *,
        source_ref: str,
        text: str,
        scope: str | None = None,
        title: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
        now_ts: int | None = None,
    ) -> dict[str, Any]:
        metadata_value = dict(metadata or {})
        metadata_value.setdefault("document_ref", source_ref)
        return self.ingest_text(
            source_kind=SemanticSourceKind.DOCUMENT,
            source_ref=source_ref,
            text=text,
            scope=scope or f"document:{source_ref}",
            title=title,
            thread_id=thread_id,
            project_id=project_id,
            pinned=pinned,
            metadata=metadata_value,
            now_ts=now_ts,
        )

    def ingest_document_file(
        self,
        *,
        path: str,
        source_ref: str | None = None,
        scope: str | None = None,
        title: str | None = None,
        project_id: str | None = None,
        thread_id: str | None = None,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
        encoding: str = "utf-8",
        now_ts: int | None = None,
    ) -> dict[str, Any]:
        file_path = Path(str(path or "").strip())
        if not str(file_path).strip():
            return {"ok": False, "ingested": False, "reason": "document_path_required", "status": "skipped"}
        try:
            resolved_path = file_path.expanduser()
            if resolved_path.exists():
                resolved_path = resolved_path.resolve()
            file_text = resolved_path.read_text(encoding=encoding, errors="replace")
        except Exception as exc:
            return {
                "ok": False,
                "ingested": False,
                "reason": exc.__class__.__name__,
                "status": "skipped",
                "path": str(file_path),
            }
        source_ref_value = str(source_ref or resolved_path)
        metadata_value = dict(metadata or {})
        metadata_value.update(
            {
                "document_path": str(resolved_path),
                "encoding": encoding,
            }
        )
        try:
            metadata_value["byte_size"] = int(resolved_path.stat().st_size)
        except Exception:
            pass
        return self.ingest_document_text(
            source_ref=source_ref_value,
            text=file_text,
            scope=scope or f"document:{source_ref_value}",
            title=title or resolved_path.name,
            project_id=project_id,
            thread_id=thread_id,
            pinned=pinned,
            metadata=metadata_value,
            now_ts=now_ts,
        )

    def ingest_text(
        self,
        *,
        source_kind: SemanticSourceKind | str,
        source_ref: str,
        text: str,
        scope: str | None = None,
        title: str | None = None,
        thread_id: str | None = None,
        project_id: str | None = None,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
        now_ts: int | None = None,
    ) -> dict[str, Any]:
        result = {
            "ok": False,
            "ingested": False,
            "reason": None,
            "source_kind": None,
            "source_ref": str(source_ref or ""),
            "source_id": None,
            "chunk_ids": [],
            "vector_count": 0,
            "embedding_dim": None,
            "status": None,
        }
        if not self.enabled():
            result["reason"] = "semantic_memory_disabled"
            return result
        target = self._resolve_target()
        if target is None or target.reason:
            result["reason"] = target.reason if target is not None else "embed_model_not_configured"
            state = self.store.set_index_state(
                scope=_SCOPE_GLOBAL,
                embed_provider=target.provider_id if target else None,
                embed_model=target.model_id if target else None,
                embedding_dim=target.dimensions if target else None,
                status="stale",
                source_count=len(self.store.list_sources()),
                chunk_count=0,
                vector_count=0,
                stale_since=int(now_ts) if now_ts is not None else int(time.time()),
                last_error_kind=result["reason"],
                last_error_message=result["reason"],
                details={"source_kind": str(source_kind)},
            )
            result["status"] = state.status
            return result
        provider = self._provider_for_target(target)
        if provider is None:
            result["reason"] = "embedding_provider_unavailable"
            self._rebuild_state(
                target=target,
                status="stale",
                source_count=len(self.store.list_sources()),
                chunk_count=0,
                vector_count=0,
                embedding_dim=None,
                stale_since=int(now_ts) if now_ts is not None else int(time.time()),
                last_error_kind=result["reason"],
                last_error_message=result["reason"],
                details={"source_kind": str(source_kind)},
            )
            return result

        normalized_text = _normalize_text(text)
        if not normalized_text:
            result["reason"] = "empty_text"
            return result

        if isinstance(source_kind, SemanticSourceKind):
            kind = source_kind
        else:
            normalized_kind = str(source_kind).strip().lower()
            kind = SemanticSourceKind(normalized_kind) if normalized_kind in {item.value for item in SemanticSourceKind} else SemanticSourceKind.CONVERSATION

        if not self._source_kind_enabled(kind):
            result["reason"] = f"{kind.value}_semantic_disabled"
            result["status"] = "disabled"
            return result
        result["source_kind"] = kind.value
        result["source_ref"] = str(source_ref or "")

        scope_value = _normalize_scope(scope)
        created_at = int(now_ts) if now_ts is not None else int(time.time())
        metadata_value = metadata or {}
        source_hash = _content_hash(kind.value, str(source_ref), scope_value, normalized_text, metadata_value)
        source_id = _source_id(kind.value, str(source_ref), scope_value, source_hash)
        source = self.store.upsert_source(
            source_id=source_id,
            source_kind=kind,
            source_ref=str(source_ref),
            scope=scope_value,
            content_hash=source_hash,
            title=title,
            thread_id=thread_id,
            project_id=project_id,
            pinned=bool(pinned),
            status="indexing",
            metadata=metadata_value,
            created_at=created_at,
            updated_at=created_at,
        )
        chunk_rows: list[dict[str, Any]] = []
        for index, (start, end, chunk_text) in enumerate(_split_chunks(normalized_text, chunk_size=self._chunk_size)):
            chunk_id = _chunk_id(source_id, index, chunk_text)
            chunk_rows.append(
                {
                    "id": chunk_id,
                    "chunk_index": index,
                    "text": chunk_text,
                    "chunk_hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                    "char_start": start,
                    "char_end": end,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "metadata": metadata_value,
                }
            )
        self.store.replace_chunks(source_id=source.id, chunks=chunk_rows)
        current_counts = self._counts_payload(scope=None)
        if not chunk_rows:
            state = self._rebuild_state(
                target=target,
                status="ready",
                source_count=current_counts["source_count"],
                chunk_count=0,
                vector_count=0,
                embedding_dim=None,
                last_indexed_at=created_at,
                details={"source_kind": kind.value, "source_id": source.id},
            )
            result.update({"ok": True, "ingested": True, "reason": None, "source_id": source.id, "status": state.status})
            return result

        vectors: list[tuple[float, ...]] = []
        try:
            response = self._embed_texts(provider, target.model_id, tuple(row["text"] for row in chunk_rows))
            vectors = list(response.vectors)
        except Exception as exc:
            self._rebuild_state(
                target=target,
                status="stale",
                source_count=current_counts["source_count"],
                chunk_count=current_counts["chunk_count"],
                vector_count=0,
                embedding_dim=None,
                stale_since=created_at,
                last_error_kind=exc.__class__.__name__,
                last_error_message=str(exc),
                details={"source_kind": kind.value, "source_id": source.id},
            )
            self.store.upsert_source(
                source_id=source.id,
                source_kind=kind,
                source_ref=str(source_ref),
                scope=scope_value,
                content_hash=source_hash,
                title=title,
                thread_id=thread_id,
                project_id=project_id,
                pinned=bool(pinned),
                status="failed",
                metadata=metadata_value,
                created_at=created_at,
                updated_at=created_at,
            )
            result["reason"] = exc.__class__.__name__
            result["status"] = "stale"
            return result

        if len(vectors) != len(chunk_rows):
            self._rebuild_state(
                target=target,
                status="partial",
                source_count=current_counts["source_count"],
                chunk_count=current_counts["chunk_count"],
                vector_count=current_counts["vector_count"],
                embedding_dim=response.dimensions if vectors else None,
                stale_since=created_at,
                last_error_kind="embedding_vector_count_mismatch",
                last_error_message="embedding count did not match chunk count",
                details={"source_kind": kind.value, "source_id": source.id},
            )
            self.store.upsert_source(
                source_id=source.id,
                source_kind=kind,
                source_ref=str(source_ref),
                scope=scope_value,
                content_hash=source_hash,
                title=title,
                thread_id=thread_id,
                project_id=project_id,
                pinned=bool(pinned),
                status="partial",
                metadata=metadata_value,
                created_at=created_at,
                updated_at=created_at,
            )
            result["reason"] = "embedding_vector_count_mismatch"
            result["status"] = "partial"
            result["embedding_dim"] = response.dimensions if vectors else None
            result["vector_count"] = len(vectors)
            return result

        embedding_dim = int(response.dimensions)
        for chunk_row, vector in zip(chunk_rows, vectors, strict=True):
            self.store.upsert_vector(
                chunk_id=str(chunk_row["id"]),
                embed_provider=target.provider_id,
                embed_model=target.model_id,
                embedding_dim=embedding_dim,
                vector=vector,
                created_at=created_at,
                updated_at=created_at,
            )
        current_counts = self._counts_payload(scope=None)
        state = self._rebuild_state(
            target=target,
            status="ready",
            source_count=current_counts["source_count"],
            chunk_count=current_counts["chunk_count"],
            vector_count=current_counts["vector_count"],
            embedding_dim=embedding_dim,
            last_indexed_at=created_at,
            details={"source_kind": kind.value, "source_id": source.id},
        )
        self.store.upsert_source(
            source_id=source.id,
            source_kind=kind,
            source_ref=str(source_ref),
            scope=scope_value,
            content_hash=source_hash,
            title=title,
            thread_id=thread_id,
            project_id=project_id,
            pinned=bool(pinned),
            status="ready",
            metadata=metadata_value,
            created_at=created_at,
            updated_at=created_at,
        )
        result.update(
            {
                "ok": True,
                "ingested": True,
                "reason": None,
                "source_id": source.id,
                "chunk_ids": [row["id"] for row in chunk_rows],
                "vector_count": len(chunk_rows),
                "embedding_dim": embedding_dim,
                "status": state.status,
            }
        )
        return result

    def repair_scope(self, *, scope: str = _SCOPE_GLOBAL, now_ts: int | None = None) -> dict[str, Any]:
        selected_scope = _normalize_scope(scope)
        result: dict[str, Any] = {
            "ok": False,
            "scope": selected_scope,
            "status": "skipped",
            "reason": None,
            "target": self._target_payload(self._resolve_target()),
            "index_state_before": self._state_payload(self._index_state()),
            "index_state_after": None,
            "counts_before": self._counts_payload(scope=selected_scope),
            "all_counts_before": self._counts_payload(scope=None, source_kinds=self._enabled_source_kinds()),
            "counts_after": None,
            "all_counts_after": None,
            "rebuilt_sources": [],
            "skipped_sources": [],
            "failed_sources": [],
        }
        if not self.enabled():
            result["reason"] = "semantic_memory_disabled"
            result["summary"] = f"scope={selected_scope}; state=skipped; reason=semantic_memory_disabled"
            return result
        target = self._resolve_target()
        if target is None or target.reason:
            result["reason"] = target.reason if target else "embed_model_not_configured"
            result["summary"] = f"scope={selected_scope}; state=skipped; reason={result['reason']}"
            return result
        provider = self._provider_for_target(target)
        if provider is None:
            result["reason"] = "embedding_provider_unavailable"
            result["summary"] = f"scope={selected_scope}; state=skipped; reason=embedding_provider_unavailable"
            return result

        source_rows = [row for row in self.store.list_sources(scope=selected_scope) if self._source_kind_enabled(row.source_kind)]
        if not source_rows:
            state = self._rebuild_state(
                target=target,
                status="missing",
                source_count=0,
                chunk_count=0,
                vector_count=0,
                embedding_dim=None,
                stale_since=int(now_ts) if now_ts is not None else int(time.time()),
                last_error_kind="index_missing",
                last_error_message="no semantic sources found for scope",
                details={"scope": selected_scope, "mode": "repair"},
            )
            result.update(
                {
                    "ok": True,
                    "status": state.status,
                    "reason": "index_missing",
                    "index_state_after": self._state_payload(state),
                    "counts_after": self._counts_payload(scope=selected_scope),
                    "all_counts_after": self._counts_payload(scope=None, source_kinds=self._enabled_source_kinds()),
                    "summary": f"scope={selected_scope}; rebuilt=0; state={state.status}; reason=index_missing",
                }
            )
            return result

        rebuilt_sources: list[dict[str, Any]] = []
        skipped_sources: list[dict[str, Any]] = []
        failed_sources: list[dict[str, Any]] = []
        embedding_dim: int | None = None
        run_ts = int(now_ts) if now_ts is not None else int(time.time())

        for source_row in source_rows:
            chunks = self.store.list_chunks(source_id=source_row.id)
            if not chunks:
                self.store.upsert_source(
                    source_id=source_row.id,
                    source_kind=source_row.source_kind,
                    source_ref=source_row.source_ref,
                    scope=source_row.scope,
                    content_hash=source_row.content_hash,
                    title=source_row.title,
                    thread_id=source_row.thread_id,
                    project_id=source_row.project_id,
                    pinned=source_row.pinned,
                    status="missing",
                    metadata=source_row.metadata,
                    created_at=source_row.created_at,
                    updated_at=run_ts,
                )
                skipped_sources.append(
                    {
                        "source_id": source_row.id,
                        "source_ref": source_row.source_ref,
                        "reason": "missing_chunks",
                    }
                )
                continue
            try:
                response = self._embed_texts(provider, target.model_id, tuple(chunk.text for chunk in chunks))
            except Exception as exc:
                self.store.upsert_source(
                    source_id=source_row.id,
                    source_kind=source_row.source_kind,
                    source_ref=source_row.source_ref,
                    scope=source_row.scope,
                    content_hash=source_row.content_hash,
                    title=source_row.title,
                    thread_id=source_row.thread_id,
                    project_id=source_row.project_id,
                    pinned=source_row.pinned,
                    status="failed",
                    metadata=source_row.metadata,
                    created_at=source_row.created_at,
                    updated_at=run_ts,
                )
                failed_sources.append(
                    {
                        "source_id": source_row.id,
                        "source_ref": source_row.source_ref,
                        "reason": exc.__class__.__name__,
                    }
                )
                continue
            if len(response.vectors) != len(chunks):
                self.store.upsert_source(
                    source_id=source_row.id,
                    source_kind=source_row.source_kind,
                    source_ref=source_row.source_ref,
                    scope=source_row.scope,
                    content_hash=source_row.content_hash,
                    title=source_row.title,
                    thread_id=source_row.thread_id,
                    project_id=source_row.project_id,
                    pinned=source_row.pinned,
                    status="partial",
                    metadata=source_row.metadata,
                    created_at=source_row.created_at,
                    updated_at=run_ts,
                )
                failed_sources.append(
                    {
                        "source_id": source_row.id,
                        "source_ref": source_row.source_ref,
                        "reason": "embedding_vector_count_mismatch",
                    }
                )
                continue
            embedding_dim = int(response.dimensions)
            for chunk_row, vector in zip(chunks, response.vectors, strict=True):
                self.store.upsert_vector(
                    chunk_id=chunk_row.id,
                    embed_provider=target.provider_id,
                    embed_model=target.model_id,
                    embedding_dim=embedding_dim,
                    vector=vector,
                    created_at=run_ts,
                    updated_at=run_ts,
                )
            self.store.upsert_source(
                source_id=source_row.id,
                source_kind=source_row.source_kind,
                source_ref=source_row.source_ref,
                scope=source_row.scope,
                content_hash=source_row.content_hash,
                title=source_row.title,
                thread_id=source_row.thread_id,
                project_id=source_row.project_id,
                pinned=source_row.pinned,
                status="ready",
                metadata=source_row.metadata,
                created_at=source_row.created_at,
                updated_at=run_ts,
            )
            rebuilt_sources.append(
                {
                    "source_id": source_row.id,
                    "source_ref": source_row.source_ref,
                    "chunk_count": len(chunks),
                }
            )

        if failed_sources or skipped_sources:
            state_status = "partial"
            reason = "partial_index"
        else:
            state_status = "ready"
            reason = None
        counts_after = self._counts_payload(scope=selected_scope)
        all_counts_after = self._counts_payload(scope=None, source_kinds=self._enabled_source_kinds())
        state = self._rebuild_state(
            target=target,
            status=state_status,
            source_count=counts_after["source_count"],
            chunk_count=counts_after["chunk_count"],
            vector_count=counts_after["vector_count"],
            embedding_dim=embedding_dim,
            last_indexed_at=run_ts,
            stale_since=run_ts if state_status != "ready" else None,
            last_error_kind=reason,
            last_error_message=reason,
            details={
                "scope": selected_scope,
                "mode": "repair",
                "rebuilt_sources": [row["source_id"] for row in rebuilt_sources],
                "skipped_sources": skipped_sources,
                "failed_sources": failed_sources,
            },
        )
        result.update(
            {
                "ok": True,
                "status": state.status,
                "reason": reason,
                "rebuilt_sources": rebuilt_sources,
                "skipped_sources": skipped_sources,
                "failed_sources": failed_sources,
                "index_state_after": self._state_payload(state),
                "counts_after": counts_after,
                "all_counts_after": all_counts_after,
                "summary": (
                    f"scope={selected_scope}; rebuilt={len(rebuilt_sources)}; skipped={len(skipped_sources)}; "
                    f"failed={len(failed_sources)}; state={state.status}; reason={reason or 'none'}"
                ),
            }
        )
        return result

    def rebuild_scope(self, *, scope: str = _SCOPE_GLOBAL, now_ts: int | None = None) -> dict[str, Any]:
        return self.repair_scope(scope=scope, now_ts=now_ts)

    def _selection_status(self, target: EmbeddingTarget | None, state: SemanticIndexState | None) -> tuple[bool, str | None]:
        if not self.enabled():
            return False, "semantic_memory_disabled"
        if target is None:
            return False, "embed_model_not_configured"
        if target.reason:
            return False, target.reason
        if state is None:
            return False, "index_missing"
        counts = self._counts_payload(scope=None)
        if (
            counts["source_count"] != state.source_count
            or counts["chunk_count"] != state.chunk_count
            or counts["vector_count"] != state.vector_count
        ):
            return False, "index_state_mismatch"
        if counts["chunk_count"] > counts["vector_count"]:
            return False, "partial_index"
        if state.status not in {"ready", "warm"}:
            return False, state.status
        if state.embed_model != target.model_id or state.embed_provider != target.provider_id:
            return False, "stale_embed_model"
        if state.embedding_dim is not None and state.embedding_dim <= 0:
            return False, "dimension_invalid"
        return True, None

    def build_context_for_payload(
        self,
        payload: dict[str, Any],
        *,
        intent: str,
        now_ts: int | None = None,
    ) -> SemanticSelection:
        target, state = self._current_target_and_state()
        enabled, reason = self._selection_status(target, state)
        if not enabled:
            debug = {
                "status": "skipped",
                "reason": reason,
                "target": self._target_payload(target),
                "index_state": self._state_payload(state),
                "selected_ids": [],
                "skipped": [],
                "query": {
                    "intent": str(intent or "").strip().lower() or "chat",
                    "text": "",
                    "scopes": [],
                },
            }
            return SemanticSelection(candidates=[], merged_context_text="", debug=debug)

        assert target is not None
        provider = self._provider_for_target(target)
        if provider is None:
            return SemanticSelection(
                candidates=[],
                merged_context_text="",
                debug={
                    "status": "skipped",
                    "reason": "embedding_provider_unavailable",
                    "target": self._target_payload(target),
                    "index_state": self._state_payload(state),
                    "selected_ids": [],
                    "skipped": [],
                    "query": {"intent": str(intent or "").strip().lower() or "chat", "text": "", "scopes": []},
                },
            )

        query_text = self._extract_query_text(payload, intent=intent)
        scopes = self._extract_scopes(payload)
        if not query_text:
            return SemanticSelection(
                candidates=[],
                merged_context_text="",
                debug={
                    "status": "skipped",
                    "reason": "empty_query",
                    "target": self._target_payload(target),
                    "index_state": self._state_payload(state),
                    "selected_ids": [],
                    "skipped": [],
                    "query": {"intent": str(intent or "").strip().lower() or "chat", "text": "", "scopes": scopes},
                },
            )

        try:
            response = self._embed_texts(provider, target.model_id, (query_text,))
        except Exception as exc:
            return SemanticSelection(
                candidates=[],
                merged_context_text="",
                debug={
                    "status": "skipped",
                    "reason": exc.__class__.__name__,
                    "target": self._target_payload(target),
                    "index_state": self._state_payload(state),
                    "selected_ids": [],
                    "skipped": [],
                    "query": {"intent": str(intent or "").strip().lower() or "chat", "text": query_text, "scopes": scopes},
                },
            )

        query_vector = response.vectors[0]
        enabled_kinds = self._enabled_source_kinds()
        if not enabled_kinds:
            return SemanticSelection(
                candidates=[],
                merged_context_text="",
                debug={
                    "status": "skipped",
                    "reason": "semantic_source_kinds_disabled",
                    "target": self._target_payload(target),
                    "index_state": self._state_payload(state),
                    "selected_ids": [],
                    "skipped": [],
                    "query": {"intent": str(intent or "").strip().lower() or "chat", "text": query_text, "scopes": scopes},
                },
            )
        candidate_rows = self.store.list_joined_rows(scopes=scopes or None, source_kinds=enabled_kinds, limit=self._query_limit)
        candidate_rows = [row for row in candidate_rows if row.get("vector_json")]
        selected: list[SemanticCandidate] = []
        skipped: list[dict[str, Any]] = []
        query_ts = int(now_ts) if now_ts is not None else int(time.time())
        query_tokens = _token_set(query_text)

        for row in candidate_rows:
            vector_raw = row.get("vector_json")
            try:
                vector_payload = json.loads(vector_raw or "[]")
            except json.JSONDecodeError:
                skipped.append({"id": row.get("chunk_id"), "reason": "invalid_vector_json"})
                continue
            if not isinstance(vector_payload, list):
                skipped.append({"id": row.get("chunk_id"), "reason": "invalid_vector_payload"})
                continue
            vector: list[float] = []
            for value in vector_payload:
                try:
                    vector.append(float(value))
                except (TypeError, ValueError):
                    continue
            if not vector:
                skipped.append({"id": row.get("chunk_id"), "reason": "empty_vector"})
                continue
            if len(vector) != len(query_vector):
                skipped.append({"id": row.get("chunk_id"), "reason": "dimension_mismatch"})
                continue
            if state is not None and (state.embedding_dim is not None and state.embedding_dim != len(vector)):
                skipped.append({"id": row.get("chunk_id"), "reason": "state_dimension_mismatch"})
                continue
            similarity = _cosine_similarity(query_vector, tuple(vector))
            if similarity <= 0.0:
                skipped.append({"id": row.get("chunk_id"), "reason": "non_positive_similarity"})
                continue
            pin_bonus = 0.05 if bool(row.get("pinned", False)) else 0.0
            scope_bonus = 0.05 if str(row.get("scope") or "") in scopes else 0.0
            age_seconds = max(0, query_ts - int(row.get("chunk_updated_at") or row.get("source_updated_at") or query_ts))
            recency_bonus = max(0.0, 0.08 - min(0.08, float(age_seconds) / (30.0 * 24.0 * 3600.0)))
            overlap_bonus = _overlap_bonus(query_text, str(row.get("chunk_text") or ""))
            score = round((similarity * 0.85) + pin_bonus + scope_bonus + recency_bonus + overlap_bonus, 6)
            if score <= 0:
                skipped.append({"id": row.get("chunk_id"), "reason": "non_positive_score"})
                continue
            metadata = dict(row.get("source_metadata") or {})
            metadata.update({"chunk_metadata": dict(row.get("chunk_metadata") or {})})
            candidate = SemanticCandidate(
                id=str(row.get("chunk_id")),
                source_id=str(row.get("source_id")),
                chunk_id=str(row.get("chunk_id")),
                source_kind=SemanticSourceKind(str(row.get("source_kind") or "conversation")),
                source_ref=str(row.get("source_ref") or ""),
                scope=str(row.get("scope") or _SCOPE_GLOBAL),
                text=str(row.get("chunk_text") or ""),
                score=score,
                similarity=round(similarity, 6),
                pin_bonus=round(pin_bonus, 6),
                scope_bonus=round(scope_bonus, 6),
                recency_bonus=round(recency_bonus, 6),
                overlap_bonus=round(overlap_bonus, 6),
                embed_provider=str(row.get("embed_provider") or ""),
                embed_model=str(row.get("embed_model") or ""),
                embedding_dim=int(row.get("embedding_dim") or 0),
                created_at=int(row.get("chunk_created_at") or row.get("source_created_at") or query_ts),
                updated_at=int(row.get("chunk_updated_at") or row.get("source_updated_at") or query_ts),
                title=str(row.get("title")).strip() if row.get("title") is not None and str(row.get("title")).strip() else None,
                thread_id=str(row.get("thread_id")).strip() if row.get("thread_id") is not None and str(row.get("thread_id")).strip() else None,
                project_id=str(row.get("project_id")).strip() if row.get("project_id") is not None and str(row.get("project_id")).strip() else None,
                metadata=metadata,
            )
            selected.append(candidate)

        selected.sort(key=lambda item: (-float(item.score), item.id))
        selected = selected[: self._max_candidates]
        blocks: list[str] = []
        remaining_budget = self._max_prompt_chars
        for candidate in selected:
            separator = 2 if blocks else 0
            if remaining_budget <= separator:
                break
            block = self._render_candidate_block(candidate, remaining_budget - separator)
            if not block:
                break
            if separator:
                blocks.append("")
            blocks.append(block)
            remaining_budget -= separator + len(block)

        merged = "\n".join(blocks).strip()
        if len(merged) > self._max_prompt_chars:
            merged = merged[: self._max_prompt_chars].rstrip()
        debug = {
            "status": "ok" if selected else "empty",
            "reason": None if selected else "no_candidates",
            "target": self._target_payload(target),
            "index_state": self._state_payload(state),
            "selected_ids": [candidate.id for candidate in selected],
            "selected": [
                {
                    "id": candidate.id,
                    "score": candidate.score,
                    "similarity": candidate.similarity,
                    "source_kind": candidate.source_kind.value,
                    "scope": candidate.scope,
                    "source_ref": candidate.source_ref,
                    "why": {
                        "pin_bonus": candidate.pin_bonus,
                        "scope_bonus": candidate.scope_bonus,
                        "recency_bonus": candidate.recency_bonus,
                        "overlap_bonus": candidate.overlap_bonus,
                    },
                }
                for candidate in selected
            ],
            "skipped": skipped,
            "query": {
                "intent": str(intent or "").strip().lower() or "chat",
                "text": query_text,
                "scopes": scopes,
                "embedding_dim": len(query_vector),
            },
        }
        return SemanticSelection(candidates=selected, merged_context_text=merged, debug=debug)

    def _render_candidate_block(self, candidate: SemanticCandidate, budget: int) -> str:
        if budget <= 0:
            return ""
        source_bits = [
            f"source={candidate.source_kind.value}",
            f"scope={candidate.scope}",
            f"ref={candidate.source_ref}",
            f"model={candidate.embed_model}",
        ]
        if candidate.title:
            source_bits.append(f"title={candidate.title}")
        header = (
            f"SEMANTIC_MEMORY[{candidate.id}] (score={candidate.score:.4f}, "
            f"sim={candidate.similarity:.4f}, {' '.join(source_bits)}):"
        )
        if len(header) + 1 >= budget:
            return _trunc(header, budget)

        text = _normalize_text(candidate.text)
        if not text:
            empty_block = f"{header}\n\"\""
            return empty_block if len(empty_block) <= budget else _trunc(header, budget)

        best_quote = "\"\""
        low = 0
        high = len(text)
        while low <= high:
            mid = (low + high) // 2
            quoted = json.dumps(_trunc(text, mid), ensure_ascii=True)
            total_len = len(header) + 1 + len(quoted)
            if total_len <= budget:
                best_quote = quoted
                low = mid + 1
            else:
                high = mid - 1

        if len(header) + 1 + len(best_quote) > budget:
            return _trunc(header, budget)
        return f"{header}\n{best_quote}"

    def _extract_query_text(self, payload: dict[str, Any], *, intent: str) -> str:
        body = payload if isinstance(payload, dict) else {}
        for key in ("semantic_query", "query", "text", "message", "content"):
            raw = body.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        raw_messages = body.get("messages")
        if isinstance(raw_messages, list):
            for row in reversed(raw_messages):
                if not isinstance(row, dict):
                    continue
                role = str(row.get("role") or "").strip().lower() or "user"
                if role != "user":
                    continue
                content = row.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        if str(intent or "").strip().lower() in {"remember", "remember_note"}:
            note_text = body.get("note") or body.get("note_text")
            if isinstance(note_text, str) and note_text.strip():
                return note_text.strip()
        return ""

    def _extract_scopes(self, payload: dict[str, Any]) -> list[str]:
        body = payload if isinstance(payload, dict) else {}
        scopes = {_SCOPE_GLOBAL}
        thread_id = str(body.get("thread_id") or "").strip()
        if thread_id:
            scopes.add(f"thread:{thread_id}")
        project = str(body.get("project") or body.get("project_id") or body.get("project_tag") or "").strip()
        if project:
            scopes.add(f"project:{project}")
        user_id = str(body.get("user_id") or "").strip()
        if user_id:
            scopes.add(f"user:{user_id}")
        document_ref = str(body.get("document_ref") or body.get("path") or "").strip()
        if document_ref:
            scopes.add(f"document:{document_ref}")
        explicit_scope = str(body.get("semantic_scope") or "").strip()
        if explicit_scope:
            scopes.add(explicit_scope)
        return sorted(scopes)


def build_semantic_memory_service(
    *,
    config: Config,
    registry: Registry,
    provider_resolver: Callable[[str], Provider | None] | None,
    db_path: str,
) -> SemanticMemoryService:
    return SemanticMemoryService(
        config=config,
        registry=registry,
        provider_resolver=provider_resolver,
        db_path=db_path,
    )
