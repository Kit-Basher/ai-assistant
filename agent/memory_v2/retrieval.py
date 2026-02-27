from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from agent.memory_v2.storage import MemoryStorage
from agent.memory_v2.types import MemoryItem, MemoryLevel, MemoryQuery, MemorySelection


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]*")
_PATH_RE = re.compile(r"(?:/[a-zA-Z0-9._-]+){1,}")
_ENDPOINT_RE = re.compile(r"\/[a-z0-9._\-/{}]+")
_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_MODEL_REF_RE = re.compile(r"\b[a-z0-9._-]+:[a-z0-9._\-/]+\b")
_LEVEL_ORDER = (
    MemoryLevel.EPISODIC,
    MemoryLevel.SEMANTIC,
    MemoryLevel.PROCEDURAL,
)
_DEFAULT_LIMITS = {
    MemoryLevel.EPISODIC.value: 3,
    MemoryLevel.SEMANTIC.value: 3,
    MemoryLevel.PROCEDURAL.value: 3,
}
_EPISODIC_RECENCY_WINDOW_SECONDS = 30 * 24 * 60 * 60


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text) if token}


def _extract_entities(text: str) -> list[str]:
    values = set()
    for pattern in (_PATH_RE, _ENDPOINT_RE, _HASH_RE, _MODEL_REF_RE):
        values.update(match.group(0).lower() for match in pattern.finditer(text))
    return sorted(values)


def _effective_limits(raw: dict[str, int] | None) -> dict[str, int]:
    output = dict(_DEFAULT_LIMITS)
    if not isinstance(raw, dict):
        return output
    for level in _DEFAULT_LIMITS:
        value = raw.get(level)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        output[level] = max(0, parsed)
    return output


def _tags_match(query_tags: dict[str, str], item_tags: dict[str, str]) -> bool:
    if not query_tags:
        return True
    for key, value in sorted(query_tags.items(), key=lambda item: str(item[0])):
        key_norm = str(key).strip().lower()
        value_norm = str(value).strip()
        if not key_norm or not value_norm:
            continue
        if str(item_tags.get(key_norm) or "") != value_norm:
            return False
    return True


def _score_item(
    *,
    item: MemoryItem,
    query_tokens: set[str],
    query_entities: list[str],
    now_ts: int,
) -> tuple[float, dict[str, Any]]:
    item_norm = normalize_text(item.text)
    item_tokens = _tokenize(item_norm)
    overlap = sorted(query_tokens.intersection(item_tokens))
    overlap_count = len(overlap)

    entity_hits = sorted(entity for entity in query_entities if entity and entity in item_norm)
    entity_hit_count = len(entity_hits)

    lexical_score = float(overlap_count * 4)
    coverage_score = 0.0
    if query_tokens:
        coverage_score = (overlap_count / float(len(query_tokens))) * 8.0

    entity_score = min(12.0, float(entity_hit_count * 4))
    pin_score = 6.0 if bool(item.pinned) else 0.0

    recency_score = 0.0
    if item.level == MemoryLevel.EPISODIC:
        age_seconds = max(0, int(now_ts) - int(item.created_at))
        age_days = age_seconds / 86400.0
        recency_score = max(0.0, 6.0 - min(6.0, age_days))

    total = lexical_score + coverage_score + entity_score + pin_score + recency_score
    debug = {
        "lexical_score": round(lexical_score, 4),
        "coverage_score": round(coverage_score, 4),
        "entity_score": round(entity_score, 4),
        "pin_score": round(pin_score, 4),
        "recency_score": round(recency_score, 4),
        "overlap_tokens": overlap,
        "entity_hits": entity_hits,
    }
    return round(total, 4), debug


def select_memory(query: MemoryQuery, store: MemoryStorage) -> MemorySelection:
    query_norm = normalize_text(query.text)
    query_tokens = _tokenize(query_norm)
    query_entities = _extract_entities(query_norm)
    now_ts = int(query.now_ts if query.now_ts is not None else int(time.time()))
    limits = _effective_limits(query.limit_per_level)

    raw_by_level: dict[MemoryLevel, list[MemoryItem]] = {
        MemoryLevel.EPISODIC: list(store.list_episodic_events(limit=500)),
        MemoryLevel.SEMANTIC: list(store.list_memory_items(level=MemoryLevel.SEMANTIC)),
        MemoryLevel.PROCEDURAL: list(store.list_memory_items(level=MemoryLevel.PROCEDURAL)),
    }

    selected_by_level: dict[MemoryLevel, list[MemoryItem]] = {
        MemoryLevel.EPISODIC: [],
        MemoryLevel.SEMANTIC: [],
        MemoryLevel.PROCEDURAL: [],
    }

    selected_debug: list[dict[str, Any]] = []
    skipped_debug: list[dict[str, Any]] = []

    for level in _LEVEL_ORDER:
        candidates: list[tuple[float, MemoryItem, dict[str, Any]]] = []
        for item in sorted(raw_by_level.get(level, []), key=lambda row: row.id):
            if not _tags_match(query.tags, item.tags):
                skipped_debug.append(
                    {
                        "id": item.id,
                        "level": level.value,
                        "reason": "tag_mismatch",
                    }
                )
                continue

            if level == MemoryLevel.EPISODIC and not item.pinned:
                age_seconds = max(0, now_ts - int(item.created_at))
                if age_seconds > _EPISODIC_RECENCY_WINDOW_SECONDS:
                    skipped_debug.append(
                        {
                            "id": item.id,
                            "level": level.value,
                            "reason": "episodic_outside_window",
                        }
                    )
                    continue

            score, score_debug = _score_item(
                item=item,
                query_tokens=query_tokens,
                query_entities=query_entities,
                now_ts=now_ts,
            )
            has_overlap = bool(score_debug.get("overlap_tokens"))
            has_entities = bool(score_debug.get("entity_hits"))
            if not has_overlap and not has_entities and not item.pinned:
                skipped_debug.append(
                    {
                        "id": item.id,
                        "level": level.value,
                        "reason": "no_relevance_signal",
                    }
                )
                continue
            candidates.append((score, item, score_debug))

        candidates.sort(key=lambda row: (-float(row[0]), row[1].id))
        limit = limits.get(level.value, 0)
        chosen = candidates[:limit] if limit > 0 else []
        selected_by_level[level] = [row[1] for row in chosen]
        for score, item, score_debug in chosen:
            selected_debug.append(
                {
                    "id": item.id,
                    "level": level.value,
                    "score": round(float(score), 4),
                    "why": score_debug,
                }
            )

    selected_ids = [item.id for level in _LEVEL_ORDER for item in selected_by_level[level]]
    debug = {
        "query": {
            "norm": query_norm,
            "tokens": sorted(query_tokens),
            "entities": list(query_entities),
            "tags": dict(sorted((query.tags or {}).items())),
        },
        "limits": {key: int(value) for key, value in sorted(limits.items())},
        "selected_ids": selected_ids,
        "selected": sorted(selected_debug, key=lambda row: (str(row.get("level")), str(row.get("id")))),
        "skipped": sorted(skipped_debug, key=lambda row: (str(row.get("level")), str(row.get("id")), str(row.get("reason")))),
    }

    return MemorySelection(
        items_by_level=selected_by_level,
        merged_context_text="",
        debug=debug,
    )
