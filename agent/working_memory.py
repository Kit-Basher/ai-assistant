from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Callable, Iterable, Literal


Role = Literal["system", "user", "assistant", "tool"]

_DEFAULT_MAX_CONTEXT_TOKENS = 32768
_DEFAULT_RESERVED_OUTPUT_TOKENS = 2000
_DEFAULT_RESERVED_SYSTEM_AND_TOOLS = 3000
_MIN_RAW_CHUNK_TURNS = 6
_TARGET_RAW_CHUNK_TURNS = 8
_MAX_RAW_CHUNK_TURNS = 12
_MIN_RAW_CHUNK_TOKENS = 1200
_MAX_RAW_CHUNK_TOKENS = 2500
_RECENCY_SHIELDED_USER_TURNS = 2
_SUMMARY_LEVEL_LIMIT = 3
_SEMANTIC_INGEST_DEDUPE_LIMIT = 128
_TOKEN_RE = re.compile(r"[A-Za-z0-9_:/.-]+|[^\w\s]", re.ASCII)
_PATH_RE = re.compile(r"(?:`([^`]+)`|(?:(?:~|/|\./|\.\./)[^\s'\"`]+)|(?:[A-Za-z0-9._-]+\.[A-Za-z0-9._/-]+))")
_PREFERENCE_RE = re.compile(
    r"(?i)\b(?:i prefer|prefer|please |don't |do not |avoid |keep it |be concise|be brief|concise|brief)\b"
)
_DECISION_RE = re.compile(
    r"(?i)\b(?:decided|decision|we will|i will|use |keep |switch to|make .* default|do not|don't|must|should)\b"
)
_OPEN_THREAD_RE = re.compile(
    r"(?i)\b(?:todo|next step|follow up|follow-up|need to|still need|unresolved|pending|investigate|fix|implement)\b"
)
_TOOL_PREFIX = "Tool result:"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_text_tokens(text: str | None) -> int:
    cleaned = str(text or "").strip()
    if not cleaned:
        return 0
    tokenish = len(_TOKEN_RE.findall(cleaned))
    char_estimate = math.ceil(len(cleaned) / 4)
    return max(1, tokenish, char_estimate)


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS
    reserved_output_tokens: int = _DEFAULT_RESERVED_OUTPUT_TOKENS
    reserved_system_and_tools: int = _DEFAULT_RESERVED_SYSTEM_AND_TOOLS

    @property
    def working_memory_budget(self) -> int:
        return max(
            1024,
            int(self.max_context_tokens)
            - int(self.reserved_output_tokens)
            - int(self.reserved_system_and_tools),
        )

    @property
    def soft_threshold(self) -> int:
        return int(self.working_memory_budget * 0.72)

    @property
    def hard_threshold(self) -> int:
        return int(self.working_memory_budget * 0.85)

    @property
    def panic_threshold(self) -> int:
        return int(self.working_memory_budget * 0.93)


@dataclass
class Turn:
    turn_id: str
    role: Role
    text: str
    token_count: int
    created_at: str
    pinned: bool = False
    topic_hint: str | None = None
    references: list[str] = field(default_factory=list)


@dataclass
class SummaryBlock:
    block_id: str
    source_turn_ids: list[str]
    start_turn_id: str
    end_turn_id: str
    token_count: int
    compression_level: int
    topic: str
    facts: list[str]
    decisions: list[str]
    open_threads: list[str]
    user_preferences: list[str]
    artifacts: list[str]
    tool_results: list[str]
    created_at: str
    child_block_ids: list[str] = field(default_factory=list)
    child_compression_levels: list[int] = field(default_factory=list)
    derived_from: Literal["raw", "summary_merge"] = "raw"
    generation_count: int = 1


@dataclass
class WorkingMemoryState:
    hot_turns: list[Turn] = field(default_factory=list)
    warm_summaries: list[SummaryBlock] = field(default_factory=list)
    cold_state_blocks: list[SummaryBlock] = field(default_factory=list)
    pinned_turn_ids: set[str] = field(default_factory=set)
    last_compaction_at: str | None = None
    last_compaction_action: str | None = None
    emergency_trim_count: int = 0
    last_token_usage: dict[str, int] = field(default_factory=dict)
    thresholds: dict[str, int] = field(default_factory=dict)
    semantic_ingest_hashes: list[str] = field(default_factory=list)
    last_compaction_debug: dict[str, Any] = field(default_factory=dict)


def _dedupe_items(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        text = " ".join(str(raw or "").strip().split())
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(text)
    return ordered


def _truncate_line(text: str, limit: int = 160) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _summary_payload(block: SummaryBlock) -> dict[str, Any]:
    return {
        "range": {
            "start_turn_id": block.start_turn_id,
            "end_turn_id": block.end_turn_id,
        },
        "topic": block.topic,
        "facts": list(block.facts),
        "decisions": list(block.decisions),
        "open_threads": list(block.open_threads),
        "user_preferences": list(block.user_preferences),
        "artifacts": list(block.artifacts),
        "tool_results": list(block.tool_results),
        "compression_level": int(block.compression_level),
    }


def _summary_text(block: SummaryBlock) -> str:
    return json.dumps(_summary_payload(block), ensure_ascii=True, sort_keys=True)


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _extract_references(text: str) -> list[str]:
    references: list[str] = []
    for match in _PATH_RE.finditer(str(text or "")):
        candidate = next((group for group in match.groups() if group), match.group(0))
        cleaned = str(candidate or "").strip("`")
        if cleaned:
            references.append(cleaned)
    return _dedupe_items(references)


def _normalize_turn(raw: Any, *, fallback_id: str) -> Turn:
    source = dict(raw or {}) if isinstance(raw, dict) else {}
    role = str(source.get("role") or "assistant").strip().lower()
    if role not in {"system", "user", "assistant", "tool"}:
        role = "assistant"
    text = str(source.get("text") or "").strip()
    created_at = str(source.get("created_at") or "").strip() or _now_iso()
    references = source.get("references") if isinstance(source.get("references"), list) else []
    token_count = int(source.get("token_count") or 0) or estimate_text_tokens(text)
    return Turn(
        turn_id=str(source.get("turn_id") or fallback_id).strip() or fallback_id,
        role=role,  # type: ignore[arg-type]
        text=text,
        token_count=max(0, token_count),
        created_at=created_at,
        pinned=bool(source.get("pinned", False)),
        topic_hint=str(source.get("topic_hint") or "").strip() or None,
        references=_dedupe_items(references or _extract_references(text)),
    )


def _normalize_summary_block(raw: Any, *, fallback_id: str) -> SummaryBlock:
    source = dict(raw or {}) if isinstance(raw, dict) else {}
    facts = _dedupe_items(source.get("facts") if isinstance(source.get("facts"), list) else [])
    decisions = _dedupe_items(source.get("decisions") if isinstance(source.get("decisions"), list) else [])
    open_threads = _dedupe_items(source.get("open_threads") if isinstance(source.get("open_threads"), list) else [])
    user_preferences = _dedupe_items(
        source.get("user_preferences") if isinstance(source.get("user_preferences"), list) else []
    )
    artifacts = _dedupe_items(source.get("artifacts") if isinstance(source.get("artifacts"), list) else [])
    tool_results = _dedupe_items(source.get("tool_results") if isinstance(source.get("tool_results"), list) else [])
    compression_level = min(
        _SUMMARY_LEVEL_LIMIT,
        max(1, int(source.get("compression_level") or 1)),
    )
    derived_from = str(source.get("derived_from") or "").strip().lower()
    if derived_from not in {"raw", "summary_merge"}:
        derived_from = "summary_merge" if source.get("child_block_ids") else "raw"
    block = SummaryBlock(
        block_id=str(source.get("block_id") or fallback_id).strip() or fallback_id,
        source_turn_ids=[
            str(item).strip()
            for item in (source.get("source_turn_ids") if isinstance(source.get("source_turn_ids"), list) else [])
            if str(item).strip()
        ],
        start_turn_id=str(source.get("start_turn_id") or "").strip() or "",
        end_turn_id=str(source.get("end_turn_id") or "").strip() or "",
        token_count=max(0, int(source.get("token_count") or 0)),
        compression_level=compression_level,
        topic=_truncate_line(str(source.get("topic") or "").strip() or "conversation"),
        facts=facts,
        decisions=decisions,
        open_threads=open_threads,
        user_preferences=user_preferences,
        artifacts=artifacts,
        tool_results=tool_results,
        created_at=str(source.get("created_at") or "").strip() or _now_iso(),
        child_block_ids=[
            str(item).strip()
            for item in (source.get("child_block_ids") if isinstance(source.get("child_block_ids"), list) else [])
            if str(item).strip()
        ],
        child_compression_levels=[
            min(_SUMMARY_LEVEL_LIMIT, max(1, int(item)))
            for item in (
                source.get("child_compression_levels")
                if isinstance(source.get("child_compression_levels"), list)
                else []
            )
            if str(item).strip()
        ],
        derived_from=derived_from,  # type: ignore[arg-type]
        generation_count=max(1, int(source.get("generation_count") or compression_level or 1)),
    )
    if block.token_count <= 0:
        block.token_count = estimate_text_tokens(_summary_text(block))
    if not block.source_turn_ids and block.start_turn_id and block.end_turn_id:
        block.source_turn_ids = [block.start_turn_id, block.end_turn_id]
    if not block.start_turn_id and block.source_turn_ids:
        block.start_turn_id = block.source_turn_ids[0]
    if not block.end_turn_id and block.source_turn_ids:
        block.end_turn_id = block.source_turn_ids[-1]
    return block


def normalize_working_memory_state(raw: Any) -> WorkingMemoryState:
    source = dict(raw or {}) if isinstance(raw, dict) else {}
    hot_turns = [
        _normalize_turn(item, fallback_id=f"turn-{index:04d}")
        for index, item in enumerate(source.get("hot_turns") if isinstance(source.get("hot_turns"), list) else [], start=1)
        if isinstance(item, dict)
    ]
    warm_summaries = [
        _normalize_summary_block(item, fallback_id=f"warm-{index:04d}")
        for index, item in enumerate(
            source.get("warm_summaries") if isinstance(source.get("warm_summaries"), list) else [],
            start=1,
        )
        if isinstance(item, dict)
    ]
    cold_state_blocks = [
        _normalize_summary_block(item, fallback_id=f"cold-{index:04d}")
        for index, item in enumerate(
            source.get("cold_state_blocks") if isinstance(source.get("cold_state_blocks"), list) else [],
            start=1,
        )
        if isinstance(item, dict)
    ]
    state = WorkingMemoryState(
        hot_turns=hot_turns,
        warm_summaries=warm_summaries,
        cold_state_blocks=cold_state_blocks,
        pinned_turn_ids={
            str(item).strip()
            for item in (source.get("pinned_turn_ids") if isinstance(source.get("pinned_turn_ids"), list) else [])
            if str(item).strip()
        },
        last_compaction_at=str(source.get("last_compaction_at") or "").strip() or None,
        last_compaction_action=str(source.get("last_compaction_action") or "").strip() or None,
        emergency_trim_count=max(0, int(source.get("emergency_trim_count") or 0)),
        last_token_usage=(
            {
                str(key): int(value)
                for key, value in source.get("last_token_usage", {}).items()
                if str(key).strip()
            }
            if isinstance(source.get("last_token_usage"), dict)
            else {}
        ),
        thresholds=(
            {
                str(key): int(value)
                for key, value in source.get("thresholds", {}).items()
                if str(key).strip()
            }
            if isinstance(source.get("thresholds"), dict)
            else {}
        ),
        semantic_ingest_hashes=[
            str(item).strip()
            for item in (
                source.get("semantic_ingest_hashes")
                if isinstance(source.get("semantic_ingest_hashes"), list)
                else []
            )
            if str(item).strip()
        ][-_SEMANTIC_INGEST_DEDUPE_LIMIT:],
        last_compaction_debug=(
            dict(source.get("last_compaction_debug"))
            if isinstance(source.get("last_compaction_debug"), dict)
            else {}
        ),
    )
    _apply_pin_policy(state)
    _refresh_usage(state, budget=None)
    return state


def working_memory_state_to_dict(state: WorkingMemoryState) -> dict[str, Any]:
    return {
        "hot_turns": [
            {
                "turn_id": turn.turn_id,
                "role": turn.role,
                "text": turn.text,
                "token_count": int(turn.token_count),
                "created_at": turn.created_at,
                "pinned": bool(turn.pinned),
                "topic_hint": turn.topic_hint,
                "references": list(turn.references),
            }
            for turn in state.hot_turns
        ],
        "warm_summaries": [
            {
                "block_id": block.block_id,
                "source_turn_ids": list(block.source_turn_ids),
                "start_turn_id": block.start_turn_id,
                "end_turn_id": block.end_turn_id,
                "token_count": int(block.token_count),
                "compression_level": int(block.compression_level),
                "topic": block.topic,
                "facts": list(block.facts),
                "decisions": list(block.decisions),
                "open_threads": list(block.open_threads),
                "user_preferences": list(block.user_preferences),
                "artifacts": list(block.artifacts),
                "tool_results": list(block.tool_results),
                "created_at": block.created_at,
                "child_block_ids": list(block.child_block_ids),
                "child_compression_levels": list(block.child_compression_levels),
                "derived_from": block.derived_from,
                "generation_count": int(block.generation_count),
            }
            for block in state.warm_summaries
        ],
        "cold_state_blocks": [
            {
                "block_id": block.block_id,
                "source_turn_ids": list(block.source_turn_ids),
                "start_turn_id": block.start_turn_id,
                "end_turn_id": block.end_turn_id,
                "token_count": int(block.token_count),
                "compression_level": int(block.compression_level),
                "topic": block.topic,
                "facts": list(block.facts),
                "decisions": list(block.decisions),
                "open_threads": list(block.open_threads),
                "user_preferences": list(block.user_preferences),
                "artifacts": list(block.artifacts),
                "tool_results": list(block.tool_results),
                "created_at": block.created_at,
                "child_block_ids": list(block.child_block_ids),
                "child_compression_levels": list(block.child_compression_levels),
                "derived_from": block.derived_from,
                "generation_count": int(block.generation_count),
            }
            for block in state.cold_state_blocks
        ],
        "pinned_turn_ids": sorted(state.pinned_turn_ids),
        "last_compaction_at": state.last_compaction_at,
        "last_compaction_action": state.last_compaction_action,
        "emergency_trim_count": int(state.emergency_trim_count),
        "last_token_usage": dict(state.last_token_usage),
        "thresholds": dict(state.thresholds),
        "semantic_ingest_hashes": list(state.semantic_ingest_hashes)[-_SEMANTIC_INGEST_DEDUPE_LIMIT:],
        "last_compaction_debug": dict(state.last_compaction_debug),
    }


def _refresh_usage(state: WorkingMemoryState, budget: ContextBudget | None) -> dict[str, int]:
    usage = {
        "hot_tokens": sum(int(turn.token_count) for turn in state.hot_turns),
        "warm_tokens": sum(int(block.token_count) for block in state.warm_summaries),
        "cold_tokens": sum(int(block.token_count) for block in state.cold_state_blocks),
    }
    usage["total_tokens"] = usage["hot_tokens"] + usage["warm_tokens"] + usage["cold_tokens"]
    state.last_token_usage = usage
    if budget is not None:
        state.thresholds = {
            "working_memory_budget": int(budget.working_memory_budget),
            "soft_threshold": int(budget.soft_threshold),
            "hard_threshold": int(budget.hard_threshold),
            "panic_threshold": int(budget.panic_threshold),
        }
    return usage


def _turn_message_text(turn: Turn) -> str:
    if turn.role == "tool":
        return f"{_TOOL_PREFIX}\n{turn.text}".strip()
    return turn.text


def append_turn(
    state: WorkingMemoryState,
    *,
    role: Role,
    text: str,
    turn_id: str | None = None,
    created_at: str | None = None,
    pinned: bool = False,
    topic_hint: str | None = None,
    references: list[str] | None = None,
) -> WorkingMemoryState:
    cleaned = str(text or "").strip()
    if not cleaned:
        return state
    if state.hot_turns:
        last = state.hot_turns[-1]
        if last.role == role and last.text == cleaned:
            return state
    computed_references = _dedupe_items((references or []) + _extract_references(cleaned))
    generated_id = turn_id or f"turn-{_stable_digest(f'{role}:{created_at or _now_iso()}:{cleaned[:120]}')}"
    turn = Turn(
        turn_id=str(generated_id).strip() or f"turn-{_stable_digest(cleaned)}",
        role=role,
        text=cleaned,
        token_count=estimate_text_tokens(cleaned),
        created_at=str(created_at or _now_iso()).strip() or _now_iso(),
        pinned=bool(pinned),
        topic_hint=str(topic_hint or "").strip() or None,
        references=computed_references,
    )
    state.hot_turns.append(turn)
    if turn.pinned:
        state.pinned_turn_ids.add(turn.turn_id)
    _apply_pin_policy(state)
    _refresh_usage(state, budget=None)
    return state


def rebuild_state_from_messages(
    messages: list[dict[str, str]],
    *,
    previous_state: WorkingMemoryState | None = None,
) -> WorkingMemoryState:
    state = WorkingMemoryState()
    if previous_state is not None:
        state.pinned_turn_ids = set(previous_state.pinned_turn_ids)
        state.emergency_trim_count = int(previous_state.emergency_trim_count)
        state.last_compaction_at = previous_state.last_compaction_at
        state.last_compaction_action = previous_state.last_compaction_action
        state.thresholds = dict(previous_state.thresholds)
        state.semantic_ingest_hashes = list(previous_state.semantic_ingest_hashes)
        state.last_compaction_debug = dict(previous_state.last_compaction_debug)
    for index, row in enumerate(messages, start=1):
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "user").strip().lower()
        if role not in {"user", "assistant", "tool"}:
            continue
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        append_turn(
            state,
            role=role,  # type: ignore[arg-type]
            text=text,
            turn_id=f"turn-{index:04d}-{_stable_digest(f'{role}:{text}')}",
        )
    _apply_pin_policy(state)
    _refresh_usage(state, budget=None)
    return state


def _apply_pin_policy(state: WorkingMemoryState) -> None:
    explicit_pins = {
        turn_id
        for turn_id in state.pinned_turn_ids
        if any(turn.turn_id == turn_id for turn in state.hot_turns)
    }
    for turn in state.hot_turns:
        turn.pinned = turn.turn_id in explicit_pins
    state.pinned_turn_ids = explicit_pins


def _protected_tail_index(turns: list[Turn]) -> int:
    user_indices = [index for index, turn in enumerate(turns) if turn.role == "user"]
    if not user_indices:
        return 0
    if len(user_indices) <= _RECENCY_SHIELDED_USER_TURNS:
        return user_indices[0]
    return user_indices[-_RECENCY_SHIELDED_USER_TURNS]


def _recency_shield_turn_ids(turns: list[Turn]) -> set[str]:
    protected_from = _protected_tail_index(turns)
    if protected_from <= 0:
        return set()
    return {
        turn.turn_id
        for turn in turns[protected_from:]
        if str(turn.turn_id or "").strip()
    }


def _state_counts(state: WorkingMemoryState) -> dict[str, int]:
    return {
        "hot_turn_count": len(state.hot_turns),
        "warm_summary_count": len(state.warm_summaries),
        "cold_block_count": len(state.cold_state_blocks),
    }


def _usage_snapshot(state: WorkingMemoryState, budget: ContextBudget | None) -> dict[str, int]:
    return dict(_refresh_usage(state, budget))


def _summary_caps_for_level(level: int) -> dict[str, int]:
    normalized = max(1, min(_SUMMARY_LEVEL_LIMIT, int(level or 1)))
    if normalized >= 3:
        return {
            "facts": 5,
            "decisions": 4,
            "open_threads": 4,
            "user_preferences": 4,
            "artifacts": 5,
            "tool_results": 4,
        }
    if normalized == 2:
        return {
            "facts": 6,
            "decisions": 5,
            "open_threads": 5,
            "user_preferences": 5,
            "artifacts": 6,
            "tool_results": 5,
        }
    return {
        "facts": 8,
        "decisions": 6,
        "open_threads": 6,
        "user_preferences": 6,
        "artifacts": 8,
        "tool_results": 6,
    }


def _apply_summary_caps(fields: dict[str, list[str] | str], *, compression_level: int) -> dict[str, list[str] | str]:
    caps = _summary_caps_for_level(compression_level)
    normalized = dict(fields)
    for key, limit in caps.items():
        values = normalized.get(key)
        if isinstance(values, list):
            normalized[key] = _dedupe_items(values)[:limit]
    return normalized


def _set_compaction_debug(
    state: WorkingMemoryState,
    *,
    action: str,
    reason: str,
    threshold_crossed: str | None = None,
    affected_turn_ids: list[str] | None = None,
    affected_block_ids: list[str] | None = None,
    affected_range: dict[str, str] | None = None,
    counts_before: dict[str, int] | None = None,
    counts_after: dict[str, int] | None = None,
    tokens_before: dict[str, int] | None = None,
    tokens_after: dict[str, int] | None = None,
    semantic_extracted: bool = False,
    semantic_dedupe_skipped: bool = False,
    panic_trim: bool = False,
    regenerated_from_raw: bool = False,
) -> None:
    state.last_compaction_at = _now_iso()
    state.last_compaction_action = action
    state.last_compaction_debug = {
        "last_action": action,
        "reason": reason,
        "threshold_crossed": threshold_crossed,
        "affected_turn_ids": list(affected_turn_ids or [])[:12],
        "affected_block_ids": list(affected_block_ids or [])[:8],
        "affected_range": dict(affected_range or {}) if isinstance(affected_range, dict) else None,
        "counts_before": dict(counts_before or {}),
        "counts_after": dict(counts_after or {}),
        "tokens_before": dict(tokens_before or {}),
        "tokens_after": dict(tokens_after or {}),
        "semantic_extracted": bool(semantic_extracted),
        "semantic_dedupe_skipped": bool(semantic_dedupe_skipped),
        "panic_trim": bool(panic_trim),
        "regenerated_from_raw": bool(regenerated_from_raw),
    }


def select_chunk_for_summarization(turns: list[Turn]) -> list[Turn]:
    if not turns:
        return []
    protected_from = _protected_tail_index(turns)
    if protected_from <= 0:
        return []
    start = 0
    while start < protected_from:
        if turns[start].pinned:
            start += 1
            continue
        chunk: list[Turn] = []
        token_total = 0
        index = start
        while index < protected_from:
            turn = turns[index]
            if turn.pinned:
                break
            chunk.append(turn)
            token_total += int(turn.token_count)
            index += 1
            if len(chunk) >= _MAX_RAW_CHUNK_TURNS or token_total >= _MAX_RAW_CHUNK_TOKENS:
                break
        if chunk and (
            len(chunk) >= _MIN_RAW_CHUNK_TURNS
            or token_total >= _MIN_RAW_CHUNK_TOKENS
            or len(chunk) >= max(2, min(_TARGET_RAW_CHUNK_TURNS, protected_from))
        ):
            return chunk
        start = max(index, start + 1)
    return []


def _chunk_topic(turns: list[Turn]) -> str:
    for turn in turns:
        if turn.topic_hint:
            return _truncate_line(turn.topic_hint, 80)
        if turn.role == "user" and turn.text:
            return _truncate_line(turn.text.splitlines()[0], 80)
    if turns:
        return _truncate_line(turns[0].text.splitlines()[0], 80)
    return "conversation"


def _extract_summary_fields(turns: list[Turn]) -> dict[str, list[str] | str]:
    facts: list[str] = []
    decisions: list[str] = []
    open_threads: list[str] = []
    user_preferences: list[str] = []
    artifacts: list[str] = []
    tool_results: list[str] = []
    for turn in turns:
        text = " ".join(str(turn.text or "").split()).strip()
        if not text:
            continue
        first_line = _truncate_line(text, 180)
        artifacts.extend(turn.references or _extract_references(text))
        if turn.role == "tool":
            tool_results.append(first_line)
        if turn.role == "user" and _PREFERENCE_RE.search(text):
            user_preferences.append(first_line)
        if _DECISION_RE.search(text):
            decisions.append(first_line)
        if turn.role == "user" and ("?" in text or _OPEN_THREAD_RE.search(text)):
            open_threads.append(first_line)
        if turn.role in {"user", "assistant"}:
            facts.append(first_line)
    return {
        "topic": _chunk_topic(turns),
        "facts": _dedupe_items(facts)[:8],
        "decisions": _dedupe_items(decisions)[:6],
        "open_threads": _dedupe_items(open_threads)[:6],
        "user_preferences": _dedupe_items(user_preferences)[:6],
        "artifacts": _dedupe_items(artifacts)[:8],
        "tool_results": _dedupe_items(tool_results)[:6],
    }


def summarize_turn_chunk(turns: list[Turn], *, compression_level: int = 1) -> SummaryBlock:
    if not turns:
        raise ValueError("turn chunk is required")
    normalized_level = max(1, min(_SUMMARY_LEVEL_LIMIT, int(compression_level)))
    fields = _apply_summary_caps(_extract_summary_fields(turns), compression_level=normalized_level)
    created_at = _now_iso()
    source_turn_ids = [turn.turn_id for turn in turns]
    block = SummaryBlock(
        block_id=f"summary-{normalized_level}-{_stable_digest('|'.join(source_turn_ids))}",
        source_turn_ids=source_turn_ids,
        start_turn_id=source_turn_ids[0],
        end_turn_id=source_turn_ids[-1],
        token_count=0,
        compression_level=normalized_level,
        topic=str(fields.get("topic") or "conversation"),
        facts=list(fields.get("facts") or []),
        decisions=list(fields.get("decisions") or []),
        open_threads=list(fields.get("open_threads") or []),
        user_preferences=list(fields.get("user_preferences") or []),
        artifacts=list(fields.get("artifacts") or []),
        tool_results=list(fields.get("tool_results") or []),
        created_at=created_at,
        derived_from="raw",
        generation_count=1,
    )
    block.token_count = estimate_text_tokens(_summary_text(block))
    return block


def _merge_two_blocks(
    left: SummaryBlock,
    right: SummaryBlock,
    *,
    to_level: int,
    raw_turn_index: dict[str, Turn] | None = None,
) -> tuple[SummaryBlock, bool]:
    normalized_level = min(_SUMMARY_LEVEL_LIMIT, max(1, int(to_level)))
    combined_turn_ids = _dedupe_items(list(left.source_turn_ids) + list(right.source_turn_ids))
    if raw_turn_index:
        raw_turns = [raw_turn_index[turn_id] for turn_id in combined_turn_ids if turn_id in raw_turn_index]
        if len(raw_turns) == len(combined_turn_ids) and raw_turns:
            regenerated = summarize_turn_chunk(raw_turns, compression_level=normalized_level)
            regenerated.block_id = f"summary-{normalized_level}-{_stable_digest('|'.join(combined_turn_ids))}"
            regenerated.source_turn_ids = combined_turn_ids
            regenerated.start_turn_id = combined_turn_ids[0]
            regenerated.end_turn_id = combined_turn_ids[-1]
            regenerated.token_count = estimate_text_tokens(_summary_text(regenerated))
            return regenerated, True
    merged = SummaryBlock(
        block_id=f"summary-{normalized_level}-{_stable_digest(left.block_id + ':' + right.block_id)}",
        source_turn_ids=combined_turn_ids,
        start_turn_id=left.start_turn_id,
        end_turn_id=right.end_turn_id,
        token_count=0,
        compression_level=normalized_level,
        topic=_truncate_line(left.topic if left.topic == right.topic else f"{left.topic}; {right.topic}", 80),
        facts=[],
        decisions=[],
        open_threads=[],
        user_preferences=[],
        artifacts=[],
        tool_results=[],
        created_at=_now_iso(),
        child_block_ids=_dedupe_items(
            list(left.child_block_ids)
            + [left.block_id]
            + list(right.child_block_ids)
            + [right.block_id]
        ),
        child_compression_levels=[int(left.compression_level), int(right.compression_level)],
        derived_from="summary_merge",
        generation_count=max(int(left.generation_count), int(right.generation_count)) + 1,
    )
    merged_fields = _apply_summary_caps(
        {
            "topic": merged.topic,
            "facts": _dedupe_items(list(left.facts) + list(right.facts)),
            "decisions": _dedupe_items(list(left.decisions) + list(right.decisions)),
            "open_threads": _dedupe_items(list(left.open_threads) + list(right.open_threads)),
            "user_preferences": _dedupe_items(list(left.user_preferences) + list(right.user_preferences)),
            "artifacts": _dedupe_items(list(left.artifacts) + list(right.artifacts)),
            "tool_results": _dedupe_items(list(left.tool_results) + list(right.tool_results)),
        },
        compression_level=normalized_level,
    )
    merged.topic = str(merged_fields.get("topic") or merged.topic)
    merged.facts = list(merged_fields.get("facts") or [])
    merged.decisions = list(merged_fields.get("decisions") or [])
    merged.open_threads = list(merged_fields.get("open_threads") or [])
    merged.user_preferences = list(merged_fields.get("user_preferences") or [])
    merged.artifacts = list(merged_fields.get("artifacts") or [])
    merged.tool_results = list(merged_fields.get("tool_results") or [])
    merged.token_count = estimate_text_tokens(_summary_text(merged))
    return merged, False


def _ingest_durable_memory(
    *,
    state: WorkingMemoryState,
    durable_ingestor: Callable[[dict[str, Any]], None] | None,
    user_id: str,
    thread_id: str | None,
    summary: SummaryBlock,
    source_turns: list[Turn],
) -> dict[str, Any]:
    result = {
        "semantic_extracted": False,
        "semantic_dedupe_skipped": False,
        "dedupe_hash": None,
    }
    if durable_ingestor is None:
        return result
    raw_text = "\n".join(
        _truncate_line(f"{turn.role}: {_turn_message_text(turn)}", 300)
        for turn in source_turns
        if str(_turn_message_text(turn) or "").strip()
    ).strip()
    payload = {
        "facts": list(summary.facts),
        "decisions": list(summary.decisions),
        "open_threads": list(summary.open_threads),
        "user_preferences": list(summary.user_preferences),
        "artifacts": list(summary.artifacts),
        "tool_results": list(summary.tool_results),
    }
    if not any(payload.values()):
        return result
    dedupe_hash = _stable_hash(
        json.dumps(
            {
                "source_ref": f"working-memory:{summary.start_turn_id}:{summary.end_turn_id}",
                "payload": payload,
                "raw_text": " ".join(raw_text.split()).strip().lower(),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    result["dedupe_hash"] = dedupe_hash
    if dedupe_hash in state.semantic_ingest_hashes:
        result["semantic_dedupe_skipped"] = True
        return result
    durable_ingestor(
        {
            "user_id": str(user_id or "").strip() or "unknown",
            "thread_id": str(thread_id or "").strip() or None,
            "source_ref": f"working-memory:{summary.start_turn_id}:{summary.end_turn_id}",
            "summary": summary,
            "payload": payload,
            "text": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            "raw_text": raw_text,
        }
    )
    state.semantic_ingest_hashes = [
        *list(state.semantic_ingest_hashes)[-(max(0, _SEMANTIC_INGEST_DEDUPE_LIMIT - 1)) :],
        dedupe_hash,
    ]
    result["semantic_extracted"] = True
    return result


def summarize_oldest_unpinned_raw_chunk(
    state: WorkingMemoryState,
    *,
    user_id: str,
    thread_id: str | None,
    durable_ingestor: Callable[[dict[str, Any]], None] | None = None,
    reason: str | None = None,
    threshold_crossed: str | None = None,
) -> WorkingMemoryState:
    before_counts = _state_counts(state)
    before_tokens = _usage_snapshot(state, budget=None)
    chunk = select_chunk_for_summarization(state.hot_turns)
    if not chunk:
        return state
    summary = summarize_turn_chunk(chunk, compression_level=1)
    ingest_result = _ingest_durable_memory(
        state=state,
        durable_ingestor=durable_ingestor,
        user_id=user_id,
        thread_id=thread_id,
        summary=summary,
        source_turns=chunk,
    )
    chunk_ids = {turn.turn_id for turn in chunk}
    state.hot_turns = [turn for turn in state.hot_turns if turn.turn_id not in chunk_ids]
    state.warm_summaries.append(summary)
    _apply_pin_policy(state)
    after_counts = _state_counts(state)
    after_tokens = _usage_snapshot(state, budget=None)
    _set_compaction_debug(
        state,
        action="summarized_raw_chunk",
        reason=str(reason or "working_memory_compaction"),
        threshold_crossed=threshold_crossed,
        affected_turn_ids=list(summary.source_turn_ids),
        affected_block_ids=[summary.block_id],
        affected_range={
            "start_turn_id": summary.start_turn_id,
            "end_turn_id": summary.end_turn_id,
        },
        counts_before=before_counts,
        counts_after=after_counts,
        tokens_before=before_tokens,
        tokens_after=after_tokens,
        semantic_extracted=bool(ingest_result.get("semantic_extracted")),
        semantic_dedupe_skipped=bool(ingest_result.get("semantic_dedupe_skipped")),
    )
    return state


def merge_summaries(
    state: WorkingMemoryState,
    *,
    from_level: int,
    to_level: int,
    reason: str | None = None,
    threshold_crossed: str | None = None,
) -> WorkingMemoryState:
    indexed_blocks = [
        (index, block)
        for index, block in enumerate(state.warm_summaries)
        if int(block.compression_level) == int(from_level)
    ]
    if len(indexed_blocks) < 2:
        return state
    before_counts = _state_counts(state)
    before_tokens = _usage_snapshot(state, budget=None)
    left_index, left = indexed_blocks[0]
    right_index, right = indexed_blocks[1]
    raw_turn_index = {turn.turn_id: turn for turn in state.hot_turns}
    merged, regenerated_from_raw = _merge_two_blocks(left, right, to_level=to_level, raw_turn_index=raw_turn_index)
    remaining = [
        block
        for block in state.warm_summaries
        if block.block_id not in {left.block_id, right.block_id}
    ]
    if to_level >= _SUMMARY_LEVEL_LIMIT:
        state.cold_state_blocks.append(merged)
    else:
        insert_at = max(0, min(left_index, right_index))
        remaining.insert(insert_at, merged)
    state.warm_summaries = remaining
    after_counts = _state_counts(state)
    after_tokens = _usage_snapshot(state, budget=None)
    _set_compaction_debug(
        state,
        action=f"merged_summaries_l{from_level}_to_l{min(_SUMMARY_LEVEL_LIMIT, max(1, int(to_level)))}",
        reason=str(reason or "working_memory_compaction"),
        threshold_crossed=threshold_crossed,
        affected_block_ids=[left.block_id, right.block_id, merged.block_id],
        affected_range={
            "start_turn_id": merged.start_turn_id,
            "end_turn_id": merged.end_turn_id,
        },
        counts_before=before_counts,
        counts_after=after_counts,
        tokens_before=before_tokens,
        tokens_after=after_tokens,
        regenerated_from_raw=regenerated_from_raw,
    )
    return state


def _drop_low_value_tool_turns(state: WorkingMemoryState) -> int:
    protected_from = _protected_tail_index(state.hot_turns)
    removed = 0
    kept: list[Turn] = []
    for index, turn in enumerate(state.hot_turns):
        if (
            index < protected_from
            and not turn.pinned
            and turn.role == "tool"
            and removed < 4
        ):
            removed += 1
            continue
        kept.append(turn)
    if removed:
        state.hot_turns = kept
    return removed


def enforce_hard_compaction(
    state: WorkingMemoryState,
    *,
    user_id: str,
    thread_id: str | None,
    budget: ContextBudget,
    durable_ingestor: Callable[[dict[str, Any]], None] | None = None,
) -> WorkingMemoryState:
    progress = True
    while progress and _refresh_usage(state, budget)["total_tokens"] > budget.hard_threshold:
        progress = False
        before = _refresh_usage(state, budget)["total_tokens"]
        summarize_oldest_unpinned_raw_chunk(
            state,
            user_id=user_id,
            thread_id=thread_id,
            durable_ingestor=durable_ingestor,
            reason="hard_threshold_exceeded",
            threshold_crossed="hard",
        )
        after_summarize = _refresh_usage(state, budget)["total_tokens"]
        if after_summarize < before:
            progress = True
            continue
        merge_summaries(state, from_level=1, to_level=2, reason="hard_threshold_exceeded", threshold_crossed="hard")
        after_merge_l1 = _refresh_usage(state, budget)["total_tokens"]
        if after_merge_l1 < before:
            progress = True
            continue
        merge_summaries(state, from_level=2, to_level=3, reason="hard_threshold_exceeded", threshold_crossed="hard")
        after_merge_l2 = _refresh_usage(state, budget)["total_tokens"]
        if after_merge_l2 < before:
            progress = True
            continue
        trim_counts_before = _state_counts(state)
        trim_tokens_before = _usage_snapshot(state, budget)
        removed = _drop_low_value_tool_turns(state)
        if removed:
            trim_counts_after = _state_counts(state)
            trim_tokens_after = _usage_snapshot(state, budget)
            _set_compaction_debug(
                state,
                action="trimmed_tool_chatter",
                reason="hard_threshold_exceeded",
                threshold_crossed="hard",
                counts_before=trim_counts_before,
                counts_after=trim_counts_after,
                tokens_before=trim_tokens_before,
                tokens_after=trim_tokens_after,
            )
            progress = True
    return state


def _turn_fingerprint(turn: Turn) -> str:
    return " ".join(str(turn.text or "").strip().lower().split())[:180]


def _is_low_value_tool_turn(turn: Turn, *, duplicate_fingerprints: set[str]) -> bool:
    if turn.role != "tool":
        return False
    normalized = _turn_fingerprint(turn)
    if normalized in duplicate_fingerprints:
        return True
    if int(turn.token_count) >= 600:
        return True
    return any(marker in normalized for marker in ("log", "trace", "stdout", "stderr", "debug", "info", "warning"))


def _panic_trim_priority(turn: Turn, *, duplicate_fingerprints: set[str]) -> tuple[int, int]:
    if turn.role == "tool" and _is_low_value_tool_turn(turn, duplicate_fingerprints=duplicate_fingerprints):
        return (0, -int(turn.token_count))
    if turn.role == "tool":
        return (1, -int(turn.token_count))
    if turn.role == "assistant":
        return (2, -int(turn.token_count))
    return (99, -int(turn.token_count))


def emergency_trim(state: WorkingMemoryState, *, budget: ContextBudget) -> WorkingMemoryState:
    before_counts = _state_counts(state)
    before_tokens = _usage_snapshot(state, budget)
    protected_from = _protected_tail_index(state.hot_turns)
    normalized_counts: dict[str, int] = {}
    for turn in state.hot_turns[:protected_from]:
        key = _turn_fingerprint(turn)
        normalized_counts[key] = normalized_counts.get(key, 0) + 1
    duplicate_fingerprints = {key for key, count in normalized_counts.items() if count > 1}
    candidates: list[tuple[int, Turn]] = [
        (index, turn)
        for index, turn in enumerate(state.hot_turns[:protected_from])
        if not turn.pinned and turn.role in {"tool", "assistant"}
    ]
    current_total = int(before_tokens.get("total_tokens") or 0)
    remove_ids: list[str] = []
    for _index, turn in sorted(
        candidates,
        key=lambda row: (_panic_trim_priority(row[1], duplicate_fingerprints=duplicate_fingerprints), row[0]),
    ):
        if current_total <= budget.panic_threshold:
            break
        remove_ids.append(turn.turn_id)
        current_total -= int(turn.token_count)
    if remove_ids:
        remove_set = set(remove_ids)
        state.hot_turns = [turn for turn in state.hot_turns if turn.turn_id not in remove_set]
        state.emergency_trim_count += 1
        after_counts = _state_counts(state)
        after_tokens = _usage_snapshot(state, budget)
        _set_compaction_debug(
            state,
            action="emergency_trim",
            reason="panic_threshold_exceeded",
            threshold_crossed="panic",
            affected_turn_ids=remove_ids,
            counts_before=before_counts,
            counts_after=after_counts,
            tokens_before=before_tokens,
            tokens_after=after_tokens,
            panic_trim=True,
        )
    return state


def manage_working_memory(
    state: WorkingMemoryState,
    *,
    budget: ContextBudget,
    user_id: str,
    thread_id: str | None,
    durable_ingestor: Callable[[dict[str, Any]], None] | None = None,
) -> WorkingMemoryState:
    _apply_pin_policy(state)
    used = _refresh_usage(state, budget)["total_tokens"]
    if used < budget.soft_threshold:
        return state
    while _refresh_usage(state, budget)["total_tokens"] > budget.soft_threshold:
        before = _refresh_usage(state, budget)["total_tokens"]
        summarize_oldest_unpinned_raw_chunk(
            state,
            user_id=user_id,
            thread_id=thread_id,
            durable_ingestor=durable_ingestor,
            reason="soft_threshold_exceeded",
            threshold_crossed="soft",
        )
        after_summarize = _refresh_usage(state, budget)["total_tokens"]
        if after_summarize < before:
            continue
        merge_summaries(state, from_level=1, to_level=2, reason="soft_threshold_exceeded", threshold_crossed="soft")
        after_merge_l1 = _refresh_usage(state, budget)["total_tokens"]
        if after_merge_l1 < before:
            continue
        merge_summaries(state, from_level=2, to_level=3, reason="soft_threshold_exceeded", threshold_crossed="soft")
        after_merge_l2 = _refresh_usage(state, budget)["total_tokens"]
        if after_merge_l2 < before:
            continue
        break
    if _refresh_usage(state, budget)["total_tokens"] > budget.hard_threshold:
        enforce_hard_compaction(
            state,
            user_id=user_id,
            thread_id=thread_id,
            budget=budget,
            durable_ingestor=durable_ingestor,
        )
    if _refresh_usage(state, budget)["total_tokens"] > budget.panic_threshold:
        emergency_trim(state, budget=budget)
    _refresh_usage(state, budget)
    return state


def _relevance_score(block: SummaryBlock, query_text: str) -> int:
    query_tokens = {
        token.lower()
        for token in _TOKEN_RE.findall(str(query_text or "").lower())
        if len(token) > 2
    }
    if not query_tokens:
        return 0
    corpus = " ".join(
        [
            block.topic,
            *block.facts,
            *block.decisions,
            *block.open_threads,
            *block.user_preferences,
            *block.artifacts,
            *block.tool_results,
        ]
    ).lower()
    return sum(1 for token in query_tokens if token in corpus)


def build_working_memory_context_text(
    state: WorkingMemoryState,
    *,
    current_query: str,
    extra_context_text: str | None = None,
) -> str:
    lines: list[str] = []
    cold_blocks = sorted(
        state.cold_state_blocks,
        key=lambda block: (_relevance_score(block, current_query), block.created_at),
        reverse=True,
    )
    selected_cold = [block for block in cold_blocks if _relevance_score(block, current_query) > 0][:2]
    if not selected_cold:
        selected_cold = [
            block
            for block in state.cold_state_blocks
            if block.open_threads or block.user_preferences
        ][:1]
    if selected_cold:
        cold_lines = "\n".join(
            _summary_text(block)
            for block in sorted(selected_cold, key=lambda block: str(block.created_at or ""))
        )
        lines.append(f"Relevant cold state blocks:\n{cold_lines}")
    if state.warm_summaries:
        warm_lines = "\n".join(_summary_text(block) for block in state.warm_summaries)
        lines.append(f"Working memory summaries:\n{warm_lines}")
    if str(extra_context_text or "").strip():
        lines.append(f"Retrieved long-term memory:\n{str(extra_context_text or '').strip()}")
    return "\n\n".join(part for part in lines if part).strip()


def build_hot_messages(state: WorkingMemoryState) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in state.hot_turns:
        role = "assistant" if turn.role == "tool" else turn.role
        if role not in {"user", "assistant", "system"}:
            role = "assistant"
        content = _turn_message_text(turn)
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def build_working_memory_summary(state: WorkingMemoryState) -> dict[str, Any]:
    usage = _refresh_usage(state, budget=None)
    all_summaries = [*state.warm_summaries, *state.cold_state_blocks]
    summary_level_counts = {
        str(level): len([block for block in all_summaries if int(block.compression_level) == level])
        for level in range(1, _SUMMARY_LEVEL_LIMIT + 1)
    }
    derived_counts = {
        "raw": len([block for block in all_summaries if block.derived_from == "raw"]),
        "summary_merge": len([block for block in all_summaries if block.derived_from == "summary_merge"]),
    }
    shielded_turn_ids = _recency_shield_turn_ids(state.hot_turns)
    return {
        "hot_turn_count": len(state.hot_turns),
        "warm_summary_count": len(state.warm_summaries),
        "cold_block_count": len(state.cold_state_blocks),
        "pinned_turn_count": len(state.pinned_turn_ids),
        "hot_tokens": int(usage["hot_tokens"]),
        "warm_tokens": int(usage["warm_tokens"]),
        "cold_tokens": int(usage["cold_tokens"]),
        "total_tokens": int(usage["total_tokens"]),
        "soft_threshold": int(state.thresholds.get("soft_threshold") or 0),
        "hard_threshold": int(state.thresholds.get("hard_threshold") or 0),
        "panic_threshold": int(state.thresholds.get("panic_threshold") or 0),
        "last_compaction_at": state.last_compaction_at,
        "last_compaction_action": state.last_compaction_action,
        "emergency_trim_count": int(state.emergency_trim_count),
        "recency_shield_user_turns": int(_RECENCY_SHIELDED_USER_TURNS),
        "recency_shielded_turn_count": len(shielded_turn_ids),
        "summary_level_counts": summary_level_counts,
        "summary_provenance_counts": derived_counts,
        "max_generation_count": max((int(block.generation_count) for block in all_summaries), default=0),
        "semantic_dedupe_marker_count": len(state.semantic_ingest_hashes),
        "debug": dict(state.last_compaction_debug) if state.last_compaction_debug else None,
    }


def default_budget(max_context_tokens: int | None = None) -> ContextBudget:
    limit = int(max_context_tokens or _DEFAULT_MAX_CONTEXT_TOKENS)
    if limit < 4096:
        limit = 4096
    return ContextBudget(max_context_tokens=limit)


__all__ = [
    "ContextBudget",
    "Role",
    "SummaryBlock",
    "Turn",
    "WorkingMemoryState",
    "append_turn",
    "build_hot_messages",
    "build_working_memory_context_text",
    "build_working_memory_summary",
    "default_budget",
    "estimate_text_tokens",
    "manage_working_memory",
    "normalize_working_memory_state",
    "rebuild_state_from_messages",
    "select_chunk_for_summarization",
    "summarize_turn_chunk",
    "working_memory_state_to_dict",
]
