from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from agent.modelops.discovery import ModelInfo


@dataclass(frozen=True)
class ModelRecommendation:
    provider: str
    model_id: str
    purpose: str
    score: float
    reason: str
    tradeoffs: list[str]
    why_better_than_current: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_purpose_list(purposes: list[str]) -> list[str]:
    allowed = {"chat", "code", "organize", "story"}
    output = sorted({str(item or "").strip().lower() for item in purposes if str(item or "").strip()})
    normalized = [item for item in output if item in allowed]
    return normalized or ["chat", "code", "organize", "story"]


def _score_model(
    *,
    model: ModelInfo,
    purpose: str,
    current: dict[str, str],
    prefer_local: bool,
) -> ModelRecommendation:
    tags = {str(item).strip().lower() for item in model.tags if str(item).strip()}
    lowered = str(model.model_id or "").lower()
    current_provider = str(current.get("provider") or "").strip().lower()
    current_model = str(current.get("model_id") or "").strip()
    score = 0.35
    reason_parts: list[str] = []
    tradeoffs: list[str] = []

    if prefer_local and model.provider == "ollama":
        score += 0.20
        reason_parts.append("local_preferred")
    elif prefer_local and model.provider != "ollama":
        tradeoffs.append("remote_provider")

    if purpose == "code":
        if "code" in tags or any(token in lowered for token in ("code", "coder", "deepseek")):
            score += 0.25
            reason_parts.append("code_fit")
        else:
            tradeoffs.append("weak_code_signal")
    elif purpose in {"chat", "story"}:
        if "chat" in tags or "instruct" in lowered:
            score += 0.12
            reason_parts.append("chat_fit")
        if purpose == "story" and ("story" in tags or "creative" in lowered):
            score += 0.10
            reason_parts.append("story_fit")
    elif purpose == "organize":
        if "chat" in tags or "instruct" in lowered:
            score += 0.10
            reason_parts.append("instruction_fit")

    context_tokens = model.context_tokens
    if context_tokens is not None and context_tokens >= 32768:
        score += 0.12
        reason_parts.append("long_context")
    elif context_tokens is not None and context_tokens >= 8192:
        score += 0.06
        reason_parts.append("context_ok")
    else:
        tradeoffs.append("unknown_context")
        score -= 0.04

    if not tags or tags == {"general"}:
        score -= 0.03
        tradeoffs.append("limited_metadata")

    score = round(_clamp01(score), 2)
    reason = "+".join(reason_parts[:3]) if reason_parts else "baseline"

    why_better: list[str] = []
    if model.provider != current_provider:
        if prefer_local and model.provider == "ollama":
            why_better.append("Matches your local-first preference.")
    if current_model and f"{model.provider}:{model.model_id}" != current_model and score >= 0.70:
        why_better.append(f"Higher fit score ({score:.2f}) for {purpose} tasks.")

    return ModelRecommendation(
        provider=model.provider,
        model_id=model.model_id,
        purpose=purpose,
        score=score,
        reason=reason,
        tradeoffs=sorted(set(tradeoffs)),
        why_better_than_current=why_better,
    )


def recommend_models(
    *,
    available: list[ModelInfo],
    current: dict[str, str],
    purposes: list[str],
    prefer_local: bool = True,
) -> dict[str, list[ModelRecommendation]]:
    normalized_purposes = _normalize_purpose_list(purposes)
    ordered_available = sorted(available, key=lambda item: (item.provider, item.model_id))
    output: dict[str, list[ModelRecommendation]] = {}
    for purpose in normalized_purposes:
        rows = [
            _score_model(
                model=model,
                purpose=purpose,
                current=current,
                prefer_local=prefer_local,
            )
            for model in ordered_available
        ]
        rows.sort(key=lambda row: (-float(row.score), row.provider, row.model_id))
        output[purpose] = rows[:3]
    return output


def recommendation_to_dict(row: ModelRecommendation) -> dict[str, Any]:
    payload = asdict(row)
    payload["score"] = round(float(payload.get("score") or 0.0), 2)
    return payload


def load_seen_model_ids(path: Path) -> set[str]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    if not isinstance(parsed, dict):
        return set()
    rows = parsed.get("seen_model_ids") if isinstance(parsed.get("seen_model_ids"), list) else []
    return {str(item).strip() for item in rows if str(item).strip()}


def save_seen_model_ids(path: Path, model_ids: set[str]) -> None:
    normalized = sorted({str(item).strip() for item in model_ids if str(item).strip()})
    payload = {
        "schema_version": 1,
        "seen_model_ids": normalized,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


__all__ = [
    "ModelRecommendation",
    "load_seen_model_ids",
    "recommend_models",
    "recommendation_to_dict",
    "save_seen_model_ids",
]
