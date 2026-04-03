from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable


_CHAT_REQUIRED = frozenset({"chat"})
_EMBED_REQUIRED = frozenset({"embed"})


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_caps(value: Any) -> frozenset[str]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(str(item).strip().lower() for item in value if str(item).strip())
    return frozenset()


def _canonical_model_id(provider_id: str, model_id: str) -> str:
    provider = str(provider_id or "").strip().lower()
    model = str(model_id or "").strip()
    if not provider or not model:
        return ""
    if ":" in model and model.split(":", 1)[0].strip().lower() == provider:
        return model
    return f"{provider}:{model}"


def _provider_enabled(provider_id: str, enabled_providers: frozenset[str]) -> bool:
    provider = str(provider_id or "").strip().lower()
    if not provider:
        return False
    if not enabled_providers:
        return True
    return provider in enabled_providers


def _required_caps(purpose: str) -> frozenset[str]:
    if str(purpose or "chat").strip().lower() == "embed":
        return _EMBED_REQUIRED
    return _CHAT_REQUIRED


def estimate_required_vram_gb(params_b: float | None) -> float | None:
    if params_b is None:
        return None
    return max(0.0, float(params_b) * 1.3)


@dataclass(frozen=True)
class RecommendationContext:
    purpose: str = "chat"
    default_model: str | None = None
    allow_remote_fallback: bool = True
    enabled_providers: frozenset[str] = frozenset()
    vram_gb: float | None = None


@dataclass(frozen=True)
class ScoreBreakdown:
    provider_id: str
    model_id: str
    canonical_model_id: str
    local: bool
    availability: bool
    params_b: float | None
    price_in: float | None
    price_out: float | None
    task_fit: float
    local_feasibility: float
    cost_efficiency: float
    quality_proxy: float
    switch_gain: float
    total_score: float
    reasons: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    hard_reject_reason: str | None


@dataclass(frozen=True)
class RankedList:
    ranked: tuple[ScoreBreakdown, ...]
    filtered: tuple[ScoreBreakdown, ...]
    confidence: float
    margin: float
    show_top_only: bool


@dataclass(frozen=True)
class PickResult:
    pick: ScoreBreakdown | None
    alternatives: tuple[ScoreBreakdown, ...]
    confidence: float
    show_top_only: bool
    reason: str


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    provider_id = str(raw.get("provider_id") or "").strip().lower()
    model_id = str(raw.get("model_id") or "").strip()
    context_tokens: int | None
    try:
        context_tokens = int(raw.get("context_tokens")) if raw.get("context_tokens") is not None else None
    except (TypeError, ValueError):
        context_tokens = None
    params_b = _safe_float(raw.get("params_b"))
    price_in = _safe_float(raw.get("price_in"))
    price_out = _safe_float(raw.get("price_out"))
    local = bool(raw.get("local"))

    raw_missing = raw.get("missing_features") if isinstance(raw.get("missing_features"), list) else []
    missing_features: list[str] = []
    for item in raw_missing:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if not text.startswith("missing:"):
            text = f"missing:{text}"
        if text not in missing_features:
            missing_features.append(text)
    if context_tokens is None and "missing:context_length" not in missing_features:
        missing_features.append("missing:context_length")
    if params_b is None and "missing:params_b" not in missing_features:
        missing_features.append("missing:params_b")
    if (not local) and (price_in is None or price_out is None) and "missing:pricing" not in missing_features:
        missing_features.append("missing:pricing")

    return {
        "provider_id": provider_id,
        "model_id": model_id,
        "canonical_model_id": _canonical_model_id(provider_id, model_id),
        "params_b": params_b,
        "context_tokens": context_tokens,
        "price_in": price_in,
        "price_out": price_out,
        "capabilities": _normalize_caps(raw.get("capabilities")),
        "local": local,
        "availability": bool(raw.get("availability", True)),
        "quality_percentile": _safe_float(raw.get("quality_percentile")),
        "missing_features": tuple(missing_features),
    }


def score_candidate(raw_candidate: dict[str, Any], context: RecommendationContext) -> ScoreBreakdown:
    candidate = _normalize_candidate(raw_candidate)
    tradeoffs: list[str] = [str(item).strip().lower() for item in candidate["missing_features"] if str(item).strip()]
    reasons: list[str] = []

    provider_id = candidate["provider_id"]
    model_id = candidate["model_id"]
    canonical_model_id = candidate["canonical_model_id"]

    if not provider_id or not model_id or not canonical_model_id:
        return ScoreBreakdown(
            provider_id=provider_id,
            model_id=model_id,
            canonical_model_id=canonical_model_id,
            local=bool(candidate["local"]),
            availability=bool(candidate["availability"]),
            params_b=candidate["params_b"],
            price_in=candidate["price_in"],
            price_out=candidate["price_out"],
            task_fit=0.0,
            local_feasibility=0.0,
            cost_efficiency=0.0,
            quality_proxy=0.0,
            switch_gain=0.0,
            total_score=0.0,
            reasons=tuple(),
            tradeoffs=tuple(tradeoffs),
            hard_reject_reason="invalid_candidate",
        )

    if not bool(candidate["availability"]):
        return ScoreBreakdown(
            provider_id=provider_id,
            model_id=model_id,
            canonical_model_id=canonical_model_id,
            local=bool(candidate["local"]),
            availability=bool(candidate["availability"]),
            params_b=candidate["params_b"],
            price_in=candidate["price_in"],
            price_out=candidate["price_out"],
            task_fit=0.0,
            local_feasibility=0.0,
            cost_efficiency=0.0,
            quality_proxy=0.0,
            switch_gain=0.0,
            total_score=0.0,
            reasons=tuple(),
            tradeoffs=tuple(tradeoffs),
            hard_reject_reason="unavailable",
        )

    required_caps = _required_caps(context.purpose)
    caps = candidate["capabilities"]
    if any(required not in caps for required in required_caps):
        return ScoreBreakdown(
            provider_id=provider_id,
            model_id=model_id,
            canonical_model_id=canonical_model_id,
            local=bool(candidate["local"]),
            availability=bool(candidate["availability"]),
            params_b=candidate["params_b"],
            price_in=candidate["price_in"],
            price_out=candidate["price_out"],
            task_fit=0.0,
            local_feasibility=0.0,
            cost_efficiency=0.0,
            quality_proxy=0.0,
            switch_gain=0.0,
            total_score=0.0,
            reasons=tuple(),
            tradeoffs=tuple(tradeoffs),
            hard_reject_reason="missing_required_capability",
        )

    has_json = "json" in caps
    has_tools = "tools" in caps
    if str(context.purpose).strip().lower() == "embed":
        task_fit = 25.0
    elif has_json and has_tools:
        task_fit = 25.0
    elif has_json:
        task_fit = 20.0
    else:
        task_fit = 15.0
    reasons.append("Capability match is acceptable.")

    local_feasibility = 0.0
    if candidate["local"]:
        required_vram = estimate_required_vram_gb(candidate["params_b"])
        if context.vram_gb is None or required_vram is None:
            local_feasibility = 12.0
            tradeoffs.append("local_fit_uncertain")
        elif required_vram <= float(context.vram_gb):
            local_feasibility = 25.0
            reasons.append("Fits local VRAM.")
        elif required_vram <= float(context.vram_gb) * 1.5:
            local_feasibility = 12.0
            tradeoffs.append("near_vram_limit")
        else:
            local_feasibility = 0.0
            tradeoffs.append("local_too_large_for_vram")
    else:
        if not context.allow_remote_fallback:
            local_feasibility = 0.0
            tradeoffs.append("remote_disabled")
        elif not _provider_enabled(provider_id, context.enabled_providers):
            local_feasibility = 0.0
            tradeoffs.append("provider_disabled")
        else:
            local_feasibility = 12.5

    cost_efficiency = 0.0
    if candidate["local"]:
        cost_efficiency = 15.0
    else:
        if not context.allow_remote_fallback or not _provider_enabled(provider_id, context.enabled_providers):
            cost_efficiency = 0.0
        elif candidate["price_in"] is None or candidate["price_out"] is None:
            cost_efficiency = 4.0
            tradeoffs.append("remote_price_unknown")
        else:
            effective_cost = float(candidate["price_in"]) + (2.0 * float(candidate["price_out"]))
            if effective_cost <= 1.0:
                cost_efficiency = 15.0
            elif effective_cost <= 3.0:
                cost_efficiency = 12.0
            elif effective_cost <= 8.0:
                cost_efficiency = 8.0
            elif effective_cost <= 15.0:
                cost_efficiency = 4.0
            else:
                cost_efficiency = 1.0

    quality_proxy = 10.0
    if candidate["quality_percentile"] is not None:
        quality_proxy = _clamp((float(candidate["quality_percentile"]) / 100.0) * 25.0, 0.0, 25.0)
    elif candidate["params_b"] is not None:
        quality_proxy = min(25.0, 5.0 + (4.0 * math.log10(1.0 + (float(candidate["params_b"]) * 10.0))))

    total_score = _clamp(task_fit + local_feasibility + cost_efficiency + quality_proxy, 0.0, 100.0)
    return ScoreBreakdown(
        provider_id=provider_id,
        model_id=model_id,
        canonical_model_id=canonical_model_id,
        local=bool(candidate["local"]),
        availability=bool(candidate["availability"]),
        params_b=candidate["params_b"],
        price_in=candidate["price_in"],
        price_out=candidate["price_out"],
        task_fit=task_fit,
        local_feasibility=local_feasibility,
        cost_efficiency=cost_efficiency,
        quality_proxy=quality_proxy,
        switch_gain=0.0,
        total_score=total_score,
        reasons=tuple(dict.fromkeys(reasons)),
        tradeoffs=tuple(dict.fromkeys(tradeoffs)),
        hard_reject_reason=None,
    )


def _switch_gain(delta: float, *, is_current: bool) -> float:
    if is_current:
        return 0.0
    if delta >= 15.0:
        return 10.0
    if delta >= 8.0:
        return 6.0
    if delta >= 3.0:
        return 3.0
    return 0.0


def rank_candidates(candidates: Iterable[dict[str, Any]], context: RecommendationContext) -> RankedList:
    prelim = [score_candidate(candidate, context) for candidate in candidates if isinstance(candidate, dict)]
    ranked_prelim = [row for row in prelim if row.hard_reject_reason is None]
    filtered = [row for row in prelim if row.hard_reject_reason is not None]

    current_base_score = 0.0
    current_found = False
    if context.default_model:
        for row in ranked_prelim:
            if row.canonical_model_id == str(context.default_model):
                current_base_score = float(row.total_score)
                current_found = True
                break

    ranked: list[ScoreBreakdown] = []
    for row in ranked_prelim:
        is_current = bool(context.default_model and row.canonical_model_id == str(context.default_model))
        delta = float(row.total_score) - (float(current_base_score) if current_found else 0.0)
        switch_gain = _switch_gain(delta, is_current=is_current)
        total = _clamp(float(row.total_score) + switch_gain, 0.0, 100.0)
        ranked.append(
            ScoreBreakdown(
                provider_id=row.provider_id,
                model_id=row.model_id,
                canonical_model_id=row.canonical_model_id,
                local=row.local,
                availability=row.availability,
                params_b=row.params_b,
                price_in=row.price_in,
                price_out=row.price_out,
                task_fit=row.task_fit,
                local_feasibility=row.local_feasibility,
                cost_efficiency=row.cost_efficiency,
                quality_proxy=row.quality_proxy,
                switch_gain=switch_gain,
                total_score=total,
                reasons=row.reasons,
                tradeoffs=row.tradeoffs,
                hard_reject_reason=None,
            )
        )

    ranked.sort(key=lambda row: (-float(row.total_score), str(row.canonical_model_id)))
    filtered.sort(
        key=lambda row: (
            str(row.hard_reject_reason or ""),
            str(row.canonical_model_id),
        )
    )

    top = float(ranked[0].total_score) if ranked else 0.0
    second = float(ranked[1].total_score) if len(ranked) > 1 else 0.0
    margin = float(top - second) if len(ranked) > 1 else float(top)
    penalty = 0.0
    if ranked:
        top_row = ranked[0]
        if top_row.local and context.vram_gb is None:
            penalty += 0.15
        if top_row.params_b is None:
            penalty += 0.15
        if (not top_row.local) and (top_row.price_in is None or top_row.price_out is None):
            penalty += 0.10
    confidence = _clamp(0.35 + (0.02 * top) + (0.03 * margin) - penalty, 0.0, 1.0)
    show_top_only = confidence >= 0.80 and margin >= 8.0

    return RankedList(
        ranked=tuple(ranked),
        filtered=tuple(filtered),
        confidence=float(round(confidence, 6)),
        margin=float(round(margin, 6)),
        show_top_only=bool(show_top_only),
    )


def pick_recommendation(ranked: RankedList, *, min_pick_score: float = 0.0) -> PickResult:
    rows = list(ranked.ranked)
    if not rows:
        return PickResult(
            pick=None,
            alternatives=tuple(),
            confidence=float(ranked.confidence),
            show_top_only=False,
            reason="no_candidates",
        )
    top = rows[0]
    if float(top.total_score) < float(min_pick_score):
        return PickResult(
            pick=None,
            alternatives=tuple(rows[:3]),
            confidence=float(ranked.confidence),
            show_top_only=False,
            reason="below_threshold",
        )
    alternatives = tuple() if ranked.show_top_only else tuple(rows[1:3])
    return PickResult(
        pick=top,
        alternatives=alternatives,
        confidence=float(ranked.confidence),
        show_top_only=bool(ranked.show_top_only),
        reason="ok",
    )


__all__ = [
    "PickResult",
    "RankedList",
    "RecommendationContext",
    "ScoreBreakdown",
    "estimate_required_vram_gb",
    "pick_recommendation",
    "rank_candidates",
    "score_candidate",
]
