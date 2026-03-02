from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


_DEFAULT_WEIGHTS = {
    "quality_weight": 1.0,
    "price_weight": 0.04,
    "latency_weight": 0.25,
    "instability_weight": 0.5,
}
_PREMIUM_WEIGHTS = {
    "quality_weight": 1.35,
    "price_weight": 0.025,
    "latency_weight": 0.2,
    "instability_weight": 0.45,
}
_HIGH_STAKES_HINTS = (
    "medical",
    "legal",
    "security",
    "incident",
    "outage",
    "breach",
    "financial",
    "production",
)
_DEEP_REASONING_HINTS = (
    "deep reasoning",
    "reason deeply",
    "step-by-step",
    "step by step",
    "rigorous proof",
    "formal proof",
)
_PREMIUM_REQUEST_HINTS = (
    "use premium",
    "premium model",
    "best model",
    "upgrade model",
    "stronger model",
)


def _safe_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _normalized_allowlist(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(
            sorted(
                {
                    str(item).strip()
                    for item in value
                    if str(item).strip()
                }
            )
        )
    text = str(value or "").strip()
    if not text:
        return tuple()
    return tuple(sorted({item.strip() for item in text.split(",") if item.strip()}))


@dataclass(frozen=True)
class ValuePolicy:
    name: str
    cost_cap_per_1m: float
    allowlist: tuple[str, ...]
    quality_weight: float
    price_weight: float
    latency_weight: float
    instability_weight: float


@dataclass(frozen=True)
class UtilityScore:
    model_id: str
    provider: str
    local: bool
    allowed: bool
    rejected_by: str | None
    quality: float
    expected_cost_per_1m: float
    latency: float
    risk: float
    utility: float


def normalize_policy(raw: dict[str, Any] | None, *, name: str) -> ValuePolicy:
    data = raw if isinstance(raw, dict) else {}
    defaults = _PREMIUM_WEIGHTS if str(name).strip().lower() == "premium" else _DEFAULT_WEIGHTS
    default_cap = 12.0 if str(name).strip().lower() == "premium" else 6.0
    return ValuePolicy(
        name=str(name or "default").strip().lower() or "default",
        cost_cap_per_1m=max(0.0, _safe_float(data.get("cost_cap_per_1m"), default_cap)),
        allowlist=_normalized_allowlist(data.get("allowlist")),
        quality_weight=max(0.0, _safe_float(data.get("quality_weight"), defaults["quality_weight"])),
        price_weight=max(0.0, _safe_float(data.get("price_weight"), defaults["price_weight"])),
        latency_weight=max(0.0, _safe_float(data.get("latency_weight"), defaults["latency_weight"])),
        instability_weight=max(0.0, _safe_float(data.get("instability_weight"), defaults["instability_weight"])),
    )


def _effective_cost_per_1m(candidate: dict[str, Any]) -> float:
    local = bool(candidate.get("local", False))
    if local:
        return 0.0
    price_in = candidate.get("price_in")
    price_out = candidate.get("price_out")
    if price_in is None or price_out is None:
        return 9.0
    return max(0.0, _safe_float(price_in, 0.0) + (2.0 * _safe_float(price_out, 0.0)))


def _latency_estimate(candidate: dict[str, Any]) -> float:
    local = bool(candidate.get("local", False))
    context_tokens = int(candidate.get("context_tokens") or 0)
    base = 0.25 if local else 0.6
    if context_tokens >= 128000:
        base += 0.15
    elif context_tokens >= 32000:
        base += 0.08
    return round(base, 6)


def _instability_risk(candidate: dict[str, Any]) -> float:
    health_status = str(candidate.get("health_status") or "unknown").strip().lower()
    if health_status == "ok":
        return 0.0
    if health_status == "degraded":
        return 0.5
    if health_status == "down":
        return 1.0
    return 0.25


def _quality_score(candidate: dict[str, Any]) -> float:
    quality_rank = _safe_float(candidate.get("quality_rank"), 0.0)
    if quality_rank > 0.0:
        return _clamp(quality_rank / 10.0, 0.0, 1.0)
    params_b = _safe_float(candidate.get("params_b"), 0.0)
    if params_b > 0.0:
        return _clamp(0.25 + (params_b / 70.0), 0.0, 1.0)
    return 0.4


def score_candidate_utility(
    candidate: dict[str, Any],
    *,
    policy: ValuePolicy,
    allow_remote_fallback: bool,
) -> UtilityScore:
    model_id = str(candidate.get("model_id") or "").strip()
    provider = str(candidate.get("provider") or "").strip().lower()
    local = bool(candidate.get("local", False))
    routable = bool(candidate.get("routable", False))
    expected_cost = _effective_cost_per_1m(candidate)
    quality = _quality_score(candidate)
    latency = _latency_estimate(candidate)
    risk = _instability_risk(candidate)

    rejected_by: str | None = None
    if not routable:
        rejected_by = "not_routable"
    elif policy.allowlist and model_id not in set(policy.allowlist):
        rejected_by = "not_in_allowlist"
    elif (not local) and (not allow_remote_fallback):
        rejected_by = "remote_disabled"
    elif (not local) and expected_cost > float(policy.cost_cap_per_1m):
        rejected_by = "cost_cap_exceeded"

    utility = (
        (float(policy.quality_weight) * quality)
        - (float(policy.price_weight) * expected_cost)
        - (float(policy.latency_weight) * latency)
        - (float(policy.instability_weight) * risk)
    )
    return UtilityScore(
        model_id=model_id,
        provider=provider,
        local=local,
        allowed=rejected_by is None,
        rejected_by=rejected_by,
        quality=round(quality, 6),
        expected_cost_per_1m=round(expected_cost, 6),
        latency=round(latency, 6),
        risk=round(risk, 6),
        utility=round(utility, 6),
    )


def rank_candidates_by_utility(
    candidates: Iterable[dict[str, Any]],
    *,
    policy: ValuePolicy,
    allow_remote_fallback: bool,
) -> tuple[list[UtilityScore], list[UtilityScore]]:
    scored = [
        score_candidate_utility(
            candidate,
            policy=policy,
            allow_remote_fallback=allow_remote_fallback,
        )
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    allowed = sorted(
        [row for row in scored if row.allowed],
        key=lambda row: (-float(row.utility), str(row.model_id)),
    )
    rejected = sorted(
        [row for row in scored if not row.allowed],
        key=lambda row: (str(row.rejected_by or ""), str(row.model_id)),
    )
    return allowed, rejected


def detect_premium_escalation_triggers(
    *,
    user_text: str,
    payload: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    data = payload if isinstance(payload, dict) else {}
    normalized = " ".join(str(user_text or "").strip().lower().split())
    triggers: list[str] = []

    if bool(data.get("high_stakes", False)) or any(hint in normalized for hint in _HIGH_STAKES_HINTS):
        triggers.append("high_stakes")
    if bool(data.get("deep_reasoning", False)) or any(hint in normalized for hint in _DEEP_REASONING_HINTS):
        triggers.append("deep_reasoning")
    min_context_tokens = int(data.get("min_context_tokens") or 0)
    if min_context_tokens >= 64000 or len(normalized) >= 1800:
        triggers.append("long_context")
    if bool(data.get("baseline_fail", False)):
        triggers.append("baseline_fail")
    if bool(data.get("premium", False)) or any(hint in normalized for hint in _PREMIUM_REQUEST_HINTS):
        triggers.append("user_request")

    return tuple(dict.fromkeys(triggers))


def utility_delta(
    *,
    current: UtilityScore | None,
    candidate: UtilityScore | None,
) -> float:
    if current is None or candidate is None:
        return 0.0
    return round(float(candidate.utility) - float(current.utility), 6)

