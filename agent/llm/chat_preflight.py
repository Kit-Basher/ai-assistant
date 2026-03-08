from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping

from agent.llm.value_policy import (
    UtilityScore,
    ValuePolicy,
    detect_premium_escalation_triggers,
    normalize_policy,
    rank_candidates_by_utility,
)


_Message = dict[str, str]


@dataclass(frozen=True)
class PreparedChatRequest:
    messages: list[_Message]
    last_user_text: str
    provider_override: str | None
    model_override: str | None
    require_tools: bool
    selection_reason: str
    escalation_reasons: tuple[str, ...]
    premium_selected_model: str | None
    direct_result: dict[str, Any] | None = None
    response_selection_policy: dict[str, Any] | None = None
    log_reason: str | None = None
    log_fallback_used: bool | None = None
    consume_premium_override_once: bool = False


@dataclass(frozen=True)
class RuntimeChatPreflightBridge:
    default_policy: dict[str, Any] | ValuePolicy | None
    premium_policy: dict[str, Any] | ValuePolicy | None
    model_policy_candidates: Callable[[], list[dict[str, Any]]]
    premium_override_active: Callable[[int], bool]
    premium_override_once: bool
    persist_premium_over_cap_prompt: Callable[..., str]
    classify_authoritative_domain: Callable[[str], set[str]]
    has_local_observations_block: Callable[[str], bool]
    collect_authoritative_observations: Callable[[set[str]], dict[str, Any]]
    authoritative_tool_failure_text: Callable[[set[str], Exception], str]


def _last_user_message_text(messages: list[_Message]) -> str:
    for item in reversed(messages):
        if str(item.get("role") or "").strip().lower() == "user":
            return str(item.get("content") or "")
    if messages:
        return str((messages[-1] or {}).get("content") or "")
    return ""


def _memory_prefix_messages(memory_context_text: str) -> list[_Message]:
    text = str(memory_context_text or "").strip()
    if not text:
        return []
    return [{"role": "system", "content": text}]


def _top_non_default(rows: list[UtilityScore], default_model: str | None) -> UtilityScore | None:
    baseline = str(default_model or "").strip()
    for row in rows:
        if str(row.model_id) != baseline:
            return row
    return None


def _short_circuit_result(
    *,
    ok: bool,
    text: str,
    provider: str | None,
    model: str | None,
    selection_reason: str,
    error_kind: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": bool(ok),
        "text": str(text or ""),
        "provider": provider,
        "model": model,
        "fallback_used": False,
        "attempts": [],
        "duration_ms": 0,
        "error_kind": str(error_kind or "").strip() or None,
        "selection_reason": str(selection_reason or "").strip() or "router_default",
    }


def prepare_chat_request(
    *,
    payload: Mapping[str, Any],
    messages: list[_Message],
    defaults: Mapping[str, Any],
    request_started_epoch: int,
    default_policy: dict[str, Any] | ValuePolicy | None,
    premium_policy: dict[str, Any] | ValuePolicy | None,
    model_policy_candidates: Callable[[], list[dict[str, Any]]],
    premium_override_active: Callable[[int], bool],
    premium_override_once: bool,
    persist_premium_over_cap_prompt: Callable[..., str],
    classify_authoritative_domain: Callable[[str], set[str]],
    has_local_observations_block: Callable[[str], bool],
    collect_authoritative_observations: Callable[[set[str]], dict[str, Any]],
    authoritative_tool_failure_text: Callable[[set[str], Exception], str],
) -> PreparedChatRequest:
    normalized_default_policy = (
        default_policy
        if isinstance(default_policy, ValuePolicy)
        else normalize_policy(default_policy if isinstance(default_policy, dict) else {}, name="default")
    )
    normalized_premium_policy = (
        premium_policy
        if isinstance(premium_policy, ValuePolicy)
        else normalize_policy(premium_policy if isinstance(premium_policy, dict) else {}, name="premium")
    )
    explicit_model_override = bool(str(payload.get("model") or "").strip())
    explicit_provider_override = bool(str(payload.get("provider") or "").strip())
    default_model = (
        str(defaults.get("chat_model") or "").strip()
        or str(defaults.get("default_model") or "").strip()
        or None
    )
    default_provider = str(defaults.get("default_provider") or "").strip().lower() or None
    allow_remote_fallback = bool(defaults.get("allow_remote_fallback", True))
    model_override = str(payload.get("model") or "").strip() or None
    provider_override = str(payload.get("provider") or "").strip().lower() or None
    explicit_require_tools = "require_tools" in payload
    require_tools = bool(payload.get("require_tools"))
    memory_prefix_messages = _memory_prefix_messages(str(payload.get("memory_context_text") or ""))
    routed_messages = [*memory_prefix_messages, *messages]
    last_user_text = _last_user_message_text(messages)
    selection_reason = (
        "explicit_override"
        if explicit_model_override or explicit_provider_override
        else "default_policy"
    )

    escalation_reasons = detect_premium_escalation_triggers(user_text=last_user_text, payload=dict(payload))
    premium_selected: UtilityScore | None = None
    consume_premium_override_once = False
    if escalation_reasons and not explicit_model_override and not explicit_provider_override:
        candidates = list(model_policy_candidates())
        default_allowed, default_rejected = rank_candidates_by_utility(
            candidates,
            policy=normalized_default_policy,
            allow_remote_fallback=allow_remote_fallback,
        )
        default_by_id = {str(row.model_id): row for row in [*default_allowed, *default_rejected]}
        default_score = default_by_id.get(str(default_model or "").strip())
        premium_allowed, _premium_rejected = rank_candidates_by_utility(
            candidates,
            policy=normalized_premium_policy,
            allow_remote_fallback=allow_remote_fallback,
        )
        premium_unbounded = ValuePolicy(
            name=normalized_premium_policy.name,
            cost_cap_per_1m=1_000_000.0,
            allowlist=normalized_premium_policy.allowlist,
            quality_weight=normalized_premium_policy.quality_weight,
            price_weight=normalized_premium_policy.price_weight,
            latency_weight=normalized_premium_policy.latency_weight,
            instability_weight=normalized_premium_policy.instability_weight,
        )
        premium_no_cap_allowed, _premium_no_cap_rejected = rank_candidates_by_utility(
            candidates,
            policy=premium_unbounded,
            allow_remote_fallback=allow_remote_fallback,
        )
        baseline_utility = float(default_score.utility) if default_score is not None else -10_000.0
        top_premium = _top_non_default(premium_allowed, default_model)
        top_no_cap = _top_non_default(premium_no_cap_allowed, default_model)
        override_active = bool(premium_override_active(int(request_started_epoch)))
        if top_no_cap is not None and (
            float(top_no_cap.expected_cost_per_1m) > float(normalized_premium_policy.cost_cap_per_1m)
        ):
            if override_active and (float(top_no_cap.utility) > baseline_utility):
                premium_selected = top_no_cap
                consume_premium_override_once = bool(premium_override_once)
            elif not override_active:
                prompt = persist_premium_over_cap_prompt(
                    baseline_model=str(default_model or ""),
                    premium_model=str(top_no_cap.model_id),
                    premium_cost=float(top_no_cap.expected_cost_per_1m),
                    premium_cap=float(normalized_premium_policy.cost_cap_per_1m),
                )
                return PreparedChatRequest(
                    messages=routed_messages,
                    last_user_text=last_user_text,
                    provider_override=default_provider,
                    model_override=default_model,
                    require_tools=require_tools,
                    selection_reason="premium_over_cap_confirmation_required",
                    escalation_reasons=escalation_reasons,
                    premium_selected_model=str(top_no_cap.model_id),
                    direct_result=_short_circuit_result(
                        ok=True,
                        text=prompt,
                        provider=default_provider,
                        model=default_model,
                        selection_reason="premium_over_cap_confirmation_required",
                    ),
                    response_selection_policy={
                        "mode": "premium_over_cap",
                        "baseline_model": str(default_model or ""),
                        "premium_candidate": str(top_no_cap.model_id),
                        "premium_cost_per_1m": float(top_no_cap.expected_cost_per_1m),
                        "premium_cap_per_1m": float(normalized_premium_policy.cost_cap_per_1m),
                        "escalation_reasons": list(escalation_reasons),
                    },
                    log_reason="premium_over_cap_confirmation_required",
                    log_fallback_used=False,
                )
        if premium_selected is None and top_premium is not None and float(top_premium.utility) > baseline_utility:
            premium_selected = top_premium
        if premium_selected is not None:
            model_override = str(premium_selected.model_id)
            provider_override = str(premium_selected.provider)
            selection_reason = "premium_escalation"

    if not explicit_require_tools:
        domains = classify_authoritative_domain(last_user_text)
        if domains and not has_local_observations_block(last_user_text):
            try:
                local_observations = collect_authoritative_observations(domains)
            except Exception as exc:
                return PreparedChatRequest(
                    messages=routed_messages,
                    last_user_text=last_user_text,
                    provider_override="tool_gate",
                    model_override=None,
                    require_tools=False,
                    selection_reason="authoritative_tool_failure",
                    escalation_reasons=escalation_reasons,
                    premium_selected_model=str(getattr(premium_selected, "model_id", "") or "") or None,
                    direct_result=_short_circuit_result(
                        ok=True,
                        text=authoritative_tool_failure_text(domains, exc),
                        provider="tool_gate",
                        model=None,
                        selection_reason="authoritative_tool_failure",
                        error_kind="authoritative_tool_failure",
                    ),
                    log_reason="authoritative_tool_failure",
                    log_fallback_used=True,
                    consume_premium_override_once=consume_premium_override_once,
                )

            observations_json = json.dumps(local_observations, ensure_ascii=True, sort_keys=True)
            routed_messages = [
                {
                    "role": "system",
                    "content": (
                        "Use LOCAL_OBSERVATIONS as authoritative local evidence. "
                        "Do not invent system facts.\n\n"
                        f"LOCAL_OBSERVATIONS\n{observations_json}"
                    ),
                },
                *memory_prefix_messages,
                *messages,
            ]
            require_tools = True

    return PreparedChatRequest(
        messages=routed_messages,
        last_user_text=last_user_text,
        provider_override=provider_override,
        model_override=model_override,
        require_tools=require_tools,
        selection_reason=selection_reason,
        escalation_reasons=escalation_reasons,
        premium_selected_model=str(getattr(premium_selected, "model_id", "") or "") or None,
        consume_premium_override_once=consume_premium_override_once,
    )


def prepare_runtime_chat_request(
    *,
    payload: Mapping[str, Any],
    messages: list[_Message],
    defaults: Mapping[str, Any],
    request_started_epoch: int,
    bridge: RuntimeChatPreflightBridge,
) -> PreparedChatRequest:
    return prepare_chat_request(
        payload=payload,
        messages=messages,
        defaults=defaults,
        request_started_epoch=request_started_epoch,
        default_policy=bridge.default_policy,
        premium_policy=bridge.premium_policy,
        model_policy_candidates=bridge.model_policy_candidates,
        premium_override_active=bridge.premium_override_active,
        premium_override_once=bridge.premium_override_once,
        persist_premium_over_cap_prompt=bridge.persist_premium_over_cap_prompt,
        classify_authoritative_domain=bridge.classify_authoritative_domain,
        has_local_observations_block=bridge.has_local_observations_block,
        collect_authoritative_observations=bridge.collect_authoritative_observations,
        authoritative_tool_failure_text=bridge.authoritative_tool_failure_text,
    )


def build_chat_selection_policy_meta(
    *,
    prepared: PreparedChatRequest,
    result: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> dict[str, Any] | None:
    if prepared.response_selection_policy is not None:
        return dict(prepared.response_selection_policy)
    error_kind = str(result.get("error_kind") or result.get("error_class") or "").strip().lower()
    if error_kind == "authoritative_tool_failure":
        return None
    return {
        "default_model": str(defaults.get("chat_model") or defaults.get("default_model") or ""),
        "selected_model": str(result.get("model") or prepared.model_override or ""),
        "selected_provider": str(result.get("provider") or prepared.provider_override or ""),
        "escalation_reasons": list(prepared.escalation_reasons),
        "premium_selected": str(prepared.premium_selected_model or ""),
    }


__all__ = [
    "PreparedChatRequest",
    "RuntimeChatPreflightBridge",
    "build_chat_selection_policy_meta",
    "prepare_chat_request",
    "prepare_runtime_chat_request",
]
