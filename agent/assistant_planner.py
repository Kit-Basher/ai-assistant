from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Mapping

from agent.llm.inference_router import route_inference

PLANNER_INTENTS = {"answer_directly", "ask_agent", "clarify"}
PLANNER_CAPABILITIES = {
    "web_search",
    "telegram",
    "chat_model",
    "local_models",
    "external_skills",
    "runtime_status",
    "file_work",
    "none",
}
PLANNER_ACTIONS = {"status", "setup", "query", "preview", "diagnose", "none"}
PLANNER_ACTIONS_BY_CAPABILITY = {
    "web_search": {"status", "setup", "query"},
    "telegram": {"status", "setup"},
    "chat_model": {"status", "setup", "diagnose"},
    "local_models": {"status", "setup", "diagnose"},
    "external_skills": {"status", "setup", "query", "preview"},
    "runtime_status": {"status", "diagnose"},
    "file_work": {"status", "query", "preview"},
    "none": {"none"},
}
MIN_PLANNER_CONFIDENCE = 0.45


@dataclass(frozen=True)
class AssistantAgentRequest:
    capability: str
    action: str
    goal: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"capability": self.capability, "action": self.action, "goal": self.goal}


@dataclass(frozen=True)
class AssistantPlan:
    intent: str
    agent_request: AssistantAgentRequest
    confidence: float
    user_facing_summary: str = ""
    valid: bool = True
    error_kind: str | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        if not self.valid:
            return False
        if self.intent == "clarify":
            return True
        if self.intent == "answer_directly":
            return self.confidence >= MIN_PLANNER_CONFIDENCE
        return self.confidence >= MIN_PLANNER_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "agent_request": self.agent_request.to_dict(),
            "confidence": self.confidence,
            "user_facing_summary": self.user_facing_summary,
            "valid": self.valid,
            "error_kind": self.error_kind,
            "errors": list(self.errors),
        }


def invalid_plan(error_kind: str, *, errors: list[str] | tuple[str, ...] | None = None, raw: Any = None) -> AssistantPlan:
    raw_dict = dict(raw) if isinstance(raw, dict) else {}
    return AssistantPlan(
        intent="clarify",
        agent_request=AssistantAgentRequest("none", "none", ""),
        confidence=0.0,
        user_facing_summary="I need a clearer request before I can safely continue.",
        valid=False,
        error_kind=error_kind,
        errors=tuple(str(item) for item in (errors or [error_kind]) if str(item).strip()),
        raw=raw_dict,
    )


def _coerce_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return dict(parsed) if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return dict(parsed) if isinstance(parsed, dict) else None


def validate_assistant_plan(payload: Mapping[str, Any] | str | None) -> AssistantPlan:
    data = _coerce_json_object(payload)
    if not isinstance(data, dict):
        return invalid_plan("planner_invalid_json")
    errors: list[str] = []
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in PLANNER_INTENTS:
        errors.append("planner_unknown_intent")
    raw_request = data.get("agent_request") if isinstance(data.get("agent_request"), dict) else {}
    capability = str(raw_request.get("capability") or "none").strip().lower() or "none"
    action = str(raw_request.get("action") or "none").strip().lower() or "none"
    goal = str(raw_request.get("goal") or "").strip()[:500]
    if capability not in PLANNER_CAPABILITIES:
        errors.append("planner_unknown_capability")
    if action not in PLANNER_ACTIONS:
        errors.append("planner_unknown_action")
    if capability in PLANNER_ACTIONS_BY_CAPABILITY and action not in PLANNER_ACTIONS_BY_CAPABILITY[capability]:
        errors.append("planner_action_not_allowed_for_capability")
    if intent == "ask_agent" and capability == "none":
        errors.append("planner_missing_agent_capability")
    if intent in {"answer_directly", "clarify"} and (capability != "none" or action != "none"):
        errors.append("planner_agent_request_not_allowed_for_intent")
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
        errors.append("planner_bad_confidence")
    if confidence < 0.0 or confidence > 1.0:
        errors.append("planner_confidence_out_of_range")
        confidence = max(0.0, min(1.0, confidence))
    summary = str(data.get("user_facing_summary") or "").strip()[:240]
    if errors:
        return invalid_plan("planner_validation_failed", errors=errors, raw=data)
    return AssistantPlan(
        intent=intent,
        agent_request=AssistantAgentRequest(capability, action, goal),
        confidence=confidence,
        user_facing_summary=summary,
        valid=True,
        raw=data,
    )


class AssistantPlanner:
    """LLM planner for interpreting normal user messages.

    The planner only emits a validated intent/request object. It never executes tools.
    """

    def __init__(self, *, route_inference_fn: Callable[..., dict[str, Any]] | None = None) -> None:
        self._route_inference = route_inference_fn or route_inference

    def plan(
        self,
        *,
        user_text: str,
        llm_client: Any,
        context: dict[str, Any] | None = None,
    ) -> AssistantPlan:
        if llm_client is None:
            return invalid_plan("planner_llm_unavailable")
        context = dict(context) if isinstance(context, dict) else {}
        system_prompt = (
            "You are the Personal Agent interpretation planner. Return only JSON matching this schema: "
            "{\"intent\":\"answer_directly|ask_agent|clarify\","
            "\"agent_request\":{\"capability\":\"web_search|telegram|chat_model|local_models|external_skills|runtime_status|file_work|none\","
            "\"action\":\"status|setup|query|preview|diagnose|none\",\"goal\":\"plain user goal\"},"
            "\"confidence\":0.0," 
            "\"user_facing_summary\":\"short internal summary\"}. "
            "You may request capabilities, but you cannot execute them. Unknown tools, Docker commands, shell, installs, OAuth, browser automation, pack import, approval, enabling, and permissions are not valid planner actions. "
            "Use ask_agent only when the agent layer needs to check status, set up an allowed capability, search metadata, or start the external skill lifecycle. "
            "Use answer_directly for normal chat or questions that need no agent state. Use clarify when the user goal is unclear."
        )
        result = self._route_inference(
            llm_client=llm_client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(user_text or "").strip()},
            ],
            user_text=str(user_text or "").strip(),
            task_hint="assistant intent planning",
            purpose="assistant_planning",
            task_type="planning",
            trace_id=str(context.get("trace_id") or "assistant-planner"),
            require_json=True,
            compute_tier="low",
            timeout_seconds=float(context.get("timeout_seconds") or 8.0),
            metadata={"component": "assistant_planner"},
        )
        if not bool(result.get("ok")):
            return invalid_plan(str(result.get("error_kind") or "planner_llm_failed"))
        data = result.get("data") if isinstance(result.get("data"), dict) else None
        plan = (
            validate_assistant_plan(data.get("json"))
            if isinstance(data, dict) and isinstance(data.get("json"), dict)
            else validate_assistant_plan(result.get("text"))
        )
        if not plan.valid:
            return invalid_plan("planner_invalid_output", errors=plan.errors, raw=plan.raw)
        return plan


__all__ = [
    "AssistantAgentRequest",
    "AssistantPlan",
    "AssistantPlanner",
    "MIN_PLANNER_CONFIDENCE",
    "PLANNER_ACTIONS_BY_CAPABILITY",
    "invalid_plan",
    "validate_assistant_plan",
]
