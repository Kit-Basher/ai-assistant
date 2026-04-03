from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from agent.llm.control_contract import normalize_task_request
from agent.llm.install_planner import build_install_plan
from agent.llm.model_inventory import build_model_inventory
from agent.llm.model_selector import select_model_for_task
from agent.llm.task_classifier import classify_task_request


def _default_task_requirements(task_type: str) -> list[str]:
    normalized = str(task_type or "chat").strip().lower() or "chat"
    if normalized == "vision":
        return ["chat", "vision"]
    if normalized == "reasoning":
        return ["chat", "long_context"]
    return ["chat"]


def _is_llm_client_available(llm_client: Any) -> bool:
    if llm_client is None:
        return False
    chat_fn = getattr(llm_client, "chat", None)
    if not callable(chat_fn):
        return False
    enabled_fn = getattr(llm_client, "enabled", None)
    if not callable(enabled_fn):
        return True
    try:
        return bool(enabled_fn())
    except Exception:
        return False


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _metadata_channel(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = str(metadata.get("channel") or metadata.get("source_surface") or "").strip().lower()
    if value in {"telegram", "api", "cli"}:
        return value
    return None


def _normalized_error_result(
    *,
    error_kind: str,
    text: str,
    task_type: str | None,
    selection_reason: str,
    trace_id: str | None,
    next_action: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "text": _normalize_text(text),
        "model": None,
        "provider": None,
        "task_type": _normalize_text(task_type) or "chat",
        "selection_reason": _normalize_text(selection_reason) or "no_selection",
        "fallback_used": True,
        "error_kind": _normalize_text(error_kind) or "llm_unavailable",
        "next_action": _normalize_text(next_action) or "Run: python -m agent doctor",
        "data": dict(data or {}),
        "trace_id": _normalize_text(trace_id) or None,
    }


def _normalize_router_result(
    *,
    raw_result: Any,
    provider: str | None,
    model: str | None,
    task_type: str | None,
    selection_reason: str,
    fallback_used: bool,
    trace_id: str | None,
    next_action: str | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(data or {})
    if isinstance(raw_result, Mapping):
        ok = bool(raw_result.get("ok", False))
        text = _normalize_text(raw_result.get("text"))
        error_kind = _normalize_text(raw_result.get("error_kind") or raw_result.get("error_class")) or None
        resolved_provider = _normalize_text(raw_result.get("provider")) or _normalize_text(provider) or None
        resolved_model = _normalize_text(raw_result.get("model")) or _normalize_text(model) or None
        attempts = raw_result.get("attempts") if isinstance(raw_result.get("attempts"), list) else []
        usage = raw_result.get("usage") if isinstance(raw_result.get("usage"), Mapping) else None
        duration_ms_value = raw_result.get("duration_ms")
        try:
            duration_ms = int(duration_ms_value) if duration_ms_value is not None else None
        except (TypeError, ValueError):
            duration_ms = None
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if error_kind:
            payload["error_kind"] = error_kind
        if usage is not None:
            payload["usage"] = dict(usage)
        normalized = {
            "ok": ok,
            "text": text,
            "model": resolved_model,
            "provider": resolved_provider,
            "task_type": _normalize_text(task_type) or "chat",
            "selection_reason": _normalize_text(selection_reason) or "router_default",
            "fallback_used": bool(raw_result.get("fallback_used", fallback_used)),
            "error_kind": error_kind,
            "error_class": error_kind,
            "next_action": _normalize_text(raw_result.get("next_action") or next_action) or None,
            "data": payload,
            "trace_id": _normalize_text(trace_id) or None,
            "attempts": attempts,
        }
        if usage is not None:
            normalized["usage"] = dict(usage)
        if duration_ms is not None:
            normalized["duration_ms"] = duration_ms
        return normalized
    return {
        "ok": True,
        "text": _normalize_text(raw_result),
        "model": _normalize_text(model) or None,
        "provider": _normalize_text(provider) or None,
        "task_type": _normalize_text(task_type) or "chat",
        "selection_reason": _normalize_text(selection_reason) or "router_default",
        "fallback_used": bool(fallback_used),
        "error_kind": None,
        "next_action": _normalize_text(next_action) or None,
        "data": payload,
        "trace_id": _normalize_text(trace_id) or None,
    }


@dataclass
class InferenceRouter:
    """Single orchestrator-facing boundary for LLM execution."""

    llm_client: Any

    def execute(
        self,
        *,
        messages: list[dict[str, Any]],
        user_text: str | None = None,
        purpose: str = "chat",
        task_hint: str | None = None,
        task_type: str | None = None,
        local_only: bool | None = None,
        structured_output: bool | None = None,
        require_json: bool | None = None,
        require_vision: bool | None = None,
        min_context_tokens: int | None = None,
        timeout_seconds: float | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        require_tools: bool | None = None,
        compute_tier: str | None = None,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        normalized_purpose = _normalize_text(purpose) or "chat"
        normalized_trace_id = _normalize_text(trace_id) or None
        if not _is_llm_client_available(self.llm_client):
            return _normalized_error_result(
                error_kind="llm_unavailable",
                text="LLM inference is unavailable right now.\nNext: Run: python -m agent doctor",
                task_type=task_type or "chat",
                selection_reason="llm_unavailable",
                trace_id=normalized_trace_id,
                next_action="Run: python -m agent doctor",
            )

        normalized_task = self._build_task_request(
            user_text=user_text,
            task_hint=task_hint,
            task_type=task_type,
        )
        selected_provider = _normalize_text(provider_override) or None
        selected_model = _normalize_text(model_override) or None
        selection_reason = "router_default"
        selection_fallbacks: list[str] = []
        allow_remote_fallback = self._allow_remote_fallback(local_only=local_only)
        route_data: dict[str, Any] = {"task_request": normalized_task}

        if normalized_purpose == "chat" and selected_model is None:
            control_result = self._resolve_chat_selection(
                task_request=normalized_task,
                allow_remote_fallback=allow_remote_fallback,
                trace_id=normalized_trace_id,
                metadata=metadata,
            )
            if control_result:
                if control_result.get("inventory") is not None:
                    route_data["inventory"] = control_result["inventory"]
                selection = control_result.get("selection")
                if isinstance(selection, dict):
                    route_data["selection"] = selection
                    selected_provider = _normalize_text(selection.get("provider")) or None
                    selected_model = _normalize_text(selection.get("selected_model")) or None
                    selection_reason = _normalize_text(selection.get("reason")) or "router_default"
                    selection_fallbacks = [
                        _normalize_text(item)
                        for item in (selection.get("fallbacks") if isinstance(selection.get("fallbacks"), list) else [])
                        if _normalize_text(item)
                    ]
                if selected_model is None:
                    plan = control_result.get("plan")
                    if isinstance(plan, dict):
                        route_data["plan"] = plan
                    next_action = _normalize_text((plan or {}).get("next_action")) or "Run: python -m agent doctor"
                    error_kind = _normalize_text((selection or {}).get("reason")) or "no_suitable_model"
                    text = "No suitable local-first model is ready for this request.\nNext: {next_action}".format(
                        next_action=next_action
                    )
                    return _normalized_error_result(
                        error_kind=error_kind,
                        text=text,
                        task_type=str(normalized_task.get("task_type") or "chat"),
                        selection_reason=selection_reason,
                        trace_id=normalized_trace_id,
                        next_action=next_action,
                        data=route_data,
                    )

        chat_kwargs = self._chat_kwargs(
            purpose=normalized_purpose,
            task_type=str(normalized_task.get("task_type") or "chat"),
            compute_tier=compute_tier,
            provider_override=selected_provider,
            model_override=selected_model,
            metadata=metadata,
            require_tools=require_tools,
            require_json=require_json if require_json is not None else structured_output,
            require_vision=require_vision,
            min_context_tokens=min_context_tokens,
            timeout_seconds=timeout_seconds,
        )
        raw_result = self._call_chat(messages, chat_kwargs)
        return _normalize_router_result(
            raw_result=raw_result,
            provider=selected_provider,
            model=selected_model,
            task_type=str(normalized_task.get("task_type") or "chat"),
            selection_reason=selection_reason,
            fallback_used=bool(selection_fallbacks),
            trace_id=normalized_trace_id,
            data=route_data,
        )

    def _build_task_request(
        self,
        *,
        user_text: str | None,
        task_hint: str | None,
        task_type: str | None,
    ) -> dict[str, Any]:
        text_for_classification = _normalize_text(task_hint) or _normalize_text(user_text)
        if text_for_classification:
            task_request = classify_task_request(text_for_classification)
        else:
            task_request = normalize_task_request(
                {
                    "task_type": _normalize_text(task_type) or "chat",
                    "requirements": _default_task_requirements(_normalize_text(task_type) or "chat"),
                    "preferred_local": True,
                }
            )
        normalized_task_type = _normalize_text(task_type)
        if normalized_task_type:
            task_request = normalize_task_request(
                {
                    "task_type": normalized_task_type,
                    "requirements": task_request.get("requirements") or _default_task_requirements(normalized_task_type),
                    "preferred_local": bool(task_request.get("preferred_local", True)),
                }
            )
        return task_request

    def _allow_remote_fallback(self, *, local_only: bool | None) -> bool:
        if local_only is not None:
            return not bool(local_only)
        llm_config = getattr(self.llm_client, "config", None)
        if bool(getattr(llm_config, "safe_mode_enabled", False)):
            return False
        registry = getattr(self.llm_client, "registry", None)
        defaults = getattr(registry, "defaults", None)
        allow_remote = getattr(defaults, "allow_remote_fallback", None)
        if allow_remote is None:
            return True
        return bool(allow_remote)

    def _resolve_chat_selection(
        self,
        *,
        task_request: dict[str, Any],
        allow_remote_fallback: bool,
        trace_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        llm_config = getattr(self.llm_client, "config", None)
        llm_registry = getattr(self.llm_client, "registry", None)
        if llm_config is None or llm_registry is None:
            return {}
        router_snapshot = None
        if hasattr(self.llm_client, "doctor_snapshot"):
            try:
                snapshot_payload = self.llm_client.doctor_snapshot()
            except Exception:
                snapshot_payload = None
            if isinstance(snapshot_payload, dict):
                router_snapshot = snapshot_payload
        inventory = build_model_inventory(
            config=llm_config,
            registry=llm_registry,
            router_snapshot=router_snapshot,
            timeout_seconds=1.0,
        )
        selection = select_model_for_task(
            inventory,
            task_request,
            allow_remote_fallback=allow_remote_fallback,
            channel=_metadata_channel(metadata),
            latency_fallback=bool((metadata or {}).get("latency_fallback")),
            trace_id=trace_id,
            policy_name="default",
            policy=getattr(llm_config, "default_policy", None),
        )
        if selection.get("selected_model"):
            return {"inventory": inventory, "selection": selection}
        plan = build_install_plan(
            inventory=inventory,
            task_request=task_request,
            selection_result=selection,
            allow_remote_fallback=allow_remote_fallback,
            policy_name="default",
            policy=getattr(llm_config, "default_policy", None),
        )
        return {
            "inventory": inventory,
            "selection": selection,
            "plan": plan,
        }

    @staticmethod
    def _chat_kwargs(
        *,
        purpose: str,
        task_type: str,
        compute_tier: str | None,
        provider_override: str | None,
        model_override: str | None,
        metadata: dict[str, Any] | None,
        require_tools: bool | None,
        require_json: bool | None,
        require_vision: bool | None,
        min_context_tokens: int | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "purpose": purpose,
            "task_type": task_type,
        }
        if compute_tier is not None:
            kwargs["compute_tier"] = compute_tier
        if provider_override is not None:
            kwargs["provider_override"] = provider_override
        if model_override is not None:
            kwargs["model_override"] = model_override
        if metadata:
            kwargs["metadata"] = dict(metadata)
        if require_tools is not None:
            kwargs["require_tools"] = bool(require_tools)
        if require_json is not None:
            kwargs["require_json"] = bool(require_json)
        if require_vision is not None:
            kwargs["require_vision"] = bool(require_vision)
        if min_context_tokens is not None:
            kwargs["min_context_tokens"] = int(min_context_tokens)
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = float(timeout_seconds)
        return kwargs

    def _call_chat(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> Any:
        chat_fn = getattr(self.llm_client, "chat")
        attempt_kwargs: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for candidate in (
            kwargs,
            {key: value for key, value in kwargs.items() if key != "metadata"},
            {
                key: value
                for key, value in kwargs.items()
                if key
                in {
                    "purpose",
                    "task_type",
                    "compute_tier",
                    "provider_override",
                    "model_override",
                    "require_tools",
                    "require_json",
                    "require_vision",
                    "min_context_tokens",
                    "timeout_seconds",
                }
            },
            {
                key: value
                for key, value in kwargs.items()
                if key in {"purpose", "provider_override", "model_override", "timeout_seconds"}
            },
            {key: value for key, value in kwargs.items() if key in {"purpose"}},
            {},
        ):
            fingerprint = tuple(sorted((str(key), type(value).__name__) for key, value in candidate.items()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            attempt_kwargs.append(candidate)

        last_type_error: TypeError | None = None
        for candidate in attempt_kwargs:
            try:
                return chat_fn(messages, **candidate)
            except TypeError as exc:
                last_type_error = exc
                continue
        if last_type_error is not None:
            raise last_type_error
        return chat_fn(messages)


def route_inference(*, llm_client: Any, **kwargs: Any) -> dict[str, Any]:
    return InferenceRouter(llm_client).execute(**kwargs)


__all__ = ["InferenceRouter", "route_inference"]
