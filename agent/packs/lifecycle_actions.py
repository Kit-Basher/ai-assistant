from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.packs.lifecycle import PackLifecycleResult, PackLifecycleService, PackLifecycleState, render_lifecycle_response


Handler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class PackLifecycleActionResult:
    ok: bool
    action: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    lifecycle_before: dict[str, Any] = field(default_factory=dict)
    lifecycle_after: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "action": self.action,
            "text": self.text,
            "payload": dict(self.payload),
            "refused": self.refused,
            "lifecycle_before": dict(self.lifecycle_before),
            "lifecycle_after": dict(self.lifecycle_after) if isinstance(self.lifecycle_after, dict) else None,
        }


class PackLifecycleActionController:
    """Runs one approved lifecycle transition at a time.

    The controller is deliberately handler-driven. It validates the lifecycle
    state/action pair, calls an existing safe handler when supplied, and refuses
    unsupported transitions without mutating state.
    """

    _ALLOWED: dict[str, set[str]] = {
        "preview": {PackLifecycleState.DISCOVERED},
        "scaffold_preview": {PackLifecycleState.MISSING},
        "import_for_review": {PackLifecycleState.PREVIEWED},
        "create_review_candidate": {PackLifecycleState.SCAFFOLD_PREVIEWED},
        "inspect": {PackLifecycleState.GENERATED_QUARANTINED, PackLifecycleState.IMPORTED_FOR_REVIEW},
        "review_approve": {PackLifecycleState.GENERATED_QUARANTINED, PackLifecycleState.IMPORTED_FOR_REVIEW},
        "enable": {PackLifecycleState.APPROVED, PackLifecycleState.DISABLED},
        "request_configuration": {PackLifecycleState.NEEDS_CONFIGURATION},
        "configure": {PackLifecycleState.NEEDS_CONFIGURATION},
        "request_permission": {PackLifecycleState.NEEDS_PERMISSION, PackLifecycleState.IMPORTED_FOR_REVIEW},
        "record_permission_grant": {PackLifecycleState.NEEDS_PERMISSION, PackLifecycleState.IMPORTED_FOR_REVIEW},
        "use_if_usable": {PackLifecycleState.USABLE},
        "use": {PackLifecycleState.USABLE},
    }

    def __init__(self, *, handlers: dict[str, Handler] | None = None, lifecycle_service: PackLifecycleService | None = None) -> None:
        self._handlers = dict(handlers or {})
        self._lifecycle_service = lifecycle_service or PackLifecycleService()

    def dispatch(
        self,
        lifecycle: PackLifecycleResult | dict[str, Any],
        *,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> PackLifecycleActionResult:
        before = _lifecycle_dict(lifecycle)
        requested_action = _clean(action) or _next_action(before)
        state = _clean(before.get("state"))
        if not requested_action:
            return self._refuse(before, "", "No lifecycle action is available for this pack.")
        if bool(before.get("blocked")) or state in {PackLifecycleState.BLOCKED, PackLifecycleState.REMOVED}:
            return self._refuse(before, requested_action, "This pack is blocked or removed, so I cannot continue its lifecycle action.")
        allowed_states = self._ALLOWED.get(requested_action)
        if not allowed_states or state not in allowed_states:
            return self._refuse(
                before,
                requested_action,
                f"I cannot run {requested_action} while the pack lifecycle state is {state or 'unknown'}. Current state is unchanged.",
            )

        handler = self._handlers.get(requested_action)
        if handler is None:
            return self._not_implemented(before, requested_action)

        body = handler({**dict(context or {}), "lifecycle": before, "action": requested_action})
        if not isinstance(body, dict):
            return self._refuse(before, requested_action, "The lifecycle action handler returned no structured result.")
        ok = bool(body.get("ok", True))
        text = _clean(body.get("text") or body.get("message") or body.get("summary"))
        if not text:
            text = self._default_success_text(requested_action, body)
        after = body.get("lifecycle_after") if isinstance(body.get("lifecycle_after"), dict) else None
        if after is None:
            after = self._derive_after(requested_action, before, body)
        if ok and after and bool(after.get("usable")) and requested_action in {"preview", "import_for_review", "create_review_candidate", "review_approve"}:
            return self._refuse(before, requested_action, "Lifecycle action refused because it would skip directly to usable.")
        return PackLifecycleActionResult(
            ok=ok,
            action=requested_action,
            text=text,
            payload=body,
            refused=not ok,
            lifecycle_before=before,
            lifecycle_after=after,
        )

    def _derive_after(self, action: str, before: dict[str, Any], body: dict[str, Any]) -> dict[str, Any] | None:
        if action == "preview":
            preview = body.get("preview") if isinstance(body.get("preview"), dict) else {}
            if preview:
                return self._lifecycle_service.evaluate(capability=before.get("capability"), catalog_preview=preview).to_dict()
        if action in {"import_for_review", "create_review_candidate", "review_approve", "enable", "record_permission_grant"}:
            pack = body.get("pack") if isinstance(body.get("pack"), dict) else {}
            grants = body.get("permission_grants") if isinstance(body.get("permission_grants"), list) else []
            if pack:
                return self._lifecycle_service.evaluate(capability=before.get("capability"), imported_pack=pack, permission_grants=grants).to_dict()
        return None

    def _default_success_text(self, action: str, body: dict[str, Any]) -> str:
        if action == "preview":
            return "I showed the pack preview. Current state: previewed. Next safe step: import it for review."
        if action == "import_for_review":
            return "I imported the pack for review only. It is not approved, enabled, granted permissions, or usable yet."
        if action == "create_review_candidate":
            return "I created a review-only scaffolded pack candidate. It is not approved, enabled, granted permissions, or usable yet."
        if action == "review_approve":
            return "I recorded review approval only. The pack is still not enabled or usable yet."
        if action == "enable":
            return "I enabled the approved pack. It is usable only if configuration and permission gates are complete."
        if action in {"request_permission", "record_permission_grant"}:
            return "I handled one managed-adapter permission step. The pack is usable only after all lifecycle gates pass."
        return "I completed one lifecycle step."

    def _not_implemented(self, before: dict[str, Any], action: str) -> PackLifecycleActionResult:
        state = _clean(before.get("state")) or "unknown"
        text = f"The next lifecycle action is {action}, but that transition is not implemented in this chat path yet. Current state remains {state}."
        return PackLifecycleActionResult(ok=False, action=action, text=text, refused=True, lifecycle_before=before, payload={"not_implemented": True})

    def _refuse(self, before: dict[str, Any], action: str, text: str) -> PackLifecycleActionResult:
        return PackLifecycleActionResult(ok=False, action=action, text=text, refused=True, lifecycle_before=before, payload={"refused": True})


def lifecycle_action_context(
    *,
    lifecycle: PackLifecycleResult | dict[str, Any],
    context: dict[str, Any] | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    before = _lifecycle_dict(lifecycle)
    return {
        "origin_tool": "pack_lifecycle_action",
        "lifecycle": before,
        "lifecycle_action": _clean(action) or _next_action(before),
        **dict(context or {}),
    }


def render_lifecycle_action_prompt(lifecycle: PackLifecycleResult | dict[str, Any]) -> str:
    row = _lifecycle_dict(lifecycle)
    action = _next_action(row)
    base = render_lifecycle_response(row)
    if action:
        return f"{base} Say yes to continue this one lifecycle step, or no to cancel."
    return base


def _lifecycle_dict(value: PackLifecycleResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, PackLifecycleResult):
        return value.to_dict()
    return dict(value) if isinstance(value, dict) else {}


def _next_action(row: dict[str, Any]) -> str:
    next_step = row.get("next_step") if isinstance(row.get("next_step"), dict) else {}
    action = _clean(next_step.get("action")) if isinstance(next_step, dict) else ""
    if action == "configure":
        return "request_configuration"
    if action == "use":
        return "use_if_usable"
    return action


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
