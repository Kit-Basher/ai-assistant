from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from agent.packs.capability_recommendation import (
    classify_capability_gap_request,
    detect_pack_capability_need,
    recommend_packs_for_capability,
)
from agent.packs.lifecycle import PackLifecycleResult, PackLifecycleService, render_lifecycle_response
from agent.packs.lifecycle_actions import PackLifecycleActionController
from agent.packs.managed_adapter_invocation import (
    ManagedAdapterInvocationRequest,
    ManagedAdapterInvoker,
    OP_DESCRIBE_CAPABILITY,
    OP_DRY_RUN,
)
from agent.packs.scaffolding import build_scaffold_preview
from agent.packs.source_leads import SourceLeadSearchResult, build_source_leads_from_safe_search


@dataclass(frozen=True)
class AcquisitionNextStep:
    action: str
    label: str
    gate: str | None = None
    requires_confirmation: bool = False
    pending_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "label": self.label,
            "gate": self.gate,
            "requires_confirmation": bool(self.requires_confirmation),
            "pending_context": dict(self.pending_context),
        }


@dataclass(frozen=True)
class AcquisitionRequest:
    text: str
    requested_capability: str | None = None
    user_id: str | None = None
    thread_id: str | None = None
    operation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcquisitionResult:
    requested_capability: str | None
    detected_capability: str | None
    source_status: str
    candidate_pack: dict[str, Any] | None
    lifecycle_state: str | None
    missing_gate: str | None
    next_step: AcquisitionNextStep | None
    requires_confirmation: bool
    user_message: str
    safe_actions: tuple[str, ...] = ()
    blocked_reason: str | None = None
    lifecycle: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] | None = None
    scaffold_preview: dict[str, Any] | None = None
    action_result: dict[str, Any] | None = None
    invocation_result: dict[str, Any] | None = None
    source_leads: tuple[dict[str, Any], ...] = ()
    lead_search: dict[str, Any] | None = None
    search_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_capability": self.requested_capability,
            "detected_capability": self.detected_capability,
            "source_status": self.source_status,
            "candidate_pack": dict(self.candidate_pack) if isinstance(self.candidate_pack, dict) else None,
            "lifecycle_state": self.lifecycle_state,
            "missing_gate": self.missing_gate,
            "next_step": self.next_step.to_dict() if self.next_step is not None else None,
            "requires_confirmation": bool(self.requires_confirmation),
            "user_message": self.user_message,
            "safe_actions": list(self.safe_actions),
            "blocked_reason": self.blocked_reason,
            "lifecycle": dict(self.lifecycle),
            "recommendation": dict(self.recommendation) if isinstance(self.recommendation, dict) else None,
            "scaffold_preview": dict(self.scaffold_preview) if isinstance(self.scaffold_preview, dict) else None,
            "action_result": dict(self.action_result) if isinstance(self.action_result, dict) else None,
            "invocation_result": dict(self.invocation_result) if isinstance(self.invocation_result, dict) else None,
            "source_leads": [dict(row) for row in self.source_leads],
            "lead_search": dict(self.lead_search) if isinstance(self.lead_search, dict) else None,
            "search_status": dict(self.search_status) if isinstance(self.search_status, dict) else None,
        }


Handler = Callable[[dict[str, Any]], dict[str, Any]]


class PackAcquisitionCoordinator:
    """Assistant-facing workflow for missing external capabilities.

    The coordinator does not bypass any gate. It finds the current lifecycle
    state and returns the next safe assistant step. Mutations only happen
    through supplied lifecycle action handlers.
    """

    def __init__(
        self,
        *,
        pack_store: Any,
        pack_registry_discovery: Any,
        lifecycle_service: PackLifecycleService | None = None,
        action_controller: PackLifecycleActionController | None = None,
        action_handlers: dict[str, Handler] | None = None,
        managed_adapter_invoker: ManagedAdapterInvoker | None = None,
        web_search_handler: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]] | None = None,
        search_status_handler: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.pack_store = pack_store
        self.pack_registry_discovery = pack_registry_discovery
        self.lifecycle_service = lifecycle_service or PackLifecycleService()
        self.action_controller = action_controller or PackLifecycleActionController(
            handlers=action_handlers or {},
            lifecycle_service=self.lifecycle_service,
        )
        self.managed_adapter_invoker = managed_adapter_invoker or ManagedAdapterInvoker(
            lifecycle_service=self.lifecycle_service
        )
        self.web_search_handler = web_search_handler
        self.search_status_handler = search_status_handler

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult | None:
        capability = _detect_capability(request)
        if not capability:
            return None
        recommendation = recommend_packs_for_capability(
            request.text,
            pack_store=self.pack_store,
            pack_registry_discovery=self.pack_registry_discovery,
            capability=capability,
        )
        if not isinstance(recommendation, dict):
            recommendation = {
                "capability_required": capability,
                "capability_label": capability.replace("_", " "),
                "source_errors": [],
            }
        return self._result_from_recommendation(request, capability, recommendation)

    def continue_step(
        self,
        lifecycle: PackLifecycleResult | dict[str, Any],
        *,
        action: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AcquisitionResult:
        before = _lifecycle_dict(lifecycle)
        action_result = self.action_controller.dispatch(before, action=action, context=context or {})
        after = action_result.lifecycle_after if isinstance(action_result.lifecycle_after, dict) else before
        message = action_result.text
        if action_result.ok and after:
            message = f"{message} Current state: {after.get('state')}. {self._next_sentence(after)}".strip()
        next_step = self._next_step_from_lifecycle(after, pending_context=dict(context or {}))
        return AcquisitionResult(
            requested_capability=_clean(before.get("capability")),
            detected_capability=_clean(after.get("capability") or before.get("capability")),
            source_status=str(after.get("source") or before.get("source") or "unknown"),
            candidate_pack=_candidate_from_lifecycle(after or before),
            lifecycle_state=_clean(after.get("state") or before.get("state")) or None,
            missing_gate=_clean(after.get("missing_gate") or before.get("missing_gate")) or None,
            next_step=next_step,
            requires_confirmation=bool(next_step and next_step.requires_confirmation),
            user_message=message,
            safe_actions=_safe_actions_for_lifecycle(after or before),
            blocked_reason="action_refused" if action_result.refused else None,
            lifecycle=dict(after or before),
            action_result=action_result.to_dict(),
        )

    def use_if_usable(
        self,
        *,
        lifecycle: PackLifecycleResult | dict[str, Any],
        pack: dict[str, Any],
        adapter_declarations: list[dict[str, Any]],
        permission_grants: list[dict[str, Any]],
        operation: str = OP_DESCRIBE_CAPABILITY,
    ) -> AcquisitionResult:
        row = _lifecycle_dict(lifecycle)
        pack_id = _clean(pack.get("pack_id") or pack.get("canonical_id") or row.get("pack_id") or row.get("canonical_id"))
        pack_name = _clean(pack.get("name") or row.get("pack_name")) or "External pack"
        adapter_kind = _clean((adapter_declarations[0] if adapter_declarations else {}).get("kind"))
        if operation not in {OP_DESCRIBE_CAPABILITY, OP_DRY_RUN, "validate_grant"}:
            message = (
                "The pack has reached the managed-adapter lane, but that requested operation is not implemented yet. "
                "No arbitrary external code was run."
            )
            return AcquisitionResult(
                requested_capability=_clean(row.get("capability")),
                detected_capability=_clean(row.get("capability")),
                source_status=_clean(row.get("source")) or "installed",
                candidate_pack=_candidate_from_lifecycle(row),
                lifecycle_state=_clean(row.get("state")) or None,
                missing_gate=_clean(row.get("missing_gate")) or None,
                next_step=self._next_step_from_lifecycle(row),
                requires_confirmation=False,
                user_message=message,
                safe_actions=_safe_actions_for_lifecycle(row),
                blocked_reason="operation_unsupported",
                lifecycle=row,
            )
        invocation = self.managed_adapter_invoker.invoke(
            ManagedAdapterInvocationRequest(
                pack_id=pack_id,
                canonical_id=pack_id,
                pack_name=pack_name,
                adapter_kind=adapter_kind,
                operation=operation,
                parameters={},
                dry_run=True,
            ),
            lifecycle=row,
            pack=pack,
            adapter_declarations=adapter_declarations,
            permission_grants=permission_grants,
        )
        return AcquisitionResult(
            requested_capability=_clean(row.get("capability")),
            detected_capability=_clean(row.get("capability")),
            source_status=_clean(row.get("source")) or "installed",
            candidate_pack=_candidate_from_lifecycle(row),
            lifecycle_state=_clean(row.get("state")) or None,
            missing_gate=_clean(row.get("missing_gate")) or None,
            next_step=self._next_step_from_lifecycle(row),
            requires_confirmation=False,
            user_message=invocation.summary,
            safe_actions=("managed_adapter_invocation", "no_arbitrary_code", "no_gate_skipping"),
            blocked_reason=None if invocation.ok else (invocation.errors[0].code if invocation.errors else "invocation_failed"),
            lifecycle=row,
            invocation_result=invocation.to_dict(),
        )

    def _result_from_recommendation(
        self,
        request: AcquisitionRequest,
        capability: str,
        recommendation: dict[str, Any],
    ) -> AcquisitionResult:
        installed = recommendation.get("installed_pack") if isinstance(recommendation.get("installed_pack"), dict) else None
        recommended = recommendation.get("recommended_pack") if isinstance(recommendation.get("recommended_pack"), dict) else None
        blocked = recommendation.get("blocked_pack") if isinstance(recommendation.get("blocked_pack"), dict) else None
        scaffold = recommendation.get("scaffold_preview") if isinstance(recommendation.get("scaffold_preview"), dict) else None
        source_errors = [dict(row) for row in recommendation.get("source_errors") if isinstance(row, dict)] if isinstance(recommendation.get("source_errors"), list) else []

        if installed is not None:
            pack_id = _clean(installed.get("pack_id") or installed.get("canonical_id"))
            pack = self.pack_store.get_external_pack(pack_id) if pack_id and callable(getattr(self.pack_store, "get_external_pack", None)) else None
            lifecycle = self.lifecycle_service.evaluate(capability=capability, imported_pack=pack or installed).to_dict()
            source_status = "installed"
            candidate = installed
        elif recommended is not None:
            lifecycle = self.lifecycle_service.evaluate(capability=capability, catalog_pack=recommended).to_dict()
            source_status = "trusted_catalog_candidate"
            candidate = recommended
        elif blocked is not None or _has_source_trust_error(source_errors):
            candidate = blocked
            lifecycle = self.lifecycle_service.evaluate(capability=capability).to_dict()
            source_status = "source_trust_required"
        else:
            lead_search = self._search_source_leads(request=request, capability=capability)
            if lead_search is not None and lead_search.leads:
                lifecycle = self.lifecycle_service.evaluate(capability=capability).to_dict()
                source_status = "untrusted_source_leads"
                source_leads = tuple(lead.to_dict() for lead in lead_search.leads)
                next_step = AcquisitionNextStep(
                    action="source_approval_preview",
                    label="Review untrusted source leads before any source approval",
                    gate="source_trust",
                    requires_confirmation=True,
                    pending_context={
                        "capability": capability,
                        "origin_tool": "pack_acquisition_source_leads",
                        "source_leads": [dict(row) for row in source_leads],
                        "lead_search": lead_search.to_dict(),
                        "lifecycle": dict(lifecycle),
                        "lifecycle_action": "source_approval_preview",
                    },
                )
                message = self._render_user_message(
                    capability=capability,
                    recommendation=recommendation,
                    source_status=source_status,
                    candidate=None,
                    scaffold=None,
                    lifecycle=lifecycle,
                    source_errors=source_errors,
                    source_leads=source_leads,
                )
                return AcquisitionResult(
                    requested_capability=request.requested_capability,
                    detected_capability=capability,
                    source_status=source_status,
                    candidate_pack=None,
                    lifecycle_state=_clean(lifecycle.get("state")) or None,
                    missing_gate=_clean(lifecycle.get("missing_gate")) or None,
                    next_step=next_step,
                    requires_confirmation=True,
                    user_message=message,
                    safe_actions=(
                        "search_approved_catalogs_only",
                        "safe_web_search_source_leads",
                        "no_fetch",
                        "no_import",
                        "source_approval_required",
                    ),
                    blocked_reason="source_approval_required",
                    lifecycle=lifecycle,
                    recommendation=recommendation,
                    source_leads=source_leads,
                    lead_search=lead_search.to_dict(),
                    search_status=lead_search.search_status if isinstance(lead_search.search_status, dict) else None,
                )
            if isinstance(lead_search, SourceLeadSearchResult) and isinstance(lead_search.search_status, dict):
                recommendation = {**recommendation, "search_status": dict(lead_search.search_status)}
            if scaffold is None:
                scaffold = build_scaffold_preview(capability, user_goal=request.text) or _generic_scaffold_preview(
                    capability,
                    label=str(recommendation.get("capability_label") or capability).strip(),
                    user_goal=request.text,
                )
            lifecycle = self.lifecycle_service.evaluate(capability=capability, scaffold_preview=scaffold).to_dict()
            source_status = "no_candidate_scaffold_available"
            candidate = None

        next_step = self._next_step_from_lifecycle(
            lifecycle,
            pending_context=self._pending_context_for(capability, candidate, scaffold, source_errors),
        )
        message = self._render_user_message(
            capability=capability,
            recommendation=recommendation,
            source_status=source_status,
            candidate=candidate,
            scaffold=scaffold,
            lifecycle=lifecycle,
            source_errors=source_errors,
        )
        return AcquisitionResult(
            requested_capability=request.requested_capability,
            detected_capability=capability,
            source_status=source_status,
            candidate_pack=candidate,
            lifecycle_state=_clean(lifecycle.get("state")) or None,
            missing_gate=_clean(lifecycle.get("missing_gate")) or None,
            next_step=next_step,
            requires_confirmation=bool(next_step and next_step.requires_confirmation),
            user_message=message,
            safe_actions=_safe_actions_for_lifecycle(lifecycle),
            blocked_reason="source_trust_required" if source_status == "source_trust_required" else None,
            lifecycle=lifecycle,
            recommendation=recommendation,
            scaffold_preview=scaffold,
            search_status=(
                recommendation.get("search_status") if isinstance(recommendation.get("search_status"), dict) else None
            ),
        )

    def _search_source_leads(self, *, request: AcquisitionRequest, capability: str) -> SourceLeadSearchResult | None:
        status = (
            self.search_status_handler()
            if callable(self.search_status_handler)
            else {"available": False, "reason": "search_unavailable"}
        )
        if isinstance(status, dict) and not bool(status.get("available", False)):
            return SourceLeadSearchResult(
                ok=False,
                leads=(),
                searched=False,
                reason=str(status.get("reason") or "search_unavailable"),
                search_status=dict(status),
            )
        if not callable(self.web_search_handler):
            return SourceLeadSearchResult(
                ok=False,
                leads=(),
                searched=False,
                reason="search_unavailable",
                search_status=dict(status) if isinstance(status, dict) else None,
            )
        ok, payload = self.web_search_handler({"query": _source_lead_query(request.text, capability), "max_results": 5})
        body = dict(payload) if isinstance(payload, dict) else {}
        body.setdefault("ok", bool(ok))
        result = build_source_leads_from_safe_search(body, limit=5)
        if result.search_status is None and isinstance(status, dict):
            return SourceLeadSearchResult(
                ok=result.ok,
                leads=result.leads,
                searched=result.searched,
                reason=result.reason,
                search_status=dict(status),
            )
        return result

    def _pending_context_for(
        self,
        capability: str,
        candidate: dict[str, Any] | None,
        scaffold: dict[str, Any] | None,
        source_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {
            "capability": capability,
            "source_errors": source_errors,
            "origin_tool": "pack_acquisition",
        }
        if isinstance(candidate, dict):
            context["candidate_pack"] = dict(candidate)
            for key in ("source_id", "remote_id", "name", "artifact_type_hint"):
                if _clean(candidate.get(key)):
                    context[key] = _clean(candidate.get(key))
        if isinstance(scaffold, dict):
            context["scaffold_preview"] = dict(scaffold)
        return context

    def _next_step_from_lifecycle(
        self,
        lifecycle: dict[str, Any],
        *,
        pending_context: dict[str, Any] | None = None,
    ) -> AcquisitionNextStep | None:
        next_row = lifecycle.get("next_step") if isinstance(lifecycle.get("next_step"), dict) else {}
        action = _clean(next_row.get("action")) if isinstance(next_row, dict) else ""
        if not action:
            return None
        return AcquisitionNextStep(
            action=action,
            label=_clean(next_row.get("label")) or action.replace("_", " "),
            gate=_clean(next_row.get("gate")) or None,
            requires_confirmation=bool(next_row.get("required_confirmation", False)),
            pending_context={
                **dict(pending_context or {}),
                "lifecycle": dict(lifecycle),
                "lifecycle_action": _action_for_pending(action),
            },
        )

    @staticmethod
    def _next_sentence(lifecycle: dict[str, Any]) -> str:
        next_row = lifecycle.get("next_step") if isinstance(lifecycle.get("next_step"), dict) else {}
        label = _clean(next_row.get("label")) if isinstance(next_row, dict) else ""
        if label:
            return f"Next safe step: {label}."
        return "No further lifecycle action is available from this state."

    def _render_user_message(
        self,
        *,
        capability: str,
        recommendation: dict[str, Any],
        source_status: str,
        candidate: dict[str, Any] | None,
        scaffold: dict[str, Any] | None,
        lifecycle: dict[str, Any],
        source_errors: list[dict[str, Any]],
        source_leads: tuple[dict[str, Any], ...] = (),
    ) -> str:
        label = _clean(recommendation.get("capability_label") or capability.replace("_", " "))
        next_label = _clean((lifecycle.get("next_step") if isinstance(lifecycle.get("next_step"), dict) else {}).get("label"))
        if source_status == "trusted_catalog_candidate" and isinstance(candidate, dict):
            name = _clean(candidate.get("name")) or "a candidate pack"
            return (
                f"I don’t have that skill active yet. I found a candidate called {name} in the approved catalog. "
                "It is not installed or usable yet. Next safe step: preview it. Say yes to preview it safely."
            )
        if source_status == "source_trust_required":
            source = _first_source_error(source_errors)
            source_id = _clean(source.get("source_id")) or "unknown"
            reason = _clean(source.get("error")) or "source_not_allowlisted"
            return (
                f"I do not have {label} as an active capability yet. I found or considered a remote pack source, but it is not trusted yet "
                f"(source_id={source_id}, reason={reason}). Source approval/trust is required before fetch or import. "
                "I will not fetch it, and GitHub or catalog metadata does not make it safe."
            )
        if source_status == "untrusted_source_leads":
            lines = [
                (
                    f"I do not have {label} as an active capability yet. I did not find a trusted catalog pack, "
                    "but safe web search found untrusted source leads. These are not trusted and cannot be fetched/imported until you approve a source."
                ),
                "Search results are metadata only; I did not open pages, download archives, import packs, approve, enable, or grant permissions.",
            ]
            for index, lead in enumerate(source_leads[:5], start=1):
                title = _clean(lead.get("title")) or "Untitled lead"
                url = _clean(lead.get("url"))
                kind = _clean(lead.get("suspected_source_kind")) or "generic_web_result"
                lines.append(f"{index}. {title} [{kind}]")
                if url:
                    lines.append(f"   {url}")
            lines.append(
                "Next safe step: source approval preview. Say yes to review what source approval would require; I will not fetch or import anything."
            )
            return "\n".join(lines)
        if isinstance(scaffold, dict):
            title = _clean(scaffold.get("title")) or f"{label} draft skill"
            return (
                f"I do not have {label} active yet, and I did not find a trusted catalog skill ready to preview. "
                f"I can create a draft skill for review called {title}. It will not read files, install anything, connect to Google, or run code. "
                "Say yes to preview it safely."
            )
        return (
            f"I do not have {label} as a usable external capability yet. "
            f"{render_lifecycle_response(lifecycle)}"
        )


def _detect_capability(request: AcquisitionRequest) -> str | None:
    requested = _clean(request.requested_capability)
    if requested:
        return requested
    need = detect_pack_capability_need(request.text)
    if isinstance(need, dict) and _clean(need.get("capability")):
        return _clean(need.get("capability"))
    assessment = classify_capability_gap_request(request.text)
    if str(assessment.get("request_kind") or "").strip().lower() == "capability":
        return _clean(assessment.get("capability")) or _generic_capability_from_text(request.text)
    if looks_like_generic_external_capability_request(request.text):
        return _generic_capability_from_text(request.text)
    return None


def looks_like_generic_external_capability_request(text: str | None) -> bool:
    cleaned = " ".join(str(text or "").strip().lower().split())
    if not cleaned:
        return False
    prefixes = (
        "add a capability for ",
        "add capability for ",
        "add a skill for ",
        "install a skill for ",
        "install a skill that ",
        "create a skill for ",
        "create a skill that ",
    )
    return any(cleaned.startswith(prefix) and len(cleaned) > len(prefix) for prefix in prefixes)


def _generic_capability_from_text(text: str | None) -> str | None:
    cleaned = " ".join(str(text or "").strip().lower().split())
    if not cleaned:
        return None
    if any(term in cleaned for term in ("webpage", "web page", "browse", "browser", "web research")):
        return "browser_automation_planning"
    if any(term in cleaned for term in ("email", "inbox")):
        return "email_access"
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"custom_capability_{digest}"


def _source_lead_query(text: str | None, capability: str) -> str:
    cleaned = _clean(text)
    if cleaned:
        return f"{cleaned} external skill pack"
    return f"{capability.replace('_', ' ')} external skill pack"


def _generic_scaffold_preview(capability: str, *, label: str, user_goal: str | None) -> dict[str, Any]:
    title = (_clean(label) or capability.replace("_", " ")).title() + " Scaffold"
    scaffold_id = "scaffold-" + hashlib.sha256(
        json.dumps({"capability": capability, "title": title}, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "type": "skill_scaffold_preview",
        "scaffold_id": scaffold_id,
        "capability": capability,
        "title": title,
        "summary": "A preview-only text scaffold for a missing external capability.",
        "user_goal": _clean(user_goal) or None,
        "files_to_create": [
            {"path": "SKILL.md", "purpose": "Human-readable guidance and safety constraints."},
            {"path": "metadata.json", "purpose": "Capability labels and non-execution metadata."},
        ],
        "proposed_manifest": {
            "schema_version": 1,
            "kind": "skill",
            "id": capability.replace("_", "-"),
            "title": title,
            "capabilities": [capability],
            "phase": "preview_only",
            "creates_files": False,
            "executes_code": False,
            "permissions_granted": [],
            "managed_adapters": [],
        },
        "proposed_skill_doc": f"# {title}\n\nPreview-only generated text skill scaffold for {capability}.\n",
        "permissions_requested": [],
        "privacy_notes": ["No private data is read in preview.", "No raw user data is uploaded or logged."],
        "blocked_actions": [
            "No handler.py or executable files.",
            "No dependency installs.",
            "No network, OAuth, scraping, or arbitrary generated code execution.",
        ],
        "next_step": "After preview, a later confirmation can create a text-only review candidate in quarantine.",
        "creates_files": False,
        "executes_code": False,
    }


def _lifecycle_dict(value: PackLifecycleResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, PackLifecycleResult):
        return value.to_dict()
    return dict(value) if isinstance(value, dict) else {}


def _candidate_from_lifecycle(lifecycle: dict[str, Any]) -> dict[str, Any] | None:
    pack_id = _clean(lifecycle.get("pack_id") or lifecycle.get("canonical_id"))
    name = _clean(lifecycle.get("pack_name"))
    if not pack_id and not name:
        return None
    return {"pack_id": pack_id or None, "name": name or None}


def _safe_actions_for_lifecycle(lifecycle: dict[str, Any]) -> tuple[str, ...]:
    action = _clean((lifecycle.get("next_step") if isinstance(lifecycle.get("next_step"), dict) else {}).get("action"))
    actions = ["search_approved_catalogs_only"]
    if action:
        actions.append(action)
    actions.extend(["no_arbitrary_code", "no_gate_skipping"])
    return tuple(dict.fromkeys(actions))


def _has_source_trust_error(rows: list[dict[str, Any]]) -> bool:
    return any(_clean(row.get("error")) in {"source_not_allowlisted", "source_blocked_by_policy"} for row in rows)


def _first_source_error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def _action_for_pending(action: str) -> str:
    if action == "use":
        return "use_if_usable"
    if action == "configure":
        return "request_configuration"
    return action


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
