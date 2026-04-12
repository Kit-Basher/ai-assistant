from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALLOWED_EXECUTION_MODES = {
    "in_process",
    "managed_background_task",
    "managed_adapter",
}
DEFAULT_EXECUTION_MODE = "in_process"
DEFAULT_SKILL_TYPE = "general"

DECLARABLE_CAPABILITIES = {
    "network_access",
    "filesystem_access",
    "secret_access",
    "background_task",
    "managed_adapter",
    "notifications",
    "model_access",
}

FORBIDDEN_PERSISTENCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "systemd_service_creation": (
        ".config/systemd",
        "systemctl --user",
        "daemon-reload",
        "override.conf",
    ),
    "detached_process_spawn": (
        "start_new_session=true",
        "preexec_fn=os.setsid",
        "os.setsid(",
        "nohup ",
    ),
    "background_daemon_thread": (
        "daemon=true",
    ),
    "independent_port_bind": (
        "socket.bind(",
        ".bind((",
    ),
}

FORBIDDEN_EXECUTION_CAPABILITIES = {
    "service_creation",
    "daemonization",
    "startup_install",
    "independent_port_bind",
    "detached_process",
}


@dataclass(frozen=True)
class SkillExecutionRequest:
    skill_id: str
    skill_type: str
    requested_execution_mode: str
    requested_capabilities: tuple[str, ...]
    persistence_requested: bool


@dataclass(frozen=True)
class SkillGovernanceDecision:
    allowed: bool
    requires_user_approval: bool
    effective_execution_mode: str
    reason: str
    source_issues: tuple[str, ...] = ()


def _normalize_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(sorted({str(item).strip().lower() for item in value if str(item).strip()}))


def parse_skill_execution_request(manifest: dict[str, Any], *, skill_id: str) -> SkillExecutionRequest:
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    requested_mode = str(execution.get("mode") or DEFAULT_EXECUTION_MODE).strip().lower() or DEFAULT_EXECUTION_MODE
    requested_capabilities = _normalize_str_list(execution.get("capabilities"))
    persistence_requested = bool(execution.get("persistence_requested", False))
    if requested_mode != DEFAULT_EXECUTION_MODE:
        persistence_requested = True
    return SkillExecutionRequest(
        skill_id=str(skill_id or "").strip() or "unknown_skill",
        skill_type=str(manifest.get("type") or DEFAULT_SKILL_TYPE).strip().lower() or DEFAULT_SKILL_TYPE,
        requested_execution_mode=requested_mode,
        requested_capabilities=requested_capabilities,
        persistence_requested=persistence_requested,
    )


def scan_skill_source_for_persistence(source_text: str) -> tuple[str, ...]:
    lowered = str(source_text or "").lower()
    issues: list[str] = []
    for issue_code, patterns in FORBIDDEN_PERSISTENCE_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            issues.append(issue_code)
    return tuple(sorted(set(issues)))


def evaluate_skill_execution_request(
    request: SkillExecutionRequest,
    *,
    source_issues: tuple[str, ...] = (),
    managed_background_task_approved: bool = False,
    managed_adapter_approved: bool = False,
) -> SkillGovernanceDecision:
    mode = str(request.requested_execution_mode or DEFAULT_EXECUTION_MODE).strip().lower() or DEFAULT_EXECUTION_MODE
    capabilities = tuple(str(item).strip().lower() for item in request.requested_capabilities if str(item).strip())
    source_issues = tuple(sorted(set(str(item).strip().lower() for item in source_issues if str(item).strip())))

    if source_issues:
        return SkillGovernanceDecision(
            allowed=False,
            requires_user_approval=False,
            effective_execution_mode=mode if mode in ALLOWED_EXECUTION_MODES else DEFAULT_EXECUTION_MODE,
            reason="forbidden_persistence_pattern",
            source_issues=source_issues,
        )
    if mode not in ALLOWED_EXECUTION_MODES:
        return SkillGovernanceDecision(
            allowed=False,
            requires_user_approval=False,
            effective_execution_mode=DEFAULT_EXECUTION_MODE,
            reason="invalid_execution_mode",
        )
    unknown_capabilities = sorted(
        item
        for item in capabilities
        if item not in DECLARABLE_CAPABILITIES and item not in FORBIDDEN_EXECUTION_CAPABILITIES
    )
    if unknown_capabilities:
        return SkillGovernanceDecision(
            allowed=False,
            requires_user_approval=False,
            effective_execution_mode=mode,
            reason="unknown_execution_capability",
        )
    forbidden_requested = sorted(item for item in capabilities if item in FORBIDDEN_EXECUTION_CAPABILITIES)
    if forbidden_requested:
        return SkillGovernanceDecision(
            allowed=False,
            requires_user_approval=False,
            effective_execution_mode=mode,
            reason="forbidden_execution_capability",
        )
    if mode == "in_process":
        if request.persistence_requested:
            return SkillGovernanceDecision(
                allowed=False,
                requires_user_approval=False,
                effective_execution_mode=mode,
                reason="persistent_behavior_not_allowed_in_process",
            )
        if any(item in {"background_task", "managed_adapter"} for item in capabilities):
            return SkillGovernanceDecision(
                allowed=False,
                requires_user_approval=False,
                effective_execution_mode=mode,
                reason="persistent_capability_not_allowed_in_process",
            )
        return SkillGovernanceDecision(
            allowed=True,
            requires_user_approval=False,
            effective_execution_mode=mode,
            reason="allowed",
        )
    if mode == "managed_background_task":
        if not managed_background_task_approved:
            return SkillGovernanceDecision(
                allowed=False,
                requires_user_approval=True,
                effective_execution_mode=mode,
                reason="managed_background_task_requires_approval",
            )
        return SkillGovernanceDecision(
            allowed=True,
            requires_user_approval=False,
            effective_execution_mode=mode,
            reason="approved_managed_background_task",
        )
    if not managed_adapter_approved:
        return SkillGovernanceDecision(
            allowed=False,
            requires_user_approval=True,
            effective_execution_mode=mode,
            reason="managed_adapter_requires_approval",
        )
    return SkillGovernanceDecision(
        allowed=True,
        requires_user_approval=False,
        effective_execution_mode=mode,
        reason="approved_managed_adapter",
    )


def serialize_execution_request(request: SkillExecutionRequest) -> dict[str, Any]:
    return {
        "skill_id": request.skill_id,
        "skill_type": request.skill_type,
        "requested_execution_mode": request.requested_execution_mode,
        "requested_capabilities": list(request.requested_capabilities),
        "persistence_requested": bool(request.persistence_requested),
    }


def serialize_governance_decision(decision: SkillGovernanceDecision) -> dict[str, Any]:
    return {
        "allowed": bool(decision.allowed),
        "requires_user_approval": bool(decision.requires_user_approval),
        "effective_execution_mode": decision.effective_execution_mode,
        "reason": decision.reason,
        "source_issues": list(decision.source_issues),
    }
