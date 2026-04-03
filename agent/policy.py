from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str | None = None


def check_permissions(skill_permissions: list[str], requested: list[str]) -> bool:
    return all(perm in skill_permissions for perm in requested)


def requires_confirmation(action: dict[str, Any]) -> bool:
    # Treat delete/overwrite and ops restarts as sensitive.
    action_type = action.get("action_type")
    if action_type in {"delete", "overwrite", "ops_restart"}:
        return True
    return False


def evaluate_policy(skill_permissions: list[str], requested: list[str], action: dict[str, Any]) -> PolicyDecision:
    if not check_permissions(skill_permissions, requested):
        return PolicyDecision(False, False, "permission_denied")
    if requires_confirmation(action):
        return PolicyDecision(True, True, None)
    return PolicyDecision(True, False, None)
