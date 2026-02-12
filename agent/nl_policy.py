from __future__ import annotations

from typing import Any

_ALLOWED_PERMISSIONS = {"db:read", "sys:read", "net:none"}


def can_run_nl_skill(
    skills: dict[str, Any],
    skill_name: str,
    function_name: str,
    requested_permissions: list[str] | None = None,
) -> tuple[bool, str]:
    skill = skills.get(skill_name)
    if not skill:
        return False, "skill_not_found"
    function = skill.functions.get(function_name) if getattr(skill, "functions", None) else None
    if not function:
        return False, "function_not_found"
    if not bool(getattr(function, "read_only", False)):
        return False, "function_not_read_only"

    permissions = list(requested_permissions or [])
    if not permissions:
        permissions = [p for p in (getattr(skill, "permissions", []) or []) if isinstance(p, str)]
    if any(p not in _ALLOWED_PERMISSIONS for p in permissions):
        return False, "permission_not_allowed"
    return True, "allowed"
