from __future__ import annotations

import os
from typing import Any


def resolve_allowed_path(raw_path: str, home_path: str) -> str | None:
    if not raw_path:
        return None
    expanded = os.path.expanduser(raw_path)
    if not expanded.startswith("/"):
        return None
    normalized = os.path.realpath(expanded)
    home_real = os.path.realpath(os.path.expanduser(home_path))
    if normalized == "/" or normalized.startswith("/var") or normalized.startswith(home_real):
        return normalized
    return None


def _run_du(target: str) -> dict[str, int]:
    return {}


def build_growth_report(
    baseline_snapshot: dict[str, Any],
    current_map: dict[str, int],
    target: str,
    label: str,
) -> str:
    if not current_map:
        return "Unable to read disk usage for that path."
    return f"No growth data available for {target} ({label})."
