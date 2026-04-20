from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_stable_runtime() -> bool:
    if os.getenv("PERSONAL_AGENT_INSTANCE", "").strip().lower() == "stable":
        return True
    prefix = Path(sys.prefix)
    if sys.prefix == sys.base_prefix:
        return False
    if not (prefix / "pyvenv.cfg").is_file():
        return False
    return "runtime" in prefix.parts


def _is_repo_checkout(path: str) -> bool:
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return (resolved / ".git").exists()


if _is_stable_runtime():
    filtered: list[str] = []
    for entry in sys.path:
        if entry == "":
            continue
        if _is_repo_checkout(entry):
            continue
        filtered.append(entry)
    sys.path[:] = filtered
