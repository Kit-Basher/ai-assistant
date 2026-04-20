from __future__ import annotations

import os
import importlib.machinery
import sys
from pathlib import Path


def _is_stable_runtime() -> bool:
    if os.getenv("PERSONAL_AGENT_INSTANCE", "").strip().lower() == "stable":
        return True
    executable = Path(sys.executable)
    executable_parts = executable.parts
    if "runtime" in executable_parts:
        return True
    if sys.prefix != sys.base_prefix:
        prefix = Path(sys.prefix)
        if (prefix / "pyvenv.cfg").is_file():
            return "runtime" in prefix.parts
    return False


def _is_repo_checkout(path: str) -> bool:
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return (resolved / ".git").exists()


def _bootstrap() -> None:
    if not _is_stable_runtime():
        return
    filtered: list[str] = []
    for entry in sys.path:
        if entry == "":
            continue
        if _is_repo_checkout(entry):
            continue
        filtered.append(entry)
    sys.path[:] = filtered

    class _StableRuntimeFinder:
        _top_level_modules = {"agent", "telegram_adapter", "memory", "skills"}

        def find_spec(self, fullname: str, path: list[str] | None = None, target: object | None = None):
            _ = target
            top_level = fullname.partition(".")[0]
            if top_level not in self._top_level_modules:
                return None
            search_paths = [entry for entry in sys.path if entry and not _is_repo_checkout(entry)]
            if path is not None:
                search_paths = [entry for entry in path if entry and not _is_repo_checkout(entry)]
            return importlib.machinery.PathFinder.find_spec(fullname, search_paths)

    if not any(type(finder).__name__ == "_StableRuntimeFinder" for finder in sys.meta_path):
        sys.meta_path.insert(0, _StableRuntimeFinder())


_bootstrap()
