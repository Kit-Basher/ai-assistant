from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _is_repo_checkout(path: str) -> bool:
    if not path:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return (resolved / ".git").exists()


def _is_stable_runtime() -> bool:
    if os.getenv("PERSONAL_AGENT_INSTANCE", "").strip().lower() == "stable":
        return True
    return "runtime" in Path(sys.executable).parts


def _stable_package_root() -> Path | None:
    for entry in sys.path:
        candidate = Path.cwd() if not entry else Path(entry)
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if _is_repo_checkout(str(resolved)):
            continue
        version_path = resolved / "agent" / "version.py"
        if version_path.is_file():
            return resolved
    return None


def _load_stable_version(package_root: Path) -> str:
    version_path = package_root / "agent" / "version.py"
    spec = importlib.util.spec_from_file_location("agent.version", version_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load installed version module from {version_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent.version"] = module
    spec.loader.exec_module(module)
    return str(getattr(module, "__version__", "unknown"))


if _is_stable_runtime():
    package_root = _stable_package_root()
    if package_root is None:
        raise RuntimeError("stable runtime could not locate its installed package root")
    __path__ = [str(package_root / "agent")]
    if __spec__ is not None and __spec__.submodule_search_locations is not None:
        __spec__.submodule_search_locations[:] = __path__
    __version__ = _load_stable_version(package_root)
else:
    from agent.version import __version__

__all__ = ["__version__"]
