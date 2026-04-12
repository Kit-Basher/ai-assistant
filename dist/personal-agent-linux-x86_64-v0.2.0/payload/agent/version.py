from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib import resources as importlib_resources
from pathlib import Path
import subprocess


DIST_NAME = "personal-agent"
_UNKNOWN_VERSION = "unknown"


@dataclass(frozen=True)
class BuildInfo:
    version: str
    version_source: str
    git_commit: str | None


def repo_root_path() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_text_file(path: Path) -> str | None:
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except (OSError, UnicodeError):
        return None
    return None


def _read_packaged_text(name: str) -> str | None:
    try:
        candidate = importlib_resources.files("agent").joinpath(name)
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    except (FileNotFoundError, ModuleNotFoundError, OSError, UnicodeError):
        return None
    return None


def read_version(*, repo_root: Path | None = None) -> tuple[str, str]:
    root = (repo_root or repo_root_path()).resolve()
    version = _read_text_file(root / "VERSION")
    if version:
        return version, "repo_file"
    version = _read_packaged_text("VERSION")
    if version:
        return version, "package_resource"
    try:
        version = str(importlib_metadata.version(DIST_NAME) or "").strip()
    except importlib_metadata.PackageNotFoundError:
        version = ""
    if version:
        return version, "distribution_metadata"
    return _UNKNOWN_VERSION, "unknown"


def read_git_commit(*, repo_root: Path | None = None, timeout_seconds: float = 0.3) -> str | None:
    root = (repo_root or repo_root_path()).resolve()
    git_marker = root / ".git"
    if not git_marker.exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout_seconds)),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = str(proc.stdout or "").strip()
    return value or None


def read_build_info(*, repo_root: Path | None = None, timeout_seconds: float = 0.3) -> BuildInfo:
    root = repo_root or repo_root_path()
    version, version_source = read_version(repo_root=root)
    git_commit = read_git_commit(repo_root=root, timeout_seconds=timeout_seconds)
    return BuildInfo(version=version, version_source=version_source, git_commit=git_commit)


__version__ = read_version()[0]
