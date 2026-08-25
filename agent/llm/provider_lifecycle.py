"""Provider-neutral, immutable lifecycle planning for local model backends.

This module intentionally never invokes sudo, touches a system provider, or
changes a selected model.  Executors receive an already verified staged tree
and can only atomically point an assistant-owned ``current`` link at it after
smoke evidence succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import os
from typing import Mapping


class ProviderLifecycleState(StrEnum):
    UNAVAILABLE = "unavailable"
    EXTERNALLY_MANAGED = "externally_managed"
    ASSISTANT_MANAGED = "assistant_managed"
    BROKEN = "installed_but_broken"
    OUTDATED = "outdated"
    INCOMPATIBLE_ARTIFACT = "incompatible_artifact"
    READY = "ready"


@dataclass(frozen=True)
class ProviderProbe:
    provider_id: str
    executable: Path | None
    executable_ok: bool
    api_ok: bool
    managed_root: Path | None
    model_compatible: bool | None
    installed_version: str | None
    available_version: str | None


def classify_provider(probe: ProviderProbe) -> ProviderLifecycleState:
    if probe.executable is None:
        return ProviderLifecycleState.UNAVAILABLE
    if not probe.executable_ok or not probe.api_ok:
        return ProviderLifecycleState.BROKEN
    if probe.model_compatible is False:
        return ProviderLifecycleState.INCOMPATIBLE_ARTIFACT
    if probe.available_version and probe.available_version != probe.installed_version:
        return ProviderLifecycleState.OUTDATED
    if probe.managed_root and probe.executable.is_relative_to(probe.managed_root):
        return ProviderLifecycleState.ASSISTANT_MANAGED
    return ProviderLifecycleState.EXTERNALLY_MANAGED


def plain_status(probe: ProviderProbe) -> str:
    state = classify_provider(probe)
    if state is ProviderLifecycleState.INCOMPATIBLE_ARTIFACT:
        return "This provider works, but this model needs a separate compatible copy. Nothing is damaged or replaced."
    if state is ProviderLifecycleState.EXTERNALLY_MANAGED:
        return "This provider is installed outside Personal Agent. I can check it, but will not replace it."
    if state is ProviderLifecycleState.BROKEN:
        return "I found this provider, but it cannot complete a readiness check. I can stage a private working copy without changing the system copy."
    if state is ProviderLifecycleState.OUTDATED:
        return "A newer verified provider release is available. I can stage and test it first; your current provider and models remain unchanged."
    return "This provider is ready." if state is ProviderLifecycleState.READY else "This provider is not available."


def activate_verified_stage(*, managed_root: Path, version: str, smoke: Mapping[str, bool]) -> Path:
    """Atomically activate an assistant-owned verified version, retaining rollback."""
    if not all(bool(smoke.get(key)) for key in ("compatibility", "privacy", "tools", "performance", "canary")):
        raise ValueError("candidate_smoke_not_verified")
    candidate = managed_root / "versions" / version
    if not candidate.is_dir():
        raise ValueError("candidate_missing")
    current = managed_root / "current"
    rollback = managed_root / "rollback"
    managed_root.mkdir(parents=True, exist_ok=True)
    if current.is_symlink():
        previous = os.readlink(current)
        temporary_rollback = managed_root / ".rollback.next"
        temporary_rollback.unlink(missing_ok=True)
        temporary_rollback.symlink_to(previous)
        os.replace(temporary_rollback, rollback)
    temporary = managed_root / ".current.next"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(candidate)
    os.replace(temporary, current)
    return current
