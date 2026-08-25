from pathlib import Path
import pytest

from agent.llm.provider_lifecycle import ProviderLifecycleState, ProviderProbe, activate_verified_stage, classify_provider


def test_external_and_incompatible_provider_are_never_assistant_activated(tmp_path: Path) -> None:
    external = ProviderProbe("ollama", Path("/usr/bin/ollama"), True, True, tmp_path, False, "1", "1")
    assert classify_provider(external) is ProviderLifecycleState.INCOMPATIBLE_ARTIFACT


def test_activation_requires_all_smoke_and_preserves_rollback(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp"; (root / "versions" / "old").mkdir(parents=True); (root / "versions" / "new").mkdir()
    (root / "current").symlink_to(root / "versions" / "old")
    with pytest.raises(ValueError, match="smoke"):
        activate_verified_stage(managed_root=root, version="new", smoke={"compatibility": True})
    activate_verified_stage(managed_root=root, version="new", smoke={key: True for key in ("compatibility", "privacy", "tools", "performance", "canary")})
    assert (root / "current").resolve().name == "new"
    assert (root / "rollback").resolve().name == "old"
