from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_intent_doc_exists_and_states_external_pack_contract() -> None:
    path = REPO_ROOT / "docs" / "product" / "PROJECT_INTENT.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "local personal AI assistant runtime" in text
    assert "External skill packs are not bundled built-in abilities" in text
    assert "discover, preview, import, review, configure, permission, enable, and use" in text
    assert "Codex, development agents, and the repo control plane are not part of the normal runtime user workflow" in text
    assert "Managed adapters are the safety boundary" in text
    assert "The assistant layer is the user-facing layer" in text
    assert "The agent layer is the grounded runtime/computer/tool layer" in text
    assert "Core deterministic/native reports are factual agent outputs" in text
    assert "presentation adapters" in text


def test_runtime_spec_links_to_project_intent() -> None:
    text = (REPO_ROOT / "PRODUCT_RUNTIME_SPEC.md").read_text(encoding="utf-8")
    assert "docs/product/PROJECT_INTENT.md" in text


def test_readme_names_project_intent_as_top_source_of_truth_and_boundary() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    source_section = text.split("## Source Of Truth", 1)[1]
    assert "1. `docs/product/PROJECT_INTENT.md`" in source_section
    assert "The user interacts with the assistant layer" in text
    assert "The agent layer is the runtime/computer/tool boundary" in text
    assert "Direct native reports are raw factual outputs" in text


def test_project_status_is_snapshot_not_sole_source_of_truth() -> None:
    text = (REPO_ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8").lower()
    assert "not the sole" in text
    assert "treat it as the source of truth" not in text


def test_external_pack_format_doc_exists_and_states_safety_contract() -> None:
    path = REPO_ROOT / "docs" / "design" / "EXTERNAL_PACK_FORMAT.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "SKILL.md" in text
    assert "metadata.json" in text
    assert "manifest.json" in text
    assert "permissions.json" in text
    lowered = text.lower()
    assert "no arbitrary code execution" in lowered
    assert "source trust is not content trust" in lowered
    assert "managed adapters" in lowered


def test_current_checkpoint_doc_exists_and_names_quality_baseline() -> None:
    path = REPO_ROOT / "docs" / "operator" / "CURRENT_CHECKPOINT.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "34ed8c1" in text
    assert "external_pack_safety_smoke" in text
    lowered = text.lower()
    assert "current truth" in lowered
    assert "assistant layer" in lowered
    assert "agent layer" in lowered
    assert "direct native report commands remain raw and deterministic" in lowered
    assert "source trust is not content trust" in lowered
    assert "no arbitrary external code execution" in lowered


def test_managed_local_services_and_docker_helper_docs_state_safety_boundaries() -> None:
    local_services = REPO_ROOT / "docs" / "design" / "MANAGED_LOCAL_SERVICES.md"
    docker_helper = REPO_ROOT / "docs" / "design" / "DOCKER_HELPER_SKILL_PACK.md"
    assert local_services.is_file()
    assert docker_helper.is_file()

    local_text = local_services.read_text(encoding="utf-8").lower()
    helper_text = docker_helper.read_text(encoding="utf-8").lower()
    combined = f"{local_text}\n{helper_text}"

    assert "arbitrary docker commands" in combined
    assert "approved image" in local_text
    assert "approved services" in helper_text or "approved service" in helper_text
    assert "not the docker executor" in helper_text
    assert "external packs cannot trigger docker actions" in local_text or "external packs must not request container execution directly" in local_text
    assert "dockerfile builds from untrusted" in combined


def test_capability_setup_ux_doc_exists_and_keeps_backends_optional() -> None:
    path = REPO_ROOT / "docs" / "design" / "CAPABILITY_SETUP_UX.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "web search needs one extra local component" in lowered
    assert "docker and llama.cpp are optional" in lowered
    assert "preview -> confirm -> action" in lowered
    assert "no silent system installs" in lowered
    assert "technical details hidden" in lowered or "technical details remain" in lowered


def test_basics_ui_uses_capability_cards_without_searxng_details() -> None:
    path = REPO_ROOT / "desktop" / "src" / "components" / "BasicsTab.jsx"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "Chat" in text
    assert "Web search" in text
    assert "Telegram" in text
    assert "Local models" in text
    assert "Skills" in text
    assert "set up web search" in text
    assert "SearXNG" not in text
    assert "127.0.0.1" not in text
    assert "searxng/searxng" not in text


def test_assistant_agent_planning_doc_exists_and_draws_boundary() -> None:
    path = REPO_ROOT / "docs" / "design" / "ASSISTANT_AGENT_PLANNING.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8").lower()
    assert "assistant planner llm" in text
    assert "deterministic code remains the safety" in text
    assert "unknown capabilities or actions are rejected" in text
    assert "cannot execute" in text
    assert "yes/no pending confirmations" in text


def test_managed_action_recovery_doc_exists_and_states_rollback_boundary() -> None:
    path = REPO_ROOT / "docs" / "design" / "MANAGED_ACTION_RECOVERY.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8").lower()
    assert "action journal" in text
    assert "rollback owned changes only" in text
    assert "never silently mutate pre-existing user resources" in text
    assert "shell=false" in text
    assert "searxng" in text
    assert "persistent_managed_action_journal.md" in text


def test_persistent_managed_action_journal_doc_exists_and_limits_recovery_claims() -> None:
    path = REPO_ROOT / "docs" / "design" / "PERSISTENT_MANAGED_ACTION_JOURNAL.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8").lower()
    assert "use sqlite, not jsonl" in text
    assert "planned" in text
    assert "running" in text
    assert "verified" in text
    assert "rolled_back" in text
    assert "failed" in text
    assert "recovery_needed" in text
    assert "never store" in text
    assert "do not mutate" in text
    assert "not implemented now" in text
    assert "crash/restart recovery is not complete" in text


def test_managed_action_reliability_docs_exist_and_cover_required_flows() -> None:
    standard = REPO_ROOT / "docs" / "design" / "MANAGED_ACTION_RELIABILITY_STANDARD.md"
    audit = REPO_ROOT / "docs" / "operator" / "MANAGED_ACTION_RELIABILITY_AUDIT.md"
    assert standard.is_file()
    assert audit.is_file()

    standard_text = standard.read_text(encoding="utf-8").lower()
    audit_text = audit.read_text(encoding="utf-8").lower()
    assert "rollback owned changes only" in standard_text
    assert "no silent" in standard_text and "background services" in standard_text
    assert "expired confirmations must not execute" in standard_text

    for required in (
        "searxng",
        "model downloads",
        "provider/api key",
        "telegram",
        "pack source approval",
        "pack removal/source deletion cleanup",
        "pack quarantine fetch/import",
        "registry prune/rollback/hygiene/autoconfig/self-heal",
        "file operations",
    ):
        assert required in audit_text

    for remaining_gap in (
        "remaining gaps",
        "package install/directory creation",
        "notification send/test",
        "notification prune",
        "memory/bootstrap writes",
        "onboarding/preferences writes",
        "scoped preference reset/clear paths",
        "support bundle writes",
        "semantic-memory indexing and observe writes",
        "semantic-memory promotion risk",
        "medium risk remaining",
        "high risk remaining",
    ):
        assert remaining_gap in audit_text
    assert "persistent managed-action journal storage" in audit_text
    assert "minimal sqlite" in audit_text
    assert "existing managed-action flows are not converted" in audit_text
    assert "bulk reset/clear paths are still not wrapped" not in audit_text


def test_release_readiness_audit_exists_and_keeps_yellow_boundary() -> None:
    path = REPO_ROOT / "docs" / "operator" / "RELEASE_READINESS_AUDIT.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "yellow" in lowered
    assert "d807cb0" in text
    assert "controlled public trial" in lowered
    assert "persistent managed-action journal storage" in lowered
    assert "current flows are not converted" in lowered
    assert "scoped bulk preference reset/clear now has in-memory journal" in lowered
    assert "semantic memory must remain off by default" in lowered
    assert "package install and directory creation shell flows" in lowered
    assert "future filesystem writes" in lowered
    assert "python scripts/external_pack_safety_smoke.py" in text


def test_known_limits_do_not_contradict_supported_packaging_paths() -> None:
    text = (REPO_ROOT / "docs" / "operator" / "KNOWN_LIMITS.md").read_text(encoding="utf-8").lower()
    assert "debian/system packaging as the supported shipping path" not in text
    assert "stable release bundle" in text
    assert "optional debian" in text
    assert "legacy root/system install scripts" in text
