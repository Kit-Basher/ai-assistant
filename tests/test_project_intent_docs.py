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


def test_runtime_spec_links_to_project_intent() -> None:
    text = (REPO_ROOT / "PRODUCT_RUNTIME_SPEC.md").read_text(encoding="utf-8")
    assert "docs/product/PROJECT_INTENT.md" in text


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
