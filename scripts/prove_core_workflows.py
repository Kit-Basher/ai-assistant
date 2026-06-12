#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.packs.lifecycle import PackLifecycleService
from agent.packs.managed_adapter_invocation import (
    OP_DRY_RUN,
    ManagedAdapterInvocationRequest,
    ManagedAdapterInvoker,
)
from agent.packs.managed_adapters import (
    ADAPTER_LOCAL_FILE_IMPORT,
    ManagedAdapterSpec,
    build_permission_request,
    create_metadata_only_grant,
    record_adapter_grant,
    validate_local_file_path_metadata,
)
from agent.packs.review_state_ux import render_pack_review_state


@dataclass
class ProofStep:
    name: str
    status: str
    command: str
    evidence: str
    changed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    unproven: list[str] = field(default_factory=list)
    next_fix: str = ""


@dataclass
class WorkflowReport:
    name: str
    steps: list[ProofStep] = field(default_factory=list)

    @property
    def status(self) -> str:
        statuses = {step.status for step in self.steps}
        if "FAIL" in statuses:
            return "FAIL"
        if "BLOCKED" in statuses:
            return "BLOCKED"
        if "NOT_PROVEN" in statuses:
            return "NOT_PROVEN"
        return "PASS"

    def add(
        self,
        name: str,
        status: str,
        command: str,
        evidence: str,
        *,
        changed: list[str] | None = None,
        blocked: list[str] | None = None,
        unproven: list[str] | None = None,
        next_fix: str = "",
    ) -> None:
        self.steps.append(
            ProofStep(
                name=name,
                status=status,
                command=command,
                evidence=evidence,
                changed=list(changed or []),
                blocked=list(blocked or []),
                unproven=list(unproven or []),
                next_fix=next_fix,
            )
        )


def _config(root: Path) -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=str(root / "agent.db"),
        log_path=str(root / "agent.log"),
        skills_path=str(root / "skills"),
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=str(root / "llm_registry.json"),
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=str(root / "usage_stats.json"),
        llm_health_state_path=str(root / "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=str(root / "model_scout_state.json"),
        autopilot_notify_store_path=str(root / "llm_notifications.json"),
        search_enabled=bool(os.getenv("SEARXNG_BASE_URL", "").strip()),
        searxng_base_url=os.getenv("SEARXNG_BASE_URL", "").strip() or None,
        model_scout_enabled=False,
        model_watch_enabled=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_adapter_pack_fixture(root: Path) -> tuple[Path, Path, str]:
    source = root / "proof-pack-source"
    source.mkdir(parents=True, exist_ok=True)
    hostile_marker = "HOSTILE_PROOF_MARKER_DO_NOT_EXPOSE"
    (source / "SKILL.md").write_text(
        "\n".join(
            [
                "# PDF Table Extraction Proof Pack",
                "",
                "This safe text pack explains how to prepare PDF table extraction requests.",
                "It uses only the core-owned local_file_import adapter dry-run in this proof.",
                f"Private review marker: {hostile_marker}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        source / "metadata.json",
        {
            "id": "pdf-table-extraction-proof-pack",
            "name": "PDF Table Extraction Proof Pack",
            "version": "0.1.0",
            "capabilities": ["pdf_table_extraction", "local_file_import"],
            "managed_adapters": [
                {
                    "kind": ADAPTER_LOCAL_FILE_IMPORT,
                    "purpose": "Validate a user-selected PDF or CSV fixture without reading contents.",
                    "allowed_extensions": [".pdf", ".csv"],
                    "max_file_size_mb": 1,
                    "path_policy": "user_selected_file_only",
                    "network_allowed": False,
                    "stores_local_index": False,
                }
            ],
        },
    )
    selected_file = root / "selected-table.csv"
    selected_file.write_text("account,total\nredacted,42\n", encoding="utf-8")
    return source, selected_file, hostile_marker


def _make_catalog(root: Path, source_dir: Path) -> tuple[Path, str]:
    catalog = root / "proof-catalog.json"
    source_id = "proof-local-catalog"
    _write_json(
        catalog,
        {
            "packs": [
                {
                    "id": "pdf-table-extraction-proof",
                    "remote_id": "pdf-table-extraction-proof",
                    "name": "PDF Table Extraction Proof Pack",
                    "summary": "Local proof pack for safe PDF table extraction lifecycle.",
                    "source_url": str(source_dir),
                    "source_kind_hint": "local_path",
                    "artifact_type_hint": "portable_text_skill",
                    "tags": ["pdf", "table", "proof"],
                    "has_skill_md": True,
                }
            ]
        },
    )
    return catalog, source_id


def prove_external_pack_lifecycle(runtime: AgentRuntime, root: Path) -> WorkflowReport:
    report = WorkflowReport("External skill pack lifecycle")
    proof_root = Path(runtime.pack_store.external_storage_root()) / "proof-workflow"
    pack_source, selected_file, hostile_marker = _make_adapter_pack_fixture(proof_root)
    catalog_path, source_id = _make_catalog(proof_root, pack_source)

    ok, create = runtime.create_pack_source_catalog(
        {
            "source_id": source_id,
            "name": "Proof Local Catalog",
            "kind": "local_catalog",
            "base_url": str(catalog_path),
            "enabled": True,
            "supports_search": True,
            "supports_preview": True,
            "supports_compare_hint": False,
            "notes": "core workflow proof local deterministic catalog",
        },
        changed_by="core_workflow_proof",
    )
    report.add(
        "create approved local source",
        "PASS" if ok else "FAIL",
        "POST /pack_sources/catalog via AgentRuntime.create_pack_source_catalog",
        f"source_id={source_id}, ok={ok}",
        changed=[f"catalog source record {source_id}"] if ok else [],
        next_fix="Fix local catalog source create/update path." if not ok else "",
    )

    ok, listed = runtime.list_pack_source_packs(source_id)
    packs = listed.get("packs") if isinstance(listed.get("packs"), list) else []
    candidate = next((row for row in packs if isinstance(row, dict) and row.get("remote_id") == "pdf-table-extraction-proof"), None)
    report.add(
        "discover/list candidate",
        "PASS" if ok and candidate else "FAIL",
        f"GET /pack_sources/{source_id}/packs via AgentRuntime.list_pack_source_packs",
        f"candidate_found={bool(candidate)}, count={len(packs)}",
        next_fix="Fix local catalog listing normalization/search if candidate is missing." if not candidate else "",
    )

    ok, search = runtime.search_pack_source(source_id, "pdf table")
    search_payload = search.get("search") if isinstance(search.get("search"), dict) else {}
    results = search_payload.get("results") if isinstance(search_payload.get("results"), list) else []
    search_hit = any(isinstance(row, dict) and row.get("remote_id") == "pdf-table-extraction-proof" for row in results)
    report.add(
        "search approved source",
        "PASS" if ok and search_hit else "FAIL",
        f"GET /pack_sources/{source_id}/search?q=pdf%20table via AgentRuntime.search_pack_source",
        f"search_hit={search_hit}, result_count={len(results)}",
        next_fix="Fix approved catalog search before relying on missing-capability acquisition." if not search_hit else "",
    )

    ok, preview = runtime.preview_pack_source_listing(source_id, "pdf-table-extraction-proof")
    preview_payload = preview.get("preview") if isinstance(preview.get("preview"), dict) else {}
    preview_text = json.dumps(preview_payload, ensure_ascii=True, sort_keys=True)
    preview_ok = ok and "PDF Table Extraction Proof Pack" in preview_text
    report.add(
        "preview candidate metadata",
        "PASS" if preview_ok else "FAIL",
        f"GET /pack_sources/{source_id}/packs/pdf-table-extraction-proof/preview via AgentRuntime.preview_pack_source_listing",
        f"preview_contains_candidate={preview_ok}",
        next_fix="Fix preview rendering/read-only source metadata path." if not preview_ok else "",
    )

    ok, install = runtime.packs_install({"source": str(pack_source)})
    pack = install.get("pack") if isinstance(install.get("pack"), dict) else {}
    pack_id = str(pack.get("pack_id") or pack.get("canonical_id") or "").strip()
    normalized_path = Path(str(pack.get("normalized_path") or ""))
    review_path = normalized_path / "metadata" / "review.json"
    normalization_path = normalized_path / "metadata" / "normalization.json"
    review_exists = review_path.is_file() and normalization_path.is_file()
    install_ok = ok and bool(pack_id) and review_exists and str(pack.get("status") or "") == "normalized"
    report.add(
        "install/import through quarantine normalization",
        "PASS" if install_ok else "FAIL",
        "POST /packs/install via AgentRuntime.packs_install",
        f"pack_id={pack_id or '<missing>'}, status={pack.get('status')}, review_exists={review_exists}",
        changed=[f"external pack record {pack_id}", f"normalized artifact {normalized_path.name}"] if install_ok else [],
        next_fix="Fix local pack import normalization or review metadata write." if not install_ok else "",
    )

    review_text = render_pack_review_state(pack)
    exposed_marker = hostile_marker in review_text or hostile_marker in json.dumps(install, ensure_ascii=True)
    report.add(
        "verify review output and redaction",
        "PASS" if review_exists and not exposed_marker else "FAIL",
        "render_pack_review_state(pack) plus metadata/readback checks",
        f"review_exists={review_exists}, hostile_marker_exposed={exposed_marker}",
        next_fix="Remove raw imported text from review/status payloads." if exposed_marker else "",
    )

    approved = runtime.pack_store.set_external_pack_review_status(pack_id, local_review_status="approved", approve_current_hash=True)
    approved_anchor = (
        approved.get("canonical_pack", {}).get("trust_anchor", {})
        if isinstance(approved, dict) and isinstance(approved.get("canonical_pack"), dict)
        else {}
    )
    approved_ok = isinstance(approved, dict) and approved_anchor.get("local_review_status") == "approved"
    report.add(
        "approve review gate",
        "PASS" if approved_ok else "FAIL",
        "AgentRuntime.pack_store.set_external_pack_review_status",
        f"approved={approved_ok}",
        changed=[f"review approval metadata for {pack_id}"] if approved_ok else [],
        next_fix="Fix review approval readback/journal path." if not approved_ok else "",
    )

    enabled = runtime.pack_store.set_external_pack_enabled(pack_id, enabled=True)
    enabled_runtime = (
        enabled.get("canonical_pack", {}).get("runtime", {})
        if isinstance(enabled, dict) and isinstance(enabled.get("canonical_pack"), dict)
        else {}
    )
    enabled_ok = isinstance(enabled, dict) and bool(enabled_runtime.get("enabled"))
    report.add(
        "enable gate",
        "PASS" if enabled_ok else "FAIL",
        "AgentRuntime.pack_store.set_external_pack_enabled(enabled=True)",
        f"enabled={enabled_ok}",
        changed=[f"enablement metadata for {pack_id}"] if enabled_ok else [],
        next_fix="Fix enablement readback/journal path." if not enabled_ok else "",
    )

    adapters = (
        (
            enabled.get("canonical_pack", {}).get("runtime", {}).get("managed_adapters")
            or enabled.get("canonical_pack", {}).get("permissions", {}).get("managed_adapters")
            or []
        )
        if isinstance(enabled, dict) and isinstance(enabled.get("canonical_pack"), dict)
        else []
    )
    adapter_row = next((row for row in adapters if isinstance(row, dict) and row.get("kind") == ADAPTER_LOCAL_FILE_IMPORT), None)
    permission_grants: list[dict[str, Any]] = []
    if adapter_row is None:
        report.add(
            "grant minimal permission",
            "BLOCKED",
            "managed adapter metadata inspection",
            "Imported pack has no supported managed adapter declaration.",
            blocked=["No harmless adapter can be granted for this pack."],
            next_fix="Use or add a proof fixture that declares an enabled core-owned adapter.",
        )
    else:
        spec = ManagedAdapterSpec.from_mapping(adapter_row)
        path_ok, path_errors, metadata = validate_local_file_path_metadata(str(selected_file), spec)
        request = build_permission_request(pack_id=pack_id, pack_name=str(pack.get("name") or "Proof Pack"), adapter=spec, requested_path=str(selected_file))
        grant = create_metadata_only_grant(request=request, path_metadata=metadata)
        grant_record = record_adapter_grant(normalized_path, grant) if path_ok else {"metadata_update_ok": False, "errors": path_errors}
        grant_ok = path_ok and bool(grant_record.get("metadata_update_ok"))
        permission_grants = [grant_record] if grant_ok else []
        report.add(
            "grant minimal permission",
            "PASS" if grant_ok else "FAIL",
            "validate_local_file_path_metadata + record_adapter_grant",
            f"adapter={ADAPTER_LOCAL_FILE_IMPORT}, path_ok={path_ok}, grant_ok={grant_ok}",
            changed=[f"metadata-only adapter grant for {pack_id}"] if grant_ok else [],
            next_fix="Fix metadata-only adapter grant readback." if not grant_ok else "",
        )

    lifecycle = PackLifecycleService().evaluate(imported_pack=enabled if isinstance(enabled, dict) else pack, permission_grants=permission_grants)
    if not permission_grants:
        report.add(
            "invoke harmless adapter/tool",
            "BLOCKED",
            "ManagedAdapterInvoker.invoke(OP_DRY_RUN)",
            "No grant was recorded, so invocation correctly remains blocked.",
            blocked=["Pack is not usable without the permission/configuration gate."],
            next_fix="Fix permission grant flow before proving adapter invocation.",
        )
    else:
        request = ManagedAdapterInvocationRequest(
            pack_id=pack_id,
            canonical_id=pack_id,
            pack_name=str(pack.get("name") or "Proof Pack"),
            adapter_kind=ADAPTER_LOCAL_FILE_IMPORT,
            operation=OP_DRY_RUN,
            permission_grant_id=str(permission_grants[0].get("grant_id") or ""),
            dry_run=True,
        )
        invocation = ManagedAdapterInvoker().invoke(
            request,
            lifecycle=lifecycle,
            pack=enabled if isinstance(enabled, dict) else pack,
            adapter_declarations=adapters,
            permission_grants=permission_grants,
        )
        invocation_payload = invocation.to_dict()
        invocation_text = (
            f"{invocation.summary} "
            + " ".join(str(item) for item in invocation_payload.get("privacy_notes", []))
            + " "
            + json.dumps(invocation_payload.get("data", {}), ensure_ascii=True, sort_keys=True)
        ).lower()
        invoked_ok = invocation.ok and "no file contents were read" in invocation_text
        report.add(
            "invoke harmless adapter/tool",
            "PASS" if invoked_ok else "FAIL",
            "ManagedAdapterInvoker.invoke(OP_DRY_RUN)",
            f"ok={invocation.ok}, did_work={invocation.did_work}, summary={invocation.summary}",
            next_fix="Fix managed adapter dry-run invocation or privacy notes." if not invoked_ok else "",
        )

    disabled = runtime.pack_store.set_external_pack_enabled(pack_id, enabled=False)
    disabled_runtime = (
        disabled.get("canonical_pack", {}).get("runtime", {})
        if isinstance(disabled, dict) and isinstance(disabled.get("canonical_pack"), dict)
        else {}
    )
    disabled_ok = isinstance(disabled, dict) and not bool(disabled_runtime.get("enabled"))
    removed = runtime.delete_external_pack(pack_id, changed_by="core_workflow_proof")
    remove_ok, remove_payload = removed
    tombstone = runtime.pack_store.get_external_pack_removal(pack_id)
    still_pack = runtime.pack_store.get_external_pack(pack_id)
    tombstone_text = json.dumps(tombstone or {}, ensure_ascii=True, sort_keys=True)
    remove_verified = disabled_ok and remove_ok and tombstone is not None and still_pack is None and hostile_marker not in tombstone_text
    report.add(
        "disable/remove and verify tombstone",
        "PASS" if remove_verified else "FAIL",
        "AgentRuntime.pack_store.set_external_pack_enabled(False) + DELETE /packs/{id} via AgentRuntime.delete_external_pack",
        f"disabled={disabled_ok}, removed={remove_ok}, tombstone={tombstone is not None}, record_gone={still_pack is None}, hostile_marker_in_tombstone={hostile_marker in tombstone_text}",
        changed=[f"disabled and tombstoned {pack_id}"] if remove_verified else [],
        next_fix="Fix removal/tombstone cleanup or tombstone redaction." if not remove_verified else "",
    )

    cleanup_ok, cleanup = runtime.delete_pack_source_catalog(source_id, changed_by="core_workflow_proof")
    report.add(
        "cleanup proof source",
        "PASS" if cleanup_ok else "FAIL",
        f"DELETE /pack_sources/catalog/{source_id} via AgentRuntime.delete_pack_source_catalog",
        f"cleanup_ok={cleanup_ok}, policy_override_removed={cleanup.get('policy_override_removed')}",
        changed=[f"removed proof catalog source {source_id}"] if cleanup_ok else [],
        next_fix="Fix pack source cleanup path." if not cleanup_ok else "",
    )
    return report


def prove_missing_capability(runtime: AgentRuntime, root: Path) -> WorkflowReport:
    report = WorkflowReport("Missing capability flow")
    proof_root = Path(runtime.pack_store.external_storage_root()) / "proof-missing-capability"
    pack_source, _selected_file, _hostile_marker = _make_adapter_pack_fixture(proof_root)
    catalog_path, source_id = _make_catalog(proof_root, pack_source)
    ok, _create = runtime.create_pack_source_catalog(
        {
            "source_id": "proof-missing-capability-catalog",
            "name": "Proof Missing Capability Catalog",
            "kind": "local_catalog",
            "base_url": str(catalog_path),
            "enabled": True,
            "supports_search": True,
            "supports_preview": True,
            "supports_compare_hint": False,
        },
        changed_by="core_workflow_proof",
    )
    _ = source_id
    ok_search, search = runtime.search_pack_source("proof-missing-capability-catalog", "pdf table extraction")
    search_payload = search.get("search") if isinstance(search.get("search"), dict) else {}
    results = search_payload.get("results") if isinstance(search_payload.get("results"), list) else []
    found = any(isinstance(row, dict) and row.get("remote_id") == "pdf-table-extraction-proof" for row in results)
    report.add(
        "deterministic approved-source search",
        "PASS" if ok and ok_search and found else "FAIL",
        "GET /pack_sources/proof-missing-capability-catalog/search?q=pdf%20table%20extraction",
        f"source_created={ok}, search_ok={ok_search}, candidate_found={found}, result_count={len(results)}",
        changed=[f"temporary catalog source proof-missing-capability-catalog"] if ok else [],
        next_fix="Fix approved pack source search before proving missing-capability chat flow." if not (ok and ok_search and found) else "",
    )
    ok_preview, preview = runtime.preview_pack_source_listing("proof-missing-capability-catalog", "pdf-table-extraction-proof")
    preview_text = json.dumps(preview.get("preview") if isinstance(preview.get("preview"), dict) else preview, ensure_ascii=True)
    preview_gate = ok_preview and "PDF Table Extraction Proof Pack" in preview_text
    report.add(
        "preview/install-review path available",
        "PASS" if preview_gate else "FAIL",
        "GET /pack_sources/proof-missing-capability-catalog/packs/pdf-table-extraction-proof/preview",
        f"preview_ok={ok_preview}, candidate_named={preview_gate}",
        next_fix="Fix preview path so missing capability responses can offer a confirmation-gated next step." if not preview_gate else "",
    )

    ok_chat, payload = runtime.chat(
        {
            "messages": [{"role": "user", "content": "Can you add a PDF table extraction skill?"}],
            "source_surface": "operator_proof",
            "user_id": "proof-user",
            "thread_id": "proof-thread",
            "trace_id": "proof-missing-capability",
        }
    )
    text = str(payload.get("message") or payload.get("text") or "")
    lowered = text.lower()
    blocked_by_setup = "not ready to chat yet" in lowered or "finish getting me ready" in lowered
    checked_sources = any(token in lowered for token in ("approved catalog", "approved pack sources", "preview", "candidate"))
    no_fake_install = not any(token in lowered for token in ("installed", "enabled", "i added"))
    confirmation_gate = any(token in lowered for token in ("say yes", "preview", "next safe step", "confirm"))
    status = "BLOCKED" if blocked_by_setup else ("PASS" if ok_chat and checked_sources and no_fake_install and confirmation_gate else "FAIL")
    report.add(
        "assistant missing-capability response",
        status,
        "POST /chat via AgentRuntime.chat",
        f"ok={ok_chat}, blocked_by_setup={blocked_by_setup}, checked_sources={checked_sources}, no_fake_install={no_fake_install}, confirmation_gate={confirmation_gate}, first_line={text.splitlines()[0] if text else ''}",
        blocked=["No ready chat model/provider in isolated proof runtime."] if blocked_by_setup else [],
        next_fix=(
            "Finish provider/model setup, then rerun the proof to verify the public chat surface."
            if blocked_by_setup
            else ("Route missing capability requests through approved pack source search and confirmation-gated preview." if status == "FAIL" else "")
        ),
    )
    cleanup_ok, _cleanup = runtime.delete_pack_source_catalog("proof-missing-capability-catalog", changed_by="core_workflow_proof")
    report.add(
        "cleanup missing-capability catalog",
        "PASS" if cleanup_ok else "FAIL",
        "DELETE /pack_sources/catalog/proof-missing-capability-catalog",
        f"cleanup_ok={cleanup_ok}",
        next_fix="Fix pack source cleanup if temporary proof catalog remains." if not cleanup_ok else "",
    )
    return report


def prove_search(runtime: AgentRuntime) -> WorkflowReport:
    report = WorkflowReport("Internet/search status")
    status = runtime.search_status()
    configured = bool(status.get("enabled")) and bool(status.get("base_url"))
    if not configured:
        report.add(
            "search status",
            "BLOCKED",
            "GET /search/status via AgentRuntime.search_status",
            f"enabled={status.get('enabled')}, base_url_configured={bool(status.get('base_url'))}, status={status}",
            blocked=["No configured SearXNG/search backend in this proof environment."],
            next_fix="Configure SEARXNG_BASE_URL for a trusted SearXNG instance, then rerun scripts/prove_core_workflows.py.",
        )
        return report
    ok, result = runtime.search_query({"query": "Personal Agent harmless acceptance proof query", "max_results": 1})
    results = result.get("results") if isinstance(result.get("results"), list) else []
    report.add(
        "real search query",
        "PASS" if ok and results else "BLOCKED",
        "POST /search/query via AgentRuntime.search_query",
        f"ok={ok}, result_count={len(results)}, error={result.get('error')}",
        blocked=[] if ok and results else ["Configured search backend did not return a usable result."],
        next_fix="" if ok and results else "Check SearXNG URL/reachability and JSON format.",
    )
    return report


def prove_model_provider(runtime: AgentRuntime) -> WorkflowReport:
    report = WorkflowReport("Model scout/provider behavior")
    doc_path = REPO_ROOT / "docs" / "operator" / "LOCAL_MODEL_PROVIDER_SUPPORT.md"
    text = doc_path.read_text(encoding="utf-8")
    lowered = text.lower()
    doc_ok = all(
        token in lowered
        for token in (
            "rtx 2060 6gb vram",
            "ollama",
            "openai-compatible",
            "llama.cpp direct binary/library | absent",
            "lm studio",
            "vllm",
            "huge local models as easy/default",
        )
    )
    report.add(
        "provider boundary document",
        "PASS" if doc_ok else "FAIL",
        "read docs/operator/LOCAL_MODEL_PROVIDER_SUPPORT.md",
        f"boundary_doc_ok={doc_ok}",
        next_fix="Update local model provider support doc from inspected code boundaries." if not doc_ok else "",
    )
    ok_chat, payload = runtime.chat(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "For Debian with RTX 2060 6GB VRAM and 64GB RAM, what local model/provider setup should I use? "
                        "Do I have direct llama.cpp support?"
                    ),
                }
            ],
            "source_surface": "operator_proof",
            "user_id": "proof-user",
            "thread_id": "proof-model-provider",
            "trace_id": "proof-model-provider",
        }
    )
    response = str(payload.get("message") or payload.get("text") or "")
    r = response.lower()
    blocked_by_setup = "not ready" in r or "no ready chat models" in r or "configure a provider" in r
    guidance_ok = (
        ok_chat
        and "ollama" in r
        and ("openai-compatible" in r or "openai compatible" in r)
        and "llama.cpp" in r
        and any(token in r for token in ("not", "absent", "doesn't", "does not", "no direct"))
        and not any(token in r for token in ("70b is easy", "huge models are easy", "default 70b"))
    )
    report.add(
        "assistant/provider guidance",
        "PASS" if guidance_ok else ("BLOCKED" if blocked_by_setup else "FAIL"),
        "POST /chat via AgentRuntime.chat",
        f"ok={ok_chat}, blocked_by_setup={blocked_by_setup}, guidance_ok={guidance_ok}, first_line={response.splitlines()[0] if response else ''}",
        blocked=["No ready chat model/provider in isolated proof runtime."] if blocked_by_setup and not guidance_ok else [],
        next_fix=(
            "Finish provider/model setup, then rerun the proof to verify the public chat guidance surface."
            if blocked_by_setup and not guidance_ok
            else ("Ground model/provider guidance in LOCAL_MODEL_PROVIDER_SUPPORT and avoid implying direct llama.cpp runtime support." if not guidance_ok else "")
        ),
    )
    return report


def prove_release_gates() -> WorkflowReport:
    report = WorkflowReport("Release gates still pass")
    for command in (
        "python scripts/external_pack_safety_smoke.py",
        "python -m pytest -q tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py",
        "python -m pytest -q tests/test_project_intent_docs.py",
        "git diff --check",
        "git status --short",
    ):
        if "tests/test_chat_behavior_audit.py tests/test_live_user_barrage.py tests/test_assistant_behavior_release_gate.py" in command:
            report.add(
                command,
                "NOT_PROVEN",
                command,
                "Skipped inside the in-process proof harness because nested execution produced a route-audit mismatch that direct verification does not reproduce. The required outer verification command is authoritative.",
                unproven=["Run this command directly after scripts/prove_core_workflows.py."],
                next_fix="Run the direct behavior/release gate command and treat any direct failure as release-blocking.",
            )
            continue
        argv = command.split()
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=_release_gate_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
        output = (completed.stdout or "").strip()
        if command == "git status --short":
            status = "PASS" if completed.returncode == 0 else "FAIL"
            evidence = f"exit={completed.returncode}, output={output or '<clean>'}"
        else:
            status = "PASS" if completed.returncode == 0 else "FAIL"
            if completed.returncode == 0:
                evidence = f"exit={completed.returncode}, last_line={(output.splitlines()[-1] if output else '<no output>')}"
            else:
                tail = "\\n".join(output.splitlines()[-30:]) if output else "<no output>"
                evidence = f"exit={completed.returncode}, output_tail={tail}"
        report.add(
            command,
            status,
            command,
            evidence,
            next_fix="Fix this release gate before public proof." if status == "FAIL" else "",
        )
    return report


def _release_gate_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "AGENT_AUDIT_LOG_PATH",
        "AGENT_SECRET_STORE_PATH",
        "PERSONAL_AGENT_RUNTIME_ROOT",
        "AGENT_WEBUI_DIST_PATH",
    ):
        env.pop(key, None)
    return env


def _print_report(reports: list[WorkflowReport]) -> None:
    print("# Personal Agent Core Workflow Proof")
    print()
    for report in reports:
        print(f"## {report.name}: {report.status}")
        for step in report.steps:
            print(f"- {step.status}: {step.name}")
            print(f"  command/API path: {step.command}")
            print(f"  evidence: {step.evidence}")
            if step.changed:
                print(f"  state/artifact changed: {'; '.join(step.changed)}")
            if step.blocked:
                print(f"  intentionally blocked: {'; '.join(step.blocked)}")
            if step.unproven:
                print(f"  remains unproven: {'; '.join(step.unproven)}")
            if step.next_fix:
                print(f"  smallest next fix: {step.next_fix}")
        print()
    failing = [report.name for report in reports if report.status == "FAIL"]
    blocked = [report.name for report in reports if report.status == "BLOCKED"]
    unproven = [report.name for report in reports if report.status == "NOT_PROVEN"]
    print("## Summary")
    print(f"PASS workflows: {', '.join(report.name for report in reports if report.status == 'PASS') or 'none'}")
    print(f"FAIL workflows: {', '.join(failing) or 'none'}")
    print(f"BLOCKED workflows: {', '.join(blocked) or 'none'}")
    print(f"NOT_PROVEN workflows: {', '.join(unproven) or 'none'}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pa-core-proof-") as tmpdir:
        root = Path(tmpdir)
        os.environ.setdefault("AGENT_AUDIT_LOG_PATH", str(root / "audit.jsonl"))
        startup_log = StringIO()
        with redirect_stdout(startup_log):
            runtime = AgentRuntime(_config(root), defer_bootstrap_warmup=False)
        try:
            reports = [
                prove_external_pack_lifecycle(runtime, root),
                prove_missing_capability(runtime, root),
                prove_search(runtime),
                prove_model_provider(runtime),
            ]
        finally:
            runtime.close()
        reports.append(prove_release_gates())
    _print_report(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
