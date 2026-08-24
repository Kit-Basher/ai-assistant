#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any

from agent.mutation_plan import build_mutation_confirmation


ROOT = Path(__file__).resolve().parents[1]
LIVE_STATE = Path.home() / ".local" / "share" / "personal-agent"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def timing_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return {"samples": len(ordered), "median_ms": round(statistics.median(ordered), 3), "p95_ms": round(ordered[index], 3)}


def tiny_sprite_png() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0)
    raster = b"\x00" + b"\x00\x80\xff\xff" + b"\x00\xff\x80\xff"
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raster)) + chunk(b"IEND", b"")


def request(base: str, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"}, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read(4 * 1024 * 1024).decode("utf-8")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"_text": text}
            return int(response.status), parsed if isinstance(parsed, dict) else {}, (time.monotonic() - started) * 1000
    except urllib.error.HTTPError as exc:
        parsed = json.loads(exc.read().decode("utf-8") or "{}")
        return int(exc.code), parsed if isinstance(parsed, dict) else {}, (time.monotonic() - started) * 1000


def chat(base: str, text: str, suffix: str) -> tuple[int, dict[str, Any], float]:
    return request(base, "POST", "/chat", {
        "user_id": f"wp45-{suffix}",
        "session_id": f"wp45-{suffix}",
        "thread_id": f"wp45-{suffix}",
        "source_surface": "webui",
        "messages": [{"role": "user", "content": text}],
    })


def confirm_central_pack_mutation(base: str, action: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    scoped = {**payload, "actor_id": "wp5-isolated", "session_id": "wp5-isolated", "thread_id": "wp5-isolated"}
    status, preview, _ = request(base, "POST", f"/packs/{action}/plan", scoped)
    if status != 200 or not isinstance(preview.get("plan"), dict):
        return status, preview
    plan = preview["plan"]
    confirmation = build_mutation_confirmation(plan, confirmation_id=f"wp5-isolated-{plan['plan_id']}")
    status, applied, _ = request(base, "POST", f"/packs/{action}/apply", {**scoped, "mutation_plan": plan, "confirmation": confirmation})
    return status, applied


def confirm_capability_mutation(base: str, action: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    scoped = {**payload, "actor_id": "wp5-isolated", "session_id": "wp5-isolated", "thread_id": "wp5-isolated"}
    status, preview, _ = request(base, "POST", f"/packs/capabilities/{action}/plan", scoped)
    plan = preview.get("plan") if isinstance(preview.get("plan"), dict) else {}
    if status != 200 or not plan:
        return status, preview
    status, applied, _ = request(base, "POST", f"/packs/capabilities/{action}/apply", {
        "plan_id": plan.get("plan_id"), "binding_digest": plan.get("binding_digest"), "confirmed": True,
        "actor_id": "wp5-isolated", "session_id": "wp5-isolated", "thread_id": "wp5-isolated",
    })
    return status, applied


def make_record_usable(base: str, record: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    record_id = str(record.get("record_id") or "")
    status, snapshot, _ = request(base, "GET", "/packs/capabilities")
    full_record = next(
        (row for row in ((snapshot.get("result") or {}).get("packs") or []) if str(row.get("record_id") or "") == record_id),
        record,
    ) if status == 200 else record
    requested = list(((full_record.get("lifecycle") or {}).get("requested_permissions") or []))
    outcomes: list[dict[str, Any]] = []
    for gate, value in (("review_approved", True), ("grants", requested), ("enabled", True)):
        if gate == "grants" and not requested:
            continue
        status, body = confirm_capability_mutation(base, "gate", {"record_id": record_id, "gate": gate, "value": value})
        outcomes.append({"gate": gate, "status": status, "ok": bool(body.get("ok"))})
    return bool(record_id) and all(row["status"] == 200 and row["ok"] for row in outcomes), outcomes


def wait_ready(base: str, timeout: float = 40.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            _status, last, _elapsed = request(base, "GET", "/version", timeout=2.0)
            if last:
                return last
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"candidate_not_listening:{last}")


def start(port: int, state: Path, log: Path, *, selected_model: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update({
        "HOME": str(state.parents[2]),
        "AGENT_API_HOST": "127.0.0.1",
        "AGENT_API_PORT": str(port),
        "AGENT_DB_PATH": str(state / "agent.db"),
        "AGENT_LOG_PATH": str(state / "agent.jsonl"),
        "LLM_REGISTRY_PATH": str(state / "llm_registry.json"),
        "AGENT_SECRET_STORE_PATH": str(state / "secrets.enc.json"),
        "AGENT_PERMISSIONS_PATH": str(state / "permissions.json"),
        "AGENT_AUDIT_LOG_PATH": str(state / "audit.jsonl"),
        "AGENT_PACK_STORE_PATH": str(state / "packs"),
        "AGENT_EXTERNAL_PACKS_DIR": str(state / "external_packs"),
        "AGENT_MODEL_MANAGER_STATE_PATH": str(state / "model_manager_state.json"),
        "LLM_USAGE_STATS_PATH": str(state / "llm_usage_stats.json"),
        "LLM_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": selected_model.removeprefix("ollama:"),
        "AGENT_SAFE_MODE": "1",
        "LLM_ALLOW_REMOTE": "0",
        "TELEGRAM_ENABLED": "0",
        "TELEGRAM_REQUIRED": "0",
        "PERSONAL_AGENT_INSTANCE": "dev",
        "PERSONAL_AGENT_RUNTIME_ROOT": str(ROOT),
        "AGENT_WEBUI_DIST_PATH": str(ROOT / "agent" / "webui" / "dist"),
        "PERCEPTION_ROOTS": str(state),
        "PERSONAL_AGENT_GIT_COMMIT_OVERRIDE": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "PYTHONUNBUFFERED": "1",
    })
    handle = log.open("a", encoding="utf-8")
    try:
        return subprocess.Popen([sys.executable, "-m", "agent.api_server", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    finally:
        handle.close()


def stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="build/reports/wp45-isolated-candidate.json")
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--wp5-remote-reference-url", default="")
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="personal-agent-wp45-") as raw:
        temp = Path(raw)
        candidate_home = temp / "home"
        state = candidate_home / ".local/share/personal-agent"
        state.mkdir(parents=True)
        (candidate_home / ".config/personal-agent").mkdir(parents=True)
        (candidate_home / ".config/systemd/user").mkdir(parents=True)
        (candidate_home / ".config/systemd/user/personal-agent-api.service").write_text(
            "[Unit]\nDescription=Personal Agent isolated candidate\n", encoding="utf-8"
        )
        for name in ("agent.db", "llm_registry.json", "secrets.enc.json", "llm_usage_stats.json", "model_manager_state.json"):
            source = LIVE_STATE / name
            if source.is_file():
                shutil.copy2(source, state / name)
        (state / "packs").mkdir(exist_ok=True)
        registry_path = state / "llm_registry.json"
        before_registry = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        before_defaults = json.loads(registry_path.read_text(encoding="utf-8")).get("defaults", {})
        selected_model = str(before_defaults.get("default_model") or before_defaults.get("chat_model") or "ollama:qwen2.5:3b-instruct")
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        log = temp / "candidate.log"
        proc = start(port, state, log, selected_model=selected_model)
        try:
            version = wait_ready(base)
            try:
                with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5.0) as response:
                    observed_models = json.loads(response.read().decode("utf-8")).get("models") or []
                expected_installed = len({str(row.get("name") or row.get("model") or "").strip().lower() for row in observed_models if isinstance(row, dict) and str(row.get("name") or row.get("model") or "").strip()})
            except Exception:
                expected_installed = -1
            status, truth, elapsed = request(base, "GET", "/llm/models/truth")
            results.append({"name": "canonical_inventory", "passed": status == 200 and expected_installed >= 0 and truth.get("counts", {}).get("physically_installed") == expected_installed, "elapsed_ms": elapsed, "provider_observed": expected_installed})
            results.append({"name": "selection_truth", "passed": truth.get("selection", {}).get("effective_model") == selected_model})
            results.append({"name": "recommendation_truth", "passed": truth.get("recommendation", {}).get("default_general_assistant") == "qwen2.5:3b-instruct"})
            for index, (name, text, expected) in enumerate((
                ("presence", "u here?", "here"),
                ("model_status", "wht model r u using rn?", selected_model.removeprefix("ollama:").lower()),
                ("installed_inventory", "show models actually on this box pls", "qwen"),
                ("recommendation_chat", "which installed model is best for this assistant and why?", "__recommendation_truth__"),
                ("scout_chat", "what did model scout find?", "qwen2.5:3b"),
                ("why_selected", "why is the current model selected?", "__selection_reason__"),
                ("unavailable_models", "show unavailable or stale models", "not ready"),
                ("system_status", "give me a system status check", "ready"),
            )):
                status, body, elapsed = chat(base, text, f"case-{index}")
                message = str(body.get("message") or "").lower()
                if expected == "__recommendation_truth__":
                    matched = "qwen2.5:3b" in message or ("evidence" in message and "stale" in message and "evaluation" in message)
                elif expected == "__selection_reason__":
                    matched = selected_model.removeprefix("ollama:").lower() in message and any(term in message for term in ("selected", "using", "pinned"))
                else:
                    matched = expected in message
                results.append({"name": name, "passed": status == 200 and matched, "elapsed_ms": elapsed, "route": (body.get("meta") or {}).get("route"), "message": message[:300]})
            status, preview, _ = chat(base, "make ollama:qwen2.5:3b-instruct my default", "switch-deny")
            status2, denied, _ = chat(base, "no cancel that", "switch-deny")
            results.append({"name": "switch_preview_denial", "passed": status == 200 and status2 == 200 and any(term in str(denied.get("message") or "").lower() for term in ("cancel", "denied", "not"))})
            before_test_defaults = json.loads(registry_path.read_text(encoding="utf-8")).get("defaults", {})
            status, test_result, _ = chat(
                base,
                "test qwen2.5 3b for me but do not change my default",
                "test-deny",
            )
            after_test_defaults = json.loads(registry_path.read_text(encoding="utf-8")).get("defaults", {})
            test_message = str(test_result.get("message") or "").lower()
            results.append({
                "name": "human_spaced_model_test_without_default_change",
                "passed": (
                    status == 200
                    and "ollama:qwen2.5:3b-instruct" in test_message
                    and "couldn't find that model" not in test_message
                    and "without switching" in test_message
                    and before_test_defaults == after_test_defaults
                ),
            })
            status, generic, generic_ms = chat(base, "In one short sentence, explain why leaves look green.", "generic")
            results.append({"name": "generic_model_chat", "passed": status == 200 and bool(str(generic.get("message") or "").strip()), "elapsed_ms": generic_ms, "model": (generic.get("meta") or {}).get("model")})
            diagnostics_status, diagnostics, diagnostics_ms = request(base, "GET", "/diagnostics/export")
            diagnostics_text = json.dumps(diagnostics, sort_keys=True).lower()
            results.append({"name": "diagnostics_export", "passed": diagnostics_status == 200 and str((diagnostics.get("bundle") or {}).get("schema_version") or "") == "personal-agent.diagnostics.v1" and "bearer secret" not in diagnostics_text, "elapsed_ms": diagnostics_ms})
            backup_preview_status, _backup_preview, _ = chat(base, "back up Personal Agent", "portable-backup")
            backup_apply_status, backup_apply, backup_ms = chat(base, "yes", "portable-backup")
            backup_message = str(backup_apply.get("message") or "")
            portable_archives = list((state / "backups").glob("personal-agent-backup-*-portable-v2.tar.gz"))
            results.append({"name": "portable_backup_chat", "passed": backup_preview_status == 200 and backup_apply_status == 200 and "portable recovery archive" in backup_message.lower() and len(portable_archives) == 1, "elapsed_ms": backup_ms, "archive_count": len(portable_archives)})
            status, ui, ui_ms = request(base, "GET", "/")
            results.append({"name": "web_ui", "passed": status == 200 and "Personal Agent" in str(ui.get("_text") or ""), "elapsed_ms": ui_ms})
            status, before_packs, _ = request(base, "GET", "/packs/capabilities")
            before_pack_count = int(((before_packs.get("result") or {}).get("count") or 0))
            status, fetch_preview, _ = chat(base, "install this pack from https://example.invalid/review.zip", "pack-fetch-preview")
            results.append({
                "name": "remote_fetch_preview_only",
                "passed": status == 200 and bool((fetch_preview.get("setup") or {}).get("requires_confirmation")) and (fetch_preview.get("setup") or {}).get("mutated") is False,
            })
            status, after_preview, _ = request(base, "GET", "/packs/capabilities")
            results.append({"name": "remote_preview_zero_pack_mutation", "passed": status == 200 and int(((after_preview.get("result") or {}).get("count") or 0)) == before_pack_count})
            if args.wp5_remote_reference_url:
                remote_status, remote_created = confirm_central_pack_mutation(
                    base,
                    "fetch",
                    {"source": {"url": args.wp5_remote_reference_url, "kind": "github_archive" if "github.com" in args.wp5_remote_reference_url else "generic_archive_url"}},
                )
                remote_record = remote_created.get("record") if isinstance(remote_created.get("record"), dict) else {}
                remote_ready, remote_gates = make_record_usable(base, remote_record)
                remote_chat_status, remote_chat, _ = chat(base, "use the reference health helper to inspect this machine", "remote-pack-use")
                remote_selected = str((((remote_chat.get("setup") or {}).get("request_understanding") or {}).get("selected_capability_id") or ""))
                remote_rid = str(remote_record.get("record_id") or "")
                remote_disable_status, _ = confirm_capability_mutation(base, "gate", {"record_id": remote_rid, "gate": "enabled", "value": False})
                remote_remove_status, _ = confirm_capability_mutation(base, "remove", {"record_id": remote_rid, "private_data": "delete"})
                results.append({
                    "name": "workflow_a_remote_acquisition",
                    "passed": (
                        remote_status == 200
                        and remote_created.get("fetched_to_quarantine") is True
                        and remote_created.get("imported_for_review") is True
                        and remote_ready
                        and remote_chat_status == 200
                        and remote_selected.startswith("pack.reference-health.")
                        and remote_disable_status == 200
                        and remote_remove_status == 200
                    ),
                    "gates": remote_gates,
                    "selected": remote_selected,
                })
            else:
                results.append({"name": "workflow_a_remote_acquisition", "passed": False, "error": "--wp5-remote-reference-url is required for exact-candidate remote proof"})
            status, created = confirm_central_pack_mutation(base, "create", {"template": "declarative_native", "name": "WP5 Isolated Report", "capability_id": "system.status"})
            record = created.get("record") if isinstance(created.get("record"), dict) else {}
            record_id = str(record.get("record_id") or "")
            results.append({"name": "draft_quarantine_only", "passed": status == 200 and bool(record_id) and created.get("quarantine_only") is True and not created.get("approved") and not created.get("enabled"), "status": status, "response": created})
            gate_results: list[bool] = []
            for gate, value in (("review_approved", True), ("grants", ["system:read"]), ("enabled", True)):
                gate_status, gate_body = confirm_capability_mutation(base, "gate", {"record_id": record_id, "gate": gate, "value": value})
                gate_results.append(gate_status == 200 and bool(gate_body.get("ok")))
            results.append({"name": "separate_review_grant_enable_gates", "passed": bool(record_id) and all(gate_results), "gate_results": gate_results})
            status, pack_chat, _ = chat(base, "please use the WP5 isolated report", "pack-use")
            selected = str((((pack_chat.get("setup") or {}).get("request_understanding") or {}).get("selected_capability_id") or ""))
            results.append({"name": "dynamic_pack_chat_invocation", "passed": status == 200 and selected == "pack.wp5-isolated-report.report" and bool((pack_chat.get("setup") or {}).get("result")), "status": status, "selected": selected, "message": str(pack_chat.get("message") or "")[:500]})

            # Workflow B: a real exact-file grant, bounded derived index,
            # restart-persistent search, then immediate revocation.
            fixture = state / "library-export.json"
            fixture.write_text(json.dumps([{"title": "Dune", "author": "Frank Herbert"}, {"title": "Hyperion", "author": "Dan Simmons"}]), encoding="utf-8")
            local_status, local_created = confirm_central_pack_mutation(base, "create", {"template": "local_data_search", "name": "Isolated Library Search"})
            local_record = local_created.get("record") if isinstance(local_created.get("record"), dict) else {}
            local_ready, local_gates = make_record_usable(base, local_record)
            adapter = {"kind": "local_file_import", "purpose": "build a bounded private search index", "allowed_extensions": [".json"], "max_file_size_mb": 8, "path_policy": "user_selected_file_only", "stores_local_index": True, "network_allowed": False}
            grant_status, _ = confirm_central_pack_mutation(base, "grant", {"pack_id": "isolated-library-search", "adapter": adapter, "requested_path": str(fixture)})
            index_status, indexed = confirm_central_pack_mutation(base, "index", {"pack_id": "isolated-library-search", "record_id": str(local_record.get("record_id") or "")})
            search_status, search_body, _ = chat(base, "find Dune in my isolated library export", "local-data")
            local_selected = str((((search_body.get("setup") or {}).get("request_understanding") or {}).get("selected_capability_id") or ""))
            raw_fixture_leaked = fixture.read_bytes() in (state / "agent.db").read_bytes()
            stop(proc)
            proc = start(port, state, log, selected_model=selected_model)
            wait_ready(base)
            restart_search_status, restart_search, _ = chat(base, "search the isolated library export for Hyperion", "local-data-restart")
            revoke_status, _ = confirm_central_pack_mutation(base, "revoke", {"pack_id": "isolated-library-search"})
            denied_status, denied_search, _ = chat(base, "find Dune in my isolated library export", "local-data-denied")
            local_remove_status, _ = confirm_capability_mutation(base, "remove", {"record_id": str(local_record.get("record_id") or ""), "private_data": "delete"})
            results.append({
                "name": "workflow_b_local_data_search",
                "passed": (
                    local_status == 200 and local_ready and grant_status == 200 and index_status == 200
                    and search_status == 200 and local_selected == "pack.isolated-library-search.search"
                    and "Dune" in str((search_body.get("setup") or {}).get("result") or "")
                    and restart_search_status == 200 and "Hyperion" in str((restart_search.get("setup") or {}).get("result") or "")
                    and revoke_status == 200 and local_remove_status == 200 and not raw_fixture_leaked
                    and not bool((denied_search.get("setup") or {}).get("result"))
                ),
                "gates": local_gates,
                "indexed": bool(indexed.get("mutated")),
                "selected": local_selected,
                "revoked_status": revoke_status,
                "post_revoke_status": denied_status,
                "raw_fixture_leaked": raw_fixture_leaked,
            })

            # Workflow C: core-rendered raster metadata only. The browser
            # smoke runs while this exact visualizer is active.
            sprite = state / "presence-sprite.png"
            sprite.write_bytes(tiny_sprite_png())
            visual_status, visual_created = confirm_central_pack_mutation(base, "create", {
                "template": "presence_visualizer", "name": "Isolated Presence", "asset_path": str(sprite),
                "frame_width": 1, "frame_height": 1,
                "animations": {"idle": {"frames": [0], "frame_duration_ms": 300, "loop": True}, "thinking": {"frames": [0, 1], "frame_duration_ms": 120, "loop": True}},
            })
            visual_record = visual_created.get("record") if isinstance(visual_created.get("record"), dict) else {}
            visual_ready, visual_gates = make_record_usable(base, visual_record)
            visual_http, visual_truth, _ = request(base, "GET", "/packs/visualizer")
            results.append({"name": "workflow_c_visualizer_runtime", "passed": visual_status == 200 and visual_ready and visual_http == 200 and visual_truth.get("available") is True and visual_truth.get("scripts_allowed") is False and visual_truth.get("remote_requests") is False, "gates": visual_gates})

            # Workflow E: stage v2, compare exact authority, invalidate a
            # stale activation preview, activate atomically, then roll back.
            v1_status, v1_created = confirm_central_pack_mutation(base, "create", {"template": "declarative_native", "name": "Isolated Versioned Report", "pack_id": "isolated-versioned-report", "version": "1.0.0", "description": "Version one report", "capability_id": "system.status"})
            v1 = v1_created.get("record") if isinstance(v1_created.get("record"), dict) else {}
            v1_ready, _ = make_record_usable(base, v1)
            v2_status, v2_created = confirm_central_pack_mutation(base, "create", {"template": "declarative_native", "name": "Isolated Versioned Report", "pack_id": "isolated-versioned-report", "version": "2.0.0", "description": "Version two reviewed report", "capability_id": "models.inventory"})
            v2 = v2_created.get("record") if isinstance(v2_created.get("record"), dict) else {}
            v2_ready, _ = make_record_usable(base, v2)
            compare_status, compared, _ = request(base, "GET", f"/packs/capabilities/compare?from={urllib.parse.quote(str(v1.get('record_id') or ''))}&to={urllib.parse.quote(str(v2.get('record_id') or ''))}")
            scoped = {"record_id": str(v2.get("record_id") or ""), "actor_id": "wp5-isolated", "session_id": "wp5-isolated", "thread_id": "wp5-isolated"}
            stale_status, stale_preview, _ = request(base, "POST", "/packs/capabilities/activate/plan", scoped)
            _gate_status, _gate_body = confirm_capability_mutation(base, "gate", {"record_id": str(v2.get("record_id") or ""), "gate": "enabled", "value": False})
            stale_plan = stale_preview.get("plan") if isinstance(stale_preview.get("plan"), dict) else {}
            stale_apply_status, _stale_apply, _ = request(base, "POST", "/packs/capabilities/activate/apply", {"plan_id": stale_plan.get("plan_id"), "binding_digest": stale_plan.get("binding_digest"), "confirmed": True, **{key: scoped[key] for key in ("actor_id", "session_id", "thread_id")}})
            confirm_capability_mutation(base, "gate", {"record_id": str(v2.get("record_id") or ""), "gate": "enabled", "value": True})
            activate_status, _ = confirm_capability_mutation(base, "activate", {"record_id": str(v2.get("record_id") or "")})
            rollback_status, _ = confirm_capability_mutation(base, "rollback", {"record_id": str(v1.get("record_id") or "")})
            _, version_snapshot, _ = request(base, "GET", "/packs/capabilities")
            final_v1 = next((item for item in ((version_snapshot.get("result") or {}).get("packs") or []) if str(item.get("record_id") or "") == str(v1.get("record_id") or "")), {})
            results.append({"name": "workflow_e_update_rollback", "passed": v1_status == 200 and v2_status == 200 and v1_ready and v2_ready and compare_status == 200 and bool((compared.get("result") or {}).get("authority_changed")) and stale_status == 200 and stale_apply_status == 400 and activate_status == 200 and rollback_status == 200 and bool(final_v1.get("active"))})

            # Workflow F: the production chat planner must create one durable
            # mixed native/pack task rather than a hidden pack workflow.
            mixed_status, mixed_body, _ = chat(base, "list every installed model; then use the WP5 isolated report", "mixed-task")
            mixed_task = (mixed_body.get("setup") or {}).get("task") if isinstance((mixed_body.get("setup") or {}).get("task"), dict) else {}
            mixed_steps = mixed_task.get("steps") if isinstance(mixed_task.get("steps"), list) else []
            results.append({"name": "workflow_f_mixed_wp3_task", "passed": mixed_status == 200 and {str(step.get("capability_id") or "") for step in mixed_steps} >= {"models.inventory", "pack.wp5-isolated-report.report"} and all(str(step.get("verifier_status") or "") == "pass" for step in mixed_steps), "state": mixed_task.get("state"), "steps": [step.get("capability_id") for step in mixed_steps]})
            pack_timings: list[float] = []
            pack_latency_ok = True
            for sample in range(10):
                sample_status, sample_body, sample_ms = chat(base, "please use the WP5 isolated report", f"pack-bench-{sample}")
                pack_timings.append(sample_ms)
                pack_latency_ok = pack_latency_ok and sample_status == 200 and str((((sample_body.get("setup") or {}).get("request_understanding") or {}).get("selected_capability_id") or "")) == "pack.wp5-isolated-report.report"
            results.append({"name": "dynamic_pack_chat_latency", "passed": pack_latency_ok, **timing_summary(pack_timings)})
            browser_python = ROOT / ".venv-browser/bin/python"
            browser_output = temp / "browser.json"
            browser = subprocess.run(
                [str(browser_python), "scripts/wp5_browser_candidate_smoke.py", "--base-url", base, "--output", str(browser_output), "--expect-visualizer"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            browser_report = json.loads(browser_output.read_text(encoding="utf-8")) if browser_output.is_file() else {}
            results.append({"name": "browser_pack_ui", "passed": browser.returncode == 0 and int((browser_report.get("summary") or {}).get("failed") or 0) == 0, "summary": browser_report.get("summary") or {}, "error": browser.stderr[-500:]})
            disable_status, _disabled = confirm_capability_mutation(base, "gate", {"record_id": record_id, "gate": "enabled", "value": False})
            remove_status, removed = confirm_capability_mutation(base, "remove", {"record_id": record_id, "private_data": "delete"})
            visual_disable_status, _ = confirm_capability_mutation(base, "gate", {"record_id": str(visual_record.get("record_id") or ""), "gate": "enabled", "value": False})
            visual_remove_status, _ = confirm_capability_mutation(base, "remove", {"record_id": str(visual_record.get("record_id") or ""), "private_data": "delete"})
            version_remove_statuses = [
                confirm_capability_mutation(base, "remove", {"record_id": str(item.get("record_id") or ""), "private_data": "delete"})[0]
                for item in (v2, v1)
            ]
            _, visual_fallback, _ = request(base, "GET", "/packs/visualizer")
            status, final_packs, _ = request(base, "GET", "/packs/capabilities")
            results.append({
                "name": "disable_remove_authority_cleanup",
                "passed": disable_status == 200 and remove_status == 200 and bool(removed.get("ok")) and visual_disable_status == 200 and visual_remove_status == 200 and all(code == 200 for code in version_remove_statuses) and visual_fallback.get("available") is False and status == 200 and int(((final_packs.get("result") or {}).get("count") or 0)) == before_pack_count,
                "disable_status": disable_status,
                "remove_status": remove_status,
                "removed": removed,
            })
            stop(proc)
            proc = start(port, state, log, selected_model=selected_model)
            wait_ready(base)
            status, restarted_truth, _ = request(base, "GET", "/llm/models/truth")
            results.append({"name": "restart_reconstruction", "passed": status == 200 and expected_installed >= 0 and restarted_truth.get("counts", {}).get("physically_installed") == expected_installed})
            after_registry = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            after_defaults = json.loads(registry_path.read_text(encoding="utf-8")).get("defaults", {})
            results.append({"name": "zero_model_mutation", "passed": before_registry == after_registry and before_defaults == after_defaults})
            latency_output = temp / "latency.json"
            latency = subprocess.run([
                sys.executable, "scripts/wp45_latency_probe.py", "--base-url", base,
                "--label", "isolated-candidate", "--samples", str(args.latency_samples), "--output", str(latency_output),
            ], cwd=ROOT, check=False, capture_output=True, text=True, timeout=120)
            latency_report = json.loads(latency_output.read_text(encoding="utf-8")) if latency_output.is_file() else {}
            presence_p95 = ((latency_report.get("routes") or {}).get("presence") or {}).get("body_complete", {}).get("p95_ms")
            status_p95 = ((latency_report.get("routes") or {}).get("system_status") or {}).get("body_complete", {}).get("p95_ms")
            results.append({"name": "latency_budgets", "passed": latency.returncode == 0 and float(presence_p95 or 1e9) < 250 and float(status_p95 or 1e9) < 1000, "presence_p95_ms": presence_p95, "system_status_p95_ms": status_p95})
        finally:
            stop(proc)
        port_closed = True
        with socket.socket() as sock:
            port_closed = sock.connect_ex(("127.0.0.1", port)) != 0
        results.append({"name": "temporary_port_closed", "passed": port_closed})
        report = {
            "contract": "personal-agent.wp45-isolated-candidate.v2",
            "version": version,
            "results": results,
            "summary": {"passed": sum(bool(row.get("passed")) for row in results), "failed": sum(not bool(row.get("passed")) for row in results), "total": len(results)},
            "temporary_cleanup": {"port_closed": port_closed, "temporary_directory_removed_on_exit": True},
        }
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
