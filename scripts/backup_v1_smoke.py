#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8765"
MAX_BACKUP_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_BACKUP_FILE_BYTES = 8 * 1024 * 1024

EXPECTED_FILES = {
    "manifest.json",
    "state_database_summary.json",
    "preferences_summary.json",
    "memory_anchors_summary.json",
    "pack_metadata_summary.json",
    "runtime_config_summary.json",
    "executor_registry_journal_summary.json",
    "diagnostics_summary.json",
    "support_bundle_style_summary.json",
    "backup_summary.json",
    "confirmation_transactions.sqlite3",
}


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    command: str


def _request_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body or "{}")
        except json.JSONDecodeError:
            parsed = {"body": body}
        if isinstance(parsed, dict):
            parsed.setdefault("http_status", exc.code)
            return parsed
        return {"http_status": exc.code, "body": body}
    parsed = json.loads(body or "{}")
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _post_chat(base_url: str, message: str, *, thread_id: str = "backup-v1", timeout: float = 30.0) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url.rstrip('/')}/chat",
        payload={"message": message, "user_id": "backup-v1-smoke", "thread_id": thread_id},
        timeout=timeout,
    )


def _assistant_text(payload: dict[str, Any]) -> str:
    for key in ("response", "message", "text", "assistant"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ("response", "message", "text", "summary"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, sort_keys=True)[:1000]


def _runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else {}
    if setup:
        return setup
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    runtime_payload = data.get("runtime_payload") if isinstance(data.get("runtime_payload"), dict) else {}
    if runtime_payload:
        return runtime_payload
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    runtime_payload = meta.get("runtime_payload") if isinstance(meta.get("runtime_payload"), dict) else {}
    return runtime_payload if isinstance(runtime_payload, dict) else {}


def _pass(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=True, evidence=evidence.strip()[:1000], command=command)


def _fail(name: str, evidence: str, command: str) -> Check:
    return Check(name=name, ok=False, evidence=evidence.strip()[:1600], command=command)


def _git_status_short() -> str:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True, timeout=10)
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def _scan_json_files(path: Path) -> str:
    return "\n".join(item.read_text(encoding="utf-8", errors="replace") for item in sorted(path.glob("*.json")))


def _artifact_size(path: Path) -> tuple[int, dict[str, int]]:
    sizes: dict[str, int] = {}
    total = 0
    for item in sorted(path.glob("*.json")):
        size = item.stat().st_size
        sizes[item.name] = size
        total += size
    return total, sizes


def run(base_url: str, timeout: float) -> list[Check]:
    checks: list[Check] = []
    baseline_git = _git_status_short()

    preview = _post_chat(base_url, "back up the assistant", timeout=timeout)
    preview_text = _assistant_text(preview)
    preview_payload = _runtime_payload(preview)
    plan = preview_payload.get("canonical_plan") if isinstance(preview_payload.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("backup preview uses Plan Mode v2", f"plan_id={plan.get('plan_id')} executor_status={plan.get('executor_status')}", 'POST /chat {"message": "back up the assistant"}')
        if "Plan Mode v2" in preview_text and plan.get("action_type") == "operator.backup" and plan.get("executor_status") == "enabled"
        else _fail("backup preview uses Plan Mode v2", json.dumps(preview, sort_keys=True)[:1400], 'POST /chat {"message": "back up the assistant"}')
    )

    confirmed = _post_chat(base_url, "yes", timeout=timeout)
    confirmed_text = _assistant_text(confirmed)
    runtime = _runtime_payload(confirmed)
    executor_result = runtime.get("executor_result") if isinstance(runtime.get("executor_result"), dict) else {}
    details = executor_result.get("details") if isinstance(executor_result.get("details"), dict) else {}
    artifact_path = Path(str(details.get("artifact_path") or ""))
    checks.append(
        _pass("confirmation executes through executor registry", json.dumps(executor_result, sort_keys=True)[:1000], 'POST /chat {"message": "yes"}')
        if bool(executor_result.get("ok"))
        and bool(executor_result.get("mutated"))
        and str(executor_result.get("executor_id")) == "operator.backup.v1"
        and bool(executor_result.get("journal_id"))
        else _fail("confirmation executes through executor registry", json.dumps(confirmed, sort_keys=True)[:1600], 'POST /chat {"message": "yes"}')
    )
    checks.append(
        _pass("chat response gives local backup path", confirmed_text, 'POST /chat {"message": "yes"}')
        if str(artifact_path) and str(artifact_path) in confirmed_text
        else _fail("chat response gives local backup path", confirmed_text, 'POST /chat {"message": "yes"}')
    )
    checks.append(
        _pass("backup artifact exists", str(artifact_path), "inspect backup path")
        if artifact_path.exists() and artifact_path.is_dir()
        else _fail("backup artifact exists", str(artifact_path), "inspect backup path")
    )

    if artifact_path.exists():
        manifest_path = artifact_path / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        included = set(str(item) for item in manifest.get("included_files", []) if str(item).strip())
        missing = sorted(EXPECTED_FILES - included)
        checks.append(
            _pass("manifest exists and lists expected bounded files", json.dumps(manifest, sort_keys=True)[:1000], "inspect manifest.json")
            if manifest.get("backup_schema_version") == "backup.v1"
            and manifest.get("restore_status") in {"dry_run_only", "restore_v1_allowlisted_preferences_only"}
            and manifest.get("live_restore") in {"restore_not_enabled", "restore_v1_allowlisted_preferences_only"}
            and not missing
            else _fail("manifest exists and lists expected bounded files", f"missing={missing}; manifest={json.dumps(manifest, sort_keys=True)[:1200]}", "inspect manifest.json")
        )
        confirmation_snapshot = artifact_path / "confirmation_transactions.sqlite3"
        confirmation_ok = False
        if confirmation_snapshot.is_file() and not confirmation_snapshot.is_symlink():
            try:
                with sqlite3.connect(f"file:{confirmation_snapshot}?mode=ro", uri=True) as connection:
                    integrity = connection.execute("PRAGMA integrity_check").fetchone()
                    table = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'confirmation_transactions'"
                    ).fetchone()
                confirmation_ok = bool(integrity and integrity[0] == "ok" and table)
            except sqlite3.Error:
                confirmation_ok = False
        checks.append(
            _pass("durable confirmation snapshot is standalone and valid", "integrity=ok", "inspect confirmation snapshot")
            if confirmation_ok
            else _fail("durable confirmation snapshot is standalone and valid", "missing or invalid", "inspect confirmation snapshot")
        )
        missing_files = sorted(name for name in EXPECTED_FILES if not (artifact_path / name).is_file())
        checks.append(
            _pass("expected bounded summary files exist", ", ".join(sorted(EXPECTED_FILES)), "inspect backup files")
            if not missing_files
            else _fail("expected bounded summary files exist", f"missing_files={missing_files}", "inspect backup files")
        )
        total_size, file_sizes = _artifact_size(artifact_path)
        oversize_files = {name: size for name, size in file_sizes.items() if size > MAX_BACKUP_FILE_BYTES}
        checks.append(
            _pass("backup artifact size is bounded", f"total_bytes={total_size} file_count={len(file_sizes)}", "inspect backup file sizes")
            if total_size <= MAX_BACKUP_ARTIFACT_BYTES and not oversize_files
            else _fail(
                "backup artifact size is bounded",
                f"total_bytes={total_size} max_total={MAX_BACKUP_ARTIFACT_BYTES} oversize_files={oversize_files}",
                "inspect backup file sizes",
            )
        )
        combined = _scan_json_files(artifact_path) if total_size <= MAX_BACKUP_ARTIFACT_BYTES else ""
        forbidden_values = (
            "123456:telegram",
            "bot token",
            "bearer abc",
            "sk-test",
            "password=letmein",
            "ultrasecretkey",
            "correct horse battery staple",
        )
        checks.append(
            _pass("backup files do not contain obvious raw secrets", "no forbidden raw secret samples found", "scan backup files")
            if not any(value.lower() in combined.lower() for value in forbidden_values)
            else _fail("backup files do not contain obvious raw secrets", combined[:1600], "scan backup files")
        )
        checks.append(
            _pass("backup documents exclusions and Restore v1 boundary", "raw secret-store files excluded; restore capability documented", "inspect backup files")
            if "raw secret-store files" in combined
            and ("restore_v1_allowlisted_preferences_only" in combined or "restore_not_enabled" in combined)
            and "summary_only_raw_database_excluded" in combined
            else _fail("backup documents exclusions and Restore v1 boundary", combined[:1600], "inspect backup files")
        )
        rollback_hint = str(executor_result.get("rollback_hint") or "")
        checks.append(
            _pass("rollback hint scoped to new backup path only", rollback_hint, "inspect executor result")
            if str(artifact_path) in rollback_hint and "rm -rf /" not in rollback_hint.lower()
            else _fail("rollback hint scoped to new backup path only", rollback_hint, "inspect executor result")
        )
    else:
        checks.extend(
            [
                _fail("manifest exists and lists expected bounded files", "artifact missing", "inspect manifest.json"),
                _fail("expected bounded summary files exist", "artifact missing", "inspect backup files"),
                _fail("backup artifact size is bounded", "artifact missing", "inspect backup file sizes"),
                _fail("backup files do not contain obvious raw secrets", "artifact missing", "scan backup files"),
                _fail("backup documents exclusions and Restore v1 boundary", "artifact missing", "inspect backup files"),
                _fail("rollback hint scoped to new backup path only", "artifact missing", "inspect executor result"),
            ]
        )

    resources = executor_result.get("resources_touched") if isinstance(executor_result.get("resources_touched"), list) else []
    checks.append(
        _pass("executor result includes scoped resources_touched", f"resources={len(resources)} journal={executor_result.get('journal_id')}", "inspect executor result")
        if resources and all(str(item).startswith(str(artifact_path)) for item in resources)
        else _fail("executor result includes scoped resources_touched", json.dumps(executor_result, sort_keys=True)[:1200], "inspect executor result")
    )

    restore_preview = _post_chat(base_url, "restore from backup", thread_id="backup-v1-restore", timeout=timeout)
    restore_text = _assistant_text(restore_preview)
    restore_payload = _runtime_payload(restore_preview)
    restore_plan = restore_payload.get("canonical_plan") if isinstance(restore_payload.get("canonical_plan"), dict) else {}
    checks.append(
        _pass("restore preview is validation-gated and enabled", f"plan_id={restore_plan.get('plan_id')} executor_status={restore_plan.get('executor_status')}", 'POST /chat {"message": "restore from backup"}')
        if "safety snapshot" in restore_text.lower()
        and restore_plan.get("action_type") == "operator.restore"
        and restore_plan.get("executor_status") == "enabled"
        else _fail("restore preview is validation-gated and enabled", json.dumps(restore_preview, sort_keys=True)[:1400], 'POST /chat {"message": "restore from backup"}')
    )
    restore_cancel = _post_chat(base_url, "no", thread_id="backup-v1-restore", timeout=timeout)
    restore_cancel_text = _assistant_text(restore_cancel).lower()
    checks.append(
        _pass("restore preview cancels without live mutation", _assistant_text(restore_cancel)[:1000], 'POST /chat {"message": "no"}')
        if "cancel" in restore_cancel_text and "mutated=true" not in restore_cancel_text
        else _fail("restore preview cancels without live mutation", json.dumps(restore_cancel, sort_keys=True)[:1400], 'POST /chat {"message": "no"}')
    )

    final_git = _git_status_short()
    checks.append(
        _pass("git status unchanged", "working tree status unchanged", "git status --short")
        if final_git == baseline_git
        else _fail("git status unchanged", f"before={baseline_git!r} after={final_git!r}", "git status --short")
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Installed-product Backup v1 smoke.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    checks = run(args.base_url, args.timeout)
    passed = sum(1 for check in checks if check.ok)
    failed = len(checks) - passed

    print("# Personal Agent Backup v1 Smoke")
    for check in checks:
        print()
        print(f"## {check.name}: {'PASS' if check.ok else 'FAIL'}")
        print(f"- command/API path: {check.command}")
        print(f"- evidence: {check.evidence}")
    print()
    print("## Summary")
    print(f"PASS={passed} FAIL={failed}")
    print(f"BACKUP_V1_SMOKE: {'pass' if failed == 0 else 'fail'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
