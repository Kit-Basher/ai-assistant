#!/usr/bin/env python3
"""Read-only, redaction-safe WP6 installation/state evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import urllib.request
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = Path.home() / ".local/share/personal-agent"
DEFAULT_CONFIG = Path.home() / ".config/personal-agent"
TABLES = (
    "chat_threads", "chat_messages", "agent_tasks", "external_packs",
    "external_pack_capability_versions", "audit_log", "activity_log", "long_term_notes",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*args: str) -> tuple[int, str]:
    process = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return int(process.returncode), process.stdout.strip()


def _get(base_url: str, path: str) -> dict[str, Any]:
    last_error = "unavailable"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=12.0) as response:
                payload = json.loads(response.read(2 * 1024 * 1024).decode("utf-8"))
            return payload if isinstance(payload, dict) else {"ok": False, "error": "non_object"}
        except Exception as exc:  # noqa: BLE001 - audit records only the exception class.
            last_error = exc.__class__.__name__
            if attempt < 2:
                time.sleep(0.15)
    return {"ok": False, "error": last_error}


def _database(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"integrity": "missing", "counts": {}}
    counts: dict[str, int | None] = {}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0) as connection:
        integrity = str((connection.execute("PRAGMA integrity_check").fetchone() or ["unknown"])[0])
        existing = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in TABLES:
            counts[table] = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) if table in existing else None
    return {"integrity": integrity, "counts": counts}


def build_report(*, state_root: Path, config_root: Path, base_url: str) -> dict[str, Any]:
    head_code, head = _run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    branch_code, branch = _run("git", "-C", str(ROOT), "branch", "--show-current")
    status_code, status = _run("git", "-C", str(ROOT), "status", "--porcelain")
    remote_code, remote = _run("git", "-C", str(ROOT), "rev-parse", "@{upstream}")
    active_code, active = _run("systemctl", "--user", "is-active", "personal-agent-api.service")
    enabled_code, enabled = _run("systemctl", "--user", "is-enabled", "personal-agent-api.service")
    current = state_root / "runtime/current"
    registry_path = state_root / "llm_registry.json"
    registry: dict[str, Any] = {}
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registry = {}
    defaults = registry.get("defaults") if isinstance(registry.get("defaults"), dict) else {}
    version = _get(base_url, "/version")
    ready = _get(base_url, "/ready")
    state = _get(base_url, "/state")
    llm_status = _get(base_url, "/llm/status")
    diagnostics = _get(base_url, "/diagnostics/export")
    roots = _get(base_url, "/filesystem/roots")
    packs = _get(base_url, "/packs/state")
    protected = {
        "database": state_root / "agent.db",
        "registry": registry_path,
        "secret_store": state_root / "secrets.enc.json",
        "permissions": config_root / "permissions.json",
        "service_unit": Path.home() / ".config/systemd/user/personal-agent-api.service",
        "safe_mode_dropin": Path.home() / ".config/systemd/user/personal-agent-api.service.d/10-safe-mode.conf",
        "model_override": state_root / "llm_model_override.json",
    }
    pack_counts = packs.get("counts") if isinstance(packs.get("counts"), dict) else packs.get("summary") if isinstance(packs.get("summary"), dict) else {}
    diagnostic_bundle = diagnostics.get("bundle") if isinstance(diagnostics.get("bundle"), dict) else {}
    diagnostic_safe = diagnostic_bundle.get("safe_mode") if isinstance(diagnostic_bundle.get("safe_mode"), dict) else {}
    llm_policy = llm_status.get("policy") if isinstance(llm_status.get("policy"), dict) else {}
    safe_mode = diagnostic_safe.get("enabled")
    if safe_mode is None:
        safe_mode = llm_policy.get("safe_mode")
    if safe_mode is None:
        safe_mode = bool((ready.get("safe_mode_target") or {}).get("enabled")) if isinstance(ready.get("safe_mode_target"), dict) else None
    return {
        "schema_version": "personal-agent.wp6-state-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {"head": head if head_code == 0 else None, "branch": branch if branch_code == 0 else None, "upstream": remote if remote_code == 0 else None, "clean": status_code == 0 and not bool(status)},
        "runtime": {"version": version, "ready": bool(ready.get("ready")), "current_release": current.resolve().name if current.is_symlink() else None, "service_active": active_code == 0 and active == "active", "service_enabled": enabled_code == 0 and enabled == "enabled"},
        "model": {"default": llm_status.get("default_model") or defaults.get("default_model"), "provider": llm_status.get("default_provider") or defaults.get("default_provider"), "temporary_override": llm_status.get("temporary_override") or None, "remote_fallback": bool(llm_status.get("allow_remote_fallback", False))},
        "policy": {"safe_mode": bool(safe_mode) if safe_mode is not None else None},
        "filesystem": {"allowed_root_count": len(roots.get("allowed_roots") or []), "allowed_roots": roots.get("allowed_roots") or []},
        "packs": {"counts": pack_counts, "external_installed": int(pack_counts.get("external_installed") or pack_counts.get("external") or pack_counts.get("installed") or 0)},
        "database": _database(state_root / "agent.db"),
        "protected_hashes": {name: _sha256(path) for name, path in protected.items()},
        "redaction": {"secrets": "hash_only", "conversations": "counts_only", "pack_documents": "excluded", "environment_values": "excluded"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", default=str(DEFAULT_STATE))
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG))
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", default="build/reports/wp6-state-audit.json")
    args = parser.parse_args()
    report = build_report(state_root=Path(args.state_root).expanduser(), config_root=Path(args.config_root).expanduser(), base_url=args.base_url)
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(target.relative_to(ROOT)), "ready": report["runtime"]["ready"], "db_integrity": report["database"]["integrity"], "external_packs": report["packs"]["external_installed"]}, sort_keys=True))
    return 0 if report["database"]["integrity"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
