#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COMMANDS = ("bash", "git", "python3", "systemctl", "curl", "tar")
OPTIONAL_COMMANDS = ("ollama", "bwrap", "wasmtime")


def _os_release() -> dict[str, str]:
    rows: dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                rows[key] = value.strip().strip('"')
    return rows


def build_report() -> dict[str, object]:
    os_info = _os_release()
    commands = {name: bool(shutil.which(name)) for name in (*REQUIRED_COMMANDS, *OPTIONAL_COMMANDS)}
    required_ok = all(commands[name] for name in REQUIRED_COMMANDS)
    ubuntu_target = os_info.get("ID") == "ubuntu" and os_info.get("VERSION_ID") == "24.04"
    installer = ROOT / "scripts/install_local.sh"
    install_check = subprocess.run(["bash", "-n", str(installer)], check=False).returncode == 0
    return {
        "schema_version": "personal-agent.ubuntu-recovery-preflight.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {"distribution": os_info.get("ID"), "version": os_info.get("VERSION_ID"), "architecture": platform.machine()},
        "target": {"distribution": "ubuntu", "version": "24.04", "physical_target_observed": ubuntu_target},
        "commands": commands,
        "required_commands_ok": required_ok,
        "installer_syntax_ok": install_check,
        "user_systemd_available": commands["systemctl"],
        "optional": {
            "ollama": "required only for local model chat; restore remains inspectable without it",
            "bwrap_wasmtime": "required only for executable external packs; absence is honestly unavailable",
        },
        "backup_contract": "personal-agent.portable-backup.v2",
        "physical_fresh_host_proof": False,
        "status": "preflight_passed_physical_test_pending" if required_ok and install_check else "preflight_failed",
        "ok": required_ok and install_check,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report()
    target = ROOT / "build/reports/wp6-ubuntu-preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2 if args.json else None, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
