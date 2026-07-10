#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.primary_uninstall_policy import (  # noqa: E402
    PRIMARY_UNINSTALL_DEFAULT_DAYS,
    PRIMARY_UNINSTALL_MAX_DAYS,
    build_policy_context,
    diagnose_primary_uninstall_host_policy,
    disable_primary_uninstall_marker,
    enable_primary_uninstall_marker,
    repair_primary_uninstall_host_policy_permissions,
    validate_primary_uninstall_marker,
)


ACK_FLAG = "--acknowledge-primary-uninstall-capability"
REPAIR_ACK_FLAG = "--acknowledge-host-policy-repair"


def _print_status(status, *, inspect: bool = False) -> None:
    diagnostic = diagnose_primary_uninstall_host_policy()
    payload = status.redacted_dict()
    payload["host_policy"] = diagnostic.to_dict()
    payload["preserve_data_only"] = True
    payload["purge_supported"] = False
    if not inspect:
        payload.pop("fingerprint", None)
        payload["details"] = {k: v for k, v in payload.get("details", {}).items() if k != "filesystem"}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status.enabled:
        print(f"Primary preserve-data uninstall is enabled until {status.expires_at}. Purge is unsupported.")
    else:
        print(f"Primary preserve-data uninstall is disabled: {status.reason}. Purge is unsupported.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local operator policy for primary preserve-data uninstall.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("inspect")
    sub.add_parser("diagnose")
    enable = sub.add_parser("enable")
    enable.add_argument(ACK_FLAG, action="store_true", dest="ack")
    enable.add_argument("--expires-in-days", type=int, default=PRIMARY_UNINSTALL_DEFAULT_DAYS)
    repair = sub.add_parser("repair-permissions")
    repair.add_argument(REPAIR_ACK_FLAG, action="store_true", dest="repair_ack")
    sub.add_parser("disable")
    args = parser.parse_args()

    if args.command in {"status", "inspect"}:
        _print_status(validate_primary_uninstall_marker(build_policy_context()), inspect=args.command == "inspect")
        return 0
    if args.command == "diagnose":
        status = validate_primary_uninstall_marker(build_policy_context())
        print(json.dumps({"status": status.redacted_dict(), "host_policy": diagnose_primary_uninstall_host_policy().to_dict()}, indent=2, sort_keys=True))
        return 0
    if args.command == "repair-permissions":
        if not args.repair_ack:
            print(f"Refusing to repair host policy permissions without {REPAIR_ACK_FLAG}.", file=sys.stderr)
            return 2
        result = repair_primary_uninstall_host_policy_permissions()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1
    if args.command == "disable":
        _print_status(disable_primary_uninstall_marker())
        return 0
    if args.command == "enable":
        if not args.ack:
            print(
                f"Refusing to enable primary uninstall capability without {ACK_FLAG}.",
                file=sys.stderr,
            )
            return 2
        if args.expires_in_days < 1 or args.expires_in_days > PRIMARY_UNINSTALL_MAX_DAYS:
            print(f"Expiry must be between 1 and {PRIMARY_UNINSTALL_MAX_DAYS} days.", file=sys.stderr)
            return 2
        ctx = build_policy_context(create_identity=True)
        if ctx.repository_path != ROOT.resolve():
            print(f"Refusing copied/untrusted checkout: {ctx.repository_path} != {ROOT.resolve()}", file=sys.stderr)
            return 2
        print(
            "Enabling primary preserve-data uninstall for this local installation only. "
            "This does not uninstall anything. Purge remains unsupported."
        )
        _print_status(enable_primary_uninstall_marker(expires_in_days=args.expires_in_days), inspect=False)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
