#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.host_lifecycle import run_operation_file, status_operation_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a validated Personal Agent host lifecycle operation.")
    parser.add_argument("operation", choices=("update", "uninstall", "status", "resume", "verify"))
    parser.add_argument("--operation-record", required=True, help="Validated host lifecycle operation record JSON.")
    args = parser.parse_args()
    try:
        if args.operation == "status":
            payload = status_operation_file(args.operation_record)
        elif args.operation in {"resume", "verify"}:
            payload = run_operation_file(args.operation_record)
        else:
            payload = run_operation_file(args.operation_record, expected_type=args.operation)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must stay structured.
        print(json.dumps({"ok": False, "status": "runner_failed", "error_code": exc.__class__.__name__, "error_summary": str(exc)[:500]}, sort_keys=True))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
