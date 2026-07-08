#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import run_uninstall_helper_state_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a validated Personal Agent uninstall helper operation.")
    parser.add_argument("--operation-state", required=True, help="Path to validated uninstall operation state JSON.")
    args = parser.parse_args()
    try:
        receipt = run_uninstall_helper_state_file(args.operation_state)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "status": "helper_failed", "error_code": exc.__class__.__name__, "error_summary": str(exc)[:500]}, sort_keys=True))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt.get("status") == "completed_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
