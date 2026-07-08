#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the Personal Agent host lifecycle runner.")
    parser.add_argument("--operation-state", required=True, help="Path to validated host lifecycle operation record JSON.")
    args = parser.parse_args()
    from agent.host_lifecycle import run_operation_file

    try:
        result = run_operation_file(args.operation_state, expected_type="uninstall")
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "status": "helper_failed", "error_code": exc.__class__.__name__, "error_summary": str(exc)[:500]}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
