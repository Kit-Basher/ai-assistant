from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_product_smoke import main as run_live_product_smoke
from scripts.release_smoke import run_extended_suite


def _should_run_live_smokes(args: argparse.Namespace) -> bool:
    if bool(args.with_live_smokes):
        return True
    return str(os.environ.get("AGENT_RELEASE_VALIDATION_WITH_LIVE_SMOKES") or "").strip().lower() in {"1", "true", "yes", "on"}


def _run_optional_live_smokes() -> int:
    print("Running optional live hardware/discovery smoke hooks", flush=True)
    print("Running live_product_smoke", flush=True)
    try:
        result = int(run_live_product_smoke([]))
        if result != 0:
            print(f"Optional live smoke returned {result}; treating it as non-blocking.", flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - environment-specific live smoke guard
        print(f"Skipping optional live smoke hooks: {exc}", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the heavier Personal Agent release validation suite after the main release smoke passes."
    )
    parser.add_argument("--list", action="store_true", help="Print the exact pytest nodes without running them.")
    parser.add_argument("--no-quiet", action="store_true", help="Run pytest without -q.")
    parser.add_argument(
        "--with-live-smokes",
        action="store_true",
        help="Run optional live hardware/discovery smoke hooks after the extended pytest suite passes.",
    )
    args = parser.parse_args(argv)
    exit_code = run_extended_suite(list_only=bool(args.list), quiet=not bool(args.no_quiet))
    if exit_code != 0 or bool(args.list) or not _should_run_live_smokes(args):
        return exit_code
    live_exit_code = _run_optional_live_smokes()
    return live_exit_code or exit_code


if __name__ == "__main__":
    raise SystemExit(main())
