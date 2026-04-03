from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release_smoke import run_extended_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the heavier Personal Agent release validation suite after the main release smoke passes."
    )
    parser.add_argument("--list", action="store_true", help="Print the exact pytest nodes without running them.")
    parser.add_argument("--no-quiet", action="store_true", help="Run pytest without -q.")
    args = parser.parse_args(argv)
    return run_extended_suite(list_only=bool(args.list), quiet=not bool(args.no_quiet))


if __name__ == "__main__":
    raise SystemExit(main())
