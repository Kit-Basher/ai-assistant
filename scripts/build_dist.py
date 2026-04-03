from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_backend


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic Personal Agent wheel and sdist artifacts.")
    parser.add_argument("--outdir", default="dist", help="Output directory for build artifacts.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before building.")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    if args.clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wheel_name = build_backend.build_wheel(str(outdir))
    sdist_name = build_backend.build_sdist(str(outdir))

    print(str(outdir / wheel_name), flush=True)
    print(str(outdir / sdist_name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
