#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.doctor import default_db_path, run_doctor


def main() -> int:
    repo_root = REPO_ROOT
    db_path = default_db_path(repo_root)
    version_path = os.path.join(repo_root, "VERSION")
    results = run_doctor(repo_root=repo_root, db_path=db_path, version_path=version_path)

    failed = 0
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status} {result.name}: {result.message}")
        if not result.ok:
            failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
