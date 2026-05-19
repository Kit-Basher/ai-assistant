from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_external_pack_safety_smoke_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/external_pack_safety_smoke.py"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS external_pack_safety_smoke" in result.stdout
    assert "FAIL " not in result.stdout
