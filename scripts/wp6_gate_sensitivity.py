#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    requirements = json.loads((ROOT / "config/wp6_release_requirements.json").read_text(encoding="utf-8"))
    rows = []
    environment = {**os.environ, "WP6_SENSITIVITY": "1"}
    for defect in requirements["required_sensitivity_cases"]:
        proc = subprocess.run([sys.executable, "scripts/release_gate.py", "--wp6-only", "--wp6-defect", defect], cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        rows.append({"case": defect, "canonical_gate_rejected": proc.returncode != 0, "returncode": proc.returncode})
    clean = subprocess.run([sys.executable, "scripts/release_gate.py", "--wp6-only"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    ok = all(row["canonical_gate_rejected"] for row in rows) and clean.returncode == 0
    report = {"schema_version": "personal-agent.wp6-gate-sensitivity.v1", "ok": ok, "cases": rows, "clean_gate_passed": clean.returncode == 0}
    target = ROOT / "build/reports/wp6-gate-sensitivity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "rejected": sum(1 for row in rows if row["canonical_gate_rejected"]), "total": len(rows), "clean": clean.returncode == 0}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
