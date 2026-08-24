#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    process = subprocess.run(["npm", "audit", "--json"], cwd=ROOT / "desktop", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        payload = {"error": "audit_output_invalid", "output": process.stdout[-1000:]}
    metadata = payload.get("metadata") if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict) else {}
    counts = metadata.get("vulnerabilities") if isinstance(metadata.get("vulnerabilities"), dict) else {}
    total = int(counts.get("total") or 0)
    report = {
        "schema_version": "personal-agent.webui-dependency-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_returncode": process.returncode,
        "counts": counts,
        "advisories": payload.get("vulnerabilities") if isinstance(payload, dict) and isinstance(payload.get("vulnerabilities"), dict) else {},
        "disposition": "no_known_advisories" if total == 0 else "review_required",
        "ok": process.returncode == 0 and total == 0,
    }
    target = ROOT / "build/reports/wp6-webui-dependency-audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "counts": counts, "report": str(target.relative_to(ROOT))}, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
