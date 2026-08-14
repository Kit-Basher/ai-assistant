#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return round(ordered[max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * p))))], 3)


def probe(base_url: str, text: str, samples: int, label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(samples):
        suffix = f"wp45-{label}-{text.split()[0]}-{int(time.time())}-{index}"
        payload = json.dumps({
            "user_id": suffix,
            "session_id": suffix,
            "thread_id": suffix,
            "messages": [{"role": "user", "content": text}],
        }, separators=(",", ":"))
        completed = subprocess.run(
            [
                "curl", "--silent", "--show-error", "--output", "/tmp/wp45-latency-body.json",
                "--write-out", "%{http_code} %{http_version} %{size_download} %{time_starttransfer} %{time_total}",
                "--header", "Content-Type: application/json", "--data", payload,
                f"{base_url.rstrip('/')}/chat",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        fields = completed.stdout.strip().split()
        if completed.returncode or len(fields) != 5:
            rows.append({"ok": False, "error": completed.stderr.strip()[:200] or "curl_failed"})
            continue
        status, version, size, ttfb, total = fields
        rows.append({
            "ok": status == "200",
            "status": int(status),
            "http_version": version,
            "size_bytes": int(size),
            "ttfb_ms": round(float(ttfb) * 1000, 3),
            "body_complete_ms": round(float(total) * 1000, 3),
            "body_after_first_byte_ms": round((float(total) - float(ttfb)) * 1000, 3),
        })
    good = [row for row in rows if row.get("ok")]
    def dist(key: str) -> dict[str, Any]:
        values = [float(row[key]) for row in good]
        return {
            "samples": len(values),
            "median_ms": round(statistics.median(values), 3) if values else None,
            "p95_ms": percentile(values, 0.95) if values else None,
            "max_ms": max(values) if values else None,
        }
    return {
        "text": text,
        "successful": len(good),
        "failed": samples - len(good),
        "ttfb": dist("ttfb_ms"),
        "body_complete": dist("body_complete_ms"),
        "body_after_first_byte": dist("body_after_first_byte_ms"),
        "http_versions": sorted({str(row.get("http_version")) for row in good}),
        "content_lengths": sorted({int(row.get("size_bytes") or 0) for row in good}),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8123")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "contract": "personal-agent.wp45-latency.v1",
        "label": args.label,
        "base_url": args.base_url,
        "sample_count_per_route": args.samples,
        "routes": {
            "presence": probe(args.base_url, "u here?", args.samples, args.label),
            "system_status": probe(args.base_url, "give me a system status check", args.samples, args.label),
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, row in report["routes"].items():
        print(f"{name}: body median={row['body_complete']['median_ms']} p95={row['body_complete']['p95_ms']} ms; ttfb p95={row['ttfb']['p95_ms']} ms")
    return 0 if all(row["failed"] == 0 for row in report["routes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
