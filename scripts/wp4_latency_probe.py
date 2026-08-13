from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
import urllib.request

from wasmtime import wat2wasm


def request(base: str, method: str, path: str, payload: dict | None = None) -> tuple[dict, float]:
    started = time.perf_counter()
    req = urllib.request.Request(base.rstrip("/") + path, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Connection": "close"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        body = json.loads(response.read())
    return body, (time.perf_counter() - started) * 1000


def summary(values: list[float]) -> dict:
    ordered = sorted(values); p95 = ordered[max(0, int((len(ordered) - 1) * .95))]
    return {"samples": len(values), "median_ms": round(statistics.median(values), 3), "p95_ms": round(p95, 3)}


def chat(base: str, text: str, thread: str) -> tuple[dict, float]:
    # Each sample is an independent actor/thread so the product's deliberate
    # per-user pacing does not masquerade as routing or dispatch overhead.
    return request(base, "POST", "/chat", {"messages": [{"role": "user", "content": text}], "user_id": thread, "session_id": thread, "thread_id": thread, "source_surface": "api"})


def mutate(base: str, action: str, payload: dict) -> dict:
    binding = {"actor_id": "wp4-benchmark", "session_id": "benchmark", "thread_id": "benchmark"}
    plan, _ = request(base, "POST", f"/packs/capabilities/{action}/plan", {**payload, **binding})
    exact = plan["plan"]
    applied, _ = request(base, "POST", f"/packs/capabilities/{action}/apply", {"plan_id": exact["plan_id"], "binding_digest": exact["binding_digest"], "confirmed": True, **binding})
    return applied["result"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--base-url", required=True); parser.add_argument("--samples", type=int, default=15); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); samples = max(5, min(50, args.samples)); results = {}
    for name, text in {"native_presence": "are you here?", "native_system": "show current system status"}.items():
        values = []; routes = set()
        for index in range(samples):
            body, elapsed = chat(args.base_url, text, f"wp4-native-{name}-{index}"); values.append(elapsed); routes.add(str((body.get("meta") or {}).get("route")))
        results[name] = {"end_to_end": summary(values), "routes": sorted(routes), "worker_processes": 0, "planning_generations": 0}
    with tempfile.TemporaryDirectory(prefix="wp4-latency-") as directory:
        root = Path(directory) / "executable"; root.mkdir(); module = root / "module.wasm"
        module.write_bytes(wat2wasm('(module (func (export "invoke") (param i32) (result i32) local.get 0 i32.const 2 i32.mul))'))
        manifest = {"schema_version":"personal-agent.pack.v1","id":"benchmark-doubler","version":"1","pack_class":"sandboxed_executable","description":"Temporary exact-candidate benchmark fixture","capabilities":[{"schema_version":"personal-agent.pack-capability.v1","name":"double","description":"Double an integer in isolated pure computation","examples":["multiply a number by two","double this integer safely"],"input_schema":{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]},"output_schema":{"type":"object","properties":{"result":{"type":"integer"}},"required":["result"]},"permissions":["pure_compute"],"invocation":{"kind":"wasm","abi":"personal-agent.pack-worker.v1","module":"module.wasm","input_field":"value"},"verifier":{"kind":"integer_result"},"self_test_input":{"value":2}}]}
        (root / "personal-agent-pack.json").write_text(json.dumps(manifest))
        record = mutate(args.base_url, "import", {"path": str(root)})["record"]; rid = record["record_id"]
        mutate(args.base_url, "gate", {"record_id": rid, "gate": "review_approved", "value": True})
        mutate(args.base_url, "gate", {"record_id": rid, "gate": "grants", "value": ["pure_compute"]})
        mutate(args.base_url, "gate", {"record_id": rid, "gate": "enabled", "value": True})
        values = []; selection = []; worker = []
        for index in range(samples):
            body, elapsed = chat(args.base_url, f"use Benchmark Doubler to multiply {index + 2} by two", f"wp4-pack-{index}"); values.append(elapsed)
            timing = (body.get("meta") or {}).get("chat_timing_ms") or {}; selection.append(float(timing.get("request_understanding_ms") or 0)); worker.append(float(((body.get("setup") or {}).get("worker") or {}).get("elapsed_ms") or 0))
        results["executable_chat"] = {"end_to_end": summary(values), "selection": summary(selection), "worker_cold": summary(worker), "worker_processes_per_invocation": 1}
        mutate(args.base_url, "remove", {"record_id": rid})
        declarative = Path(directory) / "declarative"; declarative.mkdir()
        declaration = {"schema_version":"personal-agent.pack.v1","id":"benchmark-health","version":"1","pack_class":"declarative","description":"Temporary exact-candidate declarative benchmark fixture","capabilities":[{"schema_version":"personal-agent.pack-capability.v1","name":"health_report","description":"Report current system health through native status","examples":["summarize machine health with benchmark helper","inspect system status through benchmark add-on"],"input_schema":{"type":"object","properties":{},"required":[]},"output_schema":{"type":"object","properties":{"result":{"type":"string"}},"required":["result"]},"permissions":[],"invocation":{"kind":"registered_capability","capability_id":"system.status","inputs":{}},"verifier":{"kind":"nonempty"},"self_test_input":{}}]}
        (declarative / "personal-agent-pack.json").write_text(json.dumps(declaration))
        record = mutate(args.base_url, "import", {"path": str(declarative)})["record"]; rid = record["record_id"]
        mutate(args.base_url, "gate", {"record_id": rid, "gate": "review_approved", "value": True})
        mutate(args.base_url, "gate", {"record_id": rid, "gate": "grants", "value": ["system:read"]})
        mutate(args.base_url, "gate", {"record_id": rid, "gate": "enabled", "value": True})
        values = []; selection = []
        for index in range(samples):
            body, elapsed = chat(args.base_url, "summarize machine health with Benchmark Health helper", f"wp4-decl-{index}"); values.append(elapsed)
            timing = (body.get("meta") or {}).get("chat_timing_ms") or {}; selection.append(float(timing.get("request_understanding_ms") or 0))
        results["declarative_chat"] = {"end_to_end": summary(values), "selection": summary(selection), "worker_processes": 0}
        mutate(args.base_url, "remove", {"record_id": rid})
    report = {"schema_version": "wp4-latency.v1", "samples_per_case": samples, "results": results}
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output: args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(serialized + "\n")
    print(serialized); return 0


if __name__ == "__main__": raise SystemExit(main())
