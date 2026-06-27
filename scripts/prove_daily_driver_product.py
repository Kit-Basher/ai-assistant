#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Gate:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class Result:
    gate: Gate
    returncode: int
    elapsed_seconds: float
    output: str


def _summarize(output: str, *, max_lines: int = 18) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join([*lines[: max_lines // 2], "...", *lines[-(max_lines // 2) :]])


def _run(gate: Gate) -> Result:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            gate.argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=gate.timeout_seconds,
            check=False,
        )
        return Result(gate=gate, returncode=int(proc.returncode), elapsed_seconds=time.monotonic() - started, output=proc.stdout or "")
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        return Result(gate=gate, returncode=124, elapsed_seconds=time.monotonic() - started, output=f"{output}\nTIMEOUT after {gate.timeout_seconds}s")


def main() -> int:
    gates = [
        Gate("installed product abuse harness", (sys.executable, "scripts/installed_product_abuse.py"), 240),
    ]
    print("# Personal Agent Daily-Driver Product Proof")
    print(f"Repo: {ROOT}")
    failures = 0
    for gate in gates:
        result = _run(gate)
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"## {gate.name}: {status}")
        print(f"- command/API path: {' '.join(gate.argv)}")
        print(f"- elapsed_s: {result.elapsed_seconds:.1f}")
        if result.output:
            print("- output:")
            print(_summarize(result.output))
        if result.returncode != 0:
            failures += 1
    print("")
    print(f"PRODUCT_PROOF: {'pass' if failures == 0 else 'fail'}")
    print(f"FAILURES={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
