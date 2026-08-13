from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

from agent.packs.capability_contracts import WORKER_ABI, file_digest


@dataclass(frozen=True)
class WorkerHealth:
    available: bool
    reason: str | None


class SandboxedPackWorker:
    """One invocation = one Bubblewrap namespace + one no-WASI Wasm store."""

    def __init__(self, *, code_root: str | Path | None = None) -> None:
        self.code_root = Path(code_root or Path(__file__).resolve().parents[2]).resolve()

    def health(self) -> WorkerHealth:
        if shutil.which("bwrap") is None:
            return WorkerHealth(False, "bubblewrap_missing")
        try:
            import wasmtime  # noqa: F401
        except Exception:
            return WorkerHealth(False, "wasmtime_missing")
        probe = subprocess.run(
            ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--clearenv", "/usr/bin/true"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2, check=False,
        )
        return WorkerHealth(probe.returncode == 0, None if probe.returncode == 0 else "bubblewrap_user_namespace_unavailable")

    @staticmethod
    def _limits(memory_bytes: int, wall_ms: int):
        def apply() -> None:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            # Child process creation is denied structurally: the guest has no
            # WASI or host imports and runs in a private PID namespace. An
            # RLIMIT_NPROC on the bwrap launcher itself would prevent creation
            # of the namespace-init process and is therefore not used.
            resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024, 128 * 1024))
            cpu = max(1, min(6, int(wall_ms / 1000) + 1))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            # Wasmtime reserves a large sparse virtual guard region. Actual
            # guest linear memory is bounded independently by Store.set_limits;
            # this ceiling limits runaway host allocation without preventing
            # the engine's guard mapping.
            address = 8 * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (address, address))
        return apply

    def invoke(
        self, *, module_path: str | Path, artifact_digest: str, export: str,
        value: int, limits: Mapping[str, Any], cancellation_check: callable | None = None,
    ) -> dict[str, Any]:
        health = self.health()
        if not health.available:
            return {"ok": False, "error_kind": health.reason, "worker": {"isolated": False}}
        module = Path(module_path).resolve()
        if module.is_symlink() or not module.is_file() or file_digest(module) != artifact_digest:
            return {"ok": False, "error_kind": "pack_artifact_digest_changed", "worker": {"isolated": False}}
        wall_ms = min(5_000, max(50, int(limits.get("wall_ms") or 1_000)))
        memory_bytes = min(32 * 1024 * 1024, max(64 * 1024, int(limits.get("memory_bytes") or 8 * 1024 * 1024)))
        output_bytes = min(64 * 1024, max(256, int(limits.get("output_bytes") or 8 * 1024)))
        request = {"abi": WORKER_ABI, "export": export, "input": value, "fuel": int(limits.get("fuel") or 1_000_000), "memory_bytes": memory_bytes}
        python_root = Path(sys.prefix).resolve()
        with tempfile.TemporaryDirectory(prefix="pa-pack-worker-") as scratch:
            command = [
                "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                "--ro-bind", str(python_root), "/venv", "--ro-bind", str(self.code_root), "/app",
                "--ro-bind", str(module), "/module.wasm", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--dir", "/scratch", "--chdir", "/scratch", "--clearenv",
                "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "PERSONAL_AGENT_INSTANCE", "dev",
                "/venv/bin/python", "-I", "-c",
                "import sys;sys.path.insert(0,'/app');from agent.packs.worker_process import main;raise SystemExit(main())",
            ]
            started = time.perf_counter()
            try:
                proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={}, preexec_fn=self._limits(memory_bytes, wall_ms), start_new_session=True)
                stdout, stderr = proc.communicate(json.dumps(request, separators=(",", ":")).encode(), timeout=wall_ms / 1000)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.communicate()
                return {"ok": False, "error_kind": "worker_timeout", "worker": {"isolated": True, "wall_ms": wall_ms, "orphaned": False}}
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            if cancellation_check and cancellation_check():
                return {"ok": False, "error_kind": "worker_cancelled", "worker": {"isolated": True, "elapsed_ms": elapsed, "orphaned": False}}
            if len(stdout) > output_bytes:
                return {"ok": False, "error_kind": "worker_output_too_large", "worker": {"isolated": True, "elapsed_ms": elapsed, "orphaned": False}}
            try:
                response = json.loads(stdout)
            except Exception:
                return {"ok": False, "error_kind": "worker_protocol_malformed", "worker": {"isolated": True, "elapsed_ms": elapsed, "stderr_redacted": bool(stderr)}}
            if not isinstance(response, dict) or set(response) - {"ok", "abi", "untrusted_output", "error_kind"}:
                return {"ok": False, "error_kind": "worker_protocol_authority_field", "worker": {"isolated": True, "elapsed_ms": elapsed}}
            response["worker"] = {"isolated": True, "engine": "wasmtime", "namespace": "bubblewrap", "elapsed_ms": elapsed, "exit_code": proc.returncode, "orphaned": False}
            return response
