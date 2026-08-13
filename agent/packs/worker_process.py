from __future__ import annotations

"""Core-owned Wasm worker. External bytes are never imported by the API."""

import json
import resource
import sys
from typing import Any

from wasmtime import Config, Engine, Instance, Module, Store


def _reply(payload: dict[str, Any]) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    sys.stdout.flush()
    return 0 if payload.get("ok") else 2


def main() -> int:
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        raw = sys.stdin.buffer.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            return _reply({"ok": False, "error_kind": "worker_request_too_large"})
        request = json.loads(raw)
        if not isinstance(request, dict) or set(request) - {"abi", "export", "input", "fuel", "memory_bytes"}:
            return _reply({"ok": False, "error_kind": "worker_request_invalid"})
        if request.get("abi") != "personal-agent.pack-worker.v1":
            return _reply({"ok": False, "error_kind": "worker_abi_unsupported"})
        value = request.get("input")
        if not isinstance(value, int) or isinstance(value, bool):
            return _reply({"ok": False, "error_kind": "worker_input_invalid"})
        config = Config()
        config.consume_fuel = True
        config.wasm_threads = False
        config.wasm_reference_types = False
        engine = Engine(config)
        module = Module.from_file(engine, "/module.wasm")
        if list(module.imports):
            return _reply({"ok": False, "error_kind": "worker_imports_denied"})
        store = Store(engine)
        store.set_fuel(min(10_000_000, max(10_000, int(request.get("fuel") or 1_000_000))))
        store.set_limits(memory_size=min(32 * 1024 * 1024, max(64 * 1024, int(request.get("memory_bytes") or 8 * 1024 * 1024))), instances=1, memories=1, tables=1)
        instance = Instance(store, module, [])
        # Engine initialization may create bounded internal helper threads.
        # Once the guest is instantiated, forbid any subsequent child/process
        # creation before untrusted guest instructions run.
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        export_name = str(request.get("export") or "invoke")
        export = instance.exports(store).get(export_name)
        if export is None or not callable(export):
            return _reply({"ok": False, "error_kind": "worker_export_missing"})
        result = export(store, value)
        if not isinstance(result, int):
            return _reply({"ok": False, "error_kind": "worker_result_invalid"})
        # Authority-like fields cannot be emitted by this ABI: the only pack
        # value is a signed integer nested beneath untrusted_output.
        return _reply({"ok": True, "abi": "personal-agent.pack-worker.v1", "untrusted_output": {"result": result}})
    except BaseException as exc:  # worker boundary returns only a class, never hostile details
        return _reply({"ok": False, "error_kind": f"worker_failed:{exc.__class__.__name__}"})


if __name__ == "__main__":
    raise SystemExit(main())
