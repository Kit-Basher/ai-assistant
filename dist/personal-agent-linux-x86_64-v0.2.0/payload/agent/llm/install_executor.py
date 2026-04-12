from __future__ import annotations

import subprocess
import time
from typing import Any, Callable

from agent.config import Config
from agent.llm.install_approval import validate_install_approval
from agent.llm.model_inventory import build_model_inventory
from agent.llm.registry import Registry


RunFn = Callable[..., subprocess.CompletedProcess[str]]


def _trace_id() -> str:
    return f"llm-install-{int(time.time())}"


def _tail(text: str | None, *, limit: int = 400) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[-limit:]


def _verification_row(inventory: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    row = next((item for item in inventory if str(item.get("id") or "") == model_id), None)
    if not isinstance(row, dict):
        return {
            "found": False,
            "installed": False,
            "available": False,
            "healthy": False,
            "verification_status": "degraded",
        }
    installed = bool(row.get("installed", False))
    available = bool(row.get("available", False))
    healthy = bool(row.get("healthy", False))
    return {
        "found": True,
        "installed": installed,
        "available": available,
        "healthy": healthy,
        "reason": str(row.get("reason") or "").strip() or None,
        "health_reason": str(row.get("health_reason") or row.get("health_failure_kind") or row.get("reason") or "").strip() or None,
        "capability_source": str(row.get("capability_source") or "").strip() or None,
        "verification_status": "ok" if (installed and available and healthy) else "degraded",
        "provider": str(row.get("provider") or "").strip() or None,
        "capabilities": list(row.get("capabilities") or []) if isinstance(row.get("capabilities"), list) else [],
    }


def execute_install_plan(
    *,
    config: Config,
    registry: Registry,
    plan: dict[str, Any] | None,
    approve: bool,
    trace_id: str | None = None,
    run_fn: RunFn | None = None,
    inventory_builder: Callable[..., list[dict[str, Any]]] = build_model_inventory,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    active_trace_id = str(trace_id or _trace_id())
    decision = validate_install_approval(plan, approve=approve)
    model_id = str(decision.get("model_id") or "").strip() or None
    install_name = str(decision.get("install_name") or "").strip() or None
    if str(decision.get("error_kind") or "") == "already_satisfied":
        inventory = inventory_builder(config=config, registry=registry)
        verification = _verification_row(inventory, model_id or "")
        return {
            "ok": True,
            "executed": False,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": None,
            "message": "Model already installed and healthy.",
            "verification": verification,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    if not bool(decision.get("allowed", False)):
        return {
            "ok": False,
            "executed": False,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": str(decision.get("error_kind") or "install_not_allowed"),
            "message": str(decision.get("message") or "Install request was rejected."),
            "verification": {},
            "stdout_tail": "",
            "stderr_tail": "",
        }

    inventory_before = inventory_builder(config=config, registry=registry)
    pre_verification = _verification_row(inventory_before, model_id or "")
    if bool(pre_verification.get("installed", False)) and bool(pre_verification.get("healthy", False)):
        return {
            "ok": True,
            "executed": False,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": None,
            "message": "Model already installed and healthy.",
            "verification": pre_verification,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    runner = run_fn or subprocess.run
    try:
        proc = runner(
            ["ollama", "pull", str(install_name or "")],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "executed": False,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": "ollama_cli_missing",
            "message": "Ollama CLI is not available on this system.",
            "verification": pre_verification,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "executed": True,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": "timed_out",
            "message": "Ollama pull timed out.",
            "verification": pre_verification,
            "stdout_tail": _tail(getattr(exc, "stdout", "")),
            "stderr_tail": _tail(getattr(exc, "stderr", "")),
        }

    stdout_tail = _tail(proc.stdout)
    stderr_tail = _tail(proc.stderr)
    if int(proc.returncode) != 0:
        return {
            "ok": False,
            "executed": True,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": "pull_failed",
            "message": f"Ollama pull failed with exit code {int(proc.returncode)}.",
            "verification": pre_verification,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }

    inventory_after = inventory_builder(config=config, registry=registry)
    verification = _verification_row(inventory_after, model_id or "")
    if not bool(verification.get("installed", False)):
        return {
            "ok": False,
            "executed": True,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": "verification_failed",
            "message": "Install completed but the model was not visible in inventory afterwards.",
            "verification": verification,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    if not bool(verification.get("healthy", False)):
        health_reason = str(verification.get("health_reason") or "degraded").strip() or "degraded"
        return {
            "ok": True,
            "executed": True,
            "model_id": model_id,
            "install_name": install_name,
            "trace_id": active_trace_id,
            "error_kind": None,
            "message": f"Install completed, but health verification is degraded ({health_reason}).",
            "verification": verification,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    return {
        "ok": True,
        "executed": True,
        "model_id": model_id,
        "install_name": install_name,
        "trace_id": active_trace_id,
        "error_kind": None,
        "message": f"Installed and verified {model_id}.",
        "verification": verification,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


__all__ = ["execute_install_plan"]
