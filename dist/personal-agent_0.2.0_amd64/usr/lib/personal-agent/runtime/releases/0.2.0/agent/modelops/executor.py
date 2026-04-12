from __future__ import annotations

from typing import Any, Callable

from agent.modelops.safe_runner import SafeRunner


class ModelOpsExecutor:
    def __init__(
        self,
        *,
        safe_runner: SafeRunner,
        apply_defaults: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]],
        toggle_enabled: Callable[[dict[str, Any]], tuple[bool, dict[str, Any]]],
    ) -> None:
        self.safe_runner = safe_runner
        self._apply_defaults = apply_defaults
        self._toggle_enabled = toggle_enabled

    def execute_plan(self, plan: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        results: list[dict[str, Any]] = []

        for index, raw_step in enumerate(steps, start=1):
            step = raw_step if isinstance(raw_step, dict) else {}
            step_id = str(step.get("id") or f"step_{index}")
            step_type = str(step.get("type") or "unknown")

            if dry_run:
                results.append(
                    {
                        "id": step_id,
                        "type": step_type,
                        "ok": True,
                        "dry_run": True,
                        "message": "planned_only",
                    }
                )
                continue

            if step_type == "command":
                command = step.get("command") if isinstance(step.get("command"), list) else []
                timeout_seconds = float(step.get("timeout_seconds") or 600)
                command_result = self.safe_runner.run([str(part) for part in command], timeout_seconds=timeout_seconds)
                row = {
                    "id": step_id,
                    "type": step_type,
                    "ok": bool(command_result.ok),
                    "returncode": int(command_result.returncode),
                    "timed_out": bool(command_result.timed_out),
                    "truncated": bool(command_result.truncated),
                    "stdout": command_result.stdout,
                    "stderr": command_result.stderr,
                }
                results.append(row)
                if not command_result.ok:
                    return {
                        "ok": False,
                        "steps": results,
                        "error": "command_failed",
                        "failed_step": step_id,
                    }
                continue

            if step_type == "set_default_model":
                payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
                ok, body = self._apply_defaults(payload)
                row = {
                    "id": step_id,
                    "type": step_type,
                    "ok": bool(ok),
                    "response": body,
                }
                results.append(row)
                if not ok:
                    return {
                        "ok": False,
                        "steps": results,
                        "error": "registry_update_failed",
                        "failed_step": step_id,
                    }
                continue

            if step_type == "toggle_enabled":
                payload = step.get("payload") if isinstance(step.get("payload"), dict) else {}
                ok, body = self._toggle_enabled(payload)
                row = {
                    "id": step_id,
                    "type": step_type,
                    "ok": bool(ok),
                    "response": body,
                }
                results.append(row)
                if not ok:
                    return {
                        "ok": False,
                        "steps": results,
                        "error": "registry_update_failed",
                        "failed_step": step_id,
                    }
                continue

            results.append(
                {
                    "id": step_id,
                    "type": step_type,
                    "ok": False,
                    "error": "unsupported_step_type",
                }
            )
            return {
                "ok": False,
                "steps": results,
                "error": "unsupported_step_type",
                "failed_step": step_id,
            }

        return {
            "ok": True,
            "steps": results,
        }
