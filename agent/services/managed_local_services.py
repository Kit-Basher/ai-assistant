from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import secrets as secrets_lib
import shutil
import socket
import subprocess
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from agent.actions.managed_action_recovery import ManagedActionJournal


APPROVED_SEARXNG_IMAGE = "docker.io/searxng/searxng:latest"
APPROVED_SEARXNG_CONTAINER = "personal-agent-searxng"
APPROVED_SEARXNG_PORT = 8080
APPROVED_SEARXNG_FALLBACK_PORT = 8888
APPROVED_SEARXNG_VOLUME = "memory/local_services/searxng"
APPROVED_SEARXNG_CONFIG_PURPOSE = "enable_json_output_for_safe_metadata_search"
DEFAULT_SEARXNG_SECRET_KEY = "ultrasecretkey"
APPROVED_SEARXNG_SETTINGS_TEMPLATE = """\
use_default_settings: true

server:
  secret_key: "{secret_key}"

search:
  formats:
    - html
    - json
"""
APPROVED_SEARXNG_SETTINGS = APPROVED_SEARXNG_SETTINGS_TEMPLATE.format(secret_key="<generated>")
APPROVED_SEARXNG_PRIMARY_BIND = "127.0.0.1:8080:8080"
APPROVED_SEARXNG_FALLBACK_BIND = "127.0.0.1:8888:8080"
APPROVED_SEARXNG_BINDS = (APPROVED_SEARXNG_PRIMARY_BIND, APPROVED_SEARXNG_FALLBACK_BIND)
SEARXNG_CONFIG_OWNERSHIP_HANDOFF_COMMANDS = (
    'sudo chown -R "$USER:$USER" ~/.local/share/personal-agent/memory/local_services/searxng',
    "chmod -R u+rwX ~/.local/share/personal-agent/memory/local_services/searxng",
)
TRUSTED_CONTAINER_ENGINE_PATHS = {
    "podman": ("/usr/bin/podman", "/usr/local/bin/podman", "/bin/podman"),
    "docker": ("/usr/bin/docker", "/usr/local/bin/docker", "/bin/docker"),
}


def trusted_command_path(name: str) -> str | None:
    command = str(name or "").strip().lower()
    if command not in TRUSTED_CONTAINER_ENGINE_PATHS:
        return shutil.which(command)
    found = shutil.which(command)
    if found:
        return found
    for candidate in TRUSTED_CONTAINER_ENGINE_PATHS[command]:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


@dataclass(frozen=True)
class ManagedServiceSpec:
    service_id: str
    display_name: str
    purpose: str
    default_local_url: str
    approved_container_name: str
    approved_image: str
    approved_port: int
    approved_loopback_bind: str
    approved_volume_path: str | None = None
    blocked_actions: tuple[str, ...] = field(default_factory=tuple)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "default_local_url": self.default_local_url,
            "approved_container": {
                "name": self.approved_container_name,
                "image": self.approved_image,
                "port": self.approved_port,
                "bind": self.approved_loopback_bind,
                "fallback_bind": APPROVED_SEARXNG_FALLBACK_BIND if self.service_id == "searxng" else None,
                "volume_path": self.approved_volume_path,
                "volume_mount": True,
                "config_seeded": True,
                "config_purpose": APPROVED_SEARXNG_CONFIG_PURPOSE,
                "metadata_only": True,
            },
            "blocked_actions": list(self.blocked_actions),
        }


SEARXNG_SERVICE = ManagedServiceSpec(
    service_id="searxng",
    display_name="SearXNG",
    purpose="safe web search",
    default_local_url="http://127.0.0.1:8080",
    approved_container_name=APPROVED_SEARXNG_CONTAINER,
    approved_image=APPROVED_SEARXNG_IMAGE,
    approved_port=APPROVED_SEARXNG_PORT,
    approved_loopback_bind=APPROVED_SEARXNG_PRIMARY_BIND,
    approved_volume_path=APPROVED_SEARXNG_VOLUME,
    blocked_actions=(
        "docker_pull",
        "docker_run",
        "docker_stop",
        "docker_remove",
        "system_package_install",
        "config_write",
        "external_pack_triggered_container_action",
    ),
)

SERVICE_REGISTRY: dict[str, ManagedServiceSpec] = {
    SEARXNG_SERVICE.service_id: SEARXNG_SERVICE,
}


@dataclass(frozen=True)
class ManagedLocalServiceSetupPlan:
    service_id: str
    engine: str
    image: str
    container_name: str
    loopback_bind: str
    health_url: str
    host_port: int
    volume_mount: bool = True
    config_seeded: bool = True
    config_purpose: str = APPROVED_SEARXNG_CONFIG_PURPOSE
    host_volume_path: str | None = None
    container_volume_path: str | None = None

    def pull_argv(self) -> list[str]:
        return [self.engine, "pull", self.image]

    def existing_container_argv(self) -> list[str]:
        return [
            self.engine,
            "ps",
            "-a",
            "--filter",
            f"name=^/{self.container_name}$",
            "--format",
            "{{.Names}}",
        ]

    def inspect_container_argv(self) -> list[str]:
        return [self.engine, "inspect", self.container_name, "--format", "{{json .}}"]

    def start_container_argv(self) -> list[str]:
        return [self.engine, "start", self.container_name]

    def run_argv(self) -> list[str]:
        argv = [
            self.engine,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-p",
            self.loopback_bind,
        ]
        if self.volume_mount:
            argv.extend(["-v", f"{self.host_volume_path}:{self.container_volume_path}"])
        argv.append(self.image)
        return argv

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "engine": self.engine,
            "image": self.image,
            "container_name": self.container_name,
            "loopback_bind": self.loopback_bind,
            "volume_mount": self.volume_mount,
            "config_seeded": self.config_seeded,
            "config_purpose": self.config_purpose,
            "host_volume_path": self.host_volume_path,
            "container_volume_path": self.container_volume_path,
            "health_url": self.health_url,
            "host_port": self.host_port,
            "pull_argv": self.pull_argv(),
            "run_argv": self.run_argv(),
            "shell": False,
        }


@dataclass(frozen=True)
class ManagedLocalServiceStopPlan:
    service_id: str
    engine: str
    container_name: str

    def stop_argv(self) -> list[str]:
        return [self.engine, "stop", self.container_name]

    def remove_argv(self) -> list[str]:
        return [self.engine, "rm", self.container_name]

    def existing_container_argv(self) -> list[str]:
        return [
            self.engine,
            "ps",
            "-a",
            "--filter",
            f"name=^/{self.container_name}$",
            "--format",
            "{{.Names}}",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "engine": self.engine,
            "container_name": self.container_name,
            "stop_argv": self.stop_argv(),
            "remove_argv": self.remove_argv(),
            "shell": False,
        }


@dataclass(frozen=True)
class ManagedLocalServiceExecutionResult:
    ok: bool
    service_id: str
    selected_engine: str
    did_pull: bool = False
    did_run: bool = False
    did_install: bool = False
    did_configure: bool = False
    reachable: bool = False
    blocked_reason: str | None = None
    error: str | None = None
    plan: ManagedLocalServiceSetupPlan | None = None
    port_conflict: bool = False
    rollback_attempted: bool = False
    rollback_ok: bool = False
    cleanup_performed: bool = False
    cleanup_incomplete: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    journal: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "service_id": self.service_id,
            "selected_engine": self.selected_engine,
            "did_pull": self.did_pull,
            "did_run": self.did_run,
            "did_install": self.did_install,
            "did_configure": self.did_configure,
            "reachable": self.reachable,
            "blocked_reason": self.blocked_reason,
            "error": self.error,
            "plan": self.plan.to_dict() if self.plan else None,
            "port_conflict": self.port_conflict,
            "rollback_attempted": self.rollback_attempted,
            "rollback_ok": self.rollback_ok,
            "cleanup_performed": self.cleanup_performed,
            "cleanup_incomplete": self.cleanup_incomplete,
            "diagnostics": dict(self.diagnostics),
            "journal": dict(self.journal),
            "shell": False,
            "external_pack_triggered": False,
        }


@dataclass(frozen=True)
class ManagedLocalServiceStopResult:
    ok: bool
    service_id: str
    selected_engine: str
    did_stop: bool = False
    did_remove: bool = False
    blocked_reason: str | None = None
    error: str | None = None
    plan: ManagedLocalServiceStopPlan | None = None
    journal: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "service_id": self.service_id,
            "selected_engine": self.selected_engine,
            "did_stop": self.did_stop,
            "did_remove": self.did_remove,
            "blocked_reason": self.blocked_reason,
            "error": self.error,
            "plan": self.plan.to_dict() if self.plan else None,
            "journal": dict(self.journal),
            "shell": False,
            "external_pack_triggered": False,
        }


class ManagedLocalServiceExecutor:
    """Confirm-gated executor for approved core-owned local services only."""

    def __init__(
        self,
        *,
        managed_root: str | Path,
        command_finder: Callable[[str], str | None] | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        health_checker: Callable[[str], bool] | None = None,
        port_checker: Callable[[int], bool] | None = None,
        secret_generator: Callable[[], str] | None = None,
        timeout_seconds: float = 60.0,
        health_timeout_seconds: float = 12.0,
        health_poll_interval_seconds: float = 1.0,
        health_probe_timeout_seconds: float = 1.0,
    ) -> None:
        self._managed_root = Path(managed_root).expanduser().resolve()
        self._command_finder = command_finder or trusted_command_path
        self._runner = runner or subprocess.run
        self._health_checker = health_checker
        self._port_checker = port_checker or self._port_available
        self._secret_generator = secret_generator or (lambda: secrets_lib.token_urlsafe(32))
        self._timeout_seconds = max(1.0, float(timeout_seconds or 60.0))
        self._health_timeout_seconds = max(0.0, float(health_timeout_seconds or 12.0))
        self._health_poll_interval_seconds = max(0.01, float(health_poll_interval_seconds or 1.0))
        self._health_probe_timeout_seconds = max(0.1, float(health_probe_timeout_seconds or 1.0))

    def build_searxng_setup_plan(self, *, selected_engine: str, host_port: int = APPROVED_SEARXNG_PORT) -> ManagedLocalServiceSetupPlan:
        engine = str(selected_engine or "").strip().lower()
        spec = SERVICE_REGISTRY["searxng"]
        port = int(host_port)
        bind = APPROVED_SEARXNG_FALLBACK_BIND if port == APPROVED_SEARXNG_FALLBACK_PORT else APPROVED_SEARXNG_PRIMARY_BIND
        volume = (self._managed_root / str(spec.approved_volume_path)).resolve()
        return ManagedLocalServiceSetupPlan(
            service_id=spec.service_id,
            engine=engine,
            image=spec.approved_image,
            container_name=spec.approved_container_name,
            loopback_bind=bind,
            health_url=f"http://127.0.0.1:{port}",
            host_port=port,
            volume_mount=True,
            config_seeded=True,
            config_purpose=APPROVED_SEARXNG_CONFIG_PURPOSE,
            host_volume_path=str(volume),
            container_volume_path="/etc/searxng",
        )

    def preview_setup_from_status(self, *, service_id: str, selected_engine: str) -> dict[str, Any]:
        service = str(service_id or "").strip().lower()
        engine = str(selected_engine or "").strip().lower()
        if service != "searxng":
            return {"ok": False, "blocked_reason": "managed_service_unknown", "service_id": service, "selected_engine": engine}
        if engine not in {"docker", "podman"}:
            return {"ok": False, "blocked_reason": "managed_service_engine_invalid", "service_id": service, "selected_engine": engine}
        if not self._command_finder(engine):
            return {"ok": False, "blocked_reason": "managed_service_engine_missing", "service_id": service, "selected_engine": engine}
        primary_available = self._is_port_available(APPROVED_SEARXNG_PORT)
        fallback_available = self._is_port_available(APPROVED_SEARXNG_FALLBACK_PORT)
        if primary_available:
            plan = self.build_searxng_setup_plan(selected_engine=engine, host_port=APPROVED_SEARXNG_PORT)
            return {
                "ok": True,
                "service_id": service,
                "selected_engine": engine,
                "plan": plan.to_dict(),
                "preferred_port_available": True,
                "fallback_port_available": fallback_available,
                "fallback_selected": False,
                "port_conflict": False,
            }
        if fallback_available:
            plan = self.build_searxng_setup_plan(selected_engine=engine, host_port=APPROVED_SEARXNG_FALLBACK_PORT)
            return {
                "ok": True,
                "service_id": service,
                "selected_engine": engine,
                "plan": plan.to_dict(),
                "preferred_port_available": False,
                "fallback_port_available": True,
                "fallback_selected": True,
                "port_conflict": True,
                "port_conflict_message": "Port 8080 is already being used by another local app.",
            }
        return {
            "ok": False,
            "service_id": service,
            "selected_engine": engine,
            "blocked_reason": "managed_service_approved_ports_occupied",
            "preferred_port_available": False,
            "fallback_port_available": False,
            "port_conflict": True,
        }

    def build_plan_from_pending(self, params: dict[str, Any]) -> tuple[ManagedLocalServiceSetupPlan | None, str | None]:
        service_id = str(params.get("service_id") or "").strip().lower()
        if service_id != "searxng":
            return None, "managed_service_unknown"
        if str(params.get("action") or "").strip().lower() != "preview_only":
            return None, "managed_service_action_invalid"
        selected_engine = str(params.get("selected_engine") or "").strip().lower()
        pending_bind = str(params.get("loopback_bind") or "").strip()
        host_port = self._host_port_for_bind(pending_bind)
        if host_port is None:
            return None, "managed_service_bind_not_approved"
        plan = self.build_searxng_setup_plan(selected_engine=selected_engine, host_port=host_port)
        spec = SERVICE_REGISTRY["searxng"]
        checks = {
            "approved_image": spec.approved_image,
            "approved_container_name": spec.approved_container_name,
            "loopback_bind": plan.loopback_bind,
            "volume_mount": True,
            "config_seeded": True,
            "approved_volume_path": spec.approved_volume_path,
            "config_purpose": APPROVED_SEARXNG_CONFIG_PURPOSE,
        }
        for key, expected in checks.items():
            actual = bool(params.get(key)) if isinstance(expected, bool) else str(params.get(key) or "").strip()
            if actual != expected:
                return None, f"managed_service_plan_tampered_{key}"
        if any(str(params.get(key) or "").strip() for key in ("settings_yml", "settings_yaml", "config_text", "config_content")):
            return None, "managed_service_plan_tampered_config_content"
        reason = self.validate_plan(plan)
        if reason:
            return None, reason
        return plan, None

    def validate_plan(self, plan: ManagedLocalServiceSetupPlan) -> str | None:
        if plan.service_id != "searxng":
            return "managed_service_unknown"
        if plan.engine not in {"docker", "podman"}:
            return "managed_service_engine_invalid"
        if not self._command_finder(plan.engine):
            return "managed_service_engine_missing"
        spec = SERVICE_REGISTRY["searxng"]
        if plan.image != spec.approved_image:
            return "managed_service_image_not_approved"
        if plan.container_name != spec.approved_container_name:
            return "managed_service_container_not_approved"
        if plan.loopback_bind not in APPROVED_SEARXNG_BINDS:
            return "managed_service_bind_not_approved"
        try:
            host, host_port, container_port = plan.loopback_bind.split(":", 2)
        except ValueError:
            return "managed_service_bind_invalid"
        if host != "127.0.0.1" or host_port not in {"8080", "8888"} or container_port != "8080":
            return "managed_service_bind_not_loopback"
        if int(host_port) != plan.host_port:
            return "managed_service_bind_port_mismatch"
        if not plan.volume_mount:
            return "managed_service_volume_required"
        if not plan.config_seeded:
            return "managed_service_config_not_seeded"
        if plan.config_purpose != APPROVED_SEARXNG_CONFIG_PURPOSE:
            return "managed_service_config_purpose_not_approved"
        host_volume = Path(str(plan.host_volume_path or "")).expanduser().resolve()
        approved_root = (self._managed_root / str(spec.approved_volume_path)).resolve()
        if host_volume != approved_root:
            return "managed_service_volume_not_approved"
        if plan.container_volume_path != "/etc/searxng":
            return "managed_service_volume_not_approved"
        return None

    def execute_plan(self, plan: ManagedLocalServiceSetupPlan) -> ManagedLocalServiceExecutionResult:
        journal = ManagedActionJournal(action_type="managed_local_service_setup", target=plan.service_id)
        journal.plan_step("validate_plan", resource=plan.service_id)
        journal.plan_step("check_existing_container", resource=plan.container_name)
        journal.plan_step("preflight_port", resource=f"127.0.0.1:{plan.host_port}")
        journal.plan_step("preflight_config_writable", resource=plan.host_volume_path)
        journal.plan_step("seed_config", resource=plan.host_volume_path)
        journal.plan_step("pull_image", resource=plan.image)
        journal.plan_step("run_container", resource=plan.container_name)
        journal.plan_step("health_check", resource=plan.health_url)
        journal.plan_step("capture_failure_diagnostics", resource=plan.container_name)
        reason = self.validate_plan(plan)
        if reason:
            journal.record_step("validate_plan", ok=False, resource=plan.service_id, reason=reason)
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason=reason,
                plan=plan,
                journal=journal.to_dict(),
            )
        journal.record_step("validate_plan", ok=True, resource=plan.service_id)
        existing = self._run_fixed(plan.existing_container_argv())
        if existing.returncode == 0 and plan.container_name in str(existing.stdout or "").splitlines():
            return self._handle_existing_searxng_container(plan, journal)
        journal.record_step("check_existing_container", ok=True, resource=plan.container_name)
        if not self._is_port_available(plan.host_port):
            journal.record_step("preflight_port", ok=False, resource=f"127.0.0.1:{plan.host_port}")
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_port_occupied",
                error=f"Port {plan.host_port} is already being used by another local app.",
                plan=plan,
                port_conflict=True,
                journal=journal.to_dict(),
            )
        journal.record_step("preflight_port", ok=True, resource=f"127.0.0.1:{plan.host_port}")
        config_snapshot, snapshot_error = self._capture_searxng_config_state(plan)
        if snapshot_error:
            journal.record_step("preflight_config_writable", ok=False, resource=plan.host_volume_path, reason=snapshot_error)
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_config_snapshot_failed",
                error=snapshot_error,
                plan=plan,
                journal=journal.to_dict(),
            )
        writable, writable_error = self._check_searxng_config_writable(plan)
        if not writable:
            journal.record_step("preflight_config_writable", ok=False, resource=plan.host_volume_path, reason=writable_error)
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_config_dir_not_writable",
                error=writable_error or "approved_config_dir_not_writable",
                plan=plan,
                diagnostics={"operator_handoff": self._config_ownership_handoff(plan)},
                journal=journal.to_dict(),
            )
        journal.record_step("preflight_config_writable", ok=True, resource=plan.host_volume_path)
        seeded, seed_error = self._seed_searxng_config(plan)
        if not seeded:
            journal.record_step("seed_config", ok=False, resource=plan.host_volume_path, error=seed_error)
            config_rollback_ok, config_rollback_error = self._restore_searxng_config_state(plan, config_snapshot)
            journal.record_rollback_step(
                "restore_config",
                ok=config_rollback_ok,
                resource=plan.host_volume_path,
                error=config_rollback_error,
            )
            journal.mark_rollback(ok=config_rollback_ok, restored_config=config_rollback_ok, error=config_rollback_error)
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_config_dir_not_writable" if seed_error == "PermissionError" else "managed_service_config_seed_failed",
                error=seed_error,
                plan=plan,
                rollback_attempted=True,
                rollback_ok=config_rollback_ok,
                cleanup_performed=config_rollback_ok,
                cleanup_incomplete=not config_rollback_ok,
                diagnostics={"operator_handoff": self._config_ownership_handoff(plan)} if seed_error == "PermissionError" else {},
                journal=journal.to_dict(),
            )
        journal.record_step("seed_config", ok=True, resource=plan.host_volume_path, config_purpose=plan.config_purpose)
        journal.record_changed_resource("directory", str(plan.host_volume_path), owned_by="personal-agent", config_seeded=True)
        pulled = self._run_fixed(plan.pull_argv())
        if pulled.returncode != 0:
            journal.record_step("pull_image", ok=False, resource=plan.image, error=self._short_process_error(pulled))
            config_rollback_ok, config_rollback_error = self._restore_searxng_config_state(plan, config_snapshot)
            journal.record_rollback_step("restore_config", ok=config_rollback_ok, resource=plan.host_volume_path, error=config_rollback_error)
            journal.mark_rollback(ok=config_rollback_ok, restored_config=config_rollback_ok, error=config_rollback_error)
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_pull_failed",
                error=self._short_process_error(pulled),
                plan=plan,
                rollback_attempted=True,
                rollback_ok=config_rollback_ok,
                cleanup_performed=config_rollback_ok,
                cleanup_incomplete=not config_rollback_ok,
                journal=journal.to_dict(),
            )
        journal.record_step("pull_image", ok=True, resource=plan.image)
        journal.record_changed_resource("container_image", plan.image, rollback_supported=False)
        ran = self._run_fixed(plan.run_argv())
        if ran.returncode != 0:
            journal.record_step("run_container", ok=False, resource=plan.container_name, error=self._short_process_error(ran))
            config_rollback_ok, config_rollback_error = self._restore_searxng_config_state(plan, config_snapshot)
            journal.record_rollback_step("restore_config", ok=config_rollback_ok, resource=plan.host_volume_path, error=config_rollback_error)
            journal.mark_rollback(ok=config_rollback_ok, restored_config=config_rollback_ok, error=config_rollback_error)
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                did_pull=True,
                blocked_reason="managed_service_run_failed",
                error=self._short_process_error(ran),
                plan=plan,
                rollback_attempted=True,
                rollback_ok=config_rollback_ok,
                cleanup_performed=config_rollback_ok,
                cleanup_incomplete=not config_rollback_ok,
                journal=journal.to_dict(),
            )
        journal.record_step("run_container", ok=True, resource=plan.container_name)
        journal.record_created_resource("container", plan.container_name, engine=plan.engine, image=plan.image)
        reachable, health_diagnostics = self._wait_for_health(plan.health_url)
        journal.record_step("health_check", ok=reachable, resource=plan.health_url, **health_diagnostics)
        journal.mark_verification(ok=reachable, health_url=plan.health_url, **health_diagnostics)
        if not reachable:
            diagnostics = self._capture_failure_diagnostics(plan, health_diagnostics)
            journal.record_step("capture_failure_diagnostics", ok=True, resource=plan.container_name, diagnostics=diagnostics)
            rollback_ok, rollback_error = self._rollback_owned_setup(plan, journal)
            container_rollback_ok = rollback_ok
            config_rollback_ok, config_rollback_error = self._restore_searxng_config_state(plan, config_snapshot)
            journal.record_rollback_step(
                "restore_config",
                ok=config_rollback_ok,
                resource=plan.host_volume_path,
                error=config_rollback_error,
            )
            rollback_ok = rollback_ok and config_rollback_ok
            rollback_error = "; ".join(
                item for item in (rollback_error, config_rollback_error) if item
            ) or None
            journal.mark_rollback(
                ok=rollback_ok,
                removed_created_container=container_rollback_ok,
                restored_config=config_rollback_ok,
                error=rollback_error,
            )
            message = (
                "Health check failed after the approved container started. I cleaned up the failed setup. Nothing was left running."
                if rollback_ok
                else f"Health check failed after the approved container started. Cleanup was incomplete: {rollback_error or 'unknown rollback error'}."
            )
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                did_pull=True,
                did_run=True,
                reachable=False,
                blocked_reason="managed_service_health_check_failed",
                error=message,
                plan=plan,
                rollback_attempted=True,
                rollback_ok=rollback_ok,
                cleanup_performed=rollback_ok,
                cleanup_incomplete=not rollback_ok,
                diagnostics=diagnostics,
                journal=journal.to_dict(),
            )
        return ManagedLocalServiceExecutionResult(
            ok=True,
            service_id=plan.service_id,
            selected_engine=plan.engine,
            did_pull=True,
            did_run=True,
            reachable=True,
            plan=plan,
            diagnostics=health_diagnostics,
            journal=journal.to_dict(),
        )

    def _capture_searxng_config_state(
        self,
        plan: ManagedLocalServiceSetupPlan,
    ) -> tuple[dict[str, Any], str | None]:
        root = Path(str(plan.host_volume_path or "")).expanduser().resolve()
        approved = (self._managed_root / APPROVED_SEARXNG_VOLUME).resolve()
        if root != approved:
            return {}, "config_volume_not_approved"
        settings = root / "settings.yml"
        try:
            if settings.is_symlink():
                return {}, "approved_settings_symlink_forbidden"
            missing_directories: list[str] = []
            cursor = root
            while cursor != self._managed_root and self._managed_root in cursor.parents:
                if not cursor.exists():
                    missing_directories.append(str(cursor))
                cursor = cursor.parent
            return {
                "settings_existed": settings.is_file(),
                "settings_content": settings.read_bytes() if settings.is_file() else None,
                "missing_directories": missing_directories,
            }, None
        except OSError as exc:
            return {}, exc.__class__.__name__

    @staticmethod
    def _restore_searxng_config_state(
        plan: ManagedLocalServiceSetupPlan,
        snapshot: dict[str, Any],
    ) -> tuple[bool, str | None]:
        settings = Path(str(plan.host_volume_path or "")).expanduser().resolve() / "settings.yml"
        incomplete: list[str] = []
        try:
            if bool(snapshot.get("settings_existed")):
                content = snapshot.get("settings_content")
                if not isinstance(content, bytes):
                    return False, "config_snapshot_invalid"
                settings.parent.mkdir(parents=True, exist_ok=True)
                settings.write_bytes(content)
            elif settings.exists():
                settings.unlink()
            for raw_path in snapshot.get("missing_directories") or []:
                path = Path(str(raw_path)).resolve()
                try:
                    path.rmdir()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    # Never remove a directory that now contains anything else.
                    incomplete.append(f"{path.name}:{exc.__class__.__name__}")
        except OSError as exc:
            return False, exc.__class__.__name__
        if incomplete:
            return False, "config_rollback_incomplete:" + ",".join(incomplete)
        return True, None

    def execute_from_pending(self, params: dict[str, Any]) -> ManagedLocalServiceExecutionResult:
        service_id = str(params.get("service_id") or "searxng").strip().lower() or "searxng"
        engine = str(params.get("selected_engine") or "docker").strip().lower() or "docker"
        plan, reason = self.build_plan_from_pending(params)
        if reason or plan is None:
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=service_id,
                selected_engine=engine,
                blocked_reason=reason or "managed_service_plan_invalid",
            )
        return self.execute_plan(plan)

    def build_searxng_stop_plan(self, *, selected_engine: str) -> ManagedLocalServiceStopPlan:
        engine = str(selected_engine or "").strip().lower()
        return ManagedLocalServiceStopPlan(
            service_id="searxng",
            engine=engine,
            container_name=APPROVED_SEARXNG_CONTAINER,
        )

    def preview_stop_from_status(self, *, service_id: str, selected_engine: str) -> dict[str, Any]:
        service = str(service_id or "").strip().lower()
        engine = str(selected_engine or "").strip().lower()
        if service != "searxng":
            return {"ok": False, "blocked_reason": "managed_service_unknown", "service_id": service, "selected_engine": engine}
        if engine not in {"docker", "podman"}:
            return {"ok": False, "blocked_reason": "managed_service_engine_invalid", "service_id": service, "selected_engine": engine}
        if not self._command_finder(engine):
            return {"ok": False, "blocked_reason": "managed_service_engine_missing", "service_id": service, "selected_engine": engine}
        plan = self.build_searxng_stop_plan(selected_engine=engine)
        reason = self.validate_stop_plan(plan)
        if reason:
            return {"ok": False, "blocked_reason": reason, "service_id": service, "selected_engine": engine}
        return {"ok": True, "service_id": service, "selected_engine": engine, "plan": plan.to_dict(), "mutated": False}

    def validate_stop_plan(self, plan: ManagedLocalServiceStopPlan) -> str | None:
        if plan.service_id != "searxng":
            return "managed_service_unknown"
        if plan.engine not in {"docker", "podman"}:
            return "managed_service_engine_invalid"
        if not self._command_finder(plan.engine):
            return "managed_service_engine_missing"
        if plan.container_name != APPROVED_SEARXNG_CONTAINER:
            return "managed_service_container_not_approved"
        return None

    def stop_from_pending(self, params: dict[str, Any]) -> ManagedLocalServiceStopResult:
        service_id = str(params.get("service_id") or "searxng").strip().lower() or "searxng"
        engine = str(params.get("selected_engine") or "docker").strip().lower() or "docker"
        if service_id != "searxng":
            return ManagedLocalServiceStopResult(ok=False, service_id=service_id, selected_engine=engine, blocked_reason="managed_service_unknown")
        if str(params.get("action") or "").strip().lower() != "stop_preview_only":
            return ManagedLocalServiceStopResult(ok=False, service_id=service_id, selected_engine=engine, blocked_reason="managed_service_action_invalid")
        if str(params.get("approved_container_name") or "").strip() != APPROVED_SEARXNG_CONTAINER:
            return ManagedLocalServiceStopResult(
                ok=False,
                service_id=service_id,
                selected_engine=engine,
                blocked_reason="managed_service_plan_tampered_approved_container_name",
            )
        plan = self.build_searxng_stop_plan(selected_engine=engine)
        return self.stop_plan(plan)

    def stop_plan(self, plan: ManagedLocalServiceStopPlan) -> ManagedLocalServiceStopResult:
        journal = ManagedActionJournal(action_type="managed_local_service_stop", target=plan.service_id)
        journal.plan_step("validate_stop_plan", resource=plan.service_id)
        journal.plan_step("check_existing_container", resource=plan.container_name)
        journal.plan_step("stop_container", resource=plan.container_name)
        journal.plan_step("remove_container", resource=plan.container_name)
        reason = self.validate_stop_plan(plan)
        if reason:
            journal.record_step("validate_stop_plan", ok=False, resource=plan.service_id, reason=reason)
            return ManagedLocalServiceStopResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason=reason,
                plan=plan,
                journal=journal.to_dict(),
            )
        journal.record_step("validate_stop_plan", ok=True, resource=plan.service_id)
        existing = self._run_fixed(plan.existing_container_argv())
        if existing.returncode == 0 and plan.container_name not in str(existing.stdout or "").splitlines():
            journal.record_step("check_existing_container", ok=False, resource=plan.container_name, reason="missing_container")
            return ManagedLocalServiceStopResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_container_not_found",
                error="The approved Personal-Agent-managed SearXNG container was not found.",
                plan=plan,
                journal=journal.to_dict(),
            )
        journal.record_step("check_existing_container", ok=True, resource=plan.container_name)
        stopped = self._run_fixed(plan.stop_argv())
        if stopped.returncode != 0:
            journal.record_step("stop_container", ok=False, resource=plan.container_name, error=self._short_process_error(stopped))
            return ManagedLocalServiceStopResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_stop_failed",
                error=self._short_process_error(stopped),
                plan=plan,
                journal=journal.to_dict(),
            )
        journal.record_step("stop_container", ok=True, resource=plan.container_name)
        removed = self._run_fixed(plan.remove_argv())
        if removed.returncode != 0:
            journal.record_step("remove_container", ok=False, resource=plan.container_name, error=self._short_process_error(removed))
            return ManagedLocalServiceStopResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                did_stop=True,
                blocked_reason="managed_service_remove_failed",
                error=self._short_process_error(removed),
                plan=plan,
                journal=journal.to_dict(),
            )
        journal.record_step("remove_container", ok=True, resource=plan.container_name)
        journal.mark_verification(ok=True, container_removed=True)
        return ManagedLocalServiceStopResult(
            ok=True,
            service_id=plan.service_id,
            selected_engine=plan.engine,
            did_stop=True,
            did_remove=True,
            plan=plan,
            journal=journal.to_dict(),
        )

    def _rollback_owned_setup(self, plan: ManagedLocalServiceSetupPlan, journal: ManagedActionJournal) -> tuple[bool, str | None]:
        stop_result = self._run_fixed([plan.engine, "stop", plan.container_name])
        stop_ok = stop_result.returncode == 0
        journal.record_rollback_step("stop_container", ok=stop_ok, resource=plan.container_name, error=None if stop_ok else self._short_process_error(stop_result))
        remove_result = self._run_fixed([plan.engine, "rm", plan.container_name])
        remove_ok = remove_result.returncode == 0
        journal.record_rollback_step("remove_container", ok=remove_ok, resource=plan.container_name, error=None if remove_ok else self._short_process_error(remove_result))
        rollback_ok = stop_ok and remove_ok
        error = None if rollback_ok else "; ".join(
            item
            for item in [
                None if stop_ok else f"stop failed: {self._short_process_error(stop_result)}",
                None if remove_ok else f"remove failed: {self._short_process_error(remove_result)}",
            ]
            if item
        )
        journal.mark_rollback(ok=rollback_ok, stopped_container=stop_ok, removed_container=remove_ok, error=error)
        return rollback_ok, error

    def _handle_existing_searxng_container(
        self,
        plan: ManagedLocalServiceSetupPlan,
        journal: ManagedActionJournal,
    ) -> ManagedLocalServiceExecutionResult:
        info, reason = self._inspect_existing_container(plan)
        if reason or not info:
            journal.record_step("check_existing_container", ok=False, resource=plan.container_name, reason=reason or "inspect_failed")
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason=reason or "managed_service_existing_container_inspect_failed",
                error="A container with the approved name exists, but it could not be verified as an approved Personal Agent SearXNG container.",
                plan=plan,
                journal=journal.to_dict(),
            )
        effective_plan = plan
        inspected_bind = str(info.get("loopback_bind") or "").strip()
        inspected_port = self._host_port_for_bind(inspected_bind)
        if inspected_port is not None and inspected_bind != plan.loopback_bind:
            effective_plan = replace(
                plan,
                loopback_bind=inspected_bind,
                host_port=inspected_port,
                health_url=f"http://127.0.0.1:{inspected_port}",
            )
        journal.record_step(
            "check_existing_container",
            ok=True,
            resource=effective_plan.container_name,
            reused_existing=True,
            running=bool(info.get("running")),
            loopback_bind=effective_plan.loopback_bind,
        )
        if bool(info.get("running")):
            reachable, health_diagnostics = self._wait_for_health(effective_plan.health_url)
            journal.record_step("health_check", ok=reachable, resource=effective_plan.health_url, reused_existing=True, **health_diagnostics)
            journal.mark_verification(ok=reachable, reused_existing=True, health_url=effective_plan.health_url, **health_diagnostics)
            if reachable:
                return ManagedLocalServiceExecutionResult(
                    ok=True,
                    service_id=effective_plan.service_id,
                    selected_engine=effective_plan.engine,
                    reachable=True,
                    plan=effective_plan,
                    diagnostics={"reused_existing_container": True, **health_diagnostics},
                    journal=journal.to_dict(),
                )
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=effective_plan.service_id,
                selected_engine=effective_plan.engine,
                reachable=False,
                blocked_reason="managed_service_existing_container_unhealthy",
                error="The existing approved SearXNG container matched the managed-service contract, but its health check failed. I did not remove it.",
                plan=effective_plan,
                diagnostics={"reused_existing_container": True, **health_diagnostics},
                journal=journal.to_dict(),
            )
        started = self._run_fixed(effective_plan.start_container_argv())
        start_ok = int(started.returncode) == 0
        journal.record_step("run_container", ok=start_ok, resource=effective_plan.container_name, repair_action="start_existing_container")
        if not start_ok:
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=effective_plan.service_id,
                selected_engine=effective_plan.engine,
                blocked_reason="managed_service_existing_container_start_failed",
                error=self._short_process_error(started),
                plan=effective_plan,
                diagnostics={"repair_action": "start_existing_container"},
                journal=journal.to_dict(),
            )
        reachable, health_diagnostics = self._wait_for_health(effective_plan.health_url)
        journal.record_step("health_check", ok=reachable, resource=effective_plan.health_url, repaired_existing=True, **health_diagnostics)
        journal.mark_verification(ok=reachable, repaired_existing=True, health_url=effective_plan.health_url, **health_diagnostics)
        if reachable:
            return ManagedLocalServiceExecutionResult(
                ok=True,
                service_id=effective_plan.service_id,
                selected_engine=effective_plan.engine,
                reachable=True,
                plan=effective_plan,
                diagnostics={"repaired_existing_container": True, "repair_action": "start_existing_container", **health_diagnostics},
                journal=journal.to_dict(),
            )
        stopped = self._run_fixed([effective_plan.engine, "stop", effective_plan.container_name])
        rollback_ok = stopped.returncode == 0
        rollback_error = None if rollback_ok else self._short_process_error(stopped)
        journal.record_rollback_step(
            "restore_existing_container_stopped_state",
            ok=rollback_ok,
            resource=effective_plan.container_name,
            error=rollback_error,
        )
        journal.mark_rollback(
            ok=rollback_ok,
            restored_prior_stopped_state=rollback_ok,
            error=rollback_error,
        )
        return ManagedLocalServiceExecutionResult(
            ok=False,
            service_id=effective_plan.service_id,
            selected_engine=effective_plan.engine,
            reachable=False,
            blocked_reason="managed_service_startup_pending",
            error=(
                "The approved SearXNG container was started but failed the bounded chat readiness window; "
                + (
                    "its prior stopped state was restored."
                    if rollback_ok
                    else f"restoring its prior stopped state failed: {rollback_error or 'unknown rollback error'}."
                )
            ),
            plan=effective_plan,
            rollback_attempted=True,
            rollback_ok=rollback_ok,
            cleanup_performed=rollback_ok,
            cleanup_incomplete=not rollback_ok,
            diagnostics={"repaired_existing_container": True, "repair_action": "start_existing_container", **health_diagnostics},
            journal=journal.to_dict(),
        )

    def _inspect_existing_container(self, plan: ManagedLocalServiceSetupPlan) -> tuple[dict[str, Any] | None, str | None]:
        inspected = self._run_fixed(plan.inspect_container_argv())
        if inspected.returncode != 0:
            return None, "managed_service_existing_container_inspect_failed"
        try:
            payload = json.loads(str(inspected.stdout or "{}"))
        except json.JSONDecodeError:
            return None, "managed_service_existing_container_inspect_failed"
        if isinstance(payload, list):
            payload = payload[0] if payload and isinstance(payload[0], dict) else {}
        if not isinstance(payload, dict):
            return None, "managed_service_existing_container_inspect_failed"
        image = str(
            payload.get("ImageName")
            or (payload.get("Config") if isinstance(payload.get("Config"), dict) else {}).get("Image")
            or ""
        ).strip()
        if image != plan.image:
            return None, "managed_service_existing_container_image_mismatch"
        bind = self._inspect_approved_bind(payload)
        if not bind:
            return None, "managed_service_existing_container_bind_mismatch"
        if not self._inspect_has_approved_mount(payload, plan):
            return None, "managed_service_existing_container_mount_mismatch"
        state = payload.get("State") if isinstance(payload.get("State"), dict) else {}
        running = bool(state.get("Running")) or str(state.get("Status") or "").strip().lower() == "running"
        return {"running": running, "image": image, "loopback_bind": bind}, None

    @staticmethod
    def _inspect_approved_bind(payload: dict[str, Any]) -> str | None:
        network = payload.get("NetworkSettings") if isinstance(payload.get("NetworkSettings"), dict) else {}
        ports = network.get("Ports") if isinstance(network.get("Ports"), dict) else {}
        bind = ManagedLocalServiceExecutor._approved_bind_from_rows(ports)
        if bind:
            return bind
        host_config = payload.get("HostConfig") if isinstance(payload.get("HostConfig"), dict) else {}
        bindings = host_config.get("PortBindings") if isinstance(host_config.get("PortBindings"), dict) else {}
        return ManagedLocalServiceExecutor._approved_bind_from_rows(bindings)

    @staticmethod
    def _approved_bind_from_rows(port_rows: dict[str, Any]) -> str | None:
        rows = port_rows.get("8080/tcp")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict):
                continue
            bind = f"{str(row.get('HostIp') or '')}:{str(row.get('HostPort') or '')}:8080"
            if bind in APPROVED_SEARXNG_BINDS:
                return bind
        return None

    @staticmethod
    def _inspect_has_approved_mount(payload: dict[str, Any], plan: ManagedLocalServiceSetupPlan) -> bool:
        expected_source = str(Path(str(plan.host_volume_path or "")).expanduser().resolve())
        expected_destination = str(plan.container_volume_path or "")
        mounts = payload.get("Mounts")
        if not isinstance(mounts, list):
            return False
        for row in mounts:
            if not isinstance(row, dict):
                continue
            source = str(row.get("Source") or row.get("Name") or "").strip()
            destination = str(row.get("Destination") or "").strip()
            if source and str(Path(source).expanduser().resolve()) == expected_source and destination == expected_destination:
                return True
        return False

    def _check_searxng_config_writable(self, plan: ManagedLocalServiceSetupPlan) -> tuple[bool, str | None]:
        try:
            root = Path(str(plan.host_volume_path or "")).expanduser().resolve()
            approved = (self._managed_root / APPROVED_SEARXNG_VOLUME).resolve()
            if root != approved:
                return False, "config_volume_not_approved"
            if root.exists() and not root.is_dir():
                return False, "approved_config_path_not_directory"
            settings = root / "settings.yml"
            if root.exists() and not self._path_writable(root):
                return False, "approved_config_dir_not_writable"
            if settings.exists() and not self._path_writable(settings):
                return False, "approved_settings_not_writable"
            parent = root.parent
            if not root.exists() and parent.exists() and not self._path_writable(parent):
                return False, "approved_config_parent_not_writable"
        except OSError as exc:
            return False, exc.__class__.__name__
        return True, None

    @staticmethod
    def _path_writable(path: Path) -> bool:
        return bool(path.exists() and os.access(path, os.W_OK))

    def _config_ownership_handoff(self, plan: ManagedLocalServiceSetupPlan) -> dict[str, Any]:
        return {
            "kind": "visible_terminal",
            "bounded_action": "repair_searxng_config_ownership",
            "target": "memory/local_services/searxng",
            "reason": "The approved SearXNG config directory is not writable by the Personal Agent service user.",
            "commands": list(SEARXNG_CONFIG_OWNERSHIP_HANDOFF_COMMANDS),
            "command_string": " && ".join(SEARXNG_CONFIG_OWNERSHIP_HANDOFF_COMMANDS),
            "sudo_password_storage": False,
            "will_pull_image": False,
            "will_run_container": False,
            "will_enable_search": False,
            "retry_endpoint": "POST /search/setup/apply with a fresh confirmed setup plan",
            "approved_path": "memory/local_services/searxng",
            "path": str(plan.host_volume_path or ""),
        }

    def _seed_searxng_config(self, plan: ManagedLocalServiceSetupPlan) -> tuple[bool, str | None]:
        if not plan.volume_mount or not plan.config_seeded:
            return False, "config_seeded_required"
        if plan.config_purpose != APPROVED_SEARXNG_CONFIG_PURPOSE:
            return False, "config_purpose_not_approved"
        try:
            root = Path(str(plan.host_volume_path or "")).expanduser().resolve()
            approved = (self._managed_root / APPROVED_SEARXNG_VOLUME).resolve()
            if root != approved:
                return False, "config_volume_not_approved"
            root.mkdir(parents=True, exist_ok=True)
            settings = root / "settings.yml"
            existing = settings.read_text(encoding="utf-8") if settings.exists() else ""
            secret_key = self._extract_approved_searxng_secret(existing)
            if not secret_key:
                secret_key = self._generate_searxng_secret()
            settings.write_text(self._approved_searxng_settings(secret_key), encoding="utf-8")
            actual = settings.read_text(encoding="utf-8")
        except OSError as exc:
            return False, exc.__class__.__name__
        if not self._validate_approved_searxng_settings(actual):
            return False, "settings_validation_failed"
        return True, None

    def _generate_searxng_secret(self) -> str:
        for _attempt in range(3):
            value = str(self._secret_generator() or "").strip()
            if self._is_approved_searxng_secret(value):
                return value
        return secrets_lib.token_urlsafe(32)

    @staticmethod
    def _approved_searxng_settings(secret_key: str) -> str:
        escaped = str(secret_key).replace("\\", "\\\\").replace('"', '\\"')
        return APPROVED_SEARXNG_SETTINGS_TEMPLATE.format(secret_key=escaped)

    @classmethod
    def _validate_approved_searxng_settings(cls, text: str) -> bool:
        raw = str(text or "")
        return (
            "use_default_settings: true" in raw
            and cls._extract_approved_searxng_secret(raw) is not None
            and re.search(r"(?im)^\s*-\s*html\s*$", raw) is not None
            and re.search(r"(?im)^\s*-\s*json\s*$", raw) is not None
        )

    @classmethod
    def _extract_approved_searxng_secret(cls, text: str) -> str | None:
        match = re.search(r"(?im)^\s*secret_key\s*:\s*(.+?)\s*$", str(text or ""))
        if not match:
            return None
        value = match.group(1).strip().strip("\"'")
        return value if cls._is_approved_searxng_secret(value) else None

    @staticmethod
    def _is_approved_searxng_secret(value: str) -> bool:
        cleaned = str(value or "").strip()
        return bool(cleaned) and cleaned != DEFAULT_SEARXNG_SECRET_KEY

    def _run_fixed(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return self._runner(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
        )

    def _is_port_available(self, port: int) -> bool:
        try:
            return bool(self._port_checker(int(port)))
        except Exception:
            return False

    @staticmethod
    def _host_port_for_bind(bind: str) -> int | None:
        if bind == APPROVED_SEARXNG_PRIMARY_BIND:
            return APPROVED_SEARXNG_PORT
        if bind == APPROVED_SEARXNG_FALLBACK_BIND:
            return APPROVED_SEARXNG_FALLBACK_PORT
        return None

    @staticmethod
    def _port_available(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False

    def _wait_for_health(self, url: str) -> tuple[bool, dict[str, Any]]:
        started = time.monotonic()
        deadline = started + self._health_timeout_seconds
        attempts = 0
        last: dict[str, Any] = {"ok": False, "error": "not_checked"}
        while True:
            attempts += 1
            ok, last = self._check_health_once(url)
            if ok:
                return True, {
                    "attempts": attempts,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "last_health_status": last.get("status"),
                    "last_health_method": last.get("method"),
                }
            if time.monotonic() >= deadline:
                break
            time.sleep(min(self._health_poll_interval_seconds, max(0.0, deadline - time.monotonic())))
        return False, {
            "attempts": attempts,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "last_health_status": last.get("status"),
            "last_health_error": last.get("error"),
            "last_health_method": last.get("method"),
        }

    def _check_health_once(self, url: str) -> tuple[bool, dict[str, Any]]:
        if callable(self._health_checker):
            try:
                ok = bool(self._health_checker(url))
                return ok, {"ok": ok, "method": "custom", "status": 200 if ok else None, "error": None if ok else "custom_health_checker_false"}
            except Exception as exc:
                return False, {"ok": False, "method": "custom", "error": exc.__class__.__name__}
        head_ok, head_info = self._http_health_probe(url, method="HEAD")
        if head_ok:
            return True, head_info
        get_ok, get_info = self._http_health_probe(url, method="GET")
        if get_ok:
            return True, get_info
        return False, get_info if get_info.get("error") else head_info

    def _http_health_probe(self, url: str, *, method: str) -> tuple[bool, dict[str, Any]]:
        request = urllib.request.Request(url, method=method, headers={"User-Agent": "personal-agent-managed-service/1"})
        try:
            with urllib.request.urlopen(request, timeout=self._health_probe_timeout_seconds) as response:  # nosec B310 - fixed loopback URL after validation.
                status = int(getattr(response, "status", 0) or 0)
                return status == 200, {"ok": status == 200, "method": method, "status": status}
        except urllib.error.HTTPError as exc:
            return False, {"ok": False, "method": method, "status": int(getattr(exc, "code", 0) or 0), "error": "http_error"}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return False, {"ok": False, "method": method, "error": exc.__class__.__name__}

    def _capture_failure_diagnostics(self, plan: ManagedLocalServiceSetupPlan, health: dict[str, Any]) -> dict[str, Any]:
        ps_result = self._run_fixed([plan.engine, "ps", "-a", "--filter", f"name=^/{plan.container_name}$", "--no-trunc"])
        logs_result = self._run_fixed([plan.engine, "logs", "--tail", "120", plan.container_name])
        return {
            "container_name": plan.container_name,
            "selected_engine": plan.engine,
            "health": dict(health),
            "ps_returncode": int(ps_result.returncode),
            "ps_output": self._redact_diagnostic_text(str(ps_result.stdout or ps_result.stderr or "")),
            "logs_returncode": int(logs_result.returncode),
            "logs_tail": self._redact_diagnostic_text(str(logs_result.stdout or logs_result.stderr or "")),
        }

    @staticmethod
    def _redact_diagnostic_text(text: str) -> str:
        redacted = str(text or "")
        redacted = re.sub(r"(?im)^(\s*secret_key\s*:\s*)(.+)$", r"\1<redacted>", redacted)
        redacted = re.sub(r"(?i)(server\.secret_key\s*[=:]\s*)(\S+)", r"\1<redacted>", redacted)
        for marker in ("token=", "api_key=", "apikey=", "password=", "passwd=", "secret="):
            lowered = redacted.lower()
            start = lowered.find(marker)
            while start >= 0:
                end = start + len(marker)
                while end < len(redacted) and not redacted[end].isspace():
                    end += 1
                redacted = f"{redacted[:start]}{marker}<redacted>{redacted[end:]}"
                lowered = redacted.lower()
                start = lowered.find(marker, start + len(marker) + len("<redacted>"))
        return redacted[:4000]

    @staticmethod
    def _short_process_error(result: subprocess.CompletedProcess[str]) -> str:
        text = str(result.stderr or result.stdout or "").strip()
        return text[:500] if text else f"exit status {result.returncode}"


def redact_service_url(url: str | None) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return "invalid-url"
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}"
    path = parsed.path or ""
    query = parsed.query
    if query:
        redacted_pairs: list[str] = []
        for key, value in urllib.parse.parse_qsl(query, keep_blank_values=True):
            lowered = key.lower()
            if lowered in {"token", "key", "api_key", "access_token", "auth", "signature", "sig"}:
                redacted_pairs.append((key, "<redacted>"))
            else:
                redacted_pairs.append((key, value))
        query = urllib.parse.urlencode(redacted_pairs)
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, query, ""))


def _is_loopback_http_url(url: str | None) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


class ManagedLocalServiceDetector:
    """Read-only detector for approved local service prerequisites and status."""

    def __init__(
        self,
        *,
        search_status_provider: Callable[[], dict[str, Any]] | None = None,
        command_finder: Callable[[str], str | None] | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        health_checker: Callable[[str], bool] | None = None,
        searxng_url_provider: Callable[[], str | None] | None = None,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._search_status_provider = search_status_provider
        self._command_finder = command_finder or trusted_command_path
        self._command_runner = command_runner or subprocess.run
        self._health_checker = health_checker
        self._searxng_url_provider = searxng_url_provider
        self._timeout_seconds = max(0.1, float(timeout_seconds or 1.0))

    def status(self) -> dict[str, Any]:
        docker_path = self._command_finder("docker")
        podman_path = self._command_finder("podman")
        docker_available = bool(docker_path)
        podman_available = bool(podman_path)
        podman_rootless = self._podman_rootless(podman_path) if podman_path else False
        docker_rootless = self._docker_rootless(docker_path) if docker_path else False
        searxng = self._searxng_status(
            docker_available=docker_available,
            podman_available=podman_available,
            podman_rootless=podman_rootless,
            docker_rootless=docker_rootless,
        )
        return {
            "ok": True,
            "read_only": True,
            "mutating_actions_enabled": False,
            "docker_available": docker_available,
            "podman_available": podman_available,
            "docker_path": docker_path,
            "podman_path": podman_path,
            "podman_found": podman_available,
            "docker_found": docker_available,
            "podman_version": self._runtime_version(podman_path),
            "docker_version": self._runtime_version(docker_path),
            "detection_source": "PATH_OR_TRUSTED_ABSOLUTE_PATH",
            "service_path": os.environ.get("PATH", ""),
            "podman_rootless": podman_rootless,
            "docker_rootless": docker_rootless,
            "services": [searxng],
        }

    def _search_status(self) -> dict[str, Any]:
        if not callable(self._search_status_provider):
            return {}
        try:
            payload = self._search_status_provider()
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _searxng_status(
        self,
        *,
        docker_available: bool,
        podman_available: bool,
        podman_rootless: bool | None,
        docker_rootless: bool | None,
    ) -> dict[str, Any]:
        spec = SERVICE_REGISTRY["searxng"]
        search_status = self._search_status()
        enabled = bool(search_status.get("enabled", False))
        configured = bool(search_status.get("endpoint_configured", False))
        configured_url = str(search_status.get("base_url") or search_status.get("url") or "").strip()
        if not configured_url and callable(self._searxng_url_provider):
            try:
                configured_url = str(self._searxng_url_provider() or "").strip()
            except Exception:
                configured_url = ""
        expected_url = configured_url or spec.default_local_url
        should_probe_reachability = (enabled or configured) and _is_loopback_http_url(expected_url)
        reachable = self._check_reachable(expected_url) if should_probe_reachability else False
        if enabled and configured and reachable:
            next_step = "ready"
        elif not configured or not enabled:
            next_step = "setup_preview_available" if docker_available or podman_available else "install_docker_or_podman_manually"
        elif not reachable:
            next_step = "check_or_start_local_service"
        else:
            next_step = "setup_preview_available"
        return {
            **spec.to_public_dict(),
            "configured": configured,
            "enabled": enabled,
            "reachable": reachable,
            "url": redact_service_url(expected_url),
            "next_step": next_step,
            "docker_available": docker_available,
            "podman_available": podman_available,
            "podman_found": podman_available,
            "podman_rootless": podman_rootless,
            "docker_rootless": docker_rootless,
            "preferred_engine": "podman",
            "container_detection": {
                "checked": False,
                "reason": "read_only_v1_uses_cli_presence_only",
            },
        }

    def _podman_rootless(self, podman_path: str | None) -> bool | None:
        return self._runtime_rootless(podman_path, "podman")

    def _docker_rootless(self, docker_path: str | None) -> bool | None:
        return self._runtime_rootless(docker_path, "docker")

    def _runtime_rootless(self, command_path: str | None, engine: str) -> bool | None:
        if not command_path:
            return False
        argv = [command_path, "info", "--format", "{{.Host.Security.Rootless}}"]
        if engine == "docker":
            argv = [command_path, "info", "--format", "{{.SecurityOptions}}"]
        try:
            result = self._command_runner(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                shell=False,
            )
        except Exception:
            return None
        if int(getattr(result, "returncode", 1) or 0) != 0:
            return None
        output = " ".join(str(getattr(result, "stdout", "") or "").strip().lower().split())
        if engine == "docker":
            if "name=rootless" in output or "rootless" in output:
                return True
            return False if output else None
        if output in {"true", "1", "yes"}:
            return True
        if output in {"false", "0", "no"}:
            return False
        return None

    def _runtime_version(self, command_path: str | None) -> str | None:
        if not command_path:
            return None
        try:
            result = self._command_runner(
                [command_path, "--version"],
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                shell=False,
            )
        except Exception:
            return None
        if int(getattr(result, "returncode", 1) or 0) != 0:
            return None
        value = " ".join(str(getattr(result, "stdout", "") or "").strip().split())
        return value[:160] if value else None

    def _check_reachable(self, url: str) -> bool:
        if not _is_loopback_http_url(url):
            return False
        if callable(self._health_checker):
            try:
                return bool(self._health_checker(url))
            except Exception:
                return False
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": "personal-agent-service-detector/1"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:  # nosec B310 - loopback-only URL above
                return 200 <= int(getattr(response, "status", 0) or 0) < 500
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False


def build_managed_local_services_status(
    *,
    search_status_provider: Callable[[], dict[str, Any]] | None = None,
    command_finder: Callable[[str], str | None] | None = None,
    health_checker: Callable[[str], bool] | None = None,
    searxng_url_provider: Callable[[], str | None] | None = None,
    timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    started = time.monotonic()
    detector = ManagedLocalServiceDetector(
        search_status_provider=search_status_provider,
        command_finder=command_finder,
        health_checker=health_checker,
        searxng_url_provider=searxng_url_provider,
        timeout_seconds=timeout_seconds,
    )
    payload = detector.status()
    payload["checked_at_monotonic_ms"] = int(started * 1000)
    return payload
