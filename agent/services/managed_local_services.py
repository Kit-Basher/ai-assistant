from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


APPROVED_SEARXNG_IMAGE = "searxng/searxng:latest"
APPROVED_SEARXNG_CONTAINER = "personal-agent-searxng"
APPROVED_SEARXNG_PORT = 8080
APPROVED_SEARXNG_FALLBACK_PORT = 8888
APPROVED_SEARXNG_VOLUME = "memory/local_services/searxng"
APPROVED_SEARXNG_PRIMARY_BIND = "127.0.0.1:8080:8080"
APPROVED_SEARXNG_FALLBACK_BIND = "127.0.0.1:8888:8080"
APPROVED_SEARXNG_BINDS = (APPROVED_SEARXNG_PRIMARY_BIND, APPROVED_SEARXNG_FALLBACK_BIND)


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
    approved_volume_path: str
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
    host_volume_path: str
    container_volume_path: str
    health_url: str
    host_port: int

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

    def run_argv(self) -> list[str]:
        return [
            self.engine,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-p",
            self.loopback_bind,
            "-v",
            f"{self.host_volume_path}:{self.container_volume_path}",
            self.image,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "engine": self.engine,
            "image": self.image,
            "container_name": self.container_name,
            "loopback_bind": self.loopback_bind,
            "host_volume_path": self.host_volume_path,
            "container_volume_path": self.container_volume_path,
            "health_url": self.health_url,
            "host_port": self.host_port,
            "pull_argv": self.pull_argv(),
            "run_argv": self.run_argv(),
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
        timeout_seconds: float = 60.0,
    ) -> None:
        self._managed_root = Path(managed_root).expanduser().resolve()
        self._command_finder = command_finder or shutil.which
        self._runner = runner or subprocess.run
        self._health_checker = health_checker
        self._port_checker = port_checker or self._port_available
        self._timeout_seconds = max(1.0, float(timeout_seconds or 60.0))

    def build_searxng_setup_plan(self, *, selected_engine: str, host_port: int = APPROVED_SEARXNG_PORT) -> ManagedLocalServiceSetupPlan:
        engine = str(selected_engine or "").strip().lower()
        spec = SERVICE_REGISTRY["searxng"]
        port = int(host_port)
        bind = APPROVED_SEARXNG_FALLBACK_BIND if port == APPROVED_SEARXNG_FALLBACK_PORT else APPROVED_SEARXNG_PRIMARY_BIND
        volume = (self._managed_root / spec.approved_volume_path).resolve()
        return ManagedLocalServiceSetupPlan(
            service_id=spec.service_id,
            engine=engine,
            image=spec.approved_image,
            container_name=spec.approved_container_name,
            loopback_bind=bind,
            host_volume_path=str(volume),
            container_volume_path="/etc/searxng",
            health_url=f"http://127.0.0.1:{port}",
            host_port=port,
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
            "approved_volume_path": spec.approved_volume_path,
        }
        for key, expected in checks.items():
            if str(params.get(key) or "").strip() != expected:
                return None, f"managed_service_plan_tampered_{key}"
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
        host_volume = Path(plan.host_volume_path).expanduser().resolve()
        approved_root = (self._managed_root / spec.approved_volume_path).resolve()
        if host_volume != approved_root:
            return "managed_service_volume_not_approved"
        if plan.container_volume_path != "/etc/searxng":
            return "managed_service_container_volume_not_approved"
        return None

    def execute_plan(self, plan: ManagedLocalServiceSetupPlan) -> ManagedLocalServiceExecutionResult:
        reason = self.validate_plan(plan)
        if reason:
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason=reason,
                plan=plan,
            )
        if not self._is_port_available(plan.host_port):
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_port_occupied",
                error=f"Port {plan.host_port} is already being used by another local app.",
                plan=plan,
                port_conflict=True,
            )
        try:
            Path(plan.host_volume_path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_volume_create_failed",
                error=str(exc),
                plan=plan,
            )
        existing = self._run_fixed(plan.existing_container_argv())
        if existing.returncode == 0 and plan.container_name in str(existing.stdout or "").splitlines():
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_container_already_exists",
                error="A container with the approved name already exists; manual inspection is required before reuse.",
                plan=plan,
            )
        pulled = self._run_fixed(plan.pull_argv())
        if pulled.returncode != 0:
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                blocked_reason="managed_service_pull_failed",
                error=self._short_process_error(pulled),
                plan=plan,
            )
        ran = self._run_fixed(plan.run_argv())
        if ran.returncode != 0:
            return ManagedLocalServiceExecutionResult(
                ok=False,
                service_id=plan.service_id,
                selected_engine=plan.engine,
                did_pull=True,
                blocked_reason="managed_service_run_failed",
                error=self._short_process_error(ran),
                plan=plan,
            )
        reachable = self._check_health(plan.health_url)
        return ManagedLocalServiceExecutionResult(
            ok=reachable,
            service_id=plan.service_id,
            selected_engine=plan.engine,
            did_pull=True,
            did_run=True,
            reachable=reachable,
            blocked_reason=None if reachable else "managed_service_health_check_failed",
            plan=plan,
        )

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

    def _check_health(self, url: str) -> bool:
        if callable(self._health_checker):
            try:
                return bool(self._health_checker(url))
            except Exception:
                return False
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": "personal-agent-managed-service/1"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310 - fixed loopback URL after validation.
                return 200 <= int(getattr(response, "status", 0) or 0) < 500
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return False

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
        health_checker: Callable[[str], bool] | None = None,
        searxng_url_provider: Callable[[], str | None] | None = None,
        timeout_seconds: float = 1.0,
    ) -> None:
        self._search_status_provider = search_status_provider
        self._command_finder = command_finder or shutil.which
        self._health_checker = health_checker
        self._searxng_url_provider = searxng_url_provider
        self._timeout_seconds = max(0.1, float(timeout_seconds or 1.0))

    def status(self) -> dict[str, Any]:
        docker_available = bool(self._command_finder("docker"))
        podman_available = bool(self._command_finder("podman"))
        searxng = self._searxng_status(docker_available=docker_available, podman_available=podman_available)
        return {
            "ok": True,
            "read_only": True,
            "mutating_actions_enabled": False,
            "docker_available": docker_available,
            "podman_available": podman_available,
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

    def _searxng_status(self, *, docker_available: bool, podman_available: bool) -> dict[str, Any]:
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
        reachable = self._check_reachable(expected_url) if configured or _is_loopback_http_url(expected_url) else False
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
            "container_detection": {
                "checked": False,
                "reason": "read_only_v1_uses_cli_presence_only",
            },
        }

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
    timeout_seconds: float = 1.0,
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
