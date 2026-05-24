from __future__ import annotations

from dataclasses import dataclass, field
import shutil
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


APPROVED_SEARXNG_IMAGE = "searxng/searxng:latest"
APPROVED_SEARXNG_CONTAINER = "personal-agent-searxng"
APPROVED_SEARXNG_PORT = 8080
APPROVED_SEARXNG_VOLUME = "memory/local_services/searxng"


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
    approved_loopback_bind="127.0.0.1:8080:8080",
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
