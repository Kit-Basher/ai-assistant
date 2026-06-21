from __future__ import annotations

from dataclasses import replace
import io
import json
import os
import subprocess
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from agent.api_server import AgentRuntime
from agent.telegram_bridge import build_telegram_chat_payload_result
from agent.services.managed_local_services import (
    APPROVED_SEARXNG_IMAGE,
    APPROVED_SEARXNG_VOLUME,
    DEFAULT_SEARXNG_SECRET_KEY,
    ManagedLocalServiceDetector,
    ManagedLocalServiceExecutor,
    redact_service_url,
)
from tests.test_api_packs_endpoints import _HandlerForTest, _config
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text


class _FakeManagedServiceRunner:
    def __init__(
        self,
        *,
        fail_on: str | None = None,
        existing: bool = False,
        inspect_payload: dict[str, object] | None = None,
    ) -> None:
        self.fail_on = fail_on
        self.existing = existing
        self.inspect_payload = inspect_payload
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": list(argv), **kwargs})
        joined = " ".join(argv)
        if kwargs.get("shell") is not False:
            return subprocess.CompletedProcess(argv, 99, stdout="", stderr="shell must be false")
        if argv[1:3] == ["ps", "-a"]:
            stdout = "personal-agent-searxng\n" if self.existing else ""
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        if len(argv) > 1 and argv[1] == "inspect":
            if self.inspect_payload is None:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="inspect failed")
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self.inspect_payload), stderr="")
        if len(argv) > 1 and argv[1] == "start":
            self.existing = True
            if isinstance(self.inspect_payload, dict):
                state = self.inspect_payload.setdefault("State", {})
                if isinstance(state, dict):
                    state["Running"] = True
                    state["Status"] = "running"
            return subprocess.CompletedProcess(argv, 0, stdout="personal-agent-searxng\n", stderr="")
        if len(argv) > 1 and argv[1] == "logs":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="mock searxng first-boot log token=secret-value\nserver:\n  secret_key: leaked-settings-secret\n",
                stderr="",
            )
        if self.fail_on and self.fail_on in joined:
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="mock failure")
        if len(argv) > 1 and argv[1] == "run":
            self.existing = True
        if len(argv) > 1 and argv[1] == "rm":
            self.existing = False
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")


class _FakeSearchResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> "_FakeSearchResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeRawResponse:
    def __init__(self, body: bytes) -> None:
        self._body = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> "_FakeRawResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeRawOpener:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def open(self, request, timeout: float = 5.0):  # noqa: ANN001
        _ = request, timeout
        return _FakeRawResponse(self.body)


class _FakeHTTPHealthResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "_FakeHTTPHealthResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSearchOpener:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.payload = payload or {"results": []}
        self.opened_urls: list[str] = []

    def open(self, request, timeout: float = 5.0):  # noqa: ANN001
        _ = timeout
        self.opened_urls.append(getattr(request, "full_url", str(request)))
        return _FakeSearchResponse(self.payload)


class _FakePrerequisiteRunner:
    def __init__(self, *, install_ok: bool = True, rootless: bool = True) -> None:
        self.install_ok = install_ok
        self.rootless = rootless
        self.installed = False
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": list(argv), **kwargs})
        if kwargs.get("shell") is not False:
            return subprocess.CompletedProcess(argv, 99, stdout="", stderr="shell must be false")
        joined = " ".join(argv)
        if "apt-get" in joined and argv[-3:] == ["install", "-y", "podman"]:
            self.installed = bool(self.install_ok)
            return subprocess.CompletedProcess(argv, 0 if self.install_ok else 2, stdout="", stderr="mock install failed")
        if Path(str(argv[0])).name == "podman" and argv[1:3] == ["info", "--format"]:
            return subprocess.CompletedProcess(argv, 0, stdout="true" if self.rootless else "false", stderr="")
        return subprocess.CompletedProcess(argv, 3, stdout="", stderr="unexpected command")


def _approved_searxng_inspect(root: str, *, running: bool = True, image: str = APPROVED_SEARXNG_IMAGE, bind: str = "127.0.0.1:8080:8080") -> dict[str, object]:
    host, host_port, container_port = bind.split(":", 2)
    return {
        "ImageName": image,
        "Config": {"Image": image},
        "State": {"Running": running, "Status": "running" if running else "exited"},
        "NetworkSettings": {"Ports": {f"{container_port}/tcp": [{"HostIp": host, "HostPort": host_port}]}},
        "Mounts": [
            {
                "Source": str((Path(root) / APPROVED_SEARXNG_VOLUME).resolve()),
                "Destination": "/etc/searxng",
            }
        ],
    }


class TestManagedLocalServices(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_status_with_no_docker_or_podman(self) -> None:
        detector = ManagedLocalServiceDetector(
            search_status_provider=lambda: {"enabled": False, "available": False, "endpoint_configured": False},
            command_finder=lambda _name: None,
            health_checker=lambda _url: False,
        )
        payload = detector.status()

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["docker_available"])
        self.assertFalse(payload["podman_available"])
        service = payload["services"][0]
        self.assertEqual("searxng", service["service_id"])
        self.assertEqual("install_docker_or_podman_manually", service["next_step"])
        self.assertIn("docker_run", service["blocked_actions"])

    def test_status_with_mocked_docker_presence(self) -> None:
        detector = ManagedLocalServiceDetector(
            search_status_provider=lambda: {"enabled": False, "available": False, "endpoint_configured": False},
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            command_runner=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, stdout="name=seccomp", stderr=""),
            health_checker=lambda _url: False,
        )
        payload = detector.status()

        self.assertTrue(payload["docker_available"])
        self.assertFalse(payload["podman_available"])
        self.assertFalse(payload["docker_rootless"])
        self.assertEqual("setup_preview_available", payload["services"][0]["next_step"])
        self.assertEqual("127.0.0.1:8080:8080", payload["services"][0]["approved_container"]["bind"])

    def test_status_with_mocked_rootless_podman_presence(self) -> None:
        timeouts: list[float] = []

        def _runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            _ = argv
            timeouts.append(float(kwargs.get("timeout") or 0.0))
            return subprocess.CompletedProcess(argv, 0, stdout="true", stderr="")

        detector = ManagedLocalServiceDetector(
            search_status_provider=lambda: {"enabled": False, "available": False, "endpoint_configured": False},
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            command_runner=_runner,
            health_checker=lambda _url: False,
            timeout_seconds=0.25,
        )
        payload = detector.status()

        self.assertFalse(payload["docker_available"])
        self.assertTrue(payload["podman_available"])
        self.assertTrue(payload["podman_rootless"])
        self.assertEqual("podman", payload["services"][0]["preferred_engine"])
        self.assertEqual([0.25], timeouts)

    def test_status_does_not_probe_default_url_when_search_never_configured(self) -> None:
        probes: list[str] = []
        detector = ManagedLocalServiceDetector(
            search_status_provider=lambda: {"enabled": False, "available": False, "endpoint_configured": False},
            command_finder=lambda _name: None,
            health_checker=lambda url: probes.append(url) or False,
        )
        payload = detector.status()

        self.assertFalse(payload["services"][0]["reachable"])
        self.assertEqual([], probes)

    def test_url_redaction_removes_credentials_and_token_query(self) -> None:
        redacted = redact_service_url("http://user:pass@127.0.0.1:8080/search?token=secret&q=test")
        self.assertEqual("http://127.0.0.1:8080/search?token=%3Credacted%3E&q=test", redacted)

    def test_external_pack_triggered_container_actions_are_blocked_by_contract(self) -> None:
        payload = ManagedLocalServiceDetector(command_finder=lambda _name: None).status()
        service = payload["services"][0]

        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["mutating_actions_enabled"])
        self.assertIn("external_pack_triggered_container_action", service["blocked_actions"])


    def test_executor_validates_approved_docker_argv_exactly(self) -> None:
        runner = _FakeManagedServiceRunner()
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
            secret_generator=lambda: "deterministic-managed-secret",
        )
        params = {
            "service_id": "searxng",
            "selected_engine": "docker",
            "action": "preview_only",
            "approved_image": "docker.io/searxng/searxng:latest",
            "approved_container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8080:8080",
            "volume_mount": True,
            "config_seeded": True,
            "approved_volume_path": "memory/local_services/searxng",
            "config_purpose": "enable_json_output_for_safe_metadata_search",
        }

        result = executor.execute_from_pending(params)

        self.assertTrue(result.ok)
        self.assertTrue(result.did_pull)
        self.assertTrue(result.did_run)
        self.assertEqual(["docker", "pull", "docker.io/searxng/searxng:latest"], runner.calls[1]["argv"])
        run_argv = runner.calls[2]["argv"]
        self.assertEqual("docker", run_argv[0])
        self.assertIn("run", run_argv)
        self.assertIn("-d", run_argv)
        self.assertIn("--name", run_argv)
        self.assertIn("personal-agent-searxng", run_argv)
        self.assertIn("127.0.0.1:8080:8080", run_argv)
        self.assertIn("-v", run_argv)
        self.assertTrue(any(str(part).endswith("memory/local_services/searxng:/etc/searxng") for part in run_argv))
        self.assertIn("docker.io/searxng/searxng:latest", run_argv)
        settings = Path(self.tmpdir.name) / "memory/local_services/searxng/settings.yml"
        settings_text = settings.read_text(encoding="utf-8")
        self.assertIn("use_default_settings: true", settings_text)
        self.assertIn("server:", settings_text)
        self.assertIn('secret_key: "deterministic-managed-secret"', settings_text)
        self.assertIn("- json", settings_text)
        self.assertNotIn(DEFAULT_SEARXNG_SECRET_KEY, settings_text)
        result_payload = json.dumps(result.to_dict(), sort_keys=True)
        self.assertNotIn("deterministic-managed-secret", result_payload)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))

    def test_seeded_searxng_config_preserves_existing_acceptable_secret(self) -> None:
        settings = Path(self.tmpdir.name) / "memory/local_services/searxng/settings.yml"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            """\
use_default_settings: true

server:
  secret_key: "existing-good-secret"

search:
  formats:
    - html
    - json
""",
            encoding="utf-8",
        )
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
            secret_generator=lambda: "new-generated-secret",
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertTrue(result.ok)
        settings_text = settings.read_text(encoding="utf-8")
        self.assertIn('secret_key: "existing-good-secret"', settings_text)
        self.assertNotIn("new-generated-secret", settings_text)

    def test_seeded_searxng_config_replaces_missing_or_default_secret(self) -> None:
        settings = Path(self.tmpdir.name) / "memory/local_services/searxng/settings.yml"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            """\
use_default_settings: true

server:
  secret_key: "ultrasecretkey"

search:
  formats:
    - html
    - json
""",
            encoding="utf-8",
        )
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
            secret_generator=lambda: "replacement-managed-secret",
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertTrue(result.ok)
        settings_text = settings.read_text(encoding="utf-8")
        self.assertIn('secret_key: "replacement-managed-secret"', settings_text)
        self.assertNotIn(DEFAULT_SEARXNG_SECRET_KEY, settings_text)

    def test_executor_preview_uses_primary_or_fallback_port(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda port: port == 8888,
        )

        preview = executor.preview_setup_from_status(service_id="searxng", selected_engine="docker")

        self.assertTrue(preview["ok"])
        self.assertTrue(preview["fallback_selected"])
        self.assertTrue(preview["port_conflict"])
        self.assertEqual("127.0.0.1:8888:8080", preview["plan"]["loopback_bind"])
        self.assertEqual("http://127.0.0.1:8888", preview["plan"]["health_url"])

    def test_executor_preview_blocks_when_both_approved_ports_busy(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: False,
        )

        preview = executor.preview_setup_from_status(service_id="searxng", selected_engine="docker")

        self.assertFalse(preview["ok"])
        self.assertEqual("managed_service_approved_ports_occupied", preview["blocked_reason"])

    def test_podman_plan_argv_uses_fully_qualified_searxng_image(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        plan = executor.build_searxng_setup_plan(selected_engine="podman")

        self.assertEqual("docker.io/searxng/searxng:latest", plan.image)
        self.assertEqual(["podman", "pull", "docker.io/searxng/searxng:latest"], plan.pull_argv())
        self.assertTrue(plan.volume_mount)
        self.assertTrue(plan.config_seeded)
        self.assertEqual("enable_json_output_for_safe_metadata_search", plan.config_purpose)
        self.assertIn("-v", plan.run_argv())
        self.assertTrue(any(str(part).endswith("memory/local_services/searxng:/etc/searxng") for part in plan.run_argv()))
        self.assertIn("docker.io/searxng/searxng:latest", plan.run_argv())

    def test_executor_fallback_run_uses_loopback_only(self) -> None:
        runner = _FakeManagedServiceRunner()
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8888",
            port_checker=lambda port: port == 8888,
        )
        params = {
            "service_id": "searxng",
            "selected_engine": "docker",
            "action": "preview_only",
            "approved_image": "docker.io/searxng/searxng:latest",
            "approved_container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8888:8080",
            "volume_mount": True,
            "config_seeded": True,
            "approved_volume_path": "memory/local_services/searxng",
            "config_purpose": "enable_json_output_for_safe_metadata_search",
        }

        result = executor.execute_from_pending(params)

        self.assertTrue(result.ok)
        run_argv = runner.calls[2]["argv"]
        self.assertIn("127.0.0.1:8888:8080", run_argv)
        self.assertNotIn("0.0.0.0:8888:8080", run_argv)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))

    def test_executor_rejects_tampered_fallback_bind(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )
        result = executor.execute_from_pending(
            {
                "service_id": "searxng",
                "selected_engine": "docker",
                "action": "preview_only",
                "approved_image": "docker.io/searxng/searxng:latest",
                "approved_container_name": "personal-agent-searxng",
                "loopback_bind": "0.0.0.0:8888:8080",
                "volume_mount": True,
                "config_seeded": True,
                "approved_volume_path": "memory/local_services/searxng",
                "config_purpose": "enable_json_output_for_safe_metadata_search",
            }
        )

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_bind_not_approved", result.blocked_reason)

    def test_executor_port_occupied_stops_before_pull(self) -> None:
        runner = _FakeManagedServiceRunner()
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: False,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_port_occupied", result.blocked_reason)
        self.assertTrue(result.port_conflict)
        self.assertEqual([["docker", "ps", "-a", "--filter", "name=^/personal-agent-searxng$", "--format", "{{.Names}}"]], [call["argv"] for call in runner.calls])

    def test_unwritable_config_dir_blocks_before_pull_or_run_with_handoff(self) -> None:
        runner = _FakeManagedServiceRunner()
        config_dir = Path(self.tmpdir.name) / APPROVED_SEARXNG_VOLUME
        config_dir.mkdir(parents=True)
        config_dir.chmod(0o500)
        try:
            executor = ManagedLocalServiceExecutor(
                managed_root=self.tmpdir.name,
                command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
                runner=runner,
                health_checker=lambda _url: True,
                port_checker=lambda _port: True,
            )

            result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))
        finally:
            config_dir.chmod(0o700)

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_config_dir_not_writable", result.blocked_reason)
        self.assertFalse(result.did_pull)
        self.assertFalse(result.did_run)
        self.assertFalse(result.did_configure)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertNotIn(["podman", "pull", "docker.io/searxng/searxng:latest"], argv_rows)
        self.assertFalse(any(len(argv) > 1 and argv[1] == "run" for argv in argv_rows))
        handoff = result.diagnostics.get("operator_handoff", {})
        self.assertEqual("repair_searxng_config_ownership", handoff.get("bounded_action"))
        self.assertIn('sudo chown -R "$USER:$USER"', handoff.get("command_string", ""))
        self.assertIn("chmod -R u+rwX", handoff.get("command_string", ""))

    def test_existing_matching_running_container_is_reused(self) -> None:
        runner = _FakeManagedServiceRunner(
            existing=True,
            inspect_payload=_approved_searxng_inspect(self.tmpdir.name, running=True),
        )
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: False,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertTrue(result.ok)
        self.assertTrue(result.reachable)
        self.assertFalse(result.did_pull)
        self.assertFalse(result.did_run)
        self.assertTrue(result.diagnostics.get("reused_existing_container"))
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["podman", "inspect", "personal-agent-searxng", "--format", "{{json .}}"], argv_rows)
        self.assertFalse(any(len(argv) > 1 and argv[1] in {"pull", "run", "rm"} for argv in argv_rows))

    def test_existing_matching_running_fallback_container_restores_actual_port(self) -> None:
        runner = _FakeManagedServiceRunner(
            existing=True,
            inspect_payload=_approved_searxng_inspect(self.tmpdir.name, running=True, bind="127.0.0.1:8888:8080"),
        )
        health_urls: list[str] = []
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: health_urls.append(url) or url == "http://127.0.0.1:8888",
            port_checker=lambda _port: True,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertTrue(result.ok)
        self.assertEqual("127.0.0.1:8888:8080", result.plan.loopback_bind)
        self.assertEqual("http://127.0.0.1:8888", result.plan.health_url)
        self.assertEqual(["http://127.0.0.1:8888"], health_urls)

    def test_existing_matching_stopped_container_is_restarted_for_repair(self) -> None:
        runner = _FakeManagedServiceRunner(
            existing=True,
            inspect_payload=_approved_searxng_inspect(self.tmpdir.name, running=False),
        )
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertTrue(result.ok)
        self.assertTrue(result.reachable)
        self.assertTrue(result.diagnostics.get("repaired_existing_container"))
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["podman", "start", "personal-agent-searxng"], argv_rows)
        self.assertFalse(any(len(argv) > 1 and argv[1] in {"pull", "run", "rm"} for argv in argv_rows))

    def test_existing_container_mismatched_image_blocks_without_cleanup(self) -> None:
        runner = _FakeManagedServiceRunner(
            existing=True,
            inspect_payload=_approved_searxng_inspect(self.tmpdir.name, image="docker.io/other/image:latest"),
        )
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_existing_container_image_mismatch", result.blocked_reason)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertFalse(any(len(argv) > 1 and argv[1] in {"rm", "stop", "run", "pull"} for argv in argv_rows))

    def test_existing_container_public_bind_blocks_without_cleanup(self) -> None:
        runner = _FakeManagedServiceRunner(
            existing=True,
            inspect_payload=_approved_searxng_inspect(self.tmpdir.name, bind="0.0.0.0:8080:8080"),
        )
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_existing_container_bind_mismatch", result.blocked_reason)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertFalse(any(len(argv) > 1 and argv[1] in {"rm", "stop", "run", "pull"} for argv in argv_rows))

    def test_existing_container_wrong_mount_blocks_without_cleanup(self) -> None:
        inspect_payload = _approved_searxng_inspect(self.tmpdir.name)
        inspect_payload["Mounts"] = [{"Source": "/tmp/other", "Destination": "/etc/searxng"}]
        runner = _FakeManagedServiceRunner(existing=True, inspect_payload=inspect_payload)
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_existing_container_mount_mismatch", result.blocked_reason)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertFalse(any(len(argv) > 1 and argv[1] in {"rm", "stop", "run", "pull"} for argv in argv_rows))

    def test_executor_rejects_tampered_pending_fields(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )
        base = {
            "service_id": "searxng",
            "selected_engine": "docker",
            "action": "preview_only",
            "approved_image": "docker.io/searxng/searxng:latest",
            "approved_container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8080:8080",
            "volume_mount": True,
            "config_seeded": True,
            "approved_volume_path": "memory/local_services/searxng",
            "config_purpose": "enable_json_output_for_safe_metadata_search",
        }
        for key, value in {
            "approved_image": "evil/image:latest",
            "approved_container_name": "evil-name",
            "loopback_bind": "0.0.0.0:8080:8080",
            "approved_volume_path": "/tmp/random",
            "volume_mount": False,
            "config_seeded": False,
            "settings_yml": "search:\n  formats: [html, json]\n",
        }.items():
            params = dict(base)
            params[key] = value
            result = executor.execute_from_pending(params)
            self.assertFalse(result.ok, key)
            self.assertTrue(
                (result.blocked_reason or "").startswith("managed_service_plan_tampered")
                or result.blocked_reason == "managed_service_bind_not_approved",
                result.blocked_reason,
            )

    def test_executor_rejects_old_short_name_searxng_image(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.execute_from_pending(
            {
                "service_id": "searxng",
                "selected_engine": "podman",
                "action": "preview_only",
                "approved_image": "searxng/searxng:latest",
                "approved_container_name": "personal-agent-searxng",
                "loopback_bind": "127.0.0.1:8080:8080",
                "volume_mount": True,
                "config_seeded": True,
                "approved_volume_path": "memory/local_services/searxng",
                "config_purpose": "enable_json_output_for_safe_metadata_search",
            }
        )

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_plan_tampered_approved_image", result.blocked_reason)

    def test_executor_rejects_unknown_service_and_non_loopback_plan(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )
        unknown = executor.execute_from_pending({"service_id": "other", "selected_engine": "docker"})
        self.assertFalse(unknown.ok)
        self.assertEqual("managed_service_unknown", unknown.blocked_reason)

        plan = executor.build_searxng_setup_plan(selected_engine="docker")
        tampered = replace(plan, loopback_bind="0.0.0.0:8080:8080")
        self.assertEqual("managed_service_bind_not_approved", executor.validate_plan(tampered))

    def test_executor_run_failure_reports_no_config_mutation(self) -> None:
        runner = _FakeManagedServiceRunner(fail_on=" run ")
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )
        plan = executor.build_searxng_setup_plan(selected_engine="docker")
        result = executor.execute_plan(plan)

        self.assertFalse(result.ok)
        self.assertTrue(result.did_pull)
        self.assertFalse(result.did_run)
        self.assertFalse(result.did_install)
        self.assertFalse(result.did_configure)
        self.assertEqual("managed_service_run_failed", result.blocked_reason)

    def test_existing_container_without_approved_inspect_is_conservative(self) -> None:
        runner = _FakeManagedServiceRunner(existing=True)
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )
        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_existing_container_inspect_failed", result.blocked_reason)
        self.assertEqual(2, len(runner.calls))

    def test_failed_health_check_rolls_back_created_container(self) -> None:
        runner = _FakeManagedServiceRunner()
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: False,
            port_checker=lambda _port: True,
            health_timeout_seconds=0.01,
            health_poll_interval_seconds=0.01,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertFalse(result.ok)
        self.assertTrue(result.did_pull)
        self.assertTrue(result.did_run)
        self.assertEqual("managed_service_health_check_failed", result.blocked_reason)
        self.assertTrue(result.rollback_attempted)
        self.assertTrue(result.rollback_ok)
        self.assertTrue(result.cleanup_performed)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["docker", "stop", "personal-agent-searxng"], argv_rows)
        self.assertIn(["docker", "rm", "personal-agent-searxng"], argv_rows)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))
        journal = result.journal
        self.assertTrue(journal.get("created_resources"))
        self.assertTrue(journal.get("rollback_steps"))
        self.assertIn("mock searxng first-boot log", result.diagnostics["logs_tail"])
        self.assertNotIn("secret-value", result.diagnostics["logs_tail"])
        self.assertNotIn("leaked-settings-secret", result.diagnostics["logs_tail"])
        self.assertIn("secret_key: <redacted>", result.diagnostics["logs_tail"])
        self.assertEqual("custom_health_checker_false", result.diagnostics["health"]["last_health_error"])
        self.assertIn("capture_failure_diagnostics", [step["name"] for step in journal.get("executed_steps", [])])

    def test_health_check_retries_until_first_boot_succeeds(self) -> None:
        runner = _FakeManagedServiceRunner()
        health_calls: list[str] = []

        def _health(url: str) -> bool:
            health_calls.append(url)
            return len(health_calls) >= 3

        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=_health,
            port_checker=lambda _port: True,
            health_timeout_seconds=1.0,
            health_poll_interval_seconds=0.01,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="podman"))

        self.assertTrue(result.ok)
        self.assertTrue(result.reachable)
        self.assertEqual(3, len(health_calls))
        self.assertEqual(3, result.diagnostics["attempts"])
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertNotIn(["podman", "stop", "personal-agent-searxng"], argv_rows)
        self.assertNotIn(["podman", "rm", "personal-agent-searxng"], argv_rows)

    def test_http_health_probe_retries_get_after_head_failure(self) -> None:
        methods: list[str] = []

        def _urlopen(request, timeout: float = 5.0):  # noqa: ANN001
            _ = timeout
            method = str(getattr(request, "get_method")())
            methods.append(method)
            return _FakeHTTPHealthResponse(405 if method == "HEAD" else 200)

        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=_FakeManagedServiceRunner(),
            port_checker=lambda _port: True,
            health_timeout_seconds=0.01,
            health_poll_interval_seconds=0.01,
        )

        with mock.patch("agent.services.managed_local_services.urllib.request.urlopen", side_effect=_urlopen):
            ok, diagnostics = executor._wait_for_health("http://127.0.0.1:8888")  # noqa: SLF001

        self.assertTrue(ok)
        self.assertEqual(["HEAD", "GET"], methods)
        self.assertEqual("GET", diagnostics["last_health_method"])
        self.assertEqual(200, diagnostics["last_health_status"])

    def test_rollback_does_not_delete_preexisting_container(self) -> None:
        runner = _FakeManagedServiceRunner(existing=True)
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: False,
            port_checker=lambda _port: True,
            health_timeout_seconds=0.01,
            health_poll_interval_seconds=0.01,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_existing_container_inspect_failed", result.blocked_reason)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertNotIn(["docker", "stop", "personal-agent-searxng"], argv_rows)
        self.assertNotIn(["docker", "rm", "personal-agent-searxng"], argv_rows)

    def test_partial_rollback_failure_reports_remaining_cleanup(self) -> None:
        runner = _FakeManagedServiceRunner(fail_on=" rm ")
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: False,
            port_checker=lambda _port: True,
            health_timeout_seconds=0.01,
            health_poll_interval_seconds=0.01,
        )

        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertFalse(result.ok)
        self.assertTrue(result.rollback_attempted)
        self.assertFalse(result.rollback_ok)
        self.assertTrue(result.cleanup_incomplete)
        self.assertIn("Cleanup was incomplete", result.error or "")

    def test_stop_plan_removes_only_approved_managed_container(self) -> None:
        runner = _FakeManagedServiceRunner(existing=True)
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.stop_from_pending(
            {
                "service_id": "searxng",
                "selected_engine": "docker",
                "action": "stop_preview_only",
                "approved_container_name": "personal-agent-searxng",
            }
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.did_stop)
        self.assertTrue(result.did_remove)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["docker", "stop", "personal-agent-searxng"], argv_rows)
        self.assertIn(["docker", "rm", "personal-agent-searxng"], argv_rows)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))

    def test_stop_plan_rejects_tampered_container_name(self) -> None:
        runner = _FakeManagedServiceRunner(existing=True)
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        result = executor.stop_from_pending(
            {
                "service_id": "searxng",
                "selected_engine": "docker",
                "action": "stop_preview_only",
                "approved_container_name": "some-other-container",
            }
        )

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_plan_tampered_approved_container_name", result.blocked_reason)
        self.assertEqual([], runner.calls)


class TestManagedLocalServicesEndpointAndChat(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = str(Path(__file__).resolve().parents[1] / "skills")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self, *, search_enabled: bool = False, endpoint: str | None = None) -> AgentRuntime:
        config = replace(
            _config(self.registry_path, self.db_path, self.skills_path),
            search_enabled=search_enabled,
            search_provider="searxng",
            searxng_base_url=endpoint,
            search_timeout_seconds=0.2,
            search_max_results=3,
        )
        return AgentRuntime(config)

    def test_services_status_endpoint_is_read_only(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda _name: None,
            health_checker=lambda _url: False,
        )
        handler = _HandlerForTest(runtime, "/services/status")
        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["mutating_actions_enabled"])
        self.assertFalse(payload["docker_available"])
        self.assertEqual("searxng", payload["services"][0]["service_id"])

    def test_managed_services_status_uses_short_read_only_cache(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        calls = 0

        def _runner(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            calls += 1
            return subprocess.CompletedProcess(argv, 0, stdout="true", stderr="")

        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda name: "/usr/bin/podman" if name == "podman" else None,
            command_runner=_runner,
            health_checker=lambda _url: False,
        )
        runtime._managed_services_status_cache_ttl_seconds = 60.0  # noqa: SLF001

        first = runtime.managed_services_status()
        second = runtime.managed_services_status()
        fresh = runtime.managed_services_status(force_refresh=True)

        self.assertTrue(first["podman_available"])
        self.assertTrue(second["podman_available"])
        self.assertTrue(fresh["podman_available"])
        self.assertEqual(2, calls)

    def test_search_status_disabled_has_blocked_next_action(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        handler = _HandlerForTest(runtime, "/search/status")

        handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertFalse(payload["available"])
        self.assertEqual("search_disabled", payload["reason"])
        self.assertIn("Preview and confirm", payload["next_action"])
        self.assertEqual("unconfigured", payload["setup_source"])

    def test_user_provided_loopback_status_probes_availability(self) -> None:
        runtime = self._runtime(search_enabled=True, endpoint="http://127.0.0.1:8888")
        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            handler = _HandlerForTest(runtime, "/search/status")
            handler.do_GET()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertTrue(payload["available"])
        self.assertEqual("http://127.0.0.1:8888", payload["base_url"])
        self.assertEqual("managed_or_user_loopback", payload["setup_source"])

    def test_search_setup_plan_rejects_non_loopback_user_url(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        handler = _HandlerForTest(runtime, "/search/setup/plan", {"base_url": "https://search.example.test"})

        handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(400, handler.status_code)
        self.assertFalse(payload["ok"])
        self.assertEqual("searxng_endpoint_not_loopback", payload["blocked_reason"])

    def test_search_setup_plan_selects_rootless_podman_when_available(self) -> None:
        runtime = self._runtime_with_engine("podman", podman_rootless=True)
        handler = _HandlerForTest(runtime, "/search/setup/plan", {})

        handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))
        plan = payload["plan"]

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertEqual("podman", plan["selected_engine"])
        self.assertEqual("podman", plan["preferred_engine"])
        self.assertTrue(plan["rootless_expected"])
        self.assertFalse(plan["requires_docker_fallback_confirmation"])
        self.assertIsNone(plan["fallback_reason"])
        self.assertEqual("docker.io/searxng/searxng:latest", plan["image"])
        self.assertEqual("docker.io/searxng/searxng:latest", plan["executor_pending"]["approved_image"])
        self.assertTrue(plan["volume_mount"])
        self.assertTrue(plan["config_seeded"])
        self.assertEqual("memory/local_services/searxng", plan["approved_volume_path"])
        self.assertEqual("enable_json_output_for_safe_metadata_search", plan["config_purpose"])
        self.assertEqual(["memory/local_services/searxng/settings.yml"], plan["state_files_touched"])
        self.assertTrue(plan["executor_pending"]["volume_mount"])
        self.assertTrue(plan["executor_pending"]["config_seeded"])

    def test_search_setup_plan_marks_docker_as_explicit_fallback(self) -> None:
        runtime = self._runtime_with_engine("docker")
        handler = _HandlerForTest(runtime, "/search/setup/plan", {"allow_docker_fallback": True})

        handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))
        plan = payload["plan"]

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertEqual("docker", plan["selected_engine"])
        self.assertEqual("podman", plan["preferred_engine"])
        self.assertEqual("rootless_podman_not_found", plan["fallback_reason"])
        self.assertFalse(plan["rootless_expected"])
        self.assertTrue(plan["requires_docker_fallback_confirmation"])
        self.assertIn("Podman was not found", plan["engine_warning"])
        self.assertIn("root-level daemon", plan["engine_warning"])

    def test_search_setup_plan_blocks_when_no_runtime_available(self) -> None:
        runtime = self._runtime_with_engine(None)
        handler = _HandlerForTest(runtime, "/search/setup/plan", {})

        handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertEqual("podman_prerequisite", payload["plan"]["setup_mode"])
        self.assertEqual("podman", payload["plan"]["prerequisite"])
        self.assertTrue(payload["requires_confirmation"])

    def test_podman_prerequisite_setup_requires_confirmation(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite()
        handler = _HandlerForTest(runtime, "/search/setup/prerequisite/plan", {})

        handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(200, handler.status_code)
        self.assertTrue(payload["requires_confirmation"])
        self.assertEqual("podman_prerequisite", payload["plan"]["setup_mode"])
        self.assertEqual([["sudo", "apt-get", "install", "-y", "podman"]], payload["plan"]["commands"])
        self.assertEqual([], runner.calls)

    def test_podman_prerequisite_apply_rejects_invalid_and_expired_confirmation(self) -> None:
        runtime, _runner = self._runtime_with_podman_prerequisite()
        invalid = runtime.apply_podman_prerequisite({"plan_id": "missing", "confirmation_token": "bad"})
        plan_payload = runtime.podman_prerequisite_plan({})
        plan = plan_payload["plan"]
        runtime._podman_prerequisite_confirmations[plan["plan_id"]]["expires_at"] = time.time() - 1  # noqa: SLF001
        expired = runtime.apply_podman_prerequisite({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertFalse(invalid["ok"])
        self.assertEqual("invalid_confirmation", invalid["error"])
        self.assertFalse(expired["ok"])
        self.assertEqual("confirmation_expired", expired["error"])
        self.assertFalse(runtime.config.search_enabled)

    def test_search_setup_apply_returns_elevated_handoff_for_podman_prerequisite_plan_token(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite(install_ok=False)
        plan_handler = _HandlerForTest(runtime, "/search/setup/plan", {})
        plan_handler.do_POST()
        plan_payload = json.loads(plan_handler.body.decode("utf-8"))
        plan = plan_payload["plan"]
        self.assertEqual("podman_prerequisite", plan["setup_mode"])

        apply_handler = _HandlerForTest(
            runtime,
            "/search/setup/apply",
            {"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]},
        )
        apply_handler.do_POST()
        payload = json.loads(apply_handler.body.decode("utf-8"))

        self.assertEqual(400, apply_handler.status_code)
        self.assertFalse(payload["ok"])
        self.assertNotEqual("invalid_confirmation", payload.get("error"))
        self.assertEqual("elevated_handoff_required", payload["error"])
        self.assertTrue(payload["elevated_handoff_required"])
        handoff = payload["handoff"]
        self.assertEqual(["sudo", "apt-get", "install", "-y", "podman"], handoff["command"])
        self.assertEqual([["sudo", "apt-get", "install", "-y", "podman"]], handoff["allowed_commands"])
        self.assertIn("command -v podman", handoff["after_command_verification"])
        self.assertIn("podman --version", handoff["after_command_verification"])
        self.assertIn("podman info", handoff["after_command_verification"][2])
        self.assertFalse(runtime.config.search_enabled)
        self.assertEqual([], runner.calls)

    def test_search_setup_apply_podman_prerequisite_safe_mode_blocks_clearly(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite()
        runtime.config = replace(runtime.config, safe_mode_enabled=True)
        plan_payload = runtime.search_setup_plan({})
        plan = plan_payload["plan"]

        result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertFalse(result["ok"])
        self.assertEqual("safe_mode_blocked", result["error"])
        self.assertFalse(result["mutated"])
        self.assertEqual([], runner.calls)
        self.assertFalse(runtime.config.search_enabled)

    def test_podman_prerequisite_apply_uses_allowlisted_command_and_verifies_rootless(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite(running_as_root=True)

        result = self._install_podman_prerequisite(runtime)

        self.assertTrue(result["ok"])
        self.assertTrue(result["podman_present"])
        self.assertTrue(result["rootless_podman"])
        self.assertFalse(runtime.config.search_enabled)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertEqual(["/usr/bin/apt-get", "install", "-y", "podman"], argv_rows[0])
        self.assertEqual(["/usr/bin/podman", "info", "--format", "{{.Host.Security.Rootless}}"], argv_rows[1])
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))
        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("verified", row["status"])
        self.assertEqual("podman_prerequisite_setup", row["action_type"])

    def test_podman_prerequisite_apply_ignores_arbitrary_stored_command_strings(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite(running_as_root=True)
        plan_payload = runtime.podman_prerequisite_plan({})
        plan = plan_payload["plan"]
        stored = runtime._podman_prerequisite_confirmations[plan["plan_id"]]["plan"]  # noqa: SLF001
        stored["commands"] = [["sudo", "apt-get", "install", "-y", "curl"]]

        result = runtime.apply_podman_prerequisite(
            {"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]}
        )

        self.assertTrue(result["ok"])
        self.assertEqual(["/usr/bin/apt-get", "install", "-y", "podman"], runner.calls[0]["argv"])
        self.assertFalse(runtime.config.search_enabled)

    def test_failed_podman_install_does_not_enable_search(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite(install_ok=False, running_as_root=True)

        result = self._install_podman_prerequisite(runtime)

        self.assertFalse(result["ok"])
        self.assertEqual("podman_install_failed", result["error"])
        self.assertFalse(runtime.config.search_enabled)
        self.assertEqual(1, len(runner.calls))
        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("failed", row["status"])

    def test_podman_installed_but_not_rootless_does_not_enable_search(self) -> None:
        runtime, _runner = self._runtime_with_podman_prerequisite(rootless=False, running_as_root=True)

        result = self._install_podman_prerequisite(runtime)

        self.assertFalse(result["ok"])
        self.assertEqual("rootless_podman_not_usable", result["error"])
        self.assertFalse(runtime.config.search_enabled)
        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("recovery_needed", row["status"])

    def test_verified_podman_allows_searxng_plan_to_select_podman(self) -> None:
        runtime, runner = self._runtime_with_podman_prerequisite(running_as_root=True)
        result = self._install_podman_prerequisite(runtime)
        self.assertTrue(result["ok"])
        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda name: "/usr/bin/podman" if name == "podman" and runner.installed else None,
            command_runner=lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, stdout="true", stderr=""),
            health_checker=lambda _url: False,
        )
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: "/usr/bin/podman" if name == "podman" and runner.installed else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        plan_payload = runtime.search_setup_plan({})
        plan = plan_payload["plan"]

        self.assertTrue(plan_payload["ok"])
        self.assertEqual("managed_container", plan["setup_mode"])
        self.assertEqual("podman", plan["selected_engine"])

    def test_search_setup_apply_rejects_invalid_confirmation(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        handler = _HandlerForTest(runtime, "/search/setup/apply", {"plan_id": "missing", "confirmation_token": "bad"})

        handler.do_POST()
        payload = json.loads(handler.body.decode("utf-8"))

        self.assertEqual(400, handler.status_code)
        self.assertFalse(payload["ok"])
        self.assertEqual("invalid_confirmation", payload["error"])
        self.assertFalse(runtime.config.search_enabled)

    def test_search_setup_apply_user_url_verifies_and_persists_journal(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        plan_handler = _HandlerForTest(runtime, "/search/setup/plan", {"base_url": "http://127.0.0.1:8888"})
        plan_handler.do_POST()
        plan_payload = json.loads(plan_handler.body.decode("utf-8"))
        plan = plan_payload["plan"]

        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            apply_handler = _HandlerForTest(
                runtime,
                "/search/setup/apply",
                {"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]},
            )
            apply_handler.do_POST()
        payload = json.loads(apply_handler.body.decode("utf-8"))

        self.assertEqual(200, apply_handler.status_code)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["did_configure"])
        self.assertTrue(runtime.config.search_enabled)
        self.assertEqual("http://127.0.0.1:8888", runtime.config.searxng_base_url)
        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("verified", row["status"])
        self.assertEqual("searxng_managed_service_setup", row["action_type"])

    def test_verified_search_setup_persists_across_runtime_restart(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]
        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})
        self.assertTrue(result["ok"])

        restarted = self._runtime(search_enabled=False, endpoint=None)

        self.assertTrue(restarted.config.search_enabled)
        self.assertEqual("http://127.0.0.1:8888", restarted.config.searxng_base_url)
        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            status = restarted.search_status()
        self.assertTrue(status["enabled"])
        self.assertTrue(status["available"])
        self.assertTrue(status["persistent_config"]["loaded"])

    def test_persisted_search_config_reports_stopped_endpoint_as_repair_needed(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        runtime._set_runtime_search_config(enabled=True, provider="searxng", base_url="http://127.0.0.1:8888")  # noqa: SLF001

        restarted = self._runtime(search_enabled=False, endpoint=None)
        offline_opener = mock.Mock()
        offline_opener.open.side_effect = OSError("offline")
        with mock.patch("agent.search.safe_web_search.build_opener", return_value=offline_opener):
            status = restarted.search_status()

        self.assertTrue(status["enabled"])
        self.assertTrue(status["endpoint_configured"])
        self.assertFalse(status["available"])
        self.assertEqual("endpoint_unreachable", status["reason"])
        self.assertIn("Start or repair", status["next_action"])

    def test_tampered_persisted_search_config_is_not_loaded(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        path = runtime._search_runtime_config_path()  # noqa: SLF001
        self.assertIsNotNone(path)
        assert path is not None
        path.write_text(
            json.dumps(
                {
                    "search_enabled": True,
                    "search_provider": "searxng",
                    "searxng_base_url": "https://search.example.test",
                }
            ),
            encoding="utf-8",
        )

        restarted = self._runtime(search_enabled=False, endpoint=None)
        status = restarted.search_status()

        self.assertFalse(restarted.config.search_enabled)
        self.assertEqual("invalid_persisted_search_config", status["reason"])
        self.assertEqual("persisted_search_config_untrusted", status["persistent_config"]["error"])

    def test_search_setup_apply_consumes_confirmation(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]
        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            first = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})
        second = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertTrue(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual("confirmation_consumed", second["error"])

    def test_search_setup_apply_rejects_submitted_mutation_plan_tamper(self) -> None:
        runtime = self._runtime_with_engine("podman")
        plan_payload = runtime.search_setup_plan({})
        plan = plan_payload["plan"]
        mutation_plan = dict(plan["mutation_plan"])
        mutation_plan["action_type"] = "external_pack.install"

        result = runtime.apply_search_setup(
            {
                "plan_id": plan["plan_id"],
                "confirmation_token": plan["confirmation_token"],
                "mutation_plan": mutation_plan,
            }
        )

        self.assertFalse(result["ok"])
        self.assertEqual("plan_apply_mismatch", result["error"])
        self.assertFalse(runtime.config.search_enabled)

    def test_search_setup_apply_refuses_expired_confirmation(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        plan_payload = runtime.search_setup_plan({"base_url": "http://127.0.0.1:8888"})
        plan = plan_payload["plan"]
        runtime._search_setup_confirmations[plan["plan_id"]]["expires_at"] = time.time() - 1  # noqa: SLF001

        result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertFalse(result["ok"])
        self.assertEqual("confirmation_expired", result["error"])
        self.assertFalse(runtime.config.search_enabled)

    def _chat(self, runtime: AgentRuntime, prompt: str, *, history: list[dict[str, str]] | None = None) -> tuple[dict[str, object], str]:
        messages = list(history or []) + [{"role": "user", "content": prompt}]
        handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": messages,
                "session_id": "services",
                "thread_id": "services-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )
        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8"))
        return body, _assistant_text(body)

    def _runtime_with_engine(
        self,
        engine: str | None,
        *,
        ports: dict[int, bool] | None = None,
        existing: bool = False,
        podman_rootless: bool | None = True,
        docker_rootless: bool | None = False,
    ) -> AgentRuntime:
        def _info_runner(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            binary = Path(str(argv[0])).name
            if binary == "podman":
                if podman_rootless is None:
                    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unknown")
                return subprocess.CompletedProcess(argv, 0, stdout="true" if podman_rootless else "false", stderr="")
            if binary == "docker":
                if docker_rootless is None:
                    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unknown")
                return subprocess.CompletedProcess(argv, 0, stdout="name=rootless" if docker_rootless else "name=seccomp", stderr="")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unknown")

        runtime = self._runtime(search_enabled=False, endpoint=None)
        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda name: f"/usr/bin/{name}" if name == engine else None,
            command_runner=_info_runner,
            health_checker=lambda _url: False,
        )
        port_map = ports if ports is not None else {8080: True, 8888: True}
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == engine else None,
            runner=_FakeManagedServiceRunner(existing=existing),
            health_checker=lambda url: url in {"http://127.0.0.1:8080", "http://127.0.0.1:8888"},
            port_checker=lambda port: bool(port_map.get(port, False)),
        )
        return runtime

    def _runtime_with_podman_prerequisite(
        self,
        *,
        install_ok: bool = True,
        rootless: bool = True,
        running_as_root: bool = False,
    ) -> tuple[AgentRuntime, _FakePrerequisiteRunner]:
        runtime = self._runtime_with_engine(None)
        runner = _FakePrerequisiteRunner(install_ok=install_ok, rootless=rootless)

        def _finder(name: str) -> str | None:
            if name == "apt-get":
                return "/usr/bin/apt-get"
            if name == "sudo":
                return "/usr/bin/sudo"
            if name == "podman" and runner.installed:
                return "/usr/bin/podman"
            return None

        runtime._prerequisite_command_finder = _finder  # noqa: SLF001
        runtime._prerequisite_runner = runner  # noqa: SLF001
        runtime._prerequisite_running_as_root = lambda: running_as_root  # noqa: SLF001
        return runtime, runner

    def _install_podman_prerequisite(self, runtime: AgentRuntime) -> dict[str, object]:
        plan_payload = runtime.podman_prerequisite_plan({})
        plan = plan_payload["plan"]
        return runtime.apply_podman_prerequisite(
            {"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]}
        )

    def test_setup_prompt_with_docker_available_offers_podman_prerequisite(self) -> None:
        runtime = self._runtime_with_engine("docker")
        body, text = self._chat(runtime, "set up web search")

        first_line = text.splitlines()[0]
        self.assertEqual("Search is not configured.", first_line)
        self.assertIn("missing Podman", text)
        self.assertIn("preferred rootless container runtime", text)
        self.assertIn("I will not install anything until you confirm", text)
        self.assertNotIn("Approved image: docker.io/searxng/searxng:latest", text)
        self.assertNotIn("Approved container: personal-agent-searxng", text)
        self.assertNotIn("Loopback bind: 127.0.0.1:8080:8080", text)
        setup = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        plan = setup.get("plan", {}) if isinstance(setup.get("plan"), dict) else {}
        self.assertEqual("podman_prerequisite", plan.get("setup_mode"))
        self.assertEqual("podman", plan.get("prerequisite"))
        meta = body.get("meta", {}) if isinstance(body.get("meta"), dict) else {}
        self.assertIn("managed_local_service_prerequisite_preview", meta.get("used_tools", []))

    def test_setup_prompt_with_podman_available_shows_preview(self) -> None:
        runtime = self._runtime_with_engine("podman")
        body, text = self._chat(runtime, "set up SearXNG")

        self.assertTrue("Web search is not set up yet" in text or "Search is not currently working" in text)
        self.assertIn("Plan Mode confirmation:", text)
        self.assertIn("Will create: container:personal-agent-searxng.", text)
        self.assertIn("Will change: runtime_search_config, memory/local_services/searxng/settings.yml.", text)
        self.assertIn("Will delete: nothing.", text)
        self.assertIn("Rollback scope:", text)
        self.assertNotIn("using Podman", text)
        self.assertNotIn("Loopback bind: 127.0.0.1:8080:8080", text)
        setup = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        self.assertEqual("127.0.0.1:8080:8080", setup.get("loopback_bind"))
        self.assertEqual("podman", setup.get("selected_engine"))
        self.assertEqual("podman", setup.get("preferred_engine"))
        self.assertFalse(setup.get("requires_docker_fallback_confirmation"))
        plan_mode = setup.get("plan_mode") if isinstance(setup.get("plan_mode"), dict) else {}
        self.assertEqual("plan_mode", plan_mode.get("policy_layer"))
        self.assertEqual("managed_local_service.setup_apply", plan_mode.get("action_type"))
        self.assertEqual("mutating", plan_mode.get("classification"))
        self.assertTrue(plan_mode.get("requires_confirmation"))
        self.assertTrue(plan_mode.get("plan_id"))
        apply_payload = setup.get("apply_payload") if isinstance(setup.get("apply_payload"), dict) else {}
        self.assertEqual(plan_mode.get("plan_id"), apply_payload.get("plan_id"))
        self.assertTrue(apply_payload.get("confirmation_token"))
        runner = runtime._managed_local_service_executor._runner  # noqa: SLF001
        self.assertEqual([], runner.calls)

    def test_declining_search_setup_plan_does_not_mutate(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = runtime._managed_local_service_executor._runner  # noqa: SLF001
        _body, first_text = self._chat(runtime, "set up web search")

        _body, second_text = self._chat(
            runtime,
            "no",
            history=[{"role": "user", "content": "set up web search"}, {"role": "assistant", "content": first_text}],
        )

        self.assertIn("cancelled", second_text.lower())
        self.assertEqual([], runner.calls)
        self.assertFalse(runtime.config.search_enabled)

    def test_tampered_chat_search_setup_plan_is_blocked_before_executor(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = runtime._managed_local_service_executor._runner  # noqa: SLF001
        _body, first_text = self._chat(runtime, "enable web search")
        orchestrator = runtime.orchestrator()
        pending = next(iter(orchestrator.confirmations._pending.values()))  # noqa: SLF001
        params = pending.action.get("params") if isinstance(pending.action, dict) and isinstance(pending.action.get("params"), dict) else {}
        mutation_plan = params.get("mutation_plan") if isinstance(params.get("mutation_plan"), dict) else {}
        mutation_plan["action_type"] = "external_pack.install"

        _body, second_text = self._chat(
            runtime,
            "yes",
            history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}],
        )

        self.assertIn("blocked", second_text.lower())
        self.assertIn("did not pull an image", second_text)
        self.assertEqual([], runner.calls)
        self.assertFalse(runtime.config.search_enabled)

    def test_telegram_search_setup_plan_text_is_readable_and_redacted(self) -> None:
        runtime = self._runtime_with_engine("podman")
        body, _text = self._chat(runtime, "set up web search")

        result = build_telegram_chat_payload_result(
            body,
            trace_id="test-trace",
            ok=bool(body.get("ok", True)),
            handler_name="chat",
            legacy_compatibility=False,
        )

        rendered = str(result.get("text") or "")
        self.assertIn("Plan Mode confirmation:", rendered)
        self.assertIn("Will create: container:personal-agent-searxng.", rendered)
        self.assertNotIn("confirmation_token", rendered)
        self.assertNotIn("SKILL.md", rendered)
        self.assertNotIn("AGENTS.md", rendered)

    def test_yes_after_preview_runs_bounded_setup_configures_runtime_search_after_verify(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner()
        health_calls: list[str] = []

        def _health(url: str) -> bool:
            health_calls.append(url)
            return len(health_calls) >= 3 and url == "http://127.0.0.1:8080"

        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=_health,
            port_checker=lambda _port: True,
            health_timeout_seconds=1.0,
            health_poll_interval_seconds=0.01,
        )
        _body, first_text = self._chat(runtime, "enable web search")
        self.assertTrue("Web search is not set up yet" in first_text or "Search is not currently working" in first_text)

        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("SearXNG setup finished", second_text)
        self.assertIn("Pulled approved image: yes", second_text)
        self.assertIn("Started approved container: yes", second_text)
        self.assertIn("Enabled Personal Agent metadata-only search for this running service: yes", second_text)
        self.assertIn("No external pack code ran", second_text)
        self.assertIn("No host networking", second_text)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))
        self.assertIn("system package install", second_text)
        rendered = second_text.lower()
        self.assertNotIn("docker install", rendered)
        self.assertNotIn("podman install", rendered)
        self.assertTrue(runtime.config.search_enabled)
        self.assertEqual("http://127.0.0.1:8080", runtime.config.searxng_base_url)
        status = runtime.search_status()
        self.assertTrue(status["available"])
        self.assertIsNone(status["reason"])
        self.assertEqual(3, len(health_calls))
        setup_payload = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        self.assertTrue(setup_payload.get("did_configure"))

    def test_existing_fallback_container_reuse_configures_runtime_to_actual_port(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner(
            existing=True,
            inspect_payload=_approved_searxng_inspect(self.tmpdir.name, running=True, bind="127.0.0.1:8888:8080"),
        )
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8888",
            port_checker=lambda _port: True,
        )
        plan = runtime.search_setup_plan({"allow_docker_fallback": False})["plan"]

        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            result = runtime.apply_search_setup({"plan_id": plan["plan_id"], "confirmation_token": plan["confirmation_token"]})

        self.assertTrue(result["ok"])
        self.assertTrue(result["did_configure"])
        self.assertEqual("http://127.0.0.1:8888", runtime.config.searxng_base_url)
        self.assertEqual("http://127.0.0.1:8888", result["search_status"]["base_url"])

    def test_setup_verification_failure_restores_config_and_removes_owned_container(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "enable web search")

        _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("SearXNG setup did not complete", second_text)
        self.assertIn("Previous search settings restored: yes", second_text)
        self.assertFalse(runtime.config.search_enabled)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["podman", "stop", "personal-agent-searxng"], argv_rows)
        self.assertIn(["podman", "rm", "personal-agent-searxng"], argv_rows)
        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("rolled_back", row["status"])

    def test_html_only_searxng_health_is_not_enough_to_enable_search(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "enable web search")

        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeRawOpener(b"<html>ok</html>")):
            _body, second_text = self._chat(
                runtime,
                "yes",
                history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}],
            )

        self.assertIn("SearXNG setup did not complete", second_text)
        self.assertFalse(runtime.config.search_enabled)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["podman", "stop", "personal-agent-searxng"], argv_rows)
        self.assertIn(["podman", "rm", "personal-agent-searxng"], argv_rows)
        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("rolled_back", row["status"])

    def test_setup_verification_cleanup_failure_persists_recovery_needed(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner(fail_on=" rm ")
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "enable web search")

        self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        row = runtime._managed_action_journal_store.recent(limit=1)[0]  # noqa: SLF001
        self.assertEqual("recovery_needed", row["status"])
        self.assertTrue(row["recovery_needed"])

    def test_expired_yes_returns_explicit_expiry_and_does_not_execute_setup(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "enable web search")
        orchestrator = runtime.orchestrator()
        pending = next(iter(orchestrator.confirmations._pending.values()))  # noqa: SLF001
        pending.expires_at = int(time.time()) - 1
        if isinstance(pending.action, dict):
            pending.action["expires_at"] = pending.expires_at
        orchestrator._memory_runtime.clear_expired_pending_items(pending.user_id, now_ts=int(time.time()) + 601)  # noqa: SLF001

        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("confirmation expired", second_text)
        self.assertIn("didn’t make any changes", second_text)
        self.assertIn("SearXNG setup preview", second_text)
        self.assertEqual([], runner.calls)

    def test_telegram_style_delay_within_ttl_still_executes(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "enable web search")
        orchestrator = runtime.orchestrator()
        pending = next(iter(orchestrator.confirmations._pending.values()))  # noqa: SLF001
        pending.expires_at = int(time.time()) + 60
        if isinstance(pending.action, dict):
            pending.action["expires_at"] = pending.expires_at

        with mock.patch("agent.search.safe_web_search.build_opener", return_value=_FakeSearchOpener({"results": []})):
            _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("SearXNG setup finished", second_text)
        self.assertTrue(runner.calls)

    def test_failed_setup_health_check_reports_owned_cleanup(self) -> None:
        runtime = self._runtime_with_engine("podman")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "podman" else None,
            runner=runner,
            health_checker=lambda _url: False,
            port_checker=lambda _port: True,
            health_timeout_seconds=0.01,
            health_poll_interval_seconds=0.01,
        )
        _body, first_text = self._chat(runtime, "enable web search")

        _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("SearXNG setup did not complete", second_text)
        self.assertIn("cleaned up the failed setup", second_text)
        self.assertIn("Nothing was left running", second_text)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["podman", "stop", "personal-agent-searxng"], argv_rows)
        self.assertIn(["podman", "rm", "personal-agent-searxng"], argv_rows)

    def test_setup_prompt_with_port_conflict_offers_fallback(self) -> None:
        runtime = self._runtime_with_engine("podman", ports={8080: False, 8888: True})
        body, text = self._chat(runtime, "set up web search")

        self.assertIn("Port 8080 is already being used", text)
        self.assertIn("use 8888 instead", text)
        self.assertNotIn("Loopback bind: 127.0.0.1:8888:8080", text)
        self.assertIn("It will run only on this computer", text)
        setup = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        self.assertEqual("127.0.0.1:8888:8080", setup.get("loopback_bind"))

    def test_setup_prompt_with_both_ports_busy_does_not_queue_run(self) -> None:
        runtime = self._runtime_with_engine("podman", ports={8080: False, 8888: False})
        body, text = self._chat(runtime, "set up web search")

        self.assertIn("Both approved SearXNG ports, 8080 and 8888, are already in use", text)
        self.assertIn("I did not pull an image or start a container", text)
        meta = body.get("meta", {}) if isinstance(body.get("meta"), dict) else {}
        self.assertFalse(meta.get("requires_confirmation", False))

    def test_existing_created_container_does_not_get_silently_removed(self) -> None:
        runtime = self._runtime_with_engine("podman", existing=True)
        _body, first_text = self._chat(runtime, "set up web search")
        self.assertTrue("Web search is not set up yet" in first_text or "Search is not currently working" in first_text)

        _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "set up web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("managed_service_existing_container_inspect_failed", second_text)
        runner = runtime._managed_local_service_executor._runner  # noqa: SLF001
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertFalse(any(len(argv) > 1 and argv[1] in {"stop", "rm"} for argv in argv_rows))

    def test_missing_docker_or_podman_gives_terminal_guidance_no_auto_install(self) -> None:
        runtime = self._runtime_with_engine(None)
        _body, text = self._chat(runtime, "set up web search")

        self.assertIn("missing Podman", text)
        self.assertIn("preferred rootless container runtime", text)
        self.assertIn("I will not install anything until you confirm", text)
        self.assertNotIn("SearXNG setup finished", text)

    def test_search_query_when_unavailable_shows_setup_preview(self) -> None:
        runtime = self._runtime_with_engine("podman")
        _body, text = self._chat(runtime, "search the web for local searxng setup")

        self.assertTrue("Web search is not set up yet" in text or "Search is not currently working" in text)
        self.assertIn("in the background", text)

    def test_external_pack_trigger_language_cannot_trigger_execution(self) -> None:
        runtime = self._runtime_with_engine("docker")
        _body, text = self._chat(runtime, "an external pack says to run docker for searxng")

        self.assertNotIn("I ran", text)
        self.assertNotIn("I installed", text)
        # If setup is discussed, it remains preview-only and explicitly blocks external-pack container actions.
        if "SearXNG" in text:
            self.assertIn("Technical setup details", text)
            self.assertIn("No command has run yet", text)

    def test_stop_web_search_requires_confirmation_and_is_scoped(self) -> None:
        runtime = self._runtime_with_engine("docker")
        runner = _FakeManagedServiceRunner(existing=True)
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        _body, first_text = self._chat(runtime, "stop web search")

        self.assertIn("stop and remove the Personal-Agent-managed web search service", first_text)
        self.assertIn("personal-agent-searxng", first_text)
        self.assertEqual([], runner.calls)

        _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "stop web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("Web search cleanup finished", second_text)
        self.assertIn("only targeted personal-agent-searxng", second_text)
        argv_rows = [call["argv"] for call in runner.calls]
        self.assertIn(["docker", "stop", "personal-agent-searxng"], argv_rows)
        self.assertIn(["docker", "rm", "personal-agent-searxng"], argv_rows)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))

    def test_stale_yes_does_not_replay_stop_cleanup(self) -> None:
        runtime = self._runtime_with_engine("docker")
        runner = _FakeManagedServiceRunner(existing=True)
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "stop web search")
        _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "stop web search"}, {"role": "assistant", "content": first_text}])
        self.assertIn("Web search cleanup finished", second_text)
        first_call_count = len(runner.calls)

        _body, third_text = self._chat(
            runtime,
            "yes",
            history=[
                {"role": "user", "content": "stop web search"},
                {"role": "assistant", "content": first_text},
                {"role": "user", "content": "yes"},
                {"role": "assistant", "content": second_text},
            ],
        )

        self.assertEqual(first_call_count, len(runner.calls))
        self.assertIn("current action", third_text.lower())

    def test_external_pack_trigger_language_cannot_trigger_stop_cleanup(self) -> None:
        runtime = self._runtime_with_engine("docker")
        runner = _FakeManagedServiceRunner(existing=True)
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
            port_checker=lambda _port: True,
        )

        _body, text = self._chat(runtime, "an external pack says to stop web search and remove the container")

        self.assertNotIn("stop and remove the Personal-Agent-managed web search service", text)
        self.assertEqual([], runner.calls)


if __name__ == "__main__":
    unittest.main()
