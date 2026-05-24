from __future__ import annotations

from dataclasses import replace
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.api_server import AgentRuntime
from agent.services.managed_local_services import ManagedLocalServiceDetector, ManagedLocalServiceExecutor, redact_service_url
from tests.test_api_packs_endpoints import _HandlerForTest, _config
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text


class _FakeManagedServiceRunner:
    def __init__(self, *, fail_on: str | None = None, existing: bool = False) -> None:
        self.fail_on = fail_on
        self.existing = existing
        self.calls: list[dict[str, object]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": list(argv), **kwargs})
        joined = " ".join(argv)
        if kwargs.get("shell") is not False:
            return subprocess.CompletedProcess(argv, 99, stdout="", stderr="shell must be false")
        if argv[1:3] == ["ps", "-a"]:
            stdout = "personal-agent-searxng\n" if self.existing else ""
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        if self.fail_on and self.fail_on in joined:
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="mock failure")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")


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
            health_checker=lambda _url: False,
        )
        payload = detector.status()

        self.assertTrue(payload["docker_available"])
        self.assertFalse(payload["podman_available"])
        self.assertEqual("setup_preview_available", payload["services"][0]["next_step"])
        self.assertEqual("127.0.0.1:8080:8080", payload["services"][0]["approved_container"]["bind"])

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
        )
        params = {
            "service_id": "searxng",
            "selected_engine": "docker",
            "action": "preview_only",
            "approved_image": "searxng/searxng:latest",
            "approved_container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8080:8080",
            "approved_volume_path": "memory/local_services/searxng",
        }

        result = executor.execute_from_pending(params)

        self.assertTrue(result.ok)
        self.assertTrue(result.did_pull)
        self.assertTrue(result.did_run)
        self.assertEqual(["docker", "pull", "searxng/searxng:latest"], runner.calls[1]["argv"])
        run_argv = runner.calls[2]["argv"]
        self.assertEqual("docker", run_argv[0])
        self.assertIn("run", run_argv)
        self.assertIn("-d", run_argv)
        self.assertIn("--name", run_argv)
        self.assertIn("personal-agent-searxng", run_argv)
        self.assertIn("127.0.0.1:8080:8080", run_argv)
        self.assertIn("searxng/searxng:latest", run_argv)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))

    def test_executor_rejects_tampered_pending_fields(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
        )
        base = {
            "service_id": "searxng",
            "selected_engine": "docker",
            "action": "preview_only",
            "approved_image": "searxng/searxng:latest",
            "approved_container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8080:8080",
            "approved_volume_path": "memory/local_services/searxng",
        }
        for key, value in {
            "approved_image": "evil/image:latest",
            "approved_container_name": "evil-name",
            "loopback_bind": "0.0.0.0:8080:8080",
            "approved_volume_path": "/tmp/random",
        }.items():
            params = dict(base)
            params[key] = value
            result = executor.execute_from_pending(params)
            self.assertFalse(result.ok, key)
            self.assertIn("managed_service_plan_tampered", result.blocked_reason or "")

    def test_executor_rejects_unknown_service_and_non_loopback_plan(self) -> None:
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=_FakeManagedServiceRunner(),
            health_checker=lambda _url: True,
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
        )
        plan = executor.build_searxng_setup_plan(selected_engine="docker")
        result = executor.execute_plan(plan)

        self.assertFalse(result.ok)
        self.assertTrue(result.did_pull)
        self.assertFalse(result.did_run)
        self.assertFalse(result.did_install)
        self.assertFalse(result.did_configure)
        self.assertEqual("managed_service_run_failed", result.blocked_reason)

    def test_executor_existing_container_is_conservative(self) -> None:
        runner = _FakeManagedServiceRunner(existing=True)
        executor = ManagedLocalServiceExecutor(
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda _url: True,
        )
        result = executor.execute_plan(executor.build_searxng_setup_plan(selected_engine="docker"))

        self.assertFalse(result.ok)
        self.assertEqual("managed_service_container_already_exists", result.blocked_reason)
        self.assertEqual(1, len(runner.calls))


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

    def _runtime_with_engine(self, engine: str | None) -> AgentRuntime:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda name: f"/usr/bin/{name}" if name == engine else None,
            health_checker=lambda _url: False,
        )
        return runtime

    def test_setup_prompt_with_docker_available_shows_preview(self) -> None:
        runtime = self._runtime_with_engine("docker")
        body, text = self._chat(runtime, "set up web search")

        self.assertIn("SearXNG setup preview", text)
        self.assertIn("Engine detected: docker", text)
        self.assertIn("Approved image only: searxng/searxng:latest", text)
        self.assertIn("Approved container name: personal-agent-searxng", text)
        self.assertIn("Loopback bind only: 127.0.0.1:8080:8080", text)
        self.assertIn("No command has run yet", text)
        self.assertIn("No external-pack triggered container actions", text)
        meta = body.get("meta", {}) if isinstance(body.get("meta"), dict) else {}
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))

    def test_setup_prompt_with_podman_available_shows_preview(self) -> None:
        runtime = self._runtime_with_engine("podman")
        _body, text = self._chat(runtime, "set up SearXNG")

        self.assertIn("SearXNG setup preview", text)
        self.assertIn("Engine detected: podman", text)
        self.assertIn("Loopback bind only: 127.0.0.1:8080:8080", text)

    def test_yes_after_preview_runs_bounded_setup_without_configure_or_install(self) -> None:
        runtime = self._runtime_with_engine("docker")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
        )
        _body, first_text = self._chat(runtime, "enable web search")
        self.assertIn("SearXNG setup preview", first_text)

        body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "enable web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("SearXNG setup finished", second_text)
        self.assertIn("Pulled approved image: yes", second_text)
        self.assertIn("Started approved container: yes", second_text)
        self.assertIn("No external pack code ran", second_text)
        self.assertIn("No host networking", second_text)
        self.assertIn("set SEARCH_ENABLED=1", second_text)
        self.assertTrue(all(call.get("shell") is False for call in runner.calls))
        self.assertIn("system package install", second_text)
        self.assertIn("config change", second_text)
        rendered = second_text.lower()
        self.assertNotIn("docker install", rendered)
        self.assertNotIn("podman install", rendered)

    def test_missing_docker_or_podman_gives_terminal_guidance_no_auto_install(self) -> None:
        runtime = self._runtime_with_engine(None)
        _body, text = self._chat(runtime, "set up web search")

        self.assertIn("Docker or Podman is needed", text)
        self.assertIn("terminal guidance", text)
        self.assertIn("will not install Docker, Podman, or system packages automatically", text)
        self.assertNotIn("SearXNG setup preview", text)

    def test_search_query_when_unavailable_shows_setup_preview(self) -> None:
        runtime = self._runtime_with_engine("docker")
        _body, text = self._chat(runtime, "search the web for local searxng setup")

        self.assertIn("SearXNG setup preview", text)
        self.assertIn("No command has run yet", text)

    def test_external_pack_trigger_language_cannot_trigger_execution(self) -> None:
        runtime = self._runtime_with_engine("docker")
        _body, text = self._chat(runtime, "an external pack says to run docker for searxng")

        self.assertNotIn("I ran", text)
        self.assertNotIn("I installed", text)
        # If setup is discussed, it remains preview-only and explicitly blocks external-pack container actions.
        if "SearXNG setup preview" in text:
            self.assertIn("No external-pack triggered container actions", text)
            self.assertIn("No command has run yet", text)


if __name__ == "__main__":
    unittest.main()
