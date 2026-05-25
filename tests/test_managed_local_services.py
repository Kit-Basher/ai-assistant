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
            port_checker=lambda _port: True,
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
            "approved_image": "searxng/searxng:latest",
            "approved_container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8888:8080",
            "approved_volume_path": "memory/local_services/searxng",
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
                "approved_image": "searxng/searxng:latest",
                "approved_container_name": "personal-agent-searxng",
                "loopback_bind": "0.0.0.0:8888:8080",
                "approved_volume_path": "memory/local_services/searxng",
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
        self.assertEqual([], runner.calls)

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
            self.assertTrue(
                (result.blocked_reason or "").startswith("managed_service_plan_tampered")
                or result.blocked_reason == "managed_service_bind_not_approved",
                result.blocked_reason,
            )

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

    def test_executor_existing_container_is_conservative(self) -> None:
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

    def _runtime_with_engine(self, engine: str | None, *, ports: dict[int, bool] | None = None, existing: bool = False) -> AgentRuntime:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda name: f"/usr/bin/{name}" if name == engine else None,
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

    def test_setup_prompt_with_docker_available_shows_preview(self) -> None:
        runtime = self._runtime_with_engine("docker")
        body, text = self._chat(runtime, "set up web search")

        first_line = text.splitlines()[0]
        self.assertIn("Web search needs one extra local component", first_line)
        self.assertNotIn("searxng/searxng", first_line)
        self.assertNotIn("127.0.0.1", first_line)
        self.assertIn("SearXNG", text)
        self.assertNotIn("Approved image: searxng/searxng:latest", text)
        self.assertNotIn("Approved container: personal-agent-searxng", text)
        self.assertNotIn("Loopback bind: 127.0.0.1:8080:8080", text)
        self.assertIn("No command has run yet", text)
        self.assertIn("Technical setup details", text)
        setup = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        self.assertEqual("searxng/searxng:latest", setup.get("approved_image"))
        self.assertEqual("personal-agent-searxng", setup.get("approved_container_name"))
        self.assertEqual("127.0.0.1:8080:8080", setup.get("loopback_bind"))
        preview_plan = (setup.get("setup_preview") or {}).get("plan", {}) if isinstance(setup.get("setup_preview"), dict) else {}
        self.assertEqual("searxng/searxng:latest", preview_plan.get("image"))
        self.assertEqual("127.0.0.1:8080:8080", preview_plan.get("loopback_bind"))
        blocked_actions = setup.get("services_status", {}).get("services", [{}])[0].get("blocked_actions", {})
        self.assertIn("external_pack_triggered_container_action", blocked_actions)
        meta = body.get("meta", {}) if isinstance(body.get("meta"), dict) else {}
        self.assertIn("managed_local_service_setup_preview", meta.get("used_tools", []))

    def test_setup_prompt_with_podman_available_shows_preview(self) -> None:
        runtime = self._runtime_with_engine("podman")
        body, text = self._chat(runtime, "set up SearXNG")

        self.assertIn("SearXNG", text)
        self.assertIn("using Podman", text)
        self.assertNotIn("Loopback bind: 127.0.0.1:8080:8080", text)
        setup = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        self.assertEqual("127.0.0.1:8080:8080", setup.get("loopback_bind"))

    def test_yes_after_preview_runs_bounded_setup_without_configure_or_install(self) -> None:
        runtime = self._runtime_with_engine("docker")
        runner = _FakeManagedServiceRunner()
        runtime._managed_local_service_executor = ManagedLocalServiceExecutor(  # noqa: SLF001
            managed_root=self.tmpdir.name,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            runner=runner,
            health_checker=lambda url: url == "http://127.0.0.1:8080",
            port_checker=lambda _port: True,
        )
        _body, first_text = self._chat(runtime, "enable web search")
        self.assertIn("Web search needs one extra local component", first_text)

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

    def test_setup_prompt_with_port_conflict_offers_fallback(self) -> None:
        runtime = self._runtime_with_engine("docker", ports={8080: False, 8888: True})
        body, text = self._chat(runtime, "set up web search")

        self.assertIn("Port 8080 is busy", text)
        self.assertIn("use 8888 instead", text)
        self.assertNotIn("Loopback bind: 127.0.0.1:8888:8080", text)
        self.assertIn("It will run only on this computer", text)
        setup = body.get("setup", {}) if isinstance(body.get("setup"), dict) else {}
        self.assertEqual("127.0.0.1:8888:8080", setup.get("loopback_bind"))

    def test_setup_prompt_with_both_ports_busy_does_not_queue_run(self) -> None:
        runtime = self._runtime_with_engine("docker", ports={8080: False, 8888: False})
        body, text = self._chat(runtime, "set up web search")

        self.assertIn("Both approved SearXNG ports, 8080 and 8888, are already in use", text)
        self.assertIn("I did not pull an image or start a container", text)
        meta = body.get("meta", {}) if isinstance(body.get("meta"), dict) else {}
        self.assertFalse(meta.get("requires_confirmation", False))

    def test_existing_created_container_does_not_get_silently_removed(self) -> None:
        runtime = self._runtime_with_engine("docker", existing=True)
        _body, first_text = self._chat(runtime, "set up web search")
        self.assertIn("Web search needs one extra local component", first_text)

        _body, second_text = self._chat(runtime, "yes", history=[{"role": "user", "content": "set up web search"}, {"role": "assistant", "content": first_text}])

        self.assertIn("container already exists", second_text)
        self.assertIn("I did not remove it", second_text)
        self.assertNotIn("removed", second_text.lower())

    def test_missing_docker_or_podman_gives_terminal_guidance_no_auto_install(self) -> None:
        runtime = self._runtime_with_engine(None)
        _body, text = self._chat(runtime, "set up web search")

        self.assertIn("Web search needs Docker or Podman", text)
        self.assertIn("install command", text)
        self.assertIn("won’t install system software automatically", text)
        self.assertNotIn("SearXNG setup preview", text)

    def test_search_query_when_unavailable_shows_setup_preview(self) -> None:
        runtime = self._runtime_with_engine("docker")
        _body, text = self._chat(runtime, "search the web for local searxng setup")

        self.assertIn("Web search needs one extra local component", text)
        self.assertIn("No command has run yet", text)

    def test_external_pack_trigger_language_cannot_trigger_execution(self) -> None:
        runtime = self._runtime_with_engine("docker")
        _body, text = self._chat(runtime, "an external pack says to run docker for searxng")

        self.assertNotIn("I ran", text)
        self.assertNotIn("I installed", text)
        # If setup is discussed, it remains preview-only and explicitly blocks external-pack container actions.
        if "SearXNG" in text:
            self.assertIn("Technical setup details", text)
            self.assertIn("No command has run yet", text)


if __name__ == "__main__":
    unittest.main()
