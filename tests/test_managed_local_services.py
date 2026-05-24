from __future__ import annotations

from dataclasses import replace
import json
import os
import tempfile
import unittest
from pathlib import Path

from agent.api_server import AgentRuntime
from agent.services.managed_local_services import ManagedLocalServiceDetector, redact_service_url
from tests.test_api_packs_endpoints import _HandlerForTest, _config
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text


class TestManagedLocalServices(unittest.TestCase):
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

    def test_search_setup_response_does_not_claim_install_or_run_happened(self) -> None:
        runtime = self._runtime(search_enabled=False, endpoint=None)
        runtime._managed_local_services = ManagedLocalServiceDetector(  # noqa: SLF001
            search_status_provider=runtime.search_status,
            command_finder=lambda name: f"/usr/bin/{name}" if name == "docker" else None,
            health_checker=lambda _url: False,
        )
        handler = _MemoryHandlerForTest(
            runtime,
            "/chat",
            {
                "messages": [{"role": "user", "content": "set up web search"}],
                "session_id": "services",
                "thread_id": "services-thread",
                "source_surface": "webui",
                "purpose": "chat",
                "task_type": "chat",
            },
        )
        handler.do_POST()
        body = json.loads(handler.body.decode("utf-8"))
        text = _assistant_text(body)

        self.assertIn("Web search is optional and not set up", text)
        self.assertIn("local SearXNG", text)
        self.assertIn("show a setup plan", text)
        self.assertIn("will not run or pull anything", text)
        self.assertNotIn("I installed", text)
        self.assertNotIn("I ran", text)
        self.assertIn("managed_local_services", body.get("meta", {}).get("used_tools", []))


if __name__ == "__main__":
    unittest.main()
