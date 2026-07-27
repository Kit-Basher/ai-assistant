from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from agent.api_server import APIServerHandler, AgentRuntime
from tests.test_api_server import _config


class _FilesystemHandler(APIServerHandler):
    def __init__(
        self,
        runtime_obj: AgentRuntime,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        client_host: str = "127.0.0.1",
    ) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.client_address = (client_host, 43123)
        self.status_code = 0
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self.status_code = status
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = content_type, cache_control
        self.status_code = status
        self.body = body

    def _read_json(self) -> dict[str, object]:
        return dict(self._payload)

    def payload(self) -> dict[str, object]:
        parsed = json.loads(self.body.decode("utf-8"))
        assert isinstance(parsed, dict)
        return parsed


class TestFilesystemAPIContract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        fixture = Path(self.tmpdir.name)
        self.allowed_root = fixture / "allowed"
        self.notes = self.allowed_root / "notes"
        self.outside_root = fixture / "outside"
        self.notes.mkdir(parents=True)
        self.outside_root.mkdir()
        self.todo = self.notes / "todo.txt"
        self.todo.write_text("finish personal agent\n", encoding="utf-8")
        self.secret = self.notes / "secret.key"
        self.secret.write_text("harmless acceptance fixture\n", encoding="utf-8")
        self.outside = self.outside_root / "outside.txt"
        self.outside.write_text("outside\n", encoding="utf-8")
        self.escape = self.notes / "escape.txt"
        self.escape.symlink_to(self.outside)

        registry_path = fixture / "registry.json"
        db_path = fixture / "agent.db"
        config = _config(
            str(registry_path),
            str(db_path),
            perception_roots=(str(self.allowed_root),),
        )
        self.runtime = AgentRuntime(config)
        self.runtime._repo_root = self.allowed_root
        self.runtime.runtime_truth_service()._filesystem_skill_cache = None

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    @staticmethod
    def _query(route: str, **values: object) -> str:
        return f"{route}?{urllib.parse.urlencode(values)}"

    def _get(self, path: str, *, client_host: str = "127.0.0.1") -> _FilesystemHandler:
        handler = _FilesystemHandler(self.runtime, path, client_host=client_host)
        handler.do_GET()
        return handler

    def _post(self, path: str, payload: dict[str, object], *, client_host: str = "127.0.0.1") -> _FilesystemHandler:
        handler = _FilesystemHandler(self.runtime, path, payload, client_host=client_host)
        handler.do_POST()
        return handler

    def test_temp_fixture_proves_first_class_safe_filesystem_contract(self) -> None:
        roots = self._get("/filesystem/roots")
        self.assertEqual(200, roots.status_code)
        self.assertIn(str(self.allowed_root), roots.payload()["allowed_roots"])

        listed = self._get(self._query("/filesystem/list", path=str(self.notes)))
        self.assertEqual(200, listed.status_code)
        listed_names = [row["name"] for row in listed.payload()["entries"]]
        self.assertIn("todo.txt", listed_names)
        self.assertNotIn("secret.key", listed_names)

        stat = self._get(self._query("/filesystem/stat", path=str(self.todo)))
        self.assertEqual(200, stat.status_code)
        self.assertEqual("file", stat.payload()["type"])

        read = self._get(self._query("/filesystem/read", path=str(self.todo), max_bytes=8))
        self.assertEqual(200, read.status_code)
        self.assertEqual("finish p", read.payload()["text"])
        self.assertTrue(read.payload()["truncated"])

        filename_search = self._get(
            self._query("/filesystem/search", root=str(self.allowed_root), q="todo", max_results=10)
        )
        self.assertEqual(200, filename_search.status_code)
        self.assertEqual([str(self.todo)], [row["path"] for row in filename_search.payload()["results"]])

        content_search = self._post(
            "/filesystem/search_content",
            {"root": str(self.allowed_root), "q": "finish personal", "max_results": 10},
        )
        self.assertEqual(200, content_search.status_code)
        self.assertEqual([str(self.todo)], [row["path"] for row in content_search.payload()["results"]])

        for rejected_path, error_kind in (
            (self.outside, "outside_allowed_roots"),
            (self.secret, "sensitive_path_blocked"),
            (self.escape, "outside_allowed_roots"),
        ):
            rejected = self._get(self._query("/filesystem/read", path=str(rejected_path)))
            self.assertEqual(403, rejected.status_code)
            self.assertEqual(error_kind, rejected.payload()["error_kind"])

    def test_filesystem_routes_are_loopback_operator_only(self) -> None:
        get_denied = self._get("/filesystem/roots", client_host="192.0.2.10")
        post_denied = self._post(
            "/filesystem/search_content",
            {"root": str(self.allowed_root), "q": "finish"},
            client_host="192.0.2.10",
        )

        self.assertEqual(403, get_denied.status_code)
        self.assertEqual(403, post_denied.status_code)
        self.assertEqual("forbidden", get_denied.payload()["error"])
        self.assertEqual("forbidden", post_denied.payload()["error"])
        self.assertTrue(get_denied.payload()["operator_only"])
        self.assertTrue(post_denied.payload()["operator_only"])


if __name__ == "__main__":
    unittest.main()
