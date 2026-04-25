from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from urllib import error as urllib_error
from unittest.mock import patch

from telegram_adapter.bot import (
    _chat_proxy_timeout_seconds,
    _handle_message,
    _handle_telegram_text_via_local_api,
    _post_local_api_chat_json,
    _post_local_api_chat_json_async,
    _telegram_message_age_ms,
)
from agent.public_chat import build_no_llm_public_message


class _RaisingLLM:
    def chat(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("telegram adapter must not call llm directly")


class _FakeResponse:
    def __init__(self, text: str, data: dict[str, object] | None = None) -> None:
        self.text = text
        self.data = data


class _FakeOrchestrator:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []
        self.llm_client = _RaisingLLM()

    def handle_message(self, text: str, *, user_id: str) -> _FakeResponse:
        self.calls.append((text, user_id))
        return self.response


class _FakeDB:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def set_preference(self, key: str, value: str) -> None:
        self.values[str(key)] = str(value)


class _FakeMessage:
    def __init__(self, text: str, *, date: datetime | None = None) -> None:
        self.text = text
        self.message_id = 1
        self.date = date or datetime.now(timezone.utc)
        self.replies: list[dict[str, str | None]] = []
        self._next_message_id = 100

    async def reply_text(self, text: str, parse_mode: str | None = None, **_kwargs) -> object:  # type: ignore[no-untyped-def]
        entry = {"text": text, "parse_mode": parse_mode}
        self.replies.append(entry)
        self._next_message_id += 1
        return _FakeSentMessage(entry, message_id=self._next_message_id)


class _FakeSentMessage:
    def __init__(self, entry: dict[str, str | None], *, message_id: int) -> None:
        self._entry = entry
        self.message_id = message_id

    async def edit_text(self, text: str, parse_mode: str | None = None, **_kwargs) -> object:  # type: ignore[no-untyped-def]
        self._entry["text"] = text
        self._entry["parse_mode"] = parse_mode
        return self


class _FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class _FakeUpdate:
    def __init__(self, chat_id: int, text: str, *, date: datetime | None = None) -> None:
        self.effective_chat = _FakeChat(chat_id)
        self.effective_message = _FakeMessage(text, date=date)


class _FakeContext:
    def __init__(self, bot_data: dict[str, object]) -> None:
        self.application = type("App", (), {"bot_data": bot_data})()


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False

    def read(self) -> bytes:
        return self._body


def _read_log_rows(path: str) -> list[dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]


def _log_event_rows(rows: list[dict[str, object]], event_type: str) -> list[dict[str, object]]:
    return [row for row in rows if str(row.get("type") or "") == event_type]


def _chat_api_response(text: str, *, data: dict[str, object] | None = None) -> dict[str, object]:
    meta_source = dict(data) if isinstance(data, dict) else {}
    payload = {
        "ok": bool(meta_source.get("ok", True)),
        "assistant": {"content": text},
        "message": text,
        "meta": {
            "route": str(meta_source.get("route") or "generic_chat"),
            "used_llm": bool(meta_source.get("used_llm", True)),
            "used_memory": bool(meta_source.get("used_memory", False)),
            "used_runtime_state": bool(meta_source.get("used_runtime_state", False)),
            "used_tools": list(meta_source.get("used_tools") or []),
            "generic_fallback_used": bool(meta_source.get("generic_fallback_used", False)),
            "generic_fallback_reason": meta_source.get("generic_fallback_reason"),
            "runtime_state_failure_reason": meta_source.get("runtime_state_failure_reason"),
        },
        "error_kind": meta_source.get("error_kind"),
    }
    if isinstance(meta_source.get("cards_payload"), dict):
        payload["cards_payload"] = dict(meta_source["cards_payload"])
    return payload


class TestTelegramAdapter(unittest.TestCase):
    def test_telegram_message_age_ms_uses_message_timestamp(self) -> None:
        received_at = datetime(2026, 4, 19, 12, 0, 0, tzinfo=timezone.utc)
        message_date = received_at - timedelta(milliseconds=1534)
        message = _FakeMessage("hello", date=message_date)
        self.assertEqual(1534, _telegram_message_age_ms(message, received_at=received_at))

    def test_trivial_social_turn_uses_local_fast_path_without_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"

            async def _scenario() -> dict[str, object]:
                with patch(
                    "telegram_adapter.bot._post_local_api_chat_json_async",
                    side_effect=AssertionError("social turns must not call the local API"),
                ):
                    return await _handle_telegram_text_via_local_api(
                        text="hello are you working?",
                        chat_id="42",
                        trace_id="trace-1",
                        bot_data={},
                        log_path=log_path,
                        runtime=None,
                        orchestrator=None,
                        runtime_version=None,
                        runtime_git_commit=None,
                        runtime_started_ts=None,
                    )

            result = asyncio.run(_scenario())
            self.assertTrue(bool(result.get("ok")))
            self.assertFalse(bool(result.get("used_llm")))
            self.assertEqual("telegram_social_turn", result.get("handler_name"))
            chat_meta = result.get("chat_meta") if isinstance(result.get("chat_meta"), dict) else {}
            self.assertEqual("social_turn", chat_meta.get("assistant_turn_type"))
            self.assertEqual("presence_check", chat_meta.get("assistant_turn_kind"))
            self.assertTrue(bool(chat_meta.get("fast_path")))

            event_types = [str(row.get("type") or "") for row in _read_log_rows(log_path)]
            self.assertIn("telegram.social_turn.fast_path", event_types)

    def test_deterministic_status_routes_skip_placeholder_and_log_latency_breakdown(self) -> None:
        cases = [
            ("what model are you using right now", "model_status", "Current model is ollama:qwen3.5:4b."),
            ("what local models are available", "model_status", "Local models are qwen3.5:4b and llama3.1:8b."),
            ("is the agent healthy right now", "runtime_status", "Runtime is ready."),
            ("is openrouter configured?", "provider_status", "OpenRouter is configured."),
        ]

        async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
            messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
            text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
            if text == "what model are you using right now":
                return _chat_api_response(
                    "Current model is ollama:qwen3.5:4b.",
                    data={"route": "model_status", "used_runtime_state": True, "used_llm": False},
                )
            if text == "what local models are available":
                return _chat_api_response(
                    "Local models are qwen3.5:4b and llama3.1:8b.",
                    data={"route": "model_status", "used_runtime_state": True, "used_llm": False},
                )
            if text == "is the agent healthy right now":
                return _chat_api_response(
                    "Runtime is ready.",
                    data={"route": "runtime_status", "used_runtime_state": True, "used_llm": False},
                )
            return _chat_api_response(
                "OpenRouter is configured.",
                data={"route": "provider_status", "used_runtime_state": True, "used_llm": False},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )
            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            for index, (prompt, expected_route, expected_reply) in enumerate(cases, start=1):
                update = _FakeUpdate(800 + index, prompt, date=datetime.now(timezone.utc) - timedelta(seconds=2))
                asyncio.run(_handle_message(update, context))
                self.assertEqual(expected_reply, str(update.effective_message.replies[-1]["text"] or ""))
                self.assertEqual(1, len(update.effective_message.replies))
                self.assertNotIn("Thinking…", str(update.effective_message.replies[0]["text"] or ""))

            rows = _read_log_rows(log_path)
            summary_rows = _log_event_rows(rows, "telegram.latency_summary")
            self.assertEqual(len(cases), len(summary_rows))
            self.assertEqual(0, len(_log_event_rows(rows, "telegram.placeholder_send.start")))

            for summary_row, (_, expected_route, _) in zip(summary_rows, cases, strict=True):
                payload = summary_row.get("payload") if isinstance(summary_row.get("payload"), dict) else {}
                self.assertEqual(expected_route, payload.get("selected_route"))
                self.assertEqual("deferred", payload.get("placeholder_policy"))
                self.assertFalse(bool(payload.get("placeholder_used")))
                self.assertIsInstance(payload.get("telegram_message_age_ms"), int)
                self.assertIsInstance(payload.get("local_api_request_ms"), int)
                self.assertIsInstance(payload.get("response_delivery_ms"), int)
                self.assertIsInstance(payload.get("visible_total_ms"), int)

    def test_diagnostics_capture_route_renders_compact_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            update = _FakeUpdate(900, "my wifi drops after suspend")
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            async def _fake_post(_payload: dict[str, object]) -> dict[str, object]:
                return _chat_api_response(
                    "I can gather a compact diagnostics snapshot for you. Do you want me to do that?",
                    data={
                        "route": "diagnostics_capture",
                        "used_runtime_state": False,
                        "used_llm": False,
                        "cards_payload": {
                            "cards": [
                                {
                                    "title": "Diagnostics snapshot",
                                    "lines": [
                                        "OS/kernel: Linux test-host 6.8.0-1",
                                        "Network: state: up; up interfaces: wlp2s0",
                                        "Suspend/resume: matches: 2",
                                        "Assessment: Network is not fully up.",
                                        "Next action: Check NetworkManager and the Wi-Fi driver after suspend.",
                                    ],
                                    "severity": "warn",
                                }
                            ],
                            "raw_available": False,
                            "summary": "Network is not fully up.",
                            "confidence": 1.0,
                            "next_questions": [],
                        },
                    },
                )

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post
            asyncio.run(_handle_message(update, context))

            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual("Markdown", update.effective_message.replies[-1]["parse_mode"])
            self.assertIn("*Diagnostics snapshot*", reply_text)
            self.assertIn("Assessment: Network is not fully up.", reply_text)
            self.assertIn("Next action: Check NetworkManager and the Wi-Fi driver after suspend.", reply_text)

    def test_text_message_routes_through_api_proxy_and_sends_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            orchestrator = _FakeOrchestrator(_FakeResponse("adapter reply"))
            update = _FakeUpdate(42, "tell me a joke")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            with patch(
                "telegram_adapter.bot._post_local_api_chat_json_async",
                return_value=_chat_api_response("adapter reply"),
            ):
                asyncio.run(_handle_message(update, context))

            self.assertEqual([], orchestrator.calls)
            self.assertEqual("adapter reply", str(update.effective_message.replies[-1]["text"] or ""))
            event_types = [str(row.get("type") or "") for row in _read_log_rows(log_path)]
            self.assertIn("incoming_message", event_types)
            self.assertIn("telegram.route.selected", event_types)
            self.assertIn("telegram_message", event_types)
            self.assertIn("response_sent", event_types)

    def test_text_message_acknowledges_immediately_and_edits_later(self) -> None:
        async def _scenario() -> tuple[_FakeUpdate, list[dict[str, object]]]:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = f"{tmpdir}/agent.log"
                orchestrator = _FakeOrchestrator(_FakeResponse("unused"))
                update = _FakeUpdate(42, "tell me a joke")
                context = _FakeContext(
                    {
                        "orchestrator": orchestrator,
                        "db": _FakeDB(),
                        "log_path": log_path,
                    }
                )
                release = asyncio.Event()

                async def _slow_fetch(_payload: dict[str, object]) -> dict[str, object]:
                    await release.wait()
                    return _chat_api_response("final reply")

                with patch("telegram_adapter.bot._post_local_api_chat_json_async", side_effect=_slow_fetch):
                    await _handle_message(update, context)
                    self.assertEqual("Thinking…", str(update.effective_message.replies[-1]["text"] or ""))
                    release.set()
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)

                rows = _read_log_rows(log_path)
                return update, rows

        update, rows = asyncio.run(_scenario())
        self.assertEqual("final reply", str(update.effective_message.replies[-1]["text"] or ""))
        self.assertNotIn("timed out", str(update.effective_message.replies[-1]["text"] or "").lower())
        event_types = [str(row.get("type") or "") for row in rows]
        self.assertIn("telegram_async_start", event_types)
        self.assertIn("telegram_async_complete", event_types)

    def test_text_message_async_proxy_payload_uses_canonical_messages_shape(self) -> None:
        observed_payloads: list[dict[str, object]] = []

        async def _scenario() -> _FakeUpdate:
            with tempfile.TemporaryDirectory() as tmpdir:
                log_path = f"{tmpdir}/agent.log"
                update = _FakeUpdate(42, "tell me a joke")
                context = _FakeContext(
                    {
                        "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                        "db": _FakeDB(),
                        "log_path": log_path,
                    }
                )

                async def _capture(payload: dict[str, object]) -> dict[str, object]:
                    observed_payloads.append(dict(payload))
                    return _chat_api_response("final reply")

                with patch("telegram_adapter.bot._post_local_api_chat_json_async", side_effect=_capture):
                    await _handle_message(update, context)
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                return update

        update = asyncio.run(_scenario())
        self.assertEqual("final reply", str(update.effective_message.replies[-1]["text"] or ""))
        self.assertEqual(1, len(observed_payloads))
        payload = observed_payloads[0]
        self.assertEqual(
            [{"role": "user", "content": "tell me a joke"}],
            payload.get("messages"),
        )
        self.assertNotIn("message", payload)

    def test_async_chat_proxy_does_not_apply_request_timeout(self) -> None:
        captured: dict[str, object] = {}

        class _FakeWriter:
            def write(self, _data: bytes) -> None:
                return None

            async def drain(self) -> None:
                return None

            def close(self) -> None:
                return None

            async def wait_closed(self) -> None:
                return None

        async def _fake_open_connection(_host: str, _port: int):  # type: ignore[no-untyped-def]
            return object(), _FakeWriter()

        async def _fake_read_http_response_async(*, reader, timeout_seconds):  # type: ignore[no-untyped-def]
            captured["reader"] = reader
            captured["timeout_seconds"] = timeout_seconds
            await asyncio.sleep(0.01)
            return 200, {"content-length": "44"}, json.dumps(_chat_api_response("hello")).encode("utf-8")

        async def _scenario() -> dict[str, object]:
            with patch("telegram_adapter.bot.asyncio.open_connection", side_effect=_fake_open_connection):
                with patch("telegram_adapter.bot._read_http_response_async", side_effect=_fake_read_http_response_async):
                    return await _post_local_api_chat_json_async({"messages": [{"role": "user", "content": "hello"}]})

        payload = asyncio.run(_scenario())
        proxy_meta = payload.get("_proxy_meta") if isinstance(payload.get("_proxy_meta"), dict) else {}
        self.assertIsNone(captured.get("timeout_seconds"))
        self.assertIsNone(proxy_meta.get("timeout_seconds"))

    def test_async_chat_proxy_preserves_grounded_chat_body_on_http_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            update = _FakeUpdate(42, "test ollama:qwen2.5:7b-instruct without adopting it")
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            class _FakeWriter:
                def write(self, _data: bytes) -> None:
                    return None

                async def drain(self) -> None:
                    return None

                def close(self) -> None:
                    return None

                async def wait_closed(self) -> None:
                    return None

            async def _fake_open_connection(_host: str, _port: int):  # type: ignore[no-untyped-def]
                return object(), _FakeWriter()

            async def _fake_read_http_response_async(*, reader, timeout_seconds):  # type: ignore[no-untyped-def]
                _ = reader
                _ = timeout_seconds
                payload = _chat_api_response(
                    "I tested ollama:qwen2.5:7b-instruct without switching, and it is not ready right now. Reason: timeout.",
                    data={
                        "ok": False,
                        "route": "action_tool",
                        "used_llm": False,
                        "used_runtime_state": True,
                        "used_tools": ["model_controller"],
                        "error_kind": "model_test_failed",
                    },
                )
                encoded = json.dumps(payload).encode("utf-8")
                return 400, {"content-length": str(len(encoded))}, encoded

            async def _scenario() -> None:
                with patch("telegram_adapter.bot.asyncio.open_connection", side_effect=_fake_open_connection):
                    with patch("telegram_adapter.bot._read_http_response_async", side_effect=_fake_read_http_response_async):
                        await _handle_message(update, context)
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)

            asyncio.run(_scenario())

            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("I tested ollama:qwen2.5:7b-instruct without switching", reply_text)
            self.assertIn("Reason: timeout.", reply_text)
            self.assertNotIn("returned an error while handling that request", reply_text.lower())

            rows = _read_log_rows(log_path)
            message_rows = [
                row.get("payload")
                for row in rows
                if str(row.get("type") or "") == "telegram_message" and isinstance(row.get("payload"), dict)
            ]
            last_row = dict(message_rows[-1] or {})
            self.assertEqual("action_tool", str(last_row.get("selected_route") or ""))
            self.assertEqual(["model_controller"], list(last_row.get("used_tools") or []))
            self.assertIsNone(last_row.get("proxy_failure_kind"))

    def test_error_response_from_api_proxy_gets_user_friendly_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            orchestrator = _FakeOrchestrator(
                _FakeResponse(
                    "",
                    data={
                        "ok": False,
                        "error_kind": "llm_unavailable",
                    },
                )
            )
            update = _FakeUpdate(42, "tell me a joke")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            with patch(
                "telegram_adapter.bot._post_local_api_chat_json_async",
                return_value=_chat_api_response("", data={"ok": False, "error_kind": "llm_unavailable"}),
            ):
                asyncio.run(_handle_message(update, context))

            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertEqual(build_no_llm_public_message(), reply_text)
            self.assertNotIn("Reason:", reply_text)
            self.assertNotIn("llm_unavailable", reply_text.lower())

    def test_error_response_from_api_proxy_uses_assistant_voice_when_no_reply_text_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            update = _FakeUpdate(42, "tell me a joke")
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                }
            )
            with patch(
                "telegram_adapter.bot._post_local_api_chat_json_async",
                return_value=_chat_api_response(
                    "",
                    data={
                        "ok": False,
                        "error_kind": "runtime_error",
                        "message": "backend problem",
                    },
                ),
            ):
                asyncio.run(_handle_message(update, context))

            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("I couldn't finish that request", reply_text)
            self.assertIn("Please try again.", reply_text)
            self.assertNotIn("Reason:", reply_text)
            self.assertNotIn("Agent could not complete the request", reply_text)

    def test_disconnect_proxy_error_is_soft_when_backend_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            update = _FakeUpdate(42, "tell me a joke")
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                }
            )
            with patch(
                "telegram_adapter.bot._post_local_api_chat_json_async",
                return_value={
                    "_proxy_error": {
                        "kind": "disconnect",
                        "detail": "broken pipe",
                        "backend_reachable": True,
                        "backend_ready": True,
                        "backend_phase": "ready",
                    }
                },
            ):
                asyncio.run(_handle_message(update, context))

            reply_text = str(update.effective_message.replies[-1]["text"] or "")
            self.assertIn("connection to the agent dropped", reply_text.lower())
            self.assertNotIn("backend unavailable", reply_text.lower())

    def test_post_local_api_chat_json_maps_timeout_distinctly(self) -> None:
        with patch("telegram_adapter.bot.urllib_request.urlopen", side_effect=socket.timeout("timed out")):
            with patch(
                "telegram_adapter.bot._fetch_local_api_json",
                return_value={"ok": True, "ready": True, "phase": "ready"},
            ):
                payload = _post_local_api_chat_json({"messages": [{"role": "user", "content": "hello"}]})
        error = payload.get("_proxy_error") if isinstance(payload.get("_proxy_error"), dict) else {}
        self.assertEqual("timeout", error.get("kind"))
        self.assertEqual(True, error.get("backend_ready"))

    def test_post_local_api_chat_json_maps_http_error_distinctly(self) -> None:
        error = urllib_error.HTTPError(
            url="http://127.0.0.1:8765/chat",
            code=502,
            msg="Bad Gateway",
            hdrs=None,
            fp=None,
        )
        with patch("telegram_adapter.bot.urllib_request.urlopen", side_effect=error):
            with patch(
                "telegram_adapter.bot._fetch_local_api_json",
                return_value={"ok": True, "ready": True, "phase": "ready"},
            ):
                payload = _post_local_api_chat_json({"messages": [{"role": "user", "content": "hello"}]})
        proxy_error = payload.get("_proxy_error") if isinstance(payload.get("_proxy_error"), dict) else {}
        self.assertEqual("http_error", proxy_error.get("kind"))
        self.assertEqual(502, proxy_error.get("status_code"))

    def test_post_local_api_chat_json_maps_connection_refused_distinctly(self) -> None:
        with patch(
            "telegram_adapter.bot.urllib_request.urlopen",
            side_effect=urllib_error.URLError(ConnectionRefusedError("connection refused")),
        ):
            with patch("telegram_adapter.bot._fetch_local_api_json", return_value={}):
                payload = _post_local_api_chat_json({"messages": [{"role": "user", "content": "hello"}]})
        proxy_error = payload.get("_proxy_error") if isinstance(payload.get("_proxy_error"), dict) else {}
        self.assertEqual("unreachable", proxy_error.get("kind"))
        self.assertEqual(False, proxy_error.get("backend_reachable"))

    def test_post_local_api_chat_json_maps_broken_pipe_as_disconnect(self) -> None:
        with patch("telegram_adapter.bot.urllib_request.urlopen", side_effect=BrokenPipeError("broken pipe")):
            with patch(
                "telegram_adapter.bot._fetch_local_api_json",
                return_value={"ok": True, "ready": True, "phase": "ready"},
            ):
                payload = _post_local_api_chat_json({"messages": [{"role": "user", "content": "hello"}]})
        proxy_error = payload.get("_proxy_error") if isinstance(payload.get("_proxy_error"), dict) else {}
        self.assertEqual("disconnect", proxy_error.get("kind"))
        self.assertEqual(True, proxy_error.get("backend_ready"))

    def test_post_local_api_chat_json_uses_longer_timeout_for_setup_flow(self) -> None:
        observed: dict[str, object] = {}

        def _fake_urlopen(request, timeout: float):  # type: ignore[no-untyped-def]
            observed["timeout"] = timeout
            observed["url"] = getattr(request, "full_url", "")
            return _FakeHTTPResponse(_chat_api_response("Paste your OpenRouter API key and I will finish the setup."))

        with patch("telegram_adapter.bot.urllib_request.urlopen", side_effect=_fake_urlopen):
            payload = _post_local_api_chat_json({"messages": [{"role": "user", "content": "configure openrouter"}]})

        proxy_meta = payload.get("_proxy_meta") if isinstance(payload.get("_proxy_meta"), dict) else {}
        self.assertEqual(15.0, observed.get("timeout"))
        self.assertEqual("http://127.0.0.1:8765/chat", observed.get("url"))
        self.assertEqual(15.0, proxy_meta.get("timeout_seconds"))

    def test_post_local_api_chat_json_uses_default_timeout_for_non_setup_chat(self) -> None:
        observed: dict[str, object] = {}

        def _fake_urlopen(request, timeout: float):  # type: ignore[no-untyped-def]
            observed["timeout"] = timeout
            return _FakeHTTPResponse(_chat_api_response("hello"))

        with patch("telegram_adapter.bot.urllib_request.urlopen", side_effect=_fake_urlopen):
            payload = _post_local_api_chat_json({"messages": [{"role": "user", "content": "hello"}]})

        proxy_meta = payload.get("_proxy_meta") if isinstance(payload.get("_proxy_meta"), dict) else {}
        self.assertEqual(10.0, observed.get("timeout"))
        self.assertEqual(10.0, proxy_meta.get("timeout_seconds"))

    def test_chat_proxy_timeout_seconds_uses_longer_timeout_for_pending_setup_confirmation(self) -> None:
        timeout_seconds = _chat_proxy_timeout_seconds(
            {
                "messages": [{"role": "user", "content": "yes"}],
                "setup_state_hint": {
                    "step": "awaiting_openrouter_reuse_confirm",
                    "awaiting_confirmation": True,
                },
            }
        )
        self.assertEqual(30.0, timeout_seconds)

    def test_same_chat_setup_confirmation_payload_carries_setup_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            first_update = _FakeUpdate(404, "configure openrouter")
            second_update = _FakeUpdate(404, "yes")
            observed_payloads: list[dict[str, object]] = []
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                observed_payloads.append(dict(payload))
                last_text = str((((payload.get("messages") if isinstance(payload.get("messages"), list) else [{}])[-1]) or {}).get("content") or "")
                if last_text == "configure openrouter":
                    return {
                        **_chat_api_response(
                            "I already have an OpenRouter API key stored. Reply yes and I will test it now, or paste a new key to replace it.",
                            data={"route": "setup_flow", "used_runtime_state": True, "used_llm": False},
                        ),
                        "setup": {
                            "type": "confirm_reuse_secret",
                            "provider": "openrouter",
                            "prompt": "I already have an OpenRouter API key stored. Reply yes and I will test it now, or paste a new key to replace it.",
                        },
                    }
                return _chat_api_response(
                    "OpenRouter is ready. I tested it with openrouter:openai/gpt-4o-mini.",
                    data={"route": "setup_flow", "used_runtime_state": True, "used_llm": False},
                )

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post
            asyncio.run(_handle_message(first_update, context))
            asyncio.run(_handle_message(second_update, context))

            self.assertEqual(2, len(observed_payloads))
            self.assertNotIn("setup_state_hint", observed_payloads[0])
            self.assertEqual(
                {
                    "route": "setup_flow",
                    "step": "awaiting_openrouter_reuse_confirm",
                    "awaiting_secret": False,
                    "awaiting_confirmation": True,
                },
                observed_payloads[1].get("setup_state_hint"),
            )

    def test_async_placeholder_edits_to_successful_deterministic_runtime_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            runtime_update = _FakeUpdate(606, "runtime")
            setup_update = _FakeUpdate(606, "configure ollama")
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                if text == "runtime":
                    return _chat_api_response(
                        "Ready.",
                        data={"route": "runtime_status", "used_runtime_state": True, "used_llm": False},
                    )
                return _chat_api_response(
                    "Ollama is ready for chat with ollama:qwen3.5:4b.",
                    data={"route": "setup_flow", "used_runtime_state": True, "used_llm": False},
                )

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post
            asyncio.run(_handle_message(runtime_update, context))
            asyncio.run(_handle_message(setup_update, context))

            self.assertEqual("Ready.", str(runtime_update.effective_message.replies[-1]["text"] or ""))
            self.assertEqual(
                "Ollama is ready for chat with ollama:qwen3.5:4b.",
                str(setup_update.effective_message.replies[-1]["text"] or ""),
            )
            self.assertNotIn(
                "I couldn't read that from the runtime state.",
                str(runtime_update.effective_message.replies[-1]["text"] or ""),
            )

    def test_async_placeholder_uses_setup_summary_when_message_field_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            runtime_update = _FakeUpdate(606, "runtime")
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )

            async def _fake_post(_payload: dict[str, object]) -> dict[str, object]:
                return {
                    "ok": True,
                    "assistant": {"content": ""},
                    "message": "",
                    "meta": {
                        "route": "runtime_status",
                        "used_runtime_state": True,
                        "used_llm": False,
                        "used_memory": False,
                        "used_tools": [],
                    },
                    "setup": {
                        "type": "runtime_status",
                        "summary": "Ready. Using ollama / ollama:qwen3.5:4b.",
                    },
                }

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post
            asyncio.run(_handle_message(runtime_update, context))

            reply = str(runtime_update.effective_message.replies[-1]["text"] or "")
            self.assertIn("ollama:qwen3.5:4b", reply)
            self.assertNotIn("I couldn't read that from the runtime state.", reply)

    def test_slow_api_proxy_does_not_block_event_loop_or_other_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            orchestrator = _FakeOrchestrator(_FakeResponse("unused"))
            slow_update = _FakeUpdate(101, "slow question")
            fast_update = _FakeUpdate(202, "fast question")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )
            slow_started = asyncio.Event()

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                if text == "slow question":
                    slow_started.set()
                    await asyncio.sleep(0.08)
                    return _chat_api_response("slow reply", data={"route": "generic_chat"})
                return _chat_api_response("fast reply", data={"route": "generic_chat"})
            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            async def _scenario() -> None:
                slow_task = asyncio.create_task(_handle_message(slow_update, context))
                for _ in range(20):
                    if slow_started.is_set():
                        break
                    await asyncio.sleep(0.005)
                ticker_ran = {"value": False}

                async def _ticker() -> None:
                    await asyncio.sleep(0.01)
                    ticker_ran["value"] = True

                ticker_task = asyncio.create_task(_ticker())
                fast_task = asyncio.create_task(_handle_message(fast_update, context))
                await asyncio.wait_for(ticker_task, timeout=0.2)
                self.assertTrue(ticker_ran["value"])
                await asyncio.wait_for(fast_task, timeout=0.2)
                self.assertEqual("fast reply", str(fast_update.effective_message.replies[-1]["text"] or ""))
                await asyncio.wait_for(slow_task, timeout=0.3)
                for _ in range(20):
                    if str(slow_update.effective_message.replies[-1]["text"] or "") == "slow reply":
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual("slow reply", str(slow_update.effective_message.replies[-1]["text"] or ""))

            asyncio.run(_scenario())

            rows = _read_log_rows(log_path)
            event_types = [str(row.get("type") or "") for row in rows]
            self.assertIn("telegram.local_api_chat.start", event_types)
            self.assertIn("telegram.local_api_chat.finish", event_types)
            finish_rows = [row for row in rows if str(row.get("type") or "") == "telegram.local_api_chat.finish"]
            self.assertTrue(all(str((row.get("payload") or {}).get("execution_mode") or "") == "async_http" for row in finish_rows))

    def test_same_chat_two_quick_messages_are_serialized_without_busy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                }
            )
            first_update = _FakeUpdate(303, "first slow")
            second_update = _FakeUpdate(303, "second quick")
            release = asyncio.Event()
            seen: list[str] = []
            first_started = asyncio.Event()

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                seen.append(text)
                if text == "first slow":
                    first_started.set()
                    await release.wait()
                    return _chat_api_response("first reply", data={"route": "generic_chat"})
                return _chat_api_response("second reply", data={"route": "generic_chat"})

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            async def _scenario() -> None:
                first_task = asyncio.create_task(_handle_message(first_update, context))
                await asyncio.wait_for(first_started.wait(), timeout=0.2)
                second_task = asyncio.create_task(_handle_message(second_update, context))
                await asyncio.wait_for(second_task, timeout=0.2)
                release.set()
                await asyncio.wait_for(first_task, timeout=0.3)
                for _ in range(40):
                    if second_update.effective_message.replies and str(second_update.effective_message.replies[-1]["text"] or "") == "second reply":
                        break
                    await asyncio.sleep(0.01)

            asyncio.run(_scenario())

            rows = _read_log_rows(f"{tmpdir}/agent.log")
            self.assertEqual(["first slow", "second quick"], seen)
            self.assertEqual("first reply", str(first_update.effective_message.replies[-1]["text"] or ""))
            self.assertEqual("second reply", str(second_update.effective_message.replies[-1]["text"] or ""))
            self.assertNotIn("still working on your last request", str(second_update.effective_message.replies[-1]["text"] or "").lower())
            self.assertGreaterEqual(len(_log_event_rows(rows, "telegram.local_api_chat.pending_promoted")), 1)

    def test_same_chat_three_quick_messages_keep_first_and_latest_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                }
            )
            first_update = _FakeUpdate(404, "first slow")
            middle_update = _FakeUpdate(404, "middle")
            latest_update = _FakeUpdate(404, "latest")
            release = asyncio.Event()
            seen: list[str] = []
            first_started = asyncio.Event()

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                seen.append(text)
                if text == "first slow":
                    first_started.set()
                    await release.wait()
                    return _chat_api_response("first reply", data={"route": "generic_chat"})
                if text == "latest":
                    return _chat_api_response("latest reply", data={"route": "generic_chat"})
                return _chat_api_response("middle reply", data={"route": "generic_chat"})

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            async def _scenario() -> None:
                first_task = asyncio.create_task(_handle_message(first_update, context))
                await asyncio.wait_for(first_started.wait(), timeout=0.2)
                middle_task = asyncio.create_task(_handle_message(middle_update, context))
                latest_task = asyncio.create_task(_handle_message(latest_update, context))
                await asyncio.wait_for(middle_task, timeout=0.2)
                await asyncio.wait_for(latest_task, timeout=0.2)
                release.set()
                await asyncio.wait_for(first_task, timeout=0.3)
                for _ in range(40):
                    if latest_update.effective_message.replies and str(latest_update.effective_message.replies[-1]["text"] or "") == "latest reply":
                        break
                    await asyncio.sleep(0.01)

            asyncio.run(_scenario())

            rows = _read_log_rows(f"{tmpdir}/agent.log")
            self.assertEqual(["first slow", "latest"], seen)
            self.assertEqual("first reply", str(first_update.effective_message.replies[-1]["text"] or ""))
            self.assertEqual("latest reply", str(latest_update.effective_message.replies[-1]["text"] or ""))
            self.assertEqual(0, len(middle_update.effective_message.replies))
            overwritten_rows = _log_event_rows(rows, "telegram.local_api_chat.pending_overwritten")
            promoted_rows = _log_event_rows(rows, "telegram.local_api_chat.pending_promoted")
            self.assertEqual(1, len(overwritten_rows))
            self.assertEqual(1, len(promoted_rows))
            self.assertTrue(bool((overwritten_rows[0].get("payload") or {}).get("replaced_pending")))
            self.assertTrue(bool((promoted_rows[0].get("payload") or {}).get("promoted")))

    def test_same_chat_pending_request_is_processed_if_active_request_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": f"{tmpdir}/agent.log",
                }
            )
            first_update = _FakeUpdate(505, "first fail")
            second_update = _FakeUpdate(505, "second after fail")
            release = asyncio.Event()
            first_started = asyncio.Event()
            seen: list[str] = []

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                seen.append(text)
                if text == "first fail":
                    first_started.set()
                    await release.wait()
                    raise RuntimeError("boom")
                return _chat_api_response("second reply", data={"route": "generic_chat"})

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            async def _scenario() -> None:
                first_task = asyncio.create_task(_handle_message(first_update, context))
                await asyncio.wait_for(first_started.wait(), timeout=0.2)
                second_task = asyncio.create_task(_handle_message(second_update, context))
                await asyncio.wait_for(second_task, timeout=0.2)
                release.set()
                await asyncio.wait_for(first_task, timeout=0.3)
                for _ in range(40):
                    if len(second_update.effective_message.replies) and str(second_update.effective_message.replies[-1]["text"] or "") == "second reply":
                        break
                    await asyncio.sleep(0.01)

            asyncio.run(_scenario())

            rows = _read_log_rows(f"{tmpdir}/agent.log")
            self.assertIn("first fail", seen)
            self.assertIn("second after fail", seen)
            self.assertIn("agent encountered an error", str(first_update.effective_message.replies[-1]["text"] or "").lower())
            self.assertEqual("second reply", str(second_update.effective_message.replies[-1]["text"] or ""))
            self.assertGreaterEqual(len(_log_event_rows(rows, "telegram.local_api_chat.pending_promoted")), 1)


if __name__ == "__main__":
    unittest.main()
