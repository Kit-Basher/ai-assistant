from __future__ import annotations

import asyncio
import json
import socket
import tempfile
import unittest
from urllib import error as urllib_error
from unittest.mock import patch

from telegram_adapter.bot import _chat_proxy_timeout_seconds, _handle_message, _post_local_api_chat_json, _post_local_api_chat_json_async
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
    def __init__(self, text: str) -> None:
        self.text = text
        self.message_id = 1
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
    def __init__(self, chat_id: int, text: str) -> None:
        self.effective_chat = _FakeChat(chat_id)
        self.effective_message = _FakeMessage(text)


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


def _chat_api_response(text: str, *, data: dict[str, object] | None = None) -> dict[str, object]:
    meta_source = dict(data) if isinstance(data, dict) else {}
    return {
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


class TestTelegramAdapter(unittest.TestCase):
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

    def test_same_chat_proxy_requests_are_rejected_while_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            orchestrator = _FakeOrchestrator(_FakeResponse("unused"))
            first_update = _FakeUpdate(303, "first slow")
            second_update = _FakeUpdate(303, "second fast")
            context = _FakeContext(
                {
                    "orchestrator": orchestrator,
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )
            order: list[str] = []
            active_calls = {"count": 0, "max": 0}
            first_started = asyncio.Event()

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                active_calls["count"] += 1
                active_calls["max"] = max(active_calls["max"], active_calls["count"])
                order.append(f"start:{text}")
                try:
                    if text == "first slow":
                        first_started.set()
                        await asyncio.sleep(0.06)
                        return _chat_api_response("first reply", data={"route": "generic_chat"})
                    return _chat_api_response("second reply", data={"route": "generic_chat"})
                finally:
                    order.append(f"end:{text}")
                    active_calls["count"] -= 1
            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            async def _scenario() -> None:
                first_task = asyncio.create_task(_handle_message(first_update, context))
                for _ in range(20):
                    if first_started.is_set():
                        break
                    await asyncio.sleep(0.005)
                second_task = asyncio.create_task(_handle_message(second_update, context))
                await asyncio.wait_for(first_task, timeout=0.3)
                await asyncio.wait_for(second_task, timeout=0.3)

            asyncio.run(_scenario())

            self.assertEqual(1, active_calls["max"])
            self.assertEqual(["start:first slow", "end:first slow"], order)
            self.assertIn("still working on your last request", str(second_update.effective_message.replies[-1]["text"] or "").lower())
            rows = _read_log_rows(log_path)
            message_rows = [
                row.get("payload")
                for row in rows
                if str(row.get("type") or "") == "telegram_message" and isinstance(row.get("payload"), dict)
            ]
            self.assertTrue(any(str(row.get("selected_route") or "") == "chat_busy" for row in message_rows))
            self.assertTrue(any(bool(row.get("proxy_overlap_rejected")) for row in message_rows if isinstance(row, dict)))

    def test_same_chat_burst_uses_busy_reply_not_timeout_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/agent.log"
            context = _FakeContext(
                {
                    "orchestrator": _FakeOrchestrator(_FakeResponse("unused")),
                    "db": _FakeDB(),
                    "log_path": log_path,
                }
            )
            updates = [
                _FakeUpdate(404, "what model are you using?"),
                _FakeUpdate(404, "is openrouter configured?"),
                _FakeUpdate(404, "configure openrouter"),
                _FakeUpdate(404, "what managed adapters exist?"),
            ]
            first_started = asyncio.Event()

            async def _fake_post(payload: dict[str, object]) -> dict[str, object]:
                messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
                text = str(((messages[-1] if messages else {}) or {}).get("content") or "")
                if text == "what model are you using?":
                    first_started.set()
                    await asyncio.sleep(0.08)
                    return _chat_api_response("Chat is currently using ollama:qwen3.5:4b on Ollama.", data={"route": "model_status"})
                return _chat_api_response("unexpected", data={"route": "generic_chat"})

            context.application.bot_data["fetch_local_api_chat_json"] = _fake_post

            async def _scenario() -> None:
                first_task = asyncio.create_task(_handle_message(updates[0], context))
                await asyncio.wait_for(first_started.wait(), timeout=0.2)
                burst_tasks = [asyncio.create_task(_handle_message(update, context)) for update in updates[1:]]
                await asyncio.wait_for(first_task, timeout=0.3)
                for task in burst_tasks:
                    await asyncio.wait_for(task, timeout=0.3)
                for _ in range(20):
                    if "ollama:qwen3.5:4b" in str(updates[0].effective_message.replies[-1]["text"] or ""):
                        break
                    await asyncio.sleep(0.01)

            asyncio.run(_scenario())

            self.assertIn("ollama:qwen3.5:4b", str(updates[0].effective_message.replies[-1]["text"] or ""))
            for update in updates[1:]:
                reply = str(update.effective_message.replies[-1]["text"] or "")
                self.assertIn("still working on your last request", reply.lower())
                self.assertNotIn("couldn't get a reply from the agent in time", reply.lower())

            rows = _read_log_rows(log_path)
            message_rows = [
                row.get("payload")
                for row in rows
                if str(row.get("type") or "") == "telegram_message" and isinstance(row.get("payload"), dict)
            ]
            busy_rows = [row for row in message_rows if str(row.get("selected_route") or "") == "chat_busy"]
            self.assertEqual(3, len(busy_rows))
            self.assertTrue(all(bool(row.get("proxy_overlap_rejected")) for row in busy_rows))
            self.assertTrue(all(str(row.get("proxy_failure_kind") or "") == "same_chat_in_flight" for row in busy_rows))
            self.assertTrue(all(int(row.get("proxy_elapsed_ms") or 0) == 0 for row in busy_rows))


if __name__ == "__main__":
    unittest.main()
