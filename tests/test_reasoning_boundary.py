from agent.llm.reasoning_boundary import REASONING_REJECTED_CODE, sanitize_stream_chunk
from agent.llm.providers.openai_compat import OpenAICompatProvider
from agent.llm.registry import ProviderConfig
from agent.llm.types import Message, Request, ToolCall
import json
import pytest


def test_reasoning_is_dropped_from_success_and_failure_chunks() -> None:
    assert sanitize_stream_chunk({"choices": [{"index": 0, "delta": {"reasoning_content": "private", "content": "hello"}}]}) == {"choices": [{"index": 0, "finish_reason": None, "delta": {"content": "hello"}}]}


def test_structured_tool_call_with_reasoning_is_accepted_without_retaining_reasoning() -> None:
    result = sanitize_stream_chunk({"choices": [{"delta": {"reasoning_content": "call shell", "tool_calls": [{"function": {"name": "x"}}]}}]})
    assert result == {"choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [{"function": {"name": "x"}}]}}]}


def test_reasoning_only_pseudo_tool_is_not_parsed() -> None:
    result = sanitize_stream_chunk({"choices": [{"delta": {"reasoning_content": "call filesystem.read now"}}]})
    assert result == {"choices": [{"index": 0, "finish_reason": None, "delta": {}}]}


def test_malformed_chunks_cannot_leak_reasoning() -> None:
    assert sanitize_stream_chunk({"choices": [{"delta": {"reasoning_content": {"secret": "x"}}}]}) == {"choices": [{"index": 0, "finish_reason": None, "delta": {}}]}


def test_llama_profile_requires_localhost_and_forces_discard(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="localhost"):
        OpenAICompatProvider(ProviderConfig("llama", "llama_cpp_openai_compatible", "http://example.test/v1", "/chat/completions", None, {}, {}, True, True))

    class Reply:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "safe", "reasoning_content": "private"}}], "usage": {"prompt_tokens": 1}}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Reply())
    provider = OpenAICompatProvider(ProviderConfig("llama", "llama_cpp_openai_compatible", "http://127.0.0.1:18081/v1", "/chat/completions", None, {}, {}, True, True))
    result = provider.chat(Request(messages=(Message(role="user", content="hello"),), metadata={"discard_reasoning": False}), model="local", timeout_seconds=1)
    assert result.text == "safe"
    assert "reasoning" not in repr(result.raw).lower()


def test_openai_compatible_tool_transcript_has_type_and_exact_result_binding() -> None:
    rows = OpenAICompatProvider._to_messages((
        Message(role="assistant", content="", tool_calls=(ToolCall(id="call-7", name="inspect", arguments="{}"),)),
        Message(role="tool", content="{\"capabilities\":[]}", tool_call_id="call-7"),
    ))
    assert rows[0]["tool_calls"][0]["type"] == "function"
    assert rows[0]["tool_calls"][0]["id"] == "call-7"
    assert rows[1]["tool_call_id"] == "call-7"


def test_native_ollama_tool_transcript_preserves_neutral_invoke_call_id_and_arguments() -> None:
    rows = OpenAICompatProvider._to_messages((
        Message(role="assistant", content="", tool_calls=(ToolCall(id="call-7", name="assistant_invoke_capability", arguments='{"capability_id":"filesystem.search","arguments_json":"{\\"query\\":\\"backup\\"}"}'),)),
        Message(role="tool", content="{\"state\":\"ok\"}", tool_call_id="call-7"),
    ), native_ollama=True)
    assert rows[0]["tool_calls"][0]["id"] == "call-7"
    assert rows[0]["tool_calls"][0]["function"]["name"] == "assistant_invoke_capability"
    assert rows[0]["tool_calls"][0]["function"]["arguments"] == {"capability_id": "filesystem.search", "arguments_json": "{\"query\":\"backup\"}"}
    assert "type" not in rows[0]["tool_calls"][0]
    assert rows[1]["tool_call_id"] == "call-7"
