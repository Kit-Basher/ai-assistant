import json
import os
import tempfile
import unittest

from agent.config import Config
from agent.llm_router import LLMRouter


class FakeClient:
    def __init__(self, text: str = "ok", raise_error: bool = False) -> None:
        self._text = text
        self._raise = raise_error

    def generate(self, messages):
        if self._raise:
            raise RuntimeError("boom")
        return self._text


def _config(**overrides) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4.1-mini",
        openai_model_worker=None,
        agent_timezone="America/Regina",
        db_path=":memory:",
        log_path="/tmp/log.jsonl",
        skills_path="/tmp/skills",
        ollama_host=None,
        ollama_model=None,
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=20,
    )
    return base.__class__(**{**base.__dict__, **overrides})


class TestLLMRouter(unittest.TestCase):
    def test_no_providers_uses_dummy(self) -> None:
        router = LLMRouter(_config())
        result = router.generate("chat", "low", [{"role": "user", "content": "hi"}])
        self.assertEqual(result["meta"]["selected_provider"], "dummy")

    def test_ollama_minimal_config(self) -> None:
        cfg = _config(ollama_host="http://127.0.0.1:11434", ollama_model="llama3")
        router = LLMRouter(cfg, clients={"ollama": FakeClient("local")})
        result = router.generate("chat", "low", [{"role": "user", "content": "hi"}])
        self.assertEqual(result["meta"]["selected_provider"], "ollama")
        self.assertEqual(result["meta"]["selected_model"], "llama3")

    def test_watchdog_forces_local(self) -> None:
        cfg = _config(openai_api_key="key", openai_model="gpt-4.1-mini")
        router = LLMRouter(cfg, clients={"openai": FakeClient("cloud")})
        result = router.generate("watchdog", "low", [{"role": "user", "content": "hi"}])
        self.assertEqual(result["meta"]["selected_provider"], "dummy")

    def test_openai_fallback_when_local_fails(self) -> None:
        cfg = _config(
            ollama_host="http://127.0.0.1:11434",
            ollama_model="llama3",
            openai_api_key="key",
            openai_model="gpt-4.1-mini",
            allow_cloud=True,
        )
        router = LLMRouter(
            cfg,
            clients={"ollama": FakeClient(raise_error=True), "openai": FakeClient("cloud")},
        )
        result = router.generate("chat", "low", [{"role": "user", "content": "hi"}])
        self.assertEqual(result["meta"]["selected_provider"], "openai")
        self.assertEqual(result["meta"]["fallbacks_attempted"][0]["provider"], "ollama")

    def test_allow_cloud_false_never_uses_openai(self) -> None:
        cfg = _config(openai_api_key="key", openai_model="gpt-4.1-mini", allow_cloud=False)
        router = LLMRouter(cfg, clients={"openai": FakeClient("cloud")})
        result = router.generate("chat", "low", [{"role": "user", "content": "hi"}])
        self.assertEqual(result["meta"]["selected_provider"], "dummy")

    def test_routing_decision_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "agent.jsonl")
            cfg = _config(
                ollama_host="http://127.0.0.1:11434",
                ollama_model="llama3",
                log_path=log_path,
            )
            router = LLMRouter(cfg, clients={"ollama": FakeClient("local")}, log_path=log_path)
            router.generate("chat", "low", [{"role": "user", "content": "hi"}])
            with open(log_path, "r", encoding="utf-8") as handle:
                line = handle.readline().strip()
            record = json.loads(line)
            self.assertEqual(record["type"], "llm_routing_decision")
            payload = record["payload"]
            self.assertEqual(payload["task_kind"], "chat")
            self.assertEqual(payload["tier_required"], "low")
            self.assertEqual(payload["selected_provider"], "ollama")


if __name__ == "__main__":
    unittest.main()
