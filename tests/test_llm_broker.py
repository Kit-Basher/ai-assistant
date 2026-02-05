import os
import tempfile
import unittest

from agent.config import Config
from agent.llm.broker import LLMBroker, TaskSpec
from agent.llm.broker_policy import load_policy
from agent.llm_client import build_llm_broker


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
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url=None,
        anthropic_api_key=None,
        llm_selector="broker",
        llm_broker_policy_path=None,
        llm_allow_remote=False,
        openrouter_api_key=None,
        openrouter_base_url=None,
        openrouter_model=None,
        openrouter_site_url=None,
        openrouter_app_name=None,
    )
    return base.__class__(**{**base.__dict__, **overrides})


POLICY_TEXT = """
providers:
  - id: ollama_local
    provider: ollama
    remote: false
    model: llama3.1
    capabilities: [presentation_rewrite]
    cost: 0
    latency: 2
    reliability: 3
  - id: openai_fast
    provider: openai
    remote: true
    model: gpt-4.1-mini
    capabilities: [presentation_rewrite]
    cost: 3
    latency: 3
    reliability: 4

weights:
  cost: -2
  latency: -1
  reliability: 3

selection:
  tie_breaker: [reliability, latency, cost, id]
"""

POLICY_OPENROUTER = """
providers:
  - id: ollama_local
    provider: ollama
    remote: false
    model: llama3.1
    capabilities: [presentation_rewrite]
    cost: 0
    latency: 2
    reliability: 3
  - id: openrouter_fast
    provider: openrouter
    remote: true
    model: openai/gpt-4o-mini
    capabilities: [presentation_rewrite]
    cost: 3
    latency: 3
    reliability: 4

weights:
  cost: -2
  latency: -1
  reliability: 3

selection:
  tie_breaker: [reliability, latency, cost, id]
"""


class TestLLMBroker(unittest.TestCase):
    def _write_policy(self, text: str = POLICY_TEXT) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = os.path.join(tmpdir.name, "policy.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.strip())
        return path

    def test_broker_local_when_remote_disallowed(self) -> None:
        path = self._write_policy()
        policy = load_policy(path)
        cfg = _config(
            llm_allow_remote=False,
            ollama_base_url="http://127.0.0.1:11434",
            llm_broker_policy_path=path,
        )
        broker = LLMBroker(cfg, policy, client_factory=lambda c, p: {"id": p.id})
        client, decision = broker.select(TaskSpec(task="presentation_rewrite"))
        self.assertEqual(client["id"], "ollama_local")
        self.assertEqual(decision["winner_id"], "ollama_local")

    def test_broker_remote_when_allowed(self) -> None:
        path = self._write_policy()
        policy = load_policy(path)
        policy.weights["reliability"] = 8
        cfg = _config(
            llm_allow_remote=True,
            openai_api_key="key",
            ollama_base_url="http://127.0.0.1:11434",
            llm_broker_policy_path=path,
        )
        broker = LLMBroker(cfg, policy, client_factory=lambda c, p: {"id": p.id})
        client, decision = broker.select(TaskSpec(task="presentation_rewrite"))
        self.assertEqual(client["id"], "openai_fast")
        self.assertEqual(decision["winner_id"], "openai_fast")

    def test_missing_api_key_makes_candidate_unavailable(self) -> None:
        path = self._write_policy()
        policy = load_policy(path)
        cfg = _config(
            llm_allow_remote=True,
            openai_api_key=None,
            ollama_base_url="http://127.0.0.1:11434",
            llm_broker_policy_path=path,
        )
        broker = LLMBroker(cfg, policy, client_factory=lambda c, p: {"id": p.id})
        client, decision = broker.select(TaskSpec(task="presentation_rewrite"))
        self.assertEqual(client["id"], "ollama_local")
        self.assertEqual(decision["winner_id"], "ollama_local")

    def test_invalid_policy_falls_back(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = os.path.join(tmpdir.name, "policy.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("providers:\n  - id: missing_fields\n")
        cfg = _config(llm_broker_policy_path=path)
        broker, error = build_llm_broker(cfg)
        self.assertIsNone(broker)
        self.assertTrue(error)

    def test_openrouter_unavailable_without_key(self) -> None:
        path = self._write_policy(POLICY_OPENROUTER)
        policy = load_policy(path)
        cfg = _config(
            llm_allow_remote=True,
            openrouter_api_key=None,
            openrouter_base_url="https://openrouter.ai/api/v1",
            ollama_base_url="http://127.0.0.1:11434",
            llm_broker_policy_path=path,
        )
        broker = LLMBroker(cfg, policy, client_factory=lambda c, p: {"id": p.id})
        client, decision = broker.select(TaskSpec(task="presentation_rewrite"))
        self.assertEqual(client["id"], "ollama_local")
        self.assertEqual(decision["winner_id"], "ollama_local")

    def test_openrouter_available_with_key(self) -> None:
        path = self._write_policy(POLICY_OPENROUTER)
        policy = load_policy(path)
        policy.weights["reliability"] = 8
        cfg = _config(
            llm_allow_remote=True,
            openrouter_api_key="key",
            openrouter_base_url="https://openrouter.ai/api/v1",
            ollama_base_url="http://127.0.0.1:11434",
            llm_broker_policy_path=path,
        )
        broker = LLMBroker(cfg, policy, client_factory=lambda c, p: {"id": p.id})
        client, decision = broker.select(TaskSpec(task="presentation_rewrite"))
        self.assertEqual(client["id"], "openrouter_fast")
        self.assertEqual(decision["winner_id"], "openrouter_fast")

    def test_deterministic_selection(self) -> None:
        path = self._write_policy()
        policy = load_policy(path)
        cfg = _config(
            llm_allow_remote=True,
            openai_api_key="key",
            ollama_base_url="http://127.0.0.1:11434",
            llm_broker_policy_path=path,
        )
        broker = LLMBroker(cfg, policy, client_factory=lambda c, p: {"id": p.id})
        client1, decision1 = broker.select(TaskSpec(task="presentation_rewrite"))
        client2, decision2 = broker.select(TaskSpec(task="presentation_rewrite"))
        self.assertEqual(client1["id"], client2["id"])
        self.assertEqual(decision1["winner_id"], decision2["winner_id"])


if __name__ == "__main__":
    unittest.main()
