from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent.api_server import AgentRuntime
from tests.test_assistant_behavior_release_gate import _MemoryHandlerForTest, _assistant_text, _config


class TestChatBehaviorAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        self.runtime = AgentRuntime(
            _config(
                self.registry_path,
                self.db_path,
                skills_path=str(Path(__file__).resolve().parents[1] / "skills"),
            )
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _post_chat(self, prompt: str) -> tuple[int, dict[str, Any], str]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "session_id": "behavior-audit-session",
            "thread_id": "behavior-audit-thread",
            "source_surface": "webui",
            "purpose": "chat",
            "task_type": "chat",
            "trace_id": f"behavior-audit-{prompt[:8]}",
        }
        handler = _MemoryHandlerForTest(self.runtime, "/chat", payload)
        handler.do_POST()
        raw = handler.body.decode("utf-8", errors="replace")
        body = json.loads(raw) if raw.strip().startswith("{") else {}
        if not isinstance(body, dict):
            body = {}
        return int(handler.status_code), body, _assistant_text(body)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _create_pdf_table_catalog_source(self) -> None:
        root = Path(self.runtime.pack_store.external_storage_root()) / "behavior-audit-pdf-source"
        pack_source = root / "pack"
        pack_source.mkdir(parents=True, exist_ok=True)
        (pack_source / "SKILL.md").write_text(
            "# PDF Table Extraction Proof Pack\n\nHOSTILE_PROOF_MARKER_DO_NOT_EXPOSE\n",
            encoding="utf-8",
        )
        self._write_json(
            pack_source / "metadata.json",
            {
                "id": "pdf-table-extraction-proof-pack",
                "name": "PDF Table Extraction Proof Pack",
                "version": "0.1.0",
                "capabilities": ["pdf_table_extraction"],
            },
        )
        catalog = root / "catalog.json"
        self._write_json(
            catalog,
            {
                "packs": [
                    {
                        "id": "pdf-table-extraction-proof",
                        "remote_id": "pdf-table-extraction-proof",
                        "name": "PDF Table Extraction Proof Pack",
                        "summary": "Local proof pack for safe PDF table extraction lifecycle.",
                        "source_url": str(pack_source),
                        "source_kind_hint": "local_path",
                        "artifact_type_hint": "portable_text_skill",
                        "tags": ["pdf", "table", "proof"],
                        "capabilities": ["pdf_table_extraction"],
                        "has_skill_md": True,
                    }
                ]
            },
        )
        ok, body = self.runtime.create_pack_source_catalog(
            {
                "source_id": "behavior-audit-pdf-catalog",
                "name": "Behavior Audit PDF Catalog",
                "kind": "local_catalog",
                "base_url": str(catalog),
                "enabled": True,
                "supports_search": True,
                "supports_preview": True,
            },
            changed_by="test",
        )
        self.assertTrue(ok, body)

    def _assert_grounded_reply(self, prompt: str) -> tuple[int, dict[str, Any], str]:
        status, body, text = self._post_chat(prompt)
        self.assertIn(status, {200, 400}, prompt)
        self.assertTrue(text.strip(), prompt)
        self.assertEqual(text, str(body.get("message") or "").strip(), prompt)
        lowered = text.lower()
        for forbidden in (
            "runtime_payload",
            "selection_policy",
            "trace_id:",
            "source_surface:",
            "thread_id:",
            "read-only guard",
            "nl path refused",
        ):
            self.assertNotIn(forbidden, lowered, prompt)
        return status, body, text

    def test_dumb_user_prompts_stay_on_grounded_chat_path(self) -> None:
        expectations = {
            "what model am i using": ("model_status", ("model", "configured")),
            "is memory on": ("agent_memory", ("memory", "separate from system ram")),
            "why arent you working": ("runtime_status", ("chat target", "runtime")),
            "what can you do": ("assistant_capabilities", ("system inspection", "local memory")),
            "install a skill that lets you browse": ("action_tool", ("browser", "preview")),
            "fix yourself": ("runtime_status", ("diagnostics", "code changes")),
            "use the best local model": ("model_status", ("local", "model")),
            "do you remember what we were doing": ("agent_memory", ("saved", "runtime context")),
            "my computer is slow": ("operational_status", ("ram", "load")),
            "open the app": ("assistant_capabilities", ("personal agent", "127.0.0.1")),
        }

        for prompt, (expected_route, required_phrases) in expectations.items():
            with self.subTest(prompt=prompt):
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))
                lowered = text.lower()
                for phrase in required_phrases:
                    self.assertIn(phrase, lowered)
                if prompt == "install a skill that lets you browse":
                    self.assertNotIn("which model do you want me to acquire", lowered)

    def test_common_intents_accept_varied_human_phrasing(self) -> None:
        cases = (
            ("what is your runtime status", "runtime_status", ("chat", "ready")),
            ("are you actually connected to a model right now", "runtime_status", ("chat", "ready")),
            ("which provider are you using", "model_status", ("model", "ollama")),
            ("what do you remember about my preferences", "agent_memory", ("preference",)),
            ("where were we before", "agent_memory", ("saved",)),
            ("can you add browser capabilities", "action_tool", ("browser", "preview")),
            ("add a capability for reading webpages", "action_tool", ("browser", "preview")),
            ("why is my system lagging", "operational_status", ("cause",)),
            ("check ram and cpu pressure", "operational_status", ("cpu",)),
            ("what should I ask you next", "assistant_capabilities", ("runtime",)),
        )

        for prompt, expected_route, required_phrases in cases:
            with self.subTest(prompt=prompt):
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                lowered = text.lower()
                for phrase in required_phrases:
                    self.assertIn(phrase, lowered)
                self.assertNotIn("runtime_payload", lowered)
                self.assertNotIn("trace_id", lowered)

    def test_daily_driver_package_install_prompt_uses_confirmation_preview(self) -> None:
        status, body, text = self._post_chat("Can you install htop on this machine?")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        setup = body.get("setup") if isinstance(body.get("setup"), dict) else {}
        lowered = text.lower()

        self.assertEqual(200, status)
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("shell", meta.get("used_tools", []))
        self.assertFalse(meta.get("used_llm"))
        self.assertIn("install htop", lowered)
        self.assertIn("say yes to continue", lowered)
        self.assertIn("mutates the local system", lowered)
        self.assertNotIn("installing htop", lowered)
        self.assertTrue(setup.get("requires_confirmation"))

    def test_direct_questions_do_not_fall_into_stale_clarification_context(self) -> None:
        cases = {
            "is the local API healthy": ("runtime_status", ("ready", "chat")),
            "are you actually connected to a model right now": ("runtime_status", ("ready", "chat")),
            "how do i open the web UI": ("assistant_capabilities", ("127.0.0.1", "personal agent")),
            "is setup complete": ("setup_flow", ("setup", "chat")),
            "help me set this up": ("setup_flow", ("setup", "chat")),
            "where were we before": ("agent_memory", ("runtime context", "saved")),
            "this feels broken, what is wrong": ("runtime_status", ("chat target", "runtime")),
        }

        for prompt, (expected_route, required_phrases) in cases.items():
            with self.subTest(prompt=prompt):
                self._post_chat("what model am i using")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual(expected_route, meta.get("route"))
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                self.assertNotIn("i was following", text.lower())
                lowered = text.lower()
                for phrase in required_phrases:
                    self.assertIn(phrase, lowered)

    def test_assistant_does_not_claim_unperformed_agent_actions(self) -> None:
        cases = (
            ("open the app", ("i opened", "i launched")),
            ("install a skill that lets you browse", ("i installed", "installed browser support", "i ran")),
            ("make qwen3.6 the default model", ("default model updated", "i changed")),
        )

        for prompt, forbidden_phrases in cases:
            with self.subTest(prompt=prompt):
                _status, _body, text = self._assert_grounded_reply(prompt)
                lowered = text.lower()
                for phrase in forbidden_phrases:
                    self.assertNotIn(phrase, lowered)
                self.assertTrue(
                    any(token in lowered for token in ("preview", "say yes", "browse to", "desktop launcher")),
                    text,
                )

    def test_browser_skill_requests_use_pack_preview_not_apt_or_stale_followup(self) -> None:
        prompts = (
            "install a skill that lets you browse",
            "can you add browser capabilities",
            "what skills can you install for web research",
            "add a capability for reading webpages",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._post_chat("my computer is slow")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual("action_tool", meta.get("route"))
                self.assertIn("pack_acquisition", meta.get("used_tools") or [])
                lowered = text.lower()
                self.assertIn("browser", lowered)
                self.assertIn("preview", lowered)
                self.assertIn("not installed or usable", lowered)
                self.assertIn("approved catalog", lowered)
                self.assertNotIn("approved starter catalog sources and other approved/trusted catalog sources only", lowered)
                self.assertNotIn("apt-get", lowered)
                self.assertNotIn("install a using", lowered)
                self.assertNotIn("which model do you want me to acquire", lowered)
                self.assertNotIn("likely cause:", lowered)

    def test_pdf_table_missing_capability_uses_approved_pack_source_preview_path(self) -> None:
        self._create_pdf_table_catalog_source()
        _status, body, text = self._assert_grounded_reply("Can you add a PDF table extraction skill?")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        lowered = text.lower()
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("pack_acquisition", meta.get("used_tools") or [])
        self.assertIn("pdf table extraction proof pack", lowered)
        self.assertIn("approved catalog", lowered)
        self.assertIn("preview", lowered)
        self.assertIn("say yes", lowered)
        self.assertIn("not installed or usable", lowered)
        self.assertNotIn("i installed", lowered)
        self.assertNotIn("i added", lowered)
        self.assertNotIn("hostile_proof_marker_do_not_expose", lowered)

    def test_rtx_2060_local_provider_guidance_is_grounded_in_provider_boundaries(self) -> None:
        prompt = (
            "For Debian with RTX 2060 6GB VRAM and 64GB RAM, what local model/provider setup should I use? "
            "Do I have direct llama.cpp support?"
        )
        _status, body, text = self._assert_grounded_reply(prompt)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        lowered = text.lower()
        self.assertEqual("action_tool", meta.get("route"))
        self.assertIn("model_provider_support", meta.get("used_tools") or [])
        self.assertIn("ollama", lowered)
        self.assertIn("openai-compatible", lowered)
        self.assertIn("llama.cpp direct binary/library management is absent", lowered)
        self.assertIn("lm studio", lowered)
        self.assertIn("vllm", lowered)
        self.assertIn("rtx 2060 6gb vram", lowered)
        self.assertIn("64gb ram", lowered)
        self.assertIn("huge", lowered)
        self.assertIn("not", lowered)
        self.assertNotIn("70b is easy", lowered)
        self.assertFalse(lowered.startswith("current model:"), text)

    def test_open_chat_prompts_after_operational_status_do_not_reuse_stale_context(self) -> None:
        prompts = (
            "help me plan the next hour",
            "explain this project in plain english",
            "give me a concise checklist for testing this app",
            "write a short note saying the assistant is working",
            "what should I ask you next",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._post_chat("my computer is slow")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                self.assertNotEqual("interpretation_followup", meta.get("route"))
                self.assertNotIn("i was following", text.lower())
                self.assertNotIn("likely cause:", text.lower())
                if prompt == "what should I ask you next":
                    self.assertEqual("assistant_capabilities", meta.get("route"))
                if prompt == "write a short note saying the assistant is working":
                    self.assertEqual("generic_chat", meta.get("route"))
                    self.assertIn("assistant is working", text.lower())

    def test_vague_fix_it_uses_short_clarification_not_setup_summary(self) -> None:
        _status, body, text = self._assert_grounded_reply("fix it")
        lowered = text.lower()
        self.assertTrue(
            "what should i fix" in lowered
            or "reply 1, 2, or 3" in lowered
            or body.get("error_kind") == "needs_clarification"
        )
        self.assertNotIn("setup looks okay", lowered)
        self.assertNotIn("other local chat models", lowered)

    def test_vague_do_it_uses_actionable_clarification(self) -> None:
        _status, body, text = self._assert_grounded_reply("do it")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        lowered = text.lower()
        self.assertEqual("assistant_clarification", meta.get("route"))
        self.assertIn("i don’t have a current action to continue", lowered)
        self.assertIn("check runtime status", lowered)
        self.assertNotEqual("i’m not sure.", lowered.strip())

    def test_stale_yes_uses_actionable_clarification(self) -> None:
        _status, body, text = self._assert_grounded_reply("yes")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        lowered = text.lower()
        self.assertEqual("assistant_clarification", meta.get("route"))
        self.assertIn("current action", lowered)
        self.assertNotIn("i couldn't complete that yet", lowered)

    def test_setup_first_line_is_concise(self) -> None:
        _status, _body, text = self._assert_grounded_reply("is setup complete")
        first = text.splitlines()[0]
        self.assertLessEqual(len(first), 180)
        self.assertNotIn("other local chat models", first.lower())

    def test_capabilities_mentions_search_and_skill_acquisition(self) -> None:
        _status, body, text = self._assert_grounded_reply("what can you help me with right now")
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        lowered = text.lower()
        self.assertEqual("assistant_capabilities", meta.get("route"))
        self.assertIn("safe web search", lowered)
        self.assertIn("external skill acquisition", lowered)
        self.assertIn("source approval", lowered)
        self.assertIn("quarantine", lowered)

    def test_resource_prompts_after_operational_status_route_to_operational_status(self) -> None:
        prompts = (
            "is something eating resources",
            "what is using resources",
            "what is eating memory",
            "what is eating cpu",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._post_chat("my computer is slow")
                _status, body, text = self._assert_grounded_reply(prompt)
                meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
                self.assertEqual("operational_status", meta.get("route"))
                self.assertNotEqual("assistant_clarification", meta.get("route"))
                self.assertNotIn("i was following", text.lower())


if __name__ == "__main__":
    unittest.main()
