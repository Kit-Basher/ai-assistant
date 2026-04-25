from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.onboarding_flow import onboarding_completed_key
from agent.orchestrator import Orchestrator
from memory.db import MemoryDB


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeChatLLM:
    def __init__(self, *, enabled: bool, text: str = "LLM reply") -> None:
        self._enabled = bool(enabled)
        self._text = text
        self.chat_calls: list[dict[str, object]] = []

    def enabled(self) -> bool:
        return self._enabled

    def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return {"ok": True, "text": self._text, "provider": "ollama", "model": "llama3"}


def _run_script(script: Path, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class TestFirstRunReleaseSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.home = root / "home"
        self.bin_dir = root / "bin"
        self.state_dir = root / "state"
        self.home.mkdir(parents=True, exist_ok=True)
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        db_path = root / "test.db"
        self.db = MemoryDB(str(db_path))
        schema_path = REPO_ROOT / "memory" / "schema.sql"
        self.db.init_schema(str(schema_path))
        self.log_path = str(root / "events.log")
        self.skills_path = str(REPO_ROOT / "skills")

    def tearDown(self) -> None:
        self.db.close()
        self.tmpdir.cleanup()

    def _orchestrator(self) -> tuple[Orchestrator, _FakeChatLLM]:
        llm = _FakeChatLLM(enabled=True, text="LLM reply")
        orch = Orchestrator(
            db=self.db,
            skills_path=self.skills_path,
            log_path=self.log_path,
            timezone="UTC",
            llm_client=llm,
        )
        return orch, llm

    def _write_launcher_bins(self, *, ready_after: int = 1) -> tuple[Path, Path, Path]:
        systemctl_log = self.state_dir / "systemctl.log"
        curl_count = self.state_dir / "curl-count.txt"
        open_log = self.state_dir / "open.log"

        (self.bin_dir / "systemctl").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
            "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 3; fi\n"
            "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"start\" ]; then exit 0; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        (self.bin_dir / "curl").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "count=0\n"
            "if [ -f \"$CURL_COUNT\" ]; then count=$(cat \"$CURL_COUNT\"); fi\n"
            "count=$((count + 1))\n"
            "printf '%s' \"$count\" > \"$CURL_COUNT\"\n"
            f"if [ \"$count\" -lt {ready_after} ]; then exit 1; fi\n"
            "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
            encoding="utf-8",
        )
        (self.bin_dir / "xdg-open").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$*\" >> \"$OPEN_LOG\"\n",
            encoding="utf-8",
        )
        for item in ("systemctl", "curl", "xdg-open"):
            path = self.bin_dir / item
            path.chmod(0o755)
        return systemctl_log, curl_count, open_log

    def _run_launcher(self, *, ready_after: int = 1, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        systemctl_log, curl_count, open_log = self._write_launcher_bins(ready_after=ready_after)
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.home),
                "PATH": f"{self.bin_dir}:/bin:/usr/bin",
                "SYSTEMCTL_LOG": str(systemctl_log),
                "CURL_COUNT": str(curl_count),
                "OPEN_LOG": str(open_log),
                "AGENT_LAUNCHER_WAIT_SECONDS": "3",
                "AGENT_LAUNCHER_POLL_SECONDS": "0",
                "AGENT_LAUNCHER_OPEN_BROWSER": "0",
            }
        )
        if extra_env:
            env.update(extra_env)
        return _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

    def test_fresh_launch_surfaces_onboarding_and_then_suppresses_it(self) -> None:
        proc = self._run_launcher()
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("browser auto-open disabled", proc.stderr)
        self.assertFalse((self.state_dir / "open.log").exists())

        orch, llm = self._orchestrator()
        first = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertIn("tailor suggestions", first.text.lower())
        self.assertEqual([], llm.chat_calls)

        skipped = orch.handle_message("skip", "user1")
        self.assertIn("just ask me anything", skipped.text.lower())
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())

        second_proc = self._run_launcher()
        self.assertEqual(0, second_proc.returncode, second_proc.stderr)
        self.assertIn("browser auto-open disabled", second_proc.stderr)
        self.assertFalse((self.state_dir / "open.log").exists())

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            second = orch.handle_message("tell me something useful", "user1", chat_context={"source_surface": "webui"})
        self.assertEqual("LLM reply", second.text)
        self.assertEqual(1, len(llm.chat_calls))

    def test_intent_path_uses_existing_recommendation_helper_and_stays_preview_first(self) -> None:
        proc = self._run_launcher()
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("browser auto-open disabled", proc.stderr)
        self.assertFalse((self.state_dir / "open.log").exists())

        orch, llm = self._orchestrator()
        orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        orch.handle_message("yes", "user1")

        stub_recommendation = {
            "ok": True,
            "capability_required": "dev_tools",
            "capability_label": "coding tools",
            "status": "missing",
            "installed_pack": None,
            "recommended_pack": {
                "name": "Local Dev Tools",
                "reason": "best_fit_for_machine",
                "installable": True,
                "usable": False,
                "tradeoff_note": "lighter",
                "normalized_state": {"installable": True},
            },
            "alternate_pack": None,
            "comparison_mode": "single_recommendation",
            "fallback": "install_preview",
            "next_step": "If you want, say yes and I'll show the install details.",
            "warnings": [],
            "source_errors": [],
            "queries": [],
        }
        with patch("agent.orchestrator.recommend_onboarding_capability", return_value=stub_recommendation) as mock_recommend:
            response = orch.handle_message("coding", "user1", chat_context={"source_surface": "webui"})

        mock_recommend.assert_called_once()
        self.assertEqual("coding", mock_recommend.call_args.args[0])
        self.assertIn("I can add capabilities for coding.", response.text)
        self.assertIn("say yes and I'll show the install details", response.text)
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())
        self.assertEqual([], llm.chat_calls)

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            second = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertEqual("LLM reply", second.text)
        self.assertEqual(1, len(llm.chat_calls))

    def test_abandon_and_api_surface_behave_conservatively(self) -> None:
        proc = self._run_launcher()
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("browser auto-open disabled", proc.stderr)
        self.assertFalse((self.state_dir / "open.log").exists())

        orch, llm = self._orchestrator()
        prompt = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertIn("tailor suggestions", prompt.text.lower())

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            abandoned = orch.handle_message("what can you do?", "user1", chat_context={"source_surface": "webui"})
        self.assertEqual("LLM reply", abandoned.text)
        self.assertEqual("true", str(self.db.get_user_pref(onboarding_completed_key("user1")) or "").strip().lower())
        self.assertEqual(1, len(llm.chat_calls))

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            second = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertEqual("LLM reply", second.text)
        self.assertEqual(2, len(llm.chat_calls))

        with patch.object(orch, "_interpret_previous_result_followup", return_value=None), patch.object(
            orch, "_deep_system_followup_response", return_value=None
        ), patch.object(orch, "_handle_runtime_truth_chat", return_value=None), patch.object(
            orch, "_handle_action_tool_intent", return_value=None
        ), patch.object(
            orch, "_grounded_system_fallback_response", return_value=None
        ), patch.object(
            orch, "_safe_mode_containment_response", return_value=None
        ):
            api_like = orch.handle_message("hello", "user2")
        self.assertEqual("LLM reply", api_like.text)
        self.assertNotIn("tailor suggestions", api_like.text.lower())

    def test_non_fresh_state_suppresses_onboarding(self) -> None:
        self.db.set_user_pref(onboarding_completed_key("user1"), "true")
        proc = self._run_launcher()
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("browser auto-open disabled", proc.stderr)
        self.assertFalse((self.state_dir / "open.log").exists())

        orch, llm = self._orchestrator()
        response = orch.handle_message("hello", "user1", chat_context={"source_surface": "webui"})
        self.assertEqual("LLM reply", response.text)
        self.assertEqual(1, len(llm.chat_calls))
        self.assertNotIn("tailor suggestions", response.text.lower())


if __name__ == "__main__":
    unittest.main()
