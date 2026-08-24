from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.api_server import APIServerHandler, AgentRuntime
from agent.onboarding_contract import ONBOARDING_DEGRADED, ONBOARDING_LLM_MISSING, ONBOARDING_NOT_STARTED, ONBOARDING_SERVICES_DOWN, ONBOARDING_TOKEN_MISSING, onboarding_next_action, onboarding_steps
from test_api_server import _config

CORPUS_PATH = Path(__file__).parent / "held_out" / "wp6_user_scenarios.json"
CASES = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


class _Handler(APIServerHandler):
    def __init__(self, runtime: AgentRuntime, payload: dict[str, object], path: str = "/chat") -> None:
        self.runtime = runtime
        self.path = path
        self.headers = {"Content-Length": "0"}
        self._payload = payload
        self.status = 0
        self.body: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status = status
        self.body = json.loads(json.dumps(payload))


@pytest.fixture()
def runtime(tmp_path: Path) -> AgentRuntime:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "recovery-note.txt").write_text("restore proof marker\n", encoding="utf-8")
    instance = AgentRuntime(_config(str(tmp_path / "registry.json"), str(tmp_path / "agent.db"), perception_roots=(str(allowed),)))
    instance.startup_phase = "ready"
    return instance


def _chat(runtime: AgentRuntime, text: str, *, thread: str) -> dict[str, object]:
    handler = _Handler(runtime, {"messages": [{"role": "user", "content": text}], "user_id": "wp6-user", "thread_id": thread, "session_id": thread, "source_surface": "webui"})
    handler.do_POST()
    return {**handler.body, "_http_status": handler.status}


def _understanding(response: dict[str, object]) -> dict[str, object]:
    setup = response.get("setup") if isinstance(response.get("setup"), dict) else {}
    return setup.get("request_understanding") if isinstance(setup.get("request_understanding"), dict) else {}


def test_wp6_held_out_corpus_is_separate_bounded_and_complete() -> None:
    assert len(CASES) >= 24
    assert len({row["id"] for row in CASES}) == len(CASES)
    assert len({row["category"] for row in CASES}) >= 12
    production = "\n".join(path.read_text(encoding="utf-8", errors="replace") for root in (Path("agent"), Path("config")) for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".json"})
    for row in CASES:
        assert row["text"] not in production


def test_wp6_held_out_production_chat_evaluates_every_case(runtime: AgentRuntime, tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    note = allowed / "recovery-note.txt"
    results: list[dict[str, object]] = []
    for index, row in enumerate(CASES):
        text = str(row["text"]).format(root=allowed, file=note)
        try:
            response = _chat(runtime, text, thread=f"wp6-held:{index}")
            understanding = _understanding(response)
            selected = str(understanding.get("selected_capability") or understanding.get("selected_capability_id") or "")
            rendered = str(response.get("message") or response.get("text") or "").lower()
            if row.get("capability"):
                passed = selected == row["capability"]
            elif row.get("clarification"):
                meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
                passed = bool(understanding.get("clarification_needed") or response.get("needs_clarification") or meta.get("route") == "assistant_clarification") and rendered.count("?") == 1
            elif row.get("must_not_claim_completion"):
                passed = not any(token in rendered for token in ("completed successfully", "i created it", "done —"))
            elif row.get("must_not_mutate"):
                passed = not bool(response.get("did_work")) and not bool(response.get("mutated"))
            else:
                passed = bool(rendered)
            results.append({"id": row["id"], "category": row["category"], "passed": passed, "selected": selected})
        except Exception as exc:  # evaluate all blind rows even if one fails
            results.append({"id": row["id"], "category": row["category"], "passed": False, "error": exc.__class__.__name__})
    failures = [row for row in results if not row["passed"]]
    by_category: dict[str, list[bool]] = {}
    for row in results:
        by_category.setdefault(str(row["category"]), []).append(bool(row["passed"]))
    assert not failures, {"failures": failures, "by_category": {key: [sum(values), len(values)] for key, values in sorted(by_category.items())}}


def test_normal_user_onboarding_never_requires_shell_or_config_editing() -> None:
    forbidden = ("python -m", "systemctl", "curl ", "git ", ".json", "run:")
    for state in (ONBOARDING_NOT_STARTED, ONBOARDING_TOKEN_MISSING, ONBOARDING_LLM_MISSING, ONBOARDING_SERVICES_DOWN, ONBOARDING_DEGRADED):
        public = " ".join([onboarding_next_action(state), *onboarding_steps(state)]).lower()
        assert not any(marker in public for marker in forbidden), (state, public)
        assert "open " in public


def test_phrasal_backup_request_reaches_exact_operator_backup_preview(runtime: AgentRuntime) -> None:
    response = _chat(runtime, "Could you back up the assistant for recovery?", thread="wp6-backup-phrasal")
    understanding = _understanding(response)
    setup = response.get("setup") if isinstance(response.get("setup"), dict) else {}
    plan = setup.get("canonical_plan") if isinstance(setup.get("canonical_plan"), dict) else setup.get("plan") if isinstance(setup.get("plan"), dict) else {}
    assert understanding.get("selected_capability_id") == "operator.lifecycle"
    assert setup.get("requires_confirmation") is True
    assert plan.get("action_type") == "operator.backup"
    assert plan.get("executor_status") == "enabled"


def test_diagnostics_export_uses_real_route_and_redacts_hostile_secrets(runtime: AgentRuntime) -> None:
    runtime.audit_log.append(actor="Bearer wp6-secret-token", action="password=wp6", decision="deny", reason="Authorization: Basic abc", params={"api_key": "should-not-appear"}, dry_run=True, outcome="blocked", error_kind=None, duration_ms=1)
    handler = _Handler(runtime, {}, path="/diagnostics/export")
    with patch.object(runtime, "llm_health_summary", return_value={"ok": True, "health": {"providers": [], "models": [], "counts": {}}}):
        handler.do_GET()
    rendered = json.dumps(handler.body, sort_keys=True)
    assert handler.status == 200
    bundle = handler.body["bundle"]
    assert bundle["schema_version"] == "personal-agent.diagnostics.v1"
    assert bundle["capability_counts"]["total"] >= 22
    assert "wp6-secret-token" not in rendered
    assert "should-not-appear" not in rendered
    assert "Authorization: Basic abc" not in rendered
    assert "raw_conversations" in bundle["redaction"]


def test_wp6_requirements_are_not_a_hardcoded_capability_count() -> None:
    requirements = json.loads(Path("config/wp6_release_requirements.json").read_text(encoding="utf-8"))
    encoded = json.dumps(requirements, sort_keys=True)
    assert "capability_count" not in encoded
    assert len(requirements["evidence_layers"]) == 8
    assert len(requirements["required_journeys"]) == 25
    assert len(requirements["required_sensitivity_cases"]) == 16
