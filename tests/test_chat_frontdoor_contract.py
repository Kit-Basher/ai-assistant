from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.setup_chat_flow import classify_runtime_chat_route
from test_api_server import _config


class _ChatHandler(APIServerHandler):
    def __init__(self, runtime: AgentRuntime, payload: dict[str, object]) -> None:
        self.runtime = runtime
        self.path = "/chat"
        self.headers = {"Content-Length": "0"}
        self._payload = dict(payload)
        self.status_code = 0
        self.response_payload: dict[str, object] = {}

    def _read_json(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._payload)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.response_payload = json.loads(json.dumps(payload, ensure_ascii=True))


def _post_chat(runtime: AgentRuntime, text: str, *, user_id: str, thread_id: str) -> dict[str, object]:
    handler = _ChatHandler(
        runtime,
        {
            "messages": [{"role": "user", "content": text}],
            "user_id": user_id,
            "thread_id": thread_id,
            "session_id": thread_id,
            "source_surface": "webui",
        },
    )
    handler.do_POST()
    assert handler.status_code == 200, handler.response_payload
    return handler.response_payload


def _post_chat_raw(runtime: AgentRuntime, text: str, *, user_id: str, thread_id: str) -> tuple[int, dict[str, object]]:
    handler = _ChatHandler(
        runtime,
        {"messages": [{"role": "user", "content": text}], "user_id": user_id, "thread_id": thread_id},
    )
    handler.do_POST()
    return handler.status_code, handler.response_payload


def test_advertised_basic_capabilities_have_direct_deterministic_routes() -> None:
    cases = {
        "is the runtime ready?": "runtime_status",
        "search the web for today's weather in Saskatoon": "action_tool",
        "find files named release_gate.py": "action_tool",
        "can you search through my files?": "action_tool",
        "can you locate the video i just downloaded?": "action_tool",
        "read /tmp/example.txt": "action_tool",
        "what packs do you have?": "assistant_capabilities",
        "what external packs or skills are available?": "assistant_capabilities",
        "create a directory named fixture": "action_tool",
        "what model are you using": "model_status",
        "whats wrong with ollama:qwen2.5:3b-instruct can you fix it?": "model_status",
        "switch to ollama:qwen3.6:35b-a3b": "model_status",
        "what model scout sees": "action_tool",
        "what model should you use for chat?": "action_tool",
        "why are you using Gemma?": "model_policy_status",
        "run model scout now": "action_tool",
        "recommend the best model for coding/research/chat": "action_tool",
        "is Telegram working?": "runtime_status",
    }
    for phrase, expected_route in cases.items():
        decision = classify_runtime_chat_route(phrase)
        assert decision.get("route") == expected_route, (phrase, decision)
        assert decision.get("generic_allowed") is False, (phrase, decision)


def test_filesystem_list_search_read_and_sensitive_denial_work_through_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        fixture = root / "allowed"
        fixture.mkdir()
        note = fixture / "frontdoor-note.txt"
        note.write_text("front door filesystem fixture\n", encoding="utf-8")
        runtime = AgentRuntime(
            _config(
                str(root / "registry.json"),
                str(root / "agent.db"),
                perception_roots=(str(fixture),),
            )
        )
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("filesystem request reached LLM")):
            listed = _post_chat(runtime, f"list files in {fixture}", user_id="fs", thread_id="fs:t")
            searched = _post_chat(runtime, f"find files named frontdoor-note in {fixture}", user_id="fs", thread_id="fs:t")
            read = _post_chat(runtime, f"read {note}", user_id="fs", thread_id="fs:t")
            denied_status, denied = _post_chat_raw(runtime, "read ~/.ssh/id_ed25519", user_id="fs", thread_id="fs:t")
        assert "frontdoor-note.txt" in str(listed.get("message"))
        assert "frontdoor-note.txt" in str(searched.get("message"))
        assert "front door filesystem fixture" in str(read.get("message"))
        assert denied_status in {200, 400}
        assert "protected" in str(denied.get("message") or "").lower() or "outside" in str(denied.get("message") or "").lower()
        assert all(
            (response.get("meta") if isinstance(response.get("meta"), dict) else {}).get("used_llm") is False
            for response in (listed, searched, read, denied)
        )


def test_standalone_filesystem_questions_never_fall_through_to_generic_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        allowed = root / "allowed"
        allowed.mkdir()
        runtime = AgentRuntime(
            _config(
                str(root / "registry.json"),
                str(root / "agent.db"),
                perception_roots=(str(allowed),),
            )
        )
        with patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("standalone filesystem request reached generic LLM inference"),
        ):
            tools = _post_chat(
                runtime,
                "what tools do you have access to",
                user_id="standalone-fs",
                thread_id="standalone-fs:thread",
            )
            capability = _post_chat(
                runtime,
                "can you search through my files?",
                user_id="standalone-fs",
                thread_id="standalone-fs:thread",
            )
            recent_video = _post_chat(
                runtime,
                "can you locate the video i just downloaded?",
                user_id="standalone-fs",
                thread_id="standalone-fs:thread",
            )

        for response in (capability, recent_video):
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            message = str(response.get("message") or "").lower()
            assert meta.get("route") == "action_tool", response
            assert meta.get("used_llm") is False, response
            assert "sandboxed environment" not in message
            assert "unable to access your device" not in message
            assert "cannot directly access" not in message
            assert "unable to access or locate files" not in message

        capability_message = str(capability.get("message") or "")
        assert str(allowed) in capability_message
        assert "what" in capability_message.lower() and "search" in capability_message.lower()
        assert "file" in str(tools.get("message") or "").lower()

        recent_message = str(recent_video.get("message") or "").lower()
        assert "downloads" in recent_message
        assert "outside" in recent_message and "allowed" in recent_message
        assert "review" in recent_message


def test_model_scout_and_manager_questions_are_grounded_through_chat() -> None:
    prompts = (
        "what model scout sees",
        "what model should you use for chat?",
        "why are you using Gemma?",
        "run model scout now",
        "recommend the best model for coding/research/chat",
    )
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        truth = runtime.runtime_truth_service()
        candidate = {
            "model_id": "ollama:Gemma:latest",
            "provider_id": "ollama",
            "local": True,
            "usable_now": True,
            "recommendation_explanation": "it is installed, healthy, and ready for chat",
        }
        scout_payload = {
            "active_model": "ollama:Gemma:latest",
            "active_provider": "ollama",
            "current_candidate": dict(candidate),
            "recommended_candidate": dict(candidate),
            "task_recommendation": dict(candidate),
            "candidate_rows": [dict(candidate)],
            "better_candidates": [],
            "not_ready_models": [],
            "role_candidates": {"comfortable_local_default": dict(candidate)},
            "recommendation_roles": {},
            "policy": {
                "mode": "safe",
                "mode_label": "SAFE MODE",
                "safe_mode": True,
                "allow_remote_recommendation": False,
                "allow_install_pull": False,
            },
            "task_request": {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            "advisory_only": True,
        }
        policy_payload = {
            "current_candidate": dict(candidate),
            "selected_candidate": dict(candidate),
            "recommended_candidate": dict(candidate),
            "switch_recommended": False,
            "decision_detail": "the current default already matches the best ready local candidate",
            "tier_candidates": {},
        }
        with patch.object(truth, "model_scout_v2_status", return_value=scout_payload) as scout, patch.object(
            truth, "model_policy_status", return_value=policy_payload
        ) as policy, patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("model scout request reached generic LLM inference"),
        ):
            responses = [
                _post_chat(runtime, prompt, user_id="model-scout", thread_id="model-scout:thread")
                for prompt in prompts
            ]

        for prompt, response in zip(prompts, responses, strict=True):
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            message = str(response.get("message") or "").lower()
            assert meta.get("route") in {"action_tool", "model_policy_status"}, (prompt, response)
            assert meta.get("used_llm") is False, (prompt, response)
            assert "ollama:gemma:latest" in message, (prompt, response)
        assert scout.call_count == 4
        policy.assert_called_once()


def test_plain_and_upgrade_model_scout_requests_beat_shell_fallback_through_chat() -> None:
    prompts = (
        "run the model scout and see if there are any better new models we should upgrade to instead",
        "run the model scout",
    )
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        truth = runtime.runtime_truth_service()
        local_candidate = {
            "model_id": "ollama:Gemma:latest",
            "provider_id": "ollama",
            "local": True,
            "usable_now": True,
            "recommendation_explanation": "it is the strongest installed model ready for interactive chat",
        }
        scout_payload = {
            "active_model": "ollama:Gemma:latest",
            "active_provider": "ollama",
            "current_candidate": dict(local_candidate),
            "recommended_candidate": dict(local_candidate),
            "task_recommendation": dict(local_candidate),
            "candidate_rows": [dict(local_candidate)],
            "better_candidates": [],
            "not_ready_models": [],
            "role_candidates": {"comfortable_local_default": dict(local_candidate)},
            "recommendation_roles": {},
            "policy": {
                "mode": "safe",
                "mode_label": "SAFE MODE",
                "safe_mode": True,
                "allow_remote_recommendation": False,
                "allow_install_pull": False,
            },
            "task_request": {"task_type": "chat", "requirements": ["chat"], "preferred_local": True},
            "advisory_only": True,
        }
        discovery_payload = {
            "ok": True,
            "models": [
                {
                    "id": "huggingface:example/new-chat-model",
                    "provider": "huggingface",
                    "source": "huggingface",
                    "local": False,
                    "installable": True,
                }
            ],
            "sources": [{"source": "huggingface", "queried": True, "ok": True, "count": 1}],
            "debug": {"source_errors": {}},
        }
        with patch.object(truth, "model_scout_v2_status", return_value=scout_payload) as scout, patch.object(
            truth, "model_discovery_query", return_value=discovery_payload
        ) as discovery, patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("model scout request reached generic LLM inference"),
        ), patch(
            "agent.orchestrator.Orchestrator._shell_blocked_request_response",
            side_effect=AssertionError("model scout request reached shell-command fallback"),
        ):
            responses = [
                _post_chat(runtime, prompt, user_id="model-scout-shell", thread_id="model-scout-shell:thread")
                for prompt in prompts
            ]

        for prompt, response in zip(prompts, responses, strict=True):
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            message = str(response.get("message") or "").lower()
            assert meta.get("route") == "action_tool", (prompt, response)
            assert meta.get("used_llm") is False, (prompt, response)
            expected_tools = ["model_scout", "model_discovery_manager"] if "better new models" in prompt else ["model_scout"]
            assert meta.get("used_tools") == expected_tools, (prompt, response)
            assert "ollama:gemma:latest" in message, (prompt, response)
            assert "can't run that command" not in message, (prompt, response)

        upgrade_message = str(responses[0].get("message") or "").lower()
        assert "installed local" in upgrade_message
        assert "remote discovery" in upgrade_message
        assert "safe mode" in upgrade_message
        assert "no model was downloaded" in upgrade_message
        assert "no model was switched" in upgrade_message
        assert scout.call_count == 2
        discovery.assert_called_once()


def test_recent_video_location_followup_checks_videos_scope_through_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        truth = runtime.runtime_truth_service()
        downloads_result = {
            "ok": False,
            "type": "filesystem_recent_downloaded_videos",
            "downloads_path": "/home/test/Downloads",
            "resolved_path": "/data/test/Downloads",
            "scope_configured": False,
            "error_kind": "outside_allowed_roots",
        }
        videos_result = {
            "ok": False,
            "type": "filesystem_recent_videos",
            "directory_path": "/home/test/Videos",
            "resolved_path": "/data/test/Videos",
            "scope_configured": False,
            "error_kind": "outside_allowed_roots",
        }
        with patch.object(truth, "filesystem_recent_downloaded_videos", return_value=downloads_result), patch.object(
            truth, "filesystem_recent_videos_in_directory", return_value=videos_result
        ) as videos_search, patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("video location follow-up reached generic LLM inference"),
        ), patch(
            "agent.orchestrator.Orchestrator._shell_blocked_request_response",
            side_effect=AssertionError("video location follow-up reached shell-command fallback"),
        ):
            first = _post_chat(
                runtime,
                "can you locate the video i just downloaded?",
                user_id="video-followup",
                thread_id="video-followup:thread",
            )
            followup_status, followup = _post_chat_raw(
                runtime,
                "its in the videos folder",
                user_id="video-followup",
                thread_id="video-followup:thread",
            )

        first_meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
        followup_meta = followup.get("meta") if isinstance(followup.get("meta"), dict) else {}
        followup_message = str(followup.get("message") or "").lower()
        assert first_meta.get("used_llm") is False
        assert followup_status in {200, 400}
        assert followup_meta.get("route") == "action_tool"
        assert followup_meta.get("used_llm") is False
        assert followup_meta.get("used_tools") == ["filesystem"]
        assert "/home/test/videos" in followup_message
        assert "/data/test/videos" in followup_message
        assert "outside" in followup_message and "allowed" in followup_message
        videos_search.assert_called_once()
        assert str(videos_search.call_args.args[0]).endswith("Videos")


def test_local_text_pack_ingest_is_previewed_confirmed_and_reported_through_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        pack = root / "safe-local-pack"
        pack.mkdir()
        (pack / "SKILL.md").write_text(
            "---\nid: frontdoor-fixture\nname: Frontdoor Fixture\nversion: 1.0.0\n---\nSafe local guidance.\n",
            encoding="utf-8",
        )
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("pack request reached LLM")):
            state_before = _post_chat(runtime, "what external packs or skills are available?", user_id="pack", thread_id="pack:t")
            preview = _post_chat(runtime, f"install a skill pack from {pack}", user_id="pack", thread_id="pack:t")
            applied = _post_chat(runtime, "yes", user_id="pack", thread_id="pack:t")
            state_after = _post_chat(runtime, "what packs do you have?", user_id="pack", thread_id="pack:t")
        before_payload = state_before.get("setup") if isinstance(state_before.get("setup"), dict) else {}
        after_payload = state_after.get("setup") if isinstance(state_after.get("setup"), dict) else {}
        before_count = int((before_payload.get("pack_state") or {}).get("installed_count") or 0)
        after_count = int((after_payload.get("pack_state") or {}).get("installed_count") or 0)
        assert "nothing changes until you approve" in str(preview.get("message") or "").lower()
        assert "imported for review only" in str(applied.get("message") or "").lower()
        assert after_count == before_count + 1
        assert "not approved" in str(applied.get("message") or "").lower()
        assert "not enabled" in str(applied.get("message") or "").lower()


def test_remote_pack_and_unrelated_mutation_policy_remain_blocked_through_chat() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("policy request reached LLM")):
            remote_status, remote = _post_chat_raw(
                runtime,
                "install a skill pack from https://example.invalid/pack.zip",
                user_id="policy",
                thread_id="policy:t",
            )
            mutation_status, mutation = _post_chat_raw(
                runtime,
                "run sudo rm -rf /tmp/not-allowed",
                user_id="policy",
                thread_id="policy:t",
            )
        assert remote_status in {200, 400}
        assert any(term in str(remote.get("message") or "").lower() for term in ("remote", "approved", "not installed", "nothing was installed"))
        assert mutation_status in {200, 400}
        assert any(term in str(mutation.get("message") or "").lower() for term in ("blocked", "can't run", "cannot run", "unsupported"))


def test_large_local_model_is_not_chat_ready_without_a_successful_interactive_probe() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        truth = runtime.runtime_truth_service()
        inventory = {
            "models": [
                {
                    "provider_id": "ollama",
                    "model_id": "ollama:qwen3.6:35b-a3b",
                    "enabled": True,
                    "available": True,
                    "lifecycle_state": "ready",
                    "local": True,
                }
            ]
        }
        provider = {
            "configured": True,
            "connection_state": "configured_and_usable",
            "selection_state": "configured_and_usable",
            "policy_blocked": False,
            "auth_required": False,
            "secret_present": True,
            "health_status": "ok",
        }
        with patch.object(truth, "model_inventory_status", return_value=inventory), patch.object(
            truth, "_provider_status_snapshot", return_value=provider
        ), patch.object(truth, "_model_health_row", return_value={"status": "ok"}), patch.object(
            truth, "_router_snapshot", return_value={}
        ), patch.object(truth, "_router_model_status", return_value={"status": "ok", "available": True}):
            unproven = truth.model_readiness_status()
            assert not unproven.get("ready_now_models")
            assert any("interactive chat latency budget" in str(row.get("availability_reason") or "") for row in unproven.get("not_ready_models", []))
            truth._interactive_model_probe_results = {  # noqa: SLF001 - qualification contract fixture
                "ollama:qwen3.6:35b-a3b": {"ok": True, "duration_ms": 500, "timeout_seconds": 15}
            }
            truth._invalidate_snapshot_cache()  # noqa: SLF001 - qualification contract fixture
            proven = truth.model_readiness_status()
            assert any(row.get("model_id") == "ollama:qwen3.6:35b-a3b" for row in proven.get("ready_now_models", []))


def test_model_interactive_probe_maps_timeout_to_latency_budget_failure() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        truth = runtime.runtime_truth_service()
        with patch.object(
            runtime,
            "test_provider",
            return_value=(False, {"ok": False, "error": "timeout", "duration_ms": 15_000}),
        ) as provider_test:
            ok, body = truth.test_chat_model_target("ollama:qwen3.6:35b-a3b", provider_id="ollama")
        assert ok is False
        assert body.get("error") == "interactive_latency_budget_exceeded"
        assert "interactive chat timeout budget" in str(body.get("reason") or "")
        assert provider_test.call_args.args[0] == "ollama"
        assert provider_test.call_args.args[1]["timeout_seconds"] == runtime.config.llm_timeout_seconds


def test_failed_real_transcript_stays_on_deterministic_chat_frontdoor() -> None:
    transcript = (
        "can you use search",
        "can you search the internet?",
        "can you search through my files? can you locate the video i just downloaded?",
        "what tools do you have access to",
        "what model are you using",
        "what models are available?",
        "whats wrong with ollama:qwen2.5:3b-instruct can you fix it?",
        "switch to ollama:qwen3.6:35b-a3b",
        "yes",
        "are you here?",
    )
    expected_routes = (
        {"action_tool"},
        {"action_tool"},
        {"action_tool"},
        {"assistant_capabilities"},
        {"model_status"},
        {"model_status"},
        {"model_status", "action_tool"},
        {"model_status"},
        {"model_status", "assistant_clarification", "plan_mode"},
        {"social_turn", "runtime_status"},
    )

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = AgentRuntime(_config(str(root / "registry.json"), str(root / "agent.db")))
        for model in ("Gemma:latest", "qwen2.5:3b-instruct", "qwen3.6:35b-a3b"):
            runtime.add_provider_model(
                "ollama",
                {"model": model, "capabilities": ["chat"], "available": True},
            )
        runtime.set_default_chat_model("ollama:Gemma:latest")
        orchestrator = runtime.orchestrator()
        truth = runtime.runtime_truth_service()

        def _interactive_test(model_id: str, *, provider_id: str | None = None):  # type: ignore[no-untyped-def]
            if "qwen3.6:35b-a3b" in model_id:
                return False, {
                    "ok": False,
                    "provider": provider_id or "ollama",
                    "model_id": model_id,
                    "error": "interactive_latency_budget_exceeded",
                    "reason": "tiny chat probe exceeded the interactive chat timeout budget",
                    "duration_ms": 31_000,
                }
            return True, {
                "ok": True,
                "provider": provider_id or "ollama",
                "model_id": model_id,
                "duration_ms": 250,
            }

        responses: list[dict[str, object]] = []
        with patch.object(runtime, "_auto_bootstrap_local_chat_model", return_value=None), patch.object(
            orchestrator,
            "_runtime_model_catalog",
            return_value=[
                "ollama:Gemma:latest",
                "ollama:qwen2.5:3b-instruct",
                "ollama:qwen3.6:35b-a3b",
            ],
        ), patch.object(
            truth,
            "test_chat_model_target",
            side_effect=_interactive_test,
        ), patch(
            "agent.orchestrator.route_inference",
            side_effect=AssertionError("basic front-door request reached generic LLM inference"),
        ):
            for utterance in transcript:
                responses.append(
                    _post_chat(
                        runtime,
                        utterance,
                        user_id="chat-frontdoor-transcript",
                        thread_id="chat-frontdoor-transcript:thread",
                    )
                )

        for utterance, response, allowed_routes in zip(transcript, responses, expected_routes, strict=True):
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            message = str(response.get("message") or "")
            assert meta.get("route") in allowed_routes, (utterance, response)
            assert meta.get("used_llm") is False, (utterance, response)
            assert "LLM timed out before producing a response" not in message, (utterance, response)

        file_message = str(responses[2].get("message") or "").lower()
        assert "download" in file_message
        assert any(term in file_message for term in ("allowed", "configured", "outside", "search"))

        named_repair_message = str(responses[6].get("message") or "").lower()
        assert "ollama:qwen2.5:3b-instruct" in named_repair_message

        slow_switch_message = str(responses[7].get("message") or "").lower()
        assert "ollama:qwen3.6:35b-a3b" in slow_switch_message
        assert any(term in slow_switch_message for term in ("latency", "timeout", "too slow", "not ready"))
        assert "say yes" not in slow_switch_message

        final_message = str(responses[9].get("message") or "").lower()
        assert any(term in final_message for term in ("here", "ready", "available"))
