from __future__ import annotations

from pathlib import Path
import unittest

from agent.packs.capability_recommendation import (
    classify_capability_gap_request,
    build_capability_gap_response,
    detect_pack_capability_need,
    recommend_packs_for_capability,
    render_pack_capability_response,
)


class _FakePackStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def list_external_packs(self) -> list[dict[str, object]]:
        return list(self._rows)


class _FakeDiscovery:
    def __init__(self, sources: list[dict[str, object]], search_map: dict[tuple[str, str], list[dict[str, object]]]) -> None:
        self._sources = sources
        self._search_map = search_map

    def list_sources(self) -> list[dict[str, object]]:
        return list(self._sources)

    def search(self, source_id: str, query: str) -> dict[str, object]:
        source = next((row for row in self._sources if str(row.get("id") or "") == source_id), {})
        results = self._search_map.get((source_id, query), [])
        return {
            "source": dict(source) if isinstance(source, dict) else {},
            "search": {"results": list(results)},
            "from_cache": False,
            "stale": False,
        }


class TestPackCapabilityRecommendation(unittest.TestCase):
    def test_clear_capability_prompts_are_detected(self) -> None:
        cases = (
            ("Talk to me out loud", "voice_output"),
            ("Can you read this page back to me in speech?", "voice_output"),
            ("read something to me", "voice_output"),
            ("Use the avatar", "avatar_visual"),
            ("Open the robot camera feed", "camera_feed"),
            ("Look through my YouTube history and find the video about neurons differentiating during animal infancy.", "youtube_history_search"),
            ("Search my browser history for the article about CUDA setup.", "browser_history_search"),
            ("Import my Google Takeout watch history.", "google_takeout_import"),
            ("Search YouTube transcripts for this phrase.", "transcript_search"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                need = detect_pack_capability_need(text)
                self.assertIsNotNone(need)
                self.assertEqual(expected, need.get("capability"))

    def test_ambiguous_prompt_is_ignored(self) -> None:
        self.assertIsNone(detect_pack_capability_need("help me with the thing"))

    def test_knowledge_questions_do_not_trigger_capability_creation(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery([], {})
        for text in ("what is speed tool", "what is Xorg", "what is voice output"):
            with self.subTest(text=text):
                assessment = classify_capability_gap_request(text)
                self.assertEqual("knowledge", assessment.get("request_kind"))
                self.assertEqual("can_answer_locally", assessment.get("classification"))
                self.assertIsNone(detect_pack_capability_need(text))
                self.assertIsNone(
                    build_capability_gap_response(
                        text,
                        pack_store=store,
                        pack_registry_discovery=discovery,
                    )
                )

    def test_onboarding_capability_families_use_the_existing_recommendation_flow(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={
                ("local", "coding"): [
                    {
                        "remote_id": "dev-tools",
                        "name": "Local Dev Tools",
                        "summary": "Lightweight tools for coding and terminal work.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "source_url": "/tmp/dev-tools",
                    }
                ]
            },
        )
        result = recommend_packs_for_capability(None, pack_store=store, pack_registry_discovery=discovery, capability="dev_tools")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("dev_tools", result["capability_required"])
        self.assertIsNotNone(result["recommended_pack"])
        self.assertEqual("Local Dev Tools", result["recommended_pack"]["name"])

    def test_recommends_install_preview_for_clear_missing_capability(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={
                ("local", "voice"): [
                    {
                        "remote_id": "local-voice",
                        "name": "Local Voice",
                        "summary": "Local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "source_url": "/tmp/local-voice",
                    }
                ]
            },
        )
        result = recommend_packs_for_capability("Talk to me out loud", pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("voice_output", result["capability_required"])
        self.assertEqual("missing", result["status"])
        self.assertEqual("install_preview", result["fallback"])
        self.assertIsNotNone(result["recommended_pack"])
        self.assertEqual("Local Voice", result["recommended_pack"]["name"])
        self.assertEqual("discovered", result["lifecycle_state"])
        self.assertFalse(result["lifecycle"]["usable"])
        self.assertEqual("preview", result["lifecycle"]["missing_gate"])
        self.assertTrue(result["recommended_pack"]["installable"])
        self.assertEqual("lighter", result["recommended_pack"]["tradeoff_note"])
        self.assertEqual("If you want, say yes and I'll show the pack preview.", result["next_step"])

        rendered = render_pack_capability_response(result)
        self.assertIn("I don't have Voice output installed yet", rendered)
        self.assertIn("searched the approved pack sources", rendered)
        self.assertIn("safe text-only pack: Local Voice", rendered)
        self.assertIn("It is not installed yet. I can show you the preview first", rendered)
        self.assertIn("Say yes to preview it.", rendered)
        self.assertNotIn("lighter option", rendered)
        self.assertNotIn("fetch and inspect", rendered)
        self.assertEqual("single_recommendation", result["comparison_mode"])
        self.assertIsNone(result["alternate_pack"])

    def test_single_starter_candidate_renders_clear_preview_first_copy(self) -> None:
        rendered = render_pack_capability_response(
            {
                "capability_required": "qr_code_guidance",
                "capability_label": "QR code creation guidance",
                "classification": "can_partially_answer_but_capability_would_help",
                "fallback": "install_preview",
                "comparison_mode": "single_recommendation",
                "recommended_pack": {
                    "name": "QR Code Creation Guidance",
                    "source_id": "starter-safe-text",
                    "source_name": "Starter Safe Text Catalog",
                    "artifact_type_hint": "portable_text_skill",
                    "installable": True,
                    "normalized_state": {"installable": True},
                    "tradeoff_note": "lighter",
                },
            }
        )

        self.assertEqual(
            "I don't have QR-code generation installed yet, but I searched the approved starter catalog sources and found a safe text-only guidance pack: QR Code Creation Guidance. "
            "It is not installed yet. I can show you the preview first, including what it contains and any safety notes. Say yes to preview it.",
            rendered,
        )
        self.assertNotIn("helper", rendered.lower())
        self.assertNotIn("lighter option", rendered.lower())
        self.assertNotIn("fetch and inspect", rendered.lower())

    def test_capability_gap_response_includes_structured_rescue_contract(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={
                ("local", "voice"): [
                    {
                        "remote_id": "local-voice",
                        "name": "Local Voice",
                        "summary": "Local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "source_url": "/tmp/local-voice",
                    }
                ]
            },
        )
        result = build_capability_gap_response(
            "Talk to me out loud",
            pack_store=store,
            pack_registry_discovery=discovery,
        )
        self.assertIsNotNone(result)
        assert result is not None
        rescue = result.get("capability_gap_rescue")
        self.assertIsInstance(rescue, dict)
        assert isinstance(rescue, dict)
        self.assertEqual("capability_gap_rescue", rescue.get("type"))
        self.assertEqual("voice_output", rescue.get("missing_capability"))
        self.assertEqual("Talk to me out loud", rescue.get("user_goal"))
        self.assertEqual("approved_pack_sources_only", rescue.get("source_scope"))
        self.assertTrue(rescue.get("preview_required"))
        self.assertFalse(rescue.get("install_allowed_initially"))
        self.assertTrue(any("untrusted" in line for line in rescue.get("trust_warnings", [])))
        candidates = rescue.get("candidate_packs")
        self.assertIsInstance(candidates, list)
        assert isinstance(candidates, list)
        self.assertEqual(1, len(candidates))
        self.assertEqual("local", candidates[0].get("source_id"))
        self.assertEqual("local-voice", candidates[0].get("remote_id"))
        self.assertFalse(candidates[0].get("metadata_trusted"))
        self.assertTrue(candidates[0].get("preview_required"))
        actions = rescue.get("candidate_actions")
        self.assertIsInstance(actions, list)
        assert isinstance(actions, list)
        preview_actions = [row for row in actions if isinstance(row, dict) and row.get("action") == "show_preview"]
        self.assertEqual(1, len(preview_actions))
        self.assertEqual("local", preview_actions[0].get("source_id"))
        self.assertEqual("local-voice", preview_actions[0].get("remote_id"))
        self.assertFalse(preview_actions[0].get("install_allowed_initially"))

    def test_generic_unsupported_tool_request_creates_rescue_without_web_or_install(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery([], {})
        result = build_capability_gap_response(
            "Generate a QR code for https://example.com",
            pack_store=store,
            pack_registry_discovery=discovery,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("propose_new_capability", result.get("fallback"))
        rescue = result.get("capability_gap_rescue")
        self.assertIsInstance(rescue, dict)
        assert isinstance(rescue, dict)
        self.assertEqual("approved_pack_sources_only", rescue.get("source_scope"))
        self.assertTrue(rescue.get("preview_required"))
        self.assertFalse(rescue.get("install_allowed_initially"))
        self.assertIn("qr", str(rescue.get("search_query") or "").lower())
        actions = rescue.get("candidate_actions")
        self.assertIsInstance(actions, list)
        assert isinstance(actions, list)
        self.assertTrue(any(isinstance(row, dict) and row.get("action") == "sketch_helper" for row in actions))

    def test_youtube_history_gap_returns_scaffold_preview_offer_not_browser_automation(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[
                {"id": "starter-safe-text", "name": "Starter Safe Text Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={},
        )

        result = build_capability_gap_response(
            "Look through my YouTube history and find the video about neurons differentiating during animal infancy.",
            pack_store=store,
            pack_registry_discovery=discovery,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("scaffold_preview", result.get("fallback"))
        self.assertEqual("missing", result.get("lifecycle_state"))
        lifecycle = result.get("lifecycle")
        self.assertIsInstance(lifecycle, dict)
        assert isinstance(lifecycle, dict)
        self.assertEqual("scaffold_preview", (lifecycle.get("next_step") or {}).get("action"))
        self.assertEqual("youtube_history_search", result.get("capability_required"))
        self.assertIn("cannot read or search your history today", str(result.get("summary") or "").lower())
        self.assertIn("say yes to preview the scaffold", str(result.get("summary") or "").lower())
        self.assertIn("should not treat browser automation planning as the solution", str(result.get("summary") or "").lower())
        recommendation = result.get("recommendation")
        self.assertIsInstance(recommendation, dict)
        assert isinstance(recommendation, dict)
        preview = recommendation.get("scaffold_preview")
        self.assertIsInstance(preview, dict)
        assert isinstance(preview, dict)
        self.assertFalse(preview.get("creates_files"))
        self.assertFalse(preview.get("executes_code"))
        blocked = " ".join(str(item) for item in preview.get("blocked_actions", []))
        self.assertIn("No OAuth", blocked)
        self.assertIn("No browser history scraping", blocked)
        self.assertIn("No transcript fetching", blocked)
        rescue = result.get("capability_gap_rescue")
        self.assertIsInstance(rescue, dict)
        assert isinstance(rescue, dict)
        actions = rescue.get("candidate_actions")
        self.assertIsInstance(actions, list)
        assert isinstance(actions, list)
        self.assertTrue(any(isinstance(row, dict) and row.get("action") == "preview_scaffold" for row in actions))
        self.assertFalse(any(isinstance(row, dict) and row.get("action") == "show_preview" for row in actions))

    def test_real_world_starter_prompts_find_safe_text_candidates(self) -> None:
        store = _FakePackStore([])
        listings = {
            "qr": {
                "remote_id": "qr-code-guidance",
                "name": "QR Code Creation Guidance",
                "summary": "Plan QR code content, safety checks, export format, and validation.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "source_url": "memory/external_packs/starter_catalog/packs/qr-code-guidance",
                "tags": ["qr", "qrcode", "barcode", "guidance"],
            },
            "pdf": {
                "remote_id": "document-conversion-guidance",
                "name": "PDF and Document Conversion Guidance",
                "summary": "Plan safe document conversion and verification.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "source_url": "memory/external_packs/starter_catalog/packs/document-conversion-guidance",
                "tags": ["pdf", "document", "conversion", "file"],
            },
            "browser": {
                "remote_id": "browser-automation-planning",
                "name": "Browser Automation Planning Guidance",
                "summary": "Design browser automation plans without controlling a browser.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "source_url": "memory/external_packs/starter_catalog/packs/browser-automation-planning",
                "tags": ["browser", "automation", "planning", "no_execution"],
            },
            "file": {
                "remote_id": "file-organization-workflow",
                "name": "File Organization Workflow",
                "summary": "Plan file organization and dry-run review before mutation.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "source_url": "memory/external_packs/starter_catalog/packs/file-organization-workflow",
                "tags": ["file", "files", "organization", "organizing"],
            },
            "linux": {
                "remote_id": "linux-troubleshooting-workflow",
                "name": "Linux Troubleshooting Workflow",
                "summary": "Structure Linux diagnostics and rollback-safe repair plans.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "source_url": "memory/external_packs/starter_catalog/packs/linux-troubleshooting-workflow",
                "tags": ["linux", "troubleshooting", "debugging", "system"],
            },
        }
        discovery = _FakeDiscovery(
            sources=[
                {"id": "starter-safe-text", "name": "Starter Safe Text Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={
                ("starter-safe-text", key): [value]
                for key, value in listings.items()
            },
        )

        cases = (
            ("can you make me a qr code", "qr-code-guidance"),
            ("make this into a pdf", "document-conversion-guidance"),
            ("can you automate my browser", "browser-automation-planning"),
            ("find me a skill for organizing files", "file-organization-workflow"),
            ("get a skill pack for debugging linux", "linux-troubleshooting-workflow"),
        )
        for prompt, expected_remote_id in cases:
            with self.subTest(prompt=prompt):
                result = build_capability_gap_response(
                    prompt,
                    pack_store=store,
                    pack_registry_discovery=discovery,
                )
                self.assertIsNotNone(result)
                assert result is not None
                rescue = result.get("capability_gap_rescue")
                self.assertIsInstance(rescue, dict)
                assert isinstance(rescue, dict)
                self.assertEqual("install_preview", result.get("fallback"))
                self.assertTrue(rescue.get("preview_required"))
                self.assertFalse(rescue.get("install_allowed_initially"))
                candidates = rescue.get("candidate_packs")
                self.assertIsInstance(candidates, list)
                assert isinstance(candidates, list)
                self.assertEqual(expected_remote_id, candidates[0].get("remote_id"))
                actions = rescue.get("candidate_actions")
                self.assertIsInstance(actions, list)
                assert isinstance(actions, list)
                show_preview = [row for row in actions if isinstance(row, dict) and row.get("action") == "show_preview"]
                self.assertEqual(1, len(show_preview))
                self.assertFalse(show_preview[0].get("install_allowed_initially"))

    def test_knowledge_question_still_does_not_create_rescue(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery([], {})
        self.assertIsNone(
            build_capability_gap_response(
                "What is a QR code?",
                pack_store=store,
                pack_registry_discovery=discovery,
            )
        )

    def test_partial_capability_render_includes_intro_once(self) -> None:
        rendered = render_pack_capability_response(
            {
                "capability_required": "dev_tools",
                "capability_label": "coding tools",
                "classification": "can_partially_answer_but_capability_would_help",
                "fallback": "propose_new_capability",
                "helper_name": "coding helper",
                "proposal_summary": "helps with coding and terminal work",
            }
        )
        self.assertIn("I can help in text", rendered)
        self.assertEqual(1, rendered.lower().count("coding helper"))
        self.assertIn("simplest way to add it", rendered.lower())
        self.assertIn("sketch that with you", rendered.lower())

    def test_direct_helper_proposal_uses_natural_wording(self) -> None:
        rendered = render_pack_capability_response(
            {
                "capability_required": "custom_helper",
                "capability_label": "studio cue helper",
                "classification": "cannot_answer_without_new_capability",
                "fallback": "propose_new_capability",
                "helper_name": "studio cue helper",
                "proposal_summary": "coordinates studio light cues with music cues during live shows",
            }
        )
        self.assertIn("couldn't find a ready-made studio cue helper for this", rendered.lower())
        self.assertIn("simplest way to add it", rendered.lower())
        self.assertIn("sketch that with you", rendered.lower())

    def test_recommends_one_primary_plus_one_grounded_alternate(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={
                ("local", "voice"): [
                    {
                        "remote_id": "local-voice",
                        "name": "Local Voice",
                        "summary": "Lightweight local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "source_url": "/tmp/local-voice",
                        "tags": ["voice_output", "lightweight"],
                    },
                    {
                        "remote_id": "studio-voice",
                        "name": "Studio Voice",
                        "summary": "Full speech output with broader phrasing support.",
                        "artifact_type_hint": "experience_pack",
                        "installable_by_current_policy": True,
                        "source_url": "/tmp/studio-voice",
                        "tags": ["voice_output", "studio", "complete"],
                    },
                ]
            },
        )
        result = recommend_packs_for_capability("Talk to me out loud", pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("recommended_plus_alternate", result["comparison_mode"])
        self.assertIsNotNone(result["recommended_pack"])
        self.assertIsNotNone(result["alternate_pack"])
        self.assertEqual("lighter", result["recommended_pack"]["tradeoff_note"])
        self.assertEqual("may need more resources", result["alternate_pack"]["tradeoff_note"])

        rendered = render_pack_capability_response(result)
        self.assertIn("I found 2 packs that fit this machine.", rendered)
        self.assertIn("Local Voice looks like the lighter option.", rendered)
        self.assertIn("Studio Voice may need more resources.", rendered)
        self.assertIn("I'd start with Local Voice.", rendered)
        self.assertIn("If you want, say yes and I'll show the preview for Local Voice.", rendered)

    def test_blocked_discovery_source_is_not_searched(self) -> None:
        class _BlockedDiscovery:
            def __init__(self) -> None:
                self.search_calls: list[tuple[str, str]] = []

            def list_sources(self) -> list[dict[str, object]]:
                return [
                    {
                        "id": "blocked",
                        "name": "Blocked Catalog",
                        "enabled": True,
                        "allowed_by_policy": False,
                        "queryable": False,
                        "blocked_reason": "denied_by_policy",
                    }
                ]

            def search(self, source_id: str, query: str) -> dict[str, object]:
                self.search_calls.append((source_id, query))
                return {"source": {}, "search": {"results": []}, "from_cache": False, "stale": False}

        store = _FakePackStore([])
        discovery = _BlockedDiscovery()
        result = recommend_packs_for_capability(
            "Talk to me out loud",
            pack_store=store,
            pack_registry_discovery=discovery,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([], discovery.search_calls)
        self.assertEqual("propose_new_capability", result["fallback"])
        self.assertEqual("denied_by_policy", result["source_errors"][0]["error"])

    def test_weak_or_blocked_alternate_is_not_forced_into_comparison(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[
                {"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True},
            ],
            search_map={
                ("local", "voice"): [
                    {
                        "remote_id": "local-voice",
                        "name": "Local Voice",
                        "summary": "Local speech output for this machine.",
                        "artifact_type_hint": "portable_text_skill",
                        "installable_by_current_policy": True,
                        "source_url": "/tmp/local-voice",
                        "tags": ["voice_output"],
                    },
                    {
                        "remote_id": "blocked-studio",
                        "name": "Studio Voice",
                        "summary": "Studio speech output requiring more resources.",
                        "artifact_type_hint": "native_code_pack",
                        "installable_by_current_policy": False,
                        "policy_hint": "missing GPU acceleration",
                        "source_url": "/tmp/studio-voice",
                        "tags": ["voice_output", "studio"],
                    },
                ]
            },
        )
        result = recommend_packs_for_capability("Talk to me out loud", pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("single_recommendation", result["comparison_mode"])
        self.assertIsNone(result["alternate_pack"])

        rendered = render_pack_capability_response(result)
        self.assertNotIn("2 packs that fit this machine", rendered)
        self.assertNotIn("Studio Voice may need more resources.", rendered)

    def test_installed_but_disabled_pack_is_explained(self) -> None:
        store = _FakePackStore(
            [
                {
                    "pack_id": "pack.voice.local_fast",
                    "name": "Local Voice",
                    "status": "normalized",
                    "enabled": False,
                    "normalized_path": str(Path(__file__).resolve()),
                    "canonical_pack": {
                        "display_name": "Local Voice",
                        "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                        "source": {"name": "Local", "kind": "local_catalog"},
                        "capabilities": {
                            "summary": "Local speech output for this machine.",
                            "declared": ["voice_output"],
                        },
                    },
                    "review_envelope": {"pack_name": "Local Voice"},
                }
            ]
        )
        discovery = _FakeDiscovery(sources=[{"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True}], search_map={})
        result = recommend_packs_for_capability("Talk to me out loud", pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("installed_disabled", result["status"])
        self.assertIsNotNone(result["installed_pack"])
        self.assertIsNone(result["recommended_pack"])
        self.assertEqual("installed_only", result["comparison_mode"])

        rendered = render_pack_capability_response(result)
        self.assertIn("Voice output is installed, but it is disabled.", rendered)
        self.assertIn("not enabled as a live capability", rendered)
        self.assertIn("Enable it before using it.", rendered)
        self.assertIn("text", rendered.lower())

    def test_installed_and_healthy_pack_is_task_unconfirmed(self) -> None:
        store = _FakePackStore(
            [
                {
                    "pack_id": "pack.voice.local_fast",
                    "name": "Local Voice",
                    "status": "normalized",
                    "enabled": True,
                    "normalized_path": str(Path(__file__).resolve()),
                    "canonical_pack": {
                        "display_name": "Local Voice",
                        "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                        "source": {"name": "Local", "kind": "local_catalog"},
                        "capabilities": {
                            "summary": "Local speech output for this machine.",
                            "declared": ["voice_output"],
                        },
                    },
                    "review_envelope": {"pack_name": "Local Voice"},
                }
            ]
        )
        discovery = _FakeDiscovery(sources=[{"id": "local", "name": "Local Catalog", "kind": "local_catalog", "enabled": True}], search_map={})
        result = recommend_packs_for_capability("Talk to me out loud", pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("installed_healthy", result["status"])
        self.assertIsNotNone(result["installed_pack"])
        self.assertEqual("installed_only", result["comparison_mode"])
        rendered = render_pack_capability_response(result)
        self.assertIn("Voice output is installed and healthy, but I can't confirm it's usable for this task yet.", rendered)
        self.assertIn("task compatibility not confirmed", rendered)
        self.assertIn("Open the pack preview before relying on it.", rendered)

    def test_blocked_pack_remains_unusable_and_truthful(self) -> None:
        store = _FakePackStore([])
        discovery = _FakeDiscovery(
            sources=[{"id": "remote", "name": "Remote Catalog", "kind": "github_repo", "enabled": True}],
            search_map={
                ("remote", "camera"): [
                    {
                        "remote_id": "camera-pack",
                        "name": "Robot Camera",
                        "summary": "Robot camera feed integration.",
                        "artifact_type_hint": "native_code_pack",
                        "installable_by_current_policy": False,
                        "policy_hint": "missing GPU acceleration",
                        "source_url": "https://example.invalid/robot-camera",
                    }
                ]
            },
        )
        result = recommend_packs_for_capability("Open the robot camera feed", pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual("blocked", result["status"])
        self.assertIsNotNone(result["blocked_pack"])
        self.assertEqual("blocked_only", result["comparison_mode"])
        self.assertIsNone(result["alternate_pack"])

        rendered = render_pack_capability_response(result)
        self.assertIn("Camera feed isn't installed.", rendered)
        self.assertIn("isn't usable on this machine yet", rendered)
        self.assertIn("text", rendered.lower())


if __name__ == "__main__":
    unittest.main()
