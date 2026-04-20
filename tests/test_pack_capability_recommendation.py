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
            ("Use the avatar", "avatar_visual"),
            ("Open the robot camera feed", "camera_feed"),
            ("Help me code", "dev_tools"),
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
        self.assertTrue(result["recommended_pack"]["installable"])
        self.assertEqual("lighter", result["recommended_pack"]["tradeoff_note"])

        rendered = render_pack_capability_response(result)
        self.assertIn("Voice output isn't installed.", rendered)
        self.assertIn("best fit for this machine", rendered)
        self.assertIn("Local Voice looks lighter.", rendered)
        self.assertIn("Say yes and I'll show the install preview.", rendered)
        self.assertEqual("single_recommendation", result["comparison_mode"])
        self.assertIsNone(result["alternate_pack"])

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
        self.assertIn("Local Voice looks lighter.", rendered)
        self.assertIn("Studio Voice may need more resources.", rendered)
        self.assertIn("I'd start with Local Voice.", rendered)
        self.assertIn("Say yes and I'll show the install preview for Local Voice.", rendered)

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
