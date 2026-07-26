from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from scripts import pack_route_smoke, webui_smoke


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV = (
    "AGENT_DB_PATH",
    "AGENT_LOG_PATH",
    "AGENT_WEBUI_DIST_PATH",
    "LLM_REGISTRY_PATH",
    "AGENT_SECRET_STORE_PATH",
    "LLM_USAGE_STATS_PATH",
    "AGENT_PERMISSIONS_PATH",
    "AGENT_AUDIT_LOG_PATH",
)


class TestEndUserClosureContract(unittest.TestCase):
    def test_service_and_bundle_paths_share_the_canonical_state_contract(self) -> None:
        service_paths = (
            ROOT / "systemd" / "personal-agent-api.service",
            ROOT / "systemd" / "personal-agent-api-dev.service",
            ROOT / "packaging" / "debian" / "personal-agent-api.service.in",
        )
        for service_path in service_paths:
            text = service_path.read_text(encoding="utf-8")
            with self.subTest(service=service_path.name):
                for variable in REQUIRED_ENV:
                    self.assertIn(f"Environment={variable}=", text)
                self.assertIn("AGENT_PERMISSIONS_PATH=%h/.config/personal-agent/permissions.json", text)
                self.assertNotIn("AGENT_PERMISSIONS_PATH=%h/.local/share/personal-agent", text)

        stable = service_paths[0].read_text(encoding="utf-8")
        self.assertIn("WorkingDirectory=%h/.local/share/personal-agent/runtime/current", stable)
        self.assertIn(
            "AGENT_WEBUI_DIST_PATH=%h/.local/share/personal-agent/runtime/current/agent/webui/dist",
            stable,
        )
        self.assertIn("PERSONAL_AGENT_INSTANCE=stable", stable)

        service_helper = (ROOT / "scripts" / "install_user_service.sh").read_text(encoding="utf-8")
        self.assertIn("Stable runtime is not installed. Run: bash scripts/install_local.sh", service_helper)

        bundle = (ROOT / "packaging" / "release_bundle" / "install.sh").read_text(encoding="utf-8")
        for variable in REQUIRED_ENV:
            self.assertIn(f"Environment={variable}=", bundle)
        self.assertIn('config_root="${XDG_CONFIG_HOME:-$HOME/.config}/personal-agent"', bundle)
        self.assertIn("Environment=AGENT_PERMISSIONS_PATH=$config_root/permissions.json", bundle)
        self.assertIn('rm -f "$service_path"', bundle)

    def test_installed_runtime_packages_internal_writer_registry(self) -> None:
        bundle_builder = (ROOT / "scripts" / "build_release_bundle.sh").read_text(encoding="utf-8")
        deb_builder = (ROOT / "scripts" / "build_deb.sh").read_text(encoding="utf-8")
        marker = "docs/operator/INTERNAL_WRITER_REGISTRY_V1.json"
        self.assertIn(marker, bundle_builder)
        self.assertIn(marker, deb_builder)

    def test_canonical_installer_uses_reproducible_web_build_and_lockfile_is_complete(self) -> None:
        installer = (ROOT / "scripts" / "install_local.sh").read_text(encoding="utf-8")
        builder = (ROOT / "scripts" / "build_webui.sh").read_text(encoding="utf-8")
        lockfile = (ROOT / "desktop" / "package-lock.json").read_text(encoding="utf-8")
        self.assertIn('bash "$repo_root/scripts/build_webui.sh"', installer)
        self.assertIn("npm ci", builder)
        self.assertIn("webui_build_manifest.py", builder)
        self.assertIn('"node_modules/@esbuild/linux-x64"', lockfile)
        self.assertIn('"node_modules/@esbuild/win32-x64"', lockfile)

    def test_webui_live_smoke_requires_chat_first_asset_markers(self) -> None:
        good = "chat-product-shell Ask for help naturally. What can I help you with? Advanced"
        self.assertEqual([], webui_smoke._chat_first_warnings(good))
        warnings = webui_smoke._chat_first_warnings("AdminPanel DebugTab")
        self.assertTrue(any("chat-first marker" in warning for warning in warnings))

    def test_pack_smoke_confirms_an_exact_plan_before_apply(self) -> None:
        plan = {"plan_id": "plan-1"}
        preview = {
            "ok": True,
            "status": 200,
            "payload": {"ok": True, "requires_confirmation": True, "mutated": False, "plan": plan},
        }
        applied = {"ok": True, "status": 200, "payload": {"ok": True, "mutated": True}}
        with mock.patch.object(pack_route_smoke, "_request_json", side_effect=(preview, applied)) as request_json, mock.patch.object(
            pack_route_smoke,
            "build_mutation_confirmation",
            return_value={"confirmation_id": "explicit", "plan_id": "plan-1"},
        ):
            observed_preview, observed_apply = pack_route_smoke._confirmed_mutation(
                "http://127.0.0.1:8765",
                "POST",
                "/pack_sources/catalog",
                {"source_id": "fixture"},
                timeout=8.0,
            )
        self.assertIs(observed_preview, preview)
        self.assertIs(observed_apply, applied)
        self.assertEqual(2, request_json.call_count)
        apply_payload = request_json.call_args_list[1].args[3]
        self.assertEqual(plan, apply_payload["mutation_plan"])
        self.assertEqual("explicit", apply_payload["confirmation"]["confirmation_id"])

    def test_closure_fixture_is_safe_and_stable(self) -> None:
        fixture = ROOT / "tests" / "fixtures" / "end_user_closure" / "closure-note.txt"
        text = fixture.read_text(encoding="utf-8")
        self.assertIn("PERSONAL_AGENT_CLOSURE_FILE_OK", text)
        self.assertNotIn("token", text.lower())
        self.assertNotIn("password", text.lower())


if __name__ == "__main__":
    unittest.main()
