from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestInstallLocalFlow(unittest.TestCase):
    def test_recommended_local_install_targets_stable_runtime(self) -> None:
        text = (REPO_ROOT / "scripts" / "install_local.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/build_webui.sh", text)
        self.assertIn("scripts/build_release_bundle.sh", text)
        self.assertIn("personal-agent-api.service", text)
        self.assertIn("http://127.0.0.1:8765/", text)
        self.assertNotIn("personal-agent-api-dev.service", text)
        self.assertNotIn("http://127.0.0.1:18765/", text)

    def test_developer_install_is_explicit_and_isolated(self) -> None:
        text = (REPO_ROOT / "scripts" / "install_dev.sh").read_text(encoding="utf-8")
        self.assertIn("PERSONAL_AGENT_INSTANCE=dev", text)
        self.assertIn("personal-agent-api-dev.service", text)
        self.assertIn("http://127.0.0.1:18765/", text)
        self.assertNotIn("personal-agent-api.service\"", text)

    def test_install_help_names_stable_and_dev_contracts(self) -> None:
        proc = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "install_local.sh"), "--help"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("stable", proc.stdout.lower())
        self.assertIn("127.0.0.1:8765", proc.stdout)
        self.assertIn("scripts/install_dev.sh", proc.stdout)

    def test_webui_manifest_detects_source_and_output_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            desktop = root / "desktop"
            source = desktop / "src" / "App.jsx"
            dist = root / "agent" / "webui" / "dist"
            source.parent.mkdir(parents=True)
            dist.mkdir(parents=True)
            for name, content in (
                ("index.html", "<main />\n"),
                ("package.json", "{}\n"),
                ("package-lock.json", "{}\n"),
                ("vite.config.js", "export default {};\n"),
            ):
                (desktop / name).write_text(content, encoding="utf-8")
            source.write_text("export default 1;\n", encoding="utf-8")
            (dist / "index.html").write_text("built\n", encoding="utf-8")
            tool = REPO_ROOT / "scripts" / "webui_build_manifest.py"

            write = subprocess.run(
                ["python3", str(tool), "write", "--repo-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, write.returncode, write.stderr)
            verify = subprocess.run(
                ["python3", str(tool), "verify", "--repo-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, verify.returncode, verify.stderr)

            source.write_text("export default 2;\n", encoding="utf-8")
            stale_source = subprocess.run(
                ["python3", str(tool), "verify", "--repo-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, stale_source.returncode)
            self.assertIn("stale or modified", stale_source.stderr)

            source.write_text("export default 1;\n", encoding="utf-8")
            (dist / "index.html").write_text("tampered\n", encoding="utf-8")
            stale_output = subprocess.run(
                ["python3", str(tool), "verify", "--repo-root", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, stale_output.returncode)
            self.assertIn("stale or modified", stale_output.stderr)


if __name__ == "__main__":
    unittest.main()
