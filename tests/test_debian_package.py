from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: Path, *, env: dict[str, str], args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *(args or [])],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class TestDebianPackage(unittest.TestCase):
    def _build_package(self, *, repo_root: Path | None = None) -> tuple[Path, Path, Path]:
        repo_root = repo_root or REPO_ROOT
        outdir = Path(tempfile.mkdtemp(prefix="personal-agent-deb-"))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(outdir)], check=False))
        proc = _run_script(repo_root / "scripts" / "build_deb.sh", env=os.environ.copy(), args=["--outdir", str(outdir), "--clean"])
        self.assertEqual(0, proc.returncode, proc.stderr)
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith("dpkg-deb:")]
        self.assertGreaterEqual(len(lines), 3)
        stage_root = Path(lines[0])
        deb_path = Path(lines[1])
        checksum_path = Path(lines[2])
        self.assertTrue(stage_root.is_dir())
        self.assertTrue(deb_path.is_file())
        self.assertTrue(checksum_path.is_file())
        return stage_root, deb_path, checksum_path

    def test_build_deb_creates_expected_artifacts_and_metadata(self) -> None:
        stage_root, deb_path, checksum_path = self._build_package()
        version = Path(REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()

        control = subprocess.run(
            ["dpkg-deb", "-I", str(deb_path)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, control.returncode, control.stderr)
        self.assertIn("Package: personal-agent", control.stdout)
        self.assertIn(f"Version: {version}", control.stdout)
        self.assertIn("Architecture: amd64", control.stdout)
        self.assertIn("python3-openai", control.stdout)
        self.assertIn("python3-python-telegram-bot", control.stdout)
        self.assertIn("python3-keyring", control.stdout)

        listing = subprocess.run(
            ["dpkg-deb", "-c", str(deb_path)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, listing.returncode, listing.stderr)
        self.assertIn("/usr/lib/personal-agent/runtime/releases/", listing.stdout)
        self.assertIn("/usr/bin/personal-agent-webui", listing.stdout)
        self.assertIn("/usr/bin/personal-agent-uninstall", listing.stdout)
        self.assertIn("/usr/lib/systemd/user/personal-agent-api.service", listing.stdout)
        self.assertIn("/usr/share/applications/personal-agent.desktop", listing.stdout)
        self.assertIn("/usr/share/icons/hicolor/scalable/apps/personal-agent.svg", listing.stdout)

        self.assertTrue((stage_root / "usr" / "share" / "doc" / "personal-agent" / "README.Debian").is_file())
        self.assertTrue(checksum_path.read_text(encoding="utf-8").strip().endswith(deb_path.name))

    def test_deb_extracts_with_installed_paths_and_no_repo_dependency(self) -> None:
        _stage_root, deb_path, _checksum_path = self._build_package()
        version = Path(REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["dpkg-deb", "-x", str(deb_path), str(extract_root)],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, proc.returncode, proc.stderr)

            runtime_root = extract_root / "usr" / "lib" / "personal-agent" / "runtime"
            current_root = runtime_root / "current"
            release_root = runtime_root / "releases" / version
            launcher = extract_root / "usr" / "bin" / "personal-agent-webui"
            uninstaller = extract_root / "usr" / "bin" / "personal-agent-uninstall"
            service = extract_root / "usr" / "lib" / "systemd" / "user" / "personal-agent-api.service"
            desktop = extract_root / "usr" / "share" / "applications" / "personal-agent.desktop"

            self.assertTrue(release_root.is_dir())
            self.assertTrue(current_root.is_symlink())
            self.assertEqual(f"/usr/lib/personal-agent/runtime/releases/{version}", os.readlink(current_root))
            self.assertTrue(launcher.is_symlink())
            self.assertEqual("/usr/lib/personal-agent/runtime/current/bin/personal-agent-webui", os.readlink(launcher))
            self.assertTrue(uninstaller.is_symlink())
            self.assertEqual("/usr/lib/personal-agent/runtime/current/bin/personal-agent-uninstall", os.readlink(uninstaller))
            self.assertTrue(service.is_symlink())
            self.assertEqual("/usr/lib/personal-agent/runtime/current/systemd/personal-agent-api.service", os.readlink(service))
            self.assertTrue(desktop.is_file())

            desktop_text = desktop.read_text(encoding="utf-8")
            self.assertIn("/usr/bin/personal-agent-webui", desktop_text)
            self.assertNotIn(str(REPO_ROOT), desktop_text)

            service_text = (release_root / "systemd" / "personal-agent-api.service").read_text(encoding="utf-8")
            self.assertIn("/usr/lib/personal-agent/runtime/current", service_text)
            self.assertIn("%h/.local/share/personal-agent", service_text)
            self.assertNotIn(str(REPO_ROOT), service_text)

            manifest = release_root / "manifest.json"
            self.assertTrue(manifest.is_file())
            self.assertIn("/usr/lib/personal-agent/runtime", manifest.read_text(encoding="utf-8"))

    def test_package_uninstaller_cleans_user_state_and_service_registration(self) -> None:
        _stage_root, deb_path, _checksum_path = self._build_package()
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_root = Path(tmpdir) / "extract"
            extract_root.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["dpkg-deb", "-x", str(deb_path), str(extract_root)],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, proc.returncode, proc.stderr)

            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            state_root = home / ".local" / "share" / "personal-agent"
            state_root.mkdir(parents=True, exist_ok=True)
            (state_root / "marker.txt").write_text("keep", encoding="utf-8")
            service_dir = home / ".config" / "systemd" / "user"
            service_dir.mkdir(parents=True, exist_ok=True)
            service_path = service_dir / "personal-agent-api.service"
            service_path.write_text("[Unit]\nDescription=Personal Agent API\n", encoding="utf-8")

            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            systemctl_log = Path(tmpdir) / "systemctl.log"
            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "systemctl").chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "SYSTEMCTL_LOG": str(systemctl_log),
                }
            )

            uninstall = extract_root / "usr" / "lib" / "personal-agent" / "runtime" / "releases" / Path(REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip() / "bin" / "personal-agent-uninstall"
            first = subprocess.run(
                ["bash", str(uninstall)],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, first.returncode, first.stderr)
            self.assertTrue(state_root.exists())
            self.assertFalse(service_path.exists())
            self.assertIn("disable personal-agent-api.service", systemctl_log.read_text(encoding="utf-8"))

            second = subprocess.run(
                ["bash", str(uninstall), "--remove-state"],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, second.returncode, second.stderr)
            self.assertFalse(state_root.exists())
            self.assertFalse((home / ".config" / "personal-agent").exists())

    def test_build_deb_rejects_missing_runtime_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / "agent").mkdir(parents=True, exist_ok=True)
            (repo_root / "memory").mkdir(parents=True, exist_ok=True)
            (repo_root / "skills").mkdir(parents=True, exist_ok=True)
            (repo_root / "telegram_adapter").mkdir(parents=True, exist_ok=True)
            (repo_root / "assets" / "icons").mkdir(parents=True, exist_ok=True)
            (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
            (repo_root / "packaging" / "debian").mkdir(parents=True, exist_ok=True)
            (repo_root / "agent" / "webui" / "dist").mkdir(parents=True, exist_ok=True)
            (repo_root / "VERSION").write_text("0.2.0\n", encoding="utf-8")
            (repo_root / "assets" / "icons" / "personal-agent.svg").write_text("<svg />\n", encoding="utf-8")
            (repo_root / "scripts" / "launch_webui.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (repo_root / "packaging" / "debian" / "personal-agent-uninstall.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (repo_root / "packaging" / "debian" / "personal-agent-api.service.in").write_text("[Unit]\nDescription=x\n", encoding="utf-8")
            (repo_root / "packaging" / "personal-agent.desktop").write_text("[Desktop Entry]\nName=x\nExec=__PERSONAL_AGENT_LAUNCHER__\nIcon=personal-agent\nType=Application\n", encoding="utf-8")
            (repo_root / "llm_registry.json").write_text("{}\n", encoding="utf-8")

            proc = _run_script(
                REPO_ROOT / "scripts" / "build_deb.sh",
                env=os.environ.copy(),
                args=["--repo-root", str(repo_root), "--outdir", str(Path(tmpdir) / "dist"), "--clean"],
            )
            self.assertNotEqual(0, proc.returncode)
            self.assertIn("required packaging input missing", proc.stderr)


if __name__ == "__main__":
    unittest.main()
