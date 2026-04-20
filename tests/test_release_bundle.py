from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
import shutil
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


def _write_fake_python(bin_dir: Path, *, logs_dir: Path, version: str = "3.11.8") -> None:
    python_path = bin_dir / "python3"
    python_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$PYTHON_LOG\"\n"
        "case \"${1-}\" in\n"
        "  -c)\n"
        "    case \"${FAKE_PYTHON_VERSION:-3.11.8}\" in\n"
        "      3.11*|3.12*|3.13*|4.*)\n"
        "        exit 0\n"
        "        ;;\n"
        "      *)\n"
        "        exit 1\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "  -m)\n"
        "    case \"${2-}\" in\n"
        "      venv)\n"
        "        target=\"${3-}\"\n"
        "        mkdir -p \"$target/bin\"\n"
        "        cat > \"$target/bin/python\" <<'PY'\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$VENV_PYTHON_LOG\"\n"
        "if [ \"${1-}\" = \"-m\" ] && [ \"${2-}\" = \"pip\" ]; then\n"
        "  printf '%s\\n' \"$*\" >> \"$VENV_PIP_LOG\"\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"${1-}\" = \"-m\" ] && [ \"${2-}\" = \"agent\" ]; then\n"
        "  printf '%s\\n' \"$*\" >> \"$VENV_AGENT_LOG\"\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
        "PY\n"
        "        chmod +x \"$target/bin/python\"\n"
        "        ln -sf \"$target/bin/python\" \"$target/bin/pip\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      pip)\n"
        "        printf '%s\\n' \"$*\" >> \"$VENV_PIP_LOG\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      agent)\n"
        "        printf '%s\\n' \"$*\" >> \"$VENV_AGENT_LOG\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      doctor)\n"
        "        printf '%s\\n' \"$*\" >> \"$VENV_AGENT_LOG\"\n"
        "        exit 0\n"
        "        ;;\n"
        "      *)\n"
        "        exit 0\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    (logs_dir / "python.log").write_text("", encoding="utf-8")
    (logs_dir / "venv-python.log").write_text("", encoding="utf-8")
    (logs_dir / "venv-pip.log").write_text("", encoding="utf-8")
    (logs_dir / "venv-agent.log").write_text("", encoding="utf-8")


class TestReleaseBundle(unittest.TestCase):
    def test_build_release_bundle_fails_when_webui_assets_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            (repo_root / "agent").mkdir(parents=True, exist_ok=True)
            (repo_root / "memory").mkdir(parents=True, exist_ok=True)
            (repo_root / "skills").mkdir(parents=True, exist_ok=True)
            (repo_root / "telegram_adapter").mkdir(parents=True, exist_ok=True)
            (repo_root / "assets" / "icons").mkdir(parents=True, exist_ok=True)
            (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
            (repo_root / "VERSION").write_text("1.2.3\n", encoding="utf-8")
            (repo_root / "pyproject.toml").write_text("[project]\nname='x'\nversion='1.2.3'\n", encoding="utf-8")
            (repo_root / "build_backend.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
            (repo_root / "README.md").write_text("# Personal Agent\n", encoding="utf-8")
            (repo_root / "personal_agent_bootstrap.py").write_text("pass\n", encoding="utf-8")
            (repo_root / "personal_agent_bootstrap.pth").write_text("import personal_agent_bootstrap\n", encoding="utf-8")
            (repo_root / "sitecustomize.py").write_text("pass\n", encoding="utf-8")
            (repo_root / "assets" / "icons" / "personal-agent.svg").write_text("<svg />\n", encoding="utf-8")
            (repo_root / "scripts" / "launch_webui.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

            proc = _run_script(
                REPO_ROOT / "scripts" / "build_release_bundle.sh",
                env=os.environ.copy(),
                args=["--repo-root", str(repo_root), "--outdir", str(Path(tmpdir) / "dist"), "--clean"],
            )
            self.assertNotEqual(0, proc.returncode)
            self.assertIn("agent/webui/dist/index.html", proc.stderr)

    def test_build_release_bundle_creates_archive_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "dist"
            proc = _run_script(
                REPO_ROOT / "scripts" / "build_release_bundle.sh",
                env=os.environ.copy(),
                args=["--outdir", str(outdir), "--clean"],
            )
            self.assertEqual(0, proc.returncode, proc.stderr)
            bundle_dir = Path(proc.stdout.strip().splitlines()[0])
            archive_path = Path(proc.stdout.strip().splitlines()[1])
            checksum_path = Path(proc.stdout.strip().splitlines()[2])
            self.assertTrue(bundle_dir.is_dir())
            self.assertTrue((bundle_dir / "install.sh").is_file())
            self.assertTrue((bundle_dir / "uninstall.sh").is_file())
            self.assertTrue((bundle_dir / "payload" / "agent" / "api_server.py").is_file())
            self.assertTrue((bundle_dir / "payload" / "agent" / "webui" / "dist" / "index.html").is_file())
            self.assertTrue((bundle_dir / "payload" / "personal_agent_bootstrap.py").is_file())
            self.assertTrue((bundle_dir / "payload" / "personal_agent_bootstrap.pth").is_file())
            self.assertTrue((bundle_dir / "manifest.json").is_file())
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["bundle_version"], (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip())
            self.assertTrue(archive_path.is_file())
            self.assertTrue(checksum_path.is_file())

    def test_bundle_install_is_idempotent_and_uses_installed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_out = root / "bundle-out"
            install_root = root / "install-root"
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            _write_fake_python(bin_dir, logs_dir=logs_dir)
            systemctl_log = logs_dir / "systemctl.log"
            xdg_open_log = logs_dir / "xdg-open.log"
            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\nexit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$*\" >> \"$XDG_OPEN_LOG\"\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "xdg-open", "python3"):
                (bin_dir / item).chmod(0o755)

            build = _run_script(
                REPO_ROOT / "scripts" / "build_release_bundle.sh",
                env=os.environ.copy(),
                args=["--outdir", str(bundle_out), "--clean"],
            )
            self.assertEqual(0, build.returncode, build.stderr)
            bundle_dir = Path(build.stdout.strip().splitlines()[0])

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "PYTHON_LOG": str(logs_dir / "python.log"),
                    "VENV_PYTHON_LOG": str(logs_dir / "venv-python.log"),
                    "VENV_PIP_LOG": str(logs_dir / "venv-pip.log"),
                    "VENV_AGENT_LOG": str(logs_dir / "venv-agent.log"),
                    "SYSTEMCTL_LOG": str(systemctl_log),
                    "XDG_OPEN_LOG": str(xdg_open_log),
                    "FAKE_PYTHON_VERSION": "3.11.8",
                }
            )

            first = _run_script(bundle_dir / "install.sh", env=env, args=["--install-root", str(install_root)])
            self.assertEqual(0, first.returncode, first.stderr)
            second = _run_script(bundle_dir / "install.sh", env=env, args=["--install-root", str(install_root)])
            self.assertEqual(0, second.returncode, second.stderr)

            runtime_root = install_root / "runtime"
            current_root = runtime_root / "current"
            release_dir = next((runtime_root / "releases").iterdir())
            self.assertTrue((release_dir / ".venv" / "bin" / "python").exists())
            self.assertTrue(current_root.is_symlink())
            self.assertEqual(release_dir, current_root.resolve())

            launcher = install_root / "bin" / "personal-agent-webui"
            uninstall = install_root / "bin" / "personal-agent-uninstall"
            desktop = home / ".local" / "share" / "applications" / "personal-agent.desktop"
            icon = home / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps" / "personal-agent.svg"
            service = home / ".config" / "systemd" / "user" / "personal-agent-api.service"

            self.assertTrue(launcher.is_symlink())
            self.assertTrue(uninstall.is_symlink())
            self.assertTrue(desktop.is_file())
            self.assertTrue(icon.is_file())
            self.assertTrue(service.is_file())
            self.assertTrue((release_dir / "personal_agent_bootstrap.py").is_file())
            self.assertTrue((release_dir / "personal_agent_bootstrap.pth").is_file())
            self.assertIn(str(launcher), desktop.read_text(encoding="utf-8"))
            service_text = service.read_text(encoding="utf-8")
            self.assertIn(str(current_root), service_text)
            self.assertIn(str(current_root / ".venv" / "bin" / "python"), service_text)
            self.assertIn(str(current_root / "agent" / "webui" / "dist"), service_text)
            self.assertIn("PERSONAL_AGENT_RUNTIME_ROOT=", service_text)
            self.assertNotIn(str(REPO_ROOT), service_text)

            venv_pip_log = (logs_dir / "venv-pip.log").read_text(encoding="utf-8")
            self.assertIn("-m pip install ", venv_pip_log)
            self.assertNotIn("-e ", venv_pip_log)

            runtime_copy = release_dir
            self.assertTrue((runtime_copy / "bin" / "personal-agent-webui").exists())
            self.assertTrue((runtime_copy / "bin" / "personal-agent-uninstall").exists())
            self.assertTrue((runtime_copy / "manifest.json").exists() is False)

            shutil.rmtree(bundle_dir)
            self.assertFalse(bundle_dir.exists())
            self.assertTrue(launcher.exists())
            self.assertTrue(current_root.exists())

            self.assertIn("enable --now personal-agent-api.service", systemctl_log.read_text(encoding="utf-8"))
            self.assertFalse(xdg_open_log.exists())

    def test_bundle_uninstall_can_remove_state_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_out = root / "bundle-out"
            install_root = root / "install-root"
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            _write_fake_python(bin_dir, logs_dir=logs_dir)
            (bin_dir / "systemctl").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "xdg-open").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            for item in ("systemctl", "xdg-open", "python3"):
                (bin_dir / item).chmod(0o755)

            build = _run_script(
                REPO_ROOT / "scripts" / "build_release_bundle.sh",
                env=os.environ.copy(),
                args=["--outdir", str(bundle_out), "--clean"],
            )
            self.assertEqual(0, build.returncode, build.stderr)
            bundle_dir = Path(build.stdout.strip().splitlines()[0])
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "PYTHON_LOG": str(logs_dir / "python.log"),
                    "VENV_PYTHON_LOG": str(logs_dir / "venv-python.log"),
                    "VENV_PIP_LOG": str(logs_dir / "venv-pip.log"),
                    "VENV_AGENT_LOG": str(logs_dir / "venv-agent.log"),
                    "FAKE_PYTHON_VERSION": "3.11.8",
                }
            )

            first = _run_script(bundle_dir / "install.sh", env=env, args=["--install-root", str(install_root)])
            self.assertEqual(0, first.returncode, first.stderr)
            preserve = _run_script(
                install_root / "bin" / "personal-agent-uninstall",
                env=env,
                args=["--install-root", str(install_root)],
            )
            self.assertEqual(0, preserve.returncode, preserve.stderr)
            self.assertTrue(install_root.exists())
            self.assertFalse((install_root / "runtime").exists())
            self.assertFalse((home / ".local" / "share" / "applications" / "personal-agent.desktop").exists())
            self.assertFalse((home / ".config" / "systemd" / "user" / "personal-agent-api.service").exists())

            second = _run_script(bundle_dir / "install.sh", env=env, args=["--install-root", str(install_root)])
            self.assertEqual(0, second.returncode, second.stderr)
            uninstall = _run_script(
                install_root / "bin" / "personal-agent-uninstall",
                env=env,
                args=["--install-root", str(install_root), "--remove-state"],
            )
            self.assertEqual(0, uninstall.returncode, uninstall.stderr)
            self.assertFalse(install_root.exists())


if __name__ == "__main__":
    unittest.main()
