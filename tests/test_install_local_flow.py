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


def _write_fake_python(bin_dir: Path, *, version: str, logs_dir: Path) -> None:
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
        "      *)\n"
        "        exit 0\n"
        "        ;;\n"
        "    esac\n"
        "    ;;\n"
        "  -)\n"
        "    template_path=\"${2-}\"\n"
        "    desktop_path=\"${3-}\"\n"
        "    launcher_path=\"${4-}\"\n"
        "    desktop_name=\"${5-}\"\n"
        "    desktop_comment=\"${6-}\"\n"
        "    perl -0pe \"s#__PERSONAL_AGENT_LAUNCHER__#${launcher_path}#g; s#__PERSONAL_AGENT_NAME__#${desktop_name}#g; s#__PERSONAL_AGENT_COMMENT__#${desktop_comment}#g\" \"$template_path\" > \"$desktop_path\"\n"
        "    exit 0\n"
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


class TestInstallLocalFlow(unittest.TestCase):
    def test_install_local_is_idempotent_and_installs_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            _write_fake_python(bin_dir, version="3.11.8", logs_dir=logs_dir)

            systemctl_log = logs_dir / "systemctl.log"
            xdg_open_log = logs_dir / "xdg-open.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$XDG_OPEN_LOG\"\n",
                encoding="utf-8",
            )
            (bin_dir / "node").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "npm").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            for item in ("systemctl", "xdg-open", "node", "npm", "python3"):
                (bin_dir / item).chmod(0o755)

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
            script = REPO_ROOT / "scripts" / "install_local.sh"

            first = _run_script(script, env=env, args=["--desktop-launcher"])
            self.assertEqual(0, first.returncode, first.stderr)
            second = _run_script(script, env=env, args=["--desktop-launcher"])
            self.assertEqual(0, second.returncode, second.stderr)

            service_unit = home / ".config" / "systemd" / "user" / "personal-agent-api-dev.service"
            launcher_path = home / ".local" / "share" / "personal-agent" / "bin" / "personal-agent-webui-dev"
            desktop_path = home / ".local" / "share" / "applications" / "personal-agent-dev.desktop"
            icon_path = home / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps" / "personal-agent.svg"
            shell_alias = home / ".local" / "bin" / "personal-agent-webui-dev"

            self.assertTrue(service_unit.is_symlink())
            self.assertEqual(REPO_ROOT / "systemd" / "personal-agent-api-dev.service", service_unit.resolve())
            self.assertTrue(launcher_path.is_file())
            self.assertTrue(desktop_path.is_file())
            self.assertTrue(icon_path.is_file())
            self.assertTrue(shell_alias.is_symlink())
            self.assertEqual(launcher_path, shell_alias.resolve())

            rendered = desktop_path.read_text(encoding="utf-8")
            self.assertIn(str(launcher_path), rendered)
            self.assertIn("Personal Agent (Dev)", rendered)

            self.assertIn("daemon-reload", systemctl_log.read_text(encoding="utf-8"))
            self.assertIn("enable --now personal-agent-api-dev.service", systemctl_log.read_text(encoding="utf-8"))
            self.assertFalse(xdg_open_log.exists())

    def test_install_local_rejects_old_python_before_partial_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            _write_fake_python(bin_dir, version="3.10.9", logs_dir=logs_dir)
            (bin_dir / "systemctl").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "xdg-open").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            for item in ("systemctl", "xdg-open", "python3"):
                (bin_dir / item).chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "PYTHON_LOG": str(logs_dir / "python.log"),
                    "FAKE_PYTHON_VERSION": "3.10.9",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "install_local.sh", env=env)

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("Python 3.11 or newer is required", proc.stderr)
            self.assertFalse((home / ".config" / "systemd" / "user" / "personal-agent-api.service").exists())

    def test_install_local_requires_xdg_open_for_launcher_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            _write_fake_python(bin_dir, version="3.11.8", logs_dir=logs_dir)
            (bin_dir / "systemctl").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "node").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "npm").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            for item in ("systemctl", "node", "npm", "python3"):
                (bin_dir / item).chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "PYTHON_LOG": str(logs_dir / "python.log"),
                    "FAKE_PYTHON_VERSION": "3.11.8",
                    "AGENT_INSTALL_XDG_OPEN": "missing-xdg-open",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "install_local.sh", env=env, args=["--desktop-launcher"])

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("xdg-open is required for the desktop launcher", proc.stderr)
            self.assertFalse((home / ".local" / "share" / "applications" / "personal-agent.desktop").exists())

    def test_install_local_check_webui_build_requires_node_and_npm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            logs_dir = root / "logs"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)

            _write_fake_python(bin_dir, version="3.11.8", logs_dir=logs_dir)
            (bin_dir / "systemctl").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "xdg-open").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            (bin_dir / "npm").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            for item in ("systemctl", "xdg-open", "npm", "python3"):
                (bin_dir / item).chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "PYTHON_LOG": str(logs_dir / "python.log"),
                    "FAKE_PYTHON_VERSION": "3.11.8",
                    "AGENT_INSTALL_NODE": "missing-node",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "install_local.sh", env=env, args=["--check-webui-build"])

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("node is required to rebuild the web UI", proc.stderr)


if __name__ == "__main__":
    unittest.main()
