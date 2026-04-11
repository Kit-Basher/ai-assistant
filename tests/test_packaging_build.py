from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import shutil
import unittest
import zipfile
from pathlib import Path

import build_backend
from agent.version import read_build_info


class TestPackagingBuild(unittest.TestCase):
    def _build_and_install_wheel(self) -> tuple[Path, Path]:
        repo_root = Path(__file__).resolve().parents[1]
        build_dir = Path(tempfile.mkdtemp(prefix="personal-agent-wheel-"))
        venv_dir = Path(tempfile.mkdtemp(prefix="personal-agent-venv-"))
        self.addCleanup(shutil.rmtree, build_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, venv_dir, ignore_errors=True)
        proc = subprocess.run(
            [sys.executable, "scripts/build_dist.py", "--outdir", str(build_dir), "--clean"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        wheel_path = build_dir / f"personal_agent-{Path('VERSION').read_text(encoding='utf-8').strip()}-py3-none-any.whl"
        self.assertTrue(wheel_path.is_file())
        proc = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        proc = subprocess.run(
            [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--no-deps", str(wheel_path)],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        return build_dir, venv_dir

    def test_version_helper_uses_repo_version_file(self) -> None:
        info = read_build_info(repo_root=Path(__file__).resolve().parents[1])
        self.assertEqual(Path("VERSION").read_text(encoding="utf-8").strip(), info.version)
        self.assertEqual("repo_file", info.version_source)

    def test_build_backend_produces_clean_wheel_and_sdist(self) -> None:
        version = Path("VERSION").read_text(encoding="utf-8").strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_name = build_backend.build_wheel(tmpdir)
            sdist_name = build_backend.build_sdist(tmpdir)

            self.assertEqual(f"personal_agent-{version}-py3-none-any.whl", wheel_name)
            self.assertEqual(f"personal_agent-{version}.tar.gz", sdist_name)

            wheel_path = Path(tmpdir) / wheel_name
            sdist_path = Path(tmpdir) / sdist_name
            self.assertTrue(wheel_path.is_file())
            self.assertTrue(sdist_path.is_file())

            with zipfile.ZipFile(wheel_path) as zf:
                names = set(zf.namelist())
                self.assertIn("agent/api_server.py", names)
                self.assertIn("telegram_adapter/bot.py", names)
                self.assertIn("agent/bootstrap/routes.py", names)
                self.assertIn("memory/db.py", names)
                self.assertIn("skills/observe_now/handler.py", names)
                self.assertIn("agent/VERSION", names)
                self.assertIn(
                    f"personal_agent-{version}.data/data/share/personal-agent/systemd/personal-agent-api.service",
                    names,
                )
                metadata_text = zf.read(f"personal_agent-{version}.dist-info/METADATA").decode("utf-8")
                self.assertIn("Name: personal-agent", metadata_text)
                self.assertIn(f"Version: {version}", metadata_text)
                self.assertIn("Requires-Dist: openai>=1.0.0", metadata_text)
                self.assertIn("Requires-Dist: python-telegram-bot>=22.6", metadata_text)
                entry_points = zf.read(f"personal_agent-{version}.dist-info/entry_points.txt").decode("utf-8")
                self.assertIn("personal-agent = agent.cli:main", entry_points)
                self.assertIn("personal-agent-api = agent.api_server:main", entry_points)
                self.assertIn("personal-agent-telegram = telegram_adapter.bot:run", entry_points)
                self.assertNotIn("memory/agent.db", names)
                self.assertNotIn("memory/llm_usage_stats.json", names)
                self.assertNotIn("packaging/personal-agent@.service", names)
                self.assertFalse(any("__pycache__/" in name for name in names))

            with tarfile.open(sdist_path, "r:gz") as tf:
                names = set(tf.getnames())
                prefix = f"personal_agent-{version}"
                self.assertIn(f"{prefix}/pyproject.toml", names)
                self.assertIn(f"{prefix}/build_backend.py", names)
                self.assertIn(f"{prefix}/systemd/personal-agent-api.service", names)
                self.assertIn(f"{prefix}/tests/test_packaging_build.py", names)
                self.assertNotIn(f"{prefix}/packaging/personal-agent@.service", names)
                self.assertFalse(any("/__pycache__/" in name for name in names))

    def test_build_backend_supports_editable_console_script_install_path(self) -> None:
        version = Path("VERSION").read_text(encoding="utf-8").strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_name = build_backend.build_editable(tmpdir)
            wheel_path = Path(tmpdir) / wheel_name
            with zipfile.ZipFile(wheel_path) as zf:
                names = set(zf.namelist())
                self.assertIn(
                    f"personal_agent-{version}.data/purelib/personal_agent-editable.pth",
                    names,
                )
                entry_points = zf.read(f"personal_agent-{version}.dist-info/entry_points.txt").decode("utf-8")
                self.assertIn("personal-agent = agent.cli:main", entry_points)
                pth_text = zf.read(
                    f"personal_agent-{version}.data/purelib/personal_agent-editable.pth"
                ).decode("utf-8")
                self.assertIn(str(Path(build_backend.__file__).resolve().parent), pth_text)

    def test_build_script_emits_expected_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [sys.executable, "scripts/build_dist.py", "--outdir", tmpdir],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, proc.returncode, proc.stderr)
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            self.assertEqual(2, len(lines))
            self.assertTrue(lines[0].endswith(".whl"))
            self.assertTrue(lines[1].endswith(".tar.gz"))
            self.assertTrue(Path(lines[0]).is_file())
            self.assertTrue(Path(lines[1]).is_file())

    def test_fresh_wheel_install_entry_points_work_without_repo_path(self) -> None:
        _build_dir, venv_dir = self._build_and_install_wheel()
        personal_agent = venv_dir / "bin" / "personal-agent"
        personal_agent_api = venv_dir / "bin" / "personal-agent-api"
        personal_agent_telegram = venv_dir / "bin" / "personal-agent-telegram"

        version_proc = subprocess.run(
            [str(personal_agent), "version"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, version_proc.returncode, version_proc.stderr)
        self.assertIn("version=", version_proc.stdout)

        api_help_proc = subprocess.run(
            [str(personal_agent_api), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, api_help_proc.returncode, api_help_proc.stderr)
        self.assertIn("Run local Personal Agent HTTP API", api_help_proc.stdout)

        telegram_help_proc = subprocess.run(
            [str(personal_agent_telegram), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, telegram_help_proc.returncode, telegram_help_proc.stderr)
        self.assertIn("Run the Personal Agent Telegram adapter.", telegram_help_proc.stdout)
