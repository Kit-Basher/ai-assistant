from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSplitSmoke(unittest.TestCase):
    def test_launcher_summary_prefers_symlink_target_over_resolve(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "split_smoke.py", "split_smoke_script_summary")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_root = root / "runtime" / "current"
            release_root = root / "runtime" / "releases" / "0.2.0"
            stable_bin_root = root / "bin"
            current_launcher = current_root / "bin" / "personal-agent-webui"
            launcher = stable_bin_root / "personal-agent-webui"
            current_launcher.parent.mkdir(parents=True, exist_ok=True)
            current_launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            current_launcher.chmod(0o755)
            release_root.mkdir(parents=True, exist_ok=True)
            stable_bin_root.mkdir(parents=True, exist_ok=True)
            launcher.symlink_to(current_launcher)

            with patch.object(module, "STABLE_LAUNCHER", launcher):
                launcher_path, launcher_target = module._launcher_summary()

        self.assertEqual(str(launcher), launcher_path)
        self.assertEqual(str(current_launcher), launcher_target)

    def test_assert_stable_launcher_points_at_stable_runtime_accepts_current_symlink_chain(self) -> None:
        module = _load_module(REPO_ROOT / "scripts" / "split_smoke.py", "split_smoke_script_contract")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            current_root = root / "runtime" / "current"
            release_root = root / "runtime" / "releases" / "0.2.0"
            stable_bin_root = root / "bin"
            desktop_root = root / "applications"
            current_launcher = current_root / "bin" / "personal-agent-webui"
            launcher = stable_bin_root / "personal-agent-webui"
            desktop = desktop_root / "personal-agent.desktop"
            current_launcher.parent.mkdir(parents=True, exist_ok=True)
            current_launcher.write_text(
                'SERVICE_NAME="${AGENT_LAUNCHER_SERVICE_NAME:-personal-agent-api.service}"\n'
                'WEBUI_URL="${AGENT_WEBUI_URL:-http://127.0.0.1:8765/}"\n',
                encoding="utf-8",
            )
            current_launcher.chmod(0o755)
            release_root.mkdir(parents=True, exist_ok=True)
            stable_bin_root.mkdir(parents=True, exist_ok=True)
            desktop_root.mkdir(parents=True, exist_ok=True)
            launcher.symlink_to(current_launcher)
            desktop.write_text(
                "[Desktop Entry]\n"
                f"Exec={launcher}\n"
                f"TryExec={launcher}\n",
                encoding="utf-8",
            )

            with (
                patch.object(module, "STABLE_LAUNCHER", launcher),
                patch.object(module, "STABLE_DESKTOP", desktop),
            ):
                module._assert_stable_launcher_points_at_stable_runtime()


if __name__ == "__main__":
    unittest.main()
