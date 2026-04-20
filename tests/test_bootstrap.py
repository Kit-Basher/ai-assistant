from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestBootstrap(unittest.TestCase):
    def test_stable_bootstrap_prefers_installed_packages_over_checkout(self) -> None:
        original_path = sys.path[:]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                repo_checkout = root / "personal-agent"
                (repo_checkout / "agent").mkdir(parents=True, exist_ok=True)
                (repo_checkout / ".git").mkdir(parents=True, exist_ok=True)
                (repo_checkout / "agent" / "__init__.py").write_text("", encoding="utf-8")
                (repo_checkout / "agent" / "config.py").write_text('MARKER = "repo"\n', encoding="utf-8")
                site_packages = root / "site-packages"
                (site_packages / "agent").mkdir(parents=True, exist_ok=True)
                (site_packages / "agent" / "__init__.py").write_text("", encoding="utf-8")
                (site_packages / "agent" / "config.py").write_text('MARKER = "site"\n', encoding="utf-8")
                site_packages.mkdir(parents=True, exist_ok=True)

                patched_path = ["", str(repo_checkout), str(site_packages), "/usr/lib/python3.13"]
                with (
                    patch.dict(os.environ, {"PERSONAL_AGENT_INSTANCE": "stable"}, clear=False),
                    patch.object(sys, "prefix", str(root / "runtime" / "current" / ".venv")),
                    patch.object(sys, "base_prefix", "/usr"),
                    patch.object(sys, "path", patched_path),
                ):
                    sys.modules.pop("personal_agent_bootstrap", None)
                    sys.modules.pop("agent", None)
                    sys.modules.pop("agent.config", None)
                    module = importlib.import_module("personal_agent_bootstrap")
                    self.assertTrue(module._is_stable_runtime())
                    imported = importlib.import_module("agent.config")
                    self.assertEqual("site", imported.MARKER)
                    self.assertNotIn(str(repo_checkout), sys.path)
                    self.assertIn(str(site_packages), sys.path)
        finally:
            sys.path[:] = original_path
