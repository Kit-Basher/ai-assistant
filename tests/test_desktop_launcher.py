from __future__ import annotations

import configparser
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(script: Path, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class TestDesktopLauncher(unittest.TestCase):
    def test_desktop_template_has_expected_fields(self) -> None:
        desktop_path = REPO_ROOT / "packaging" / "personal-agent.desktop"
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(desktop_path, encoding="utf-8")
        self.assertIn("Desktop Entry", parser)
        entry = parser["Desktop Entry"]
        self.assertEqual("Application", entry.get("Type"))
        self.assertEqual("__PERSONAL_AGENT_NAME__", entry.get("Name"))
        self.assertEqual("__PERSONAL_AGENT_COMMENT__", entry.get("Comment"))
        self.assertEqual("false", entry.get("Terminal"))
        self.assertIn("Utility", entry.get("Categories", ""))
        self.assertEqual("__PERSONAL_AGENT_LAUNCHER__", entry.get("Exec"))
        self.assertEqual("__PERSONAL_AGENT_LAUNCHER__", entry.get("TryExec"))
        self.assertEqual("personal-agent", entry.get("Icon"))

    def test_install_helper_is_idempotent_and_renders_user_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            home.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "XDG_DATA_HOME": str(home / ".local" / "share"),
                    "PATH": f"/bin:/usr/bin:{env.get('PATH', '')}",
                }
            )
            script = REPO_ROOT / "scripts" / "install_desktop_launcher.sh"

            first = _run_script(script, env=env)
            self.assertEqual(0, first.returncode, first.stderr)
            second = _run_script(script, env=env)
            self.assertEqual(0, second.returncode, second.stderr)

            launcher_path = home / ".local" / "share" / "personal-agent" / "bin" / "personal-agent-webui-dev"
            desktop_path = home / ".local" / "share" / "applications" / "personal-agent-dev.desktop"
            icon_path = home / ".local" / "share" / "icons" / "hicolor" / "scalable" / "apps" / "personal-agent.svg"
            shell_alias = home / ".local" / "bin" / "personal-agent-webui-dev"

            self.assertTrue(launcher_path.is_file())
            self.assertTrue(desktop_path.is_file())
            self.assertTrue(icon_path.is_file())
            self.assertTrue(shell_alias.exists())
            self.assertTrue(shell_alias.is_symlink())
            self.assertEqual(launcher_path, shell_alias.resolve())

            parser = configparser.ConfigParser(interpolation=None)
            parser.read(desktop_path, encoding="utf-8")
            entry = parser["Desktop Entry"]
            self.assertEqual(str(launcher_path), entry.get("Exec"))
            self.assertEqual("personal-agent", entry.get("Icon"))
            self.assertEqual("false", entry.get("Terminal"))
            self.assertEqual("Personal Agent (Dev)", entry.get("Name"))

    def test_launcher_opens_after_service_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            systemctl_log = state_dir / "systemctl.log"
            curl_count = state_dir / "curl-count.txt"
            open_log = state_dir / "open.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "count=0\n"
                "if [ -f \"$CURL_COUNT\" ]; then count=$(cat \"$CURL_COUNT\"); fi\n"
                "count=$((count + 1))\n"
                "printf '%s' \"$count\" > \"$CURL_COUNT\"\n"
                "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$OPEN_LOG\"\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "xdg-open"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "SYSTEMCTL_LOG": str(systemctl_log),
                    "CURL_COUNT": str(curl_count),
                    "OPEN_LOG": str(open_log),
                    "AGENT_LAUNCHER_WAIT_SECONDS": "3",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertEqual(0, proc.returncode, proc.stderr)
            self.assertTrue(systemctl_log.is_file())
            self.assertFalse(systemctl_log.read_text(encoding="utf-8").count("start personal-agent-api.service"))
            self.assertTrue(open_log.is_file())
            self.assertIn("http://127.0.0.1:8765/", open_log.read_text(encoding="utf-8"))

    def test_launcher_opens_when_frontdoor_is_live_even_if_ready_lags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            systemctl_log = state_dir / "systemctl.log"
            curl_log = state_dir / "curl.log"
            open_log = state_dir / "open.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "url=\"${@: -1}\"\n"
                "printf '%s\\n' \"$url\" >> \"$CURL_LOG\"\n"
                "case \"$url\" in\n"
                "  */ready)\n"
                "    printf '{\"ready\": false, \"summary\": \"Starting up.\"}'\n"
                "    ;;\n"
                "  *)\n"
                "    printf '<!doctype html><html><head><meta name=\"personal-agent-webui\" content=\"1\"></head><body>ready</body></html>'\n"
                "    ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$OPEN_LOG\"\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "xdg-open"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "SYSTEMCTL_LOG": str(systemctl_log),
                    "CURL_LOG": str(curl_log),
                    "OPEN_LOG": str(open_log),
                    "AGENT_LAUNCHER_WAIT_SECONDS": "3",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertEqual(0, proc.returncode, proc.stderr)
            self.assertTrue(open_log.is_file())
            self.assertIn("http://127.0.0.1:8765/", open_log.read_text(encoding="utf-8"))
            self.assertIn("/ready", curl_log.read_text(encoding="utf-8"))

    def test_launcher_falls_back_to_explicit_browser_when_xdg_open_is_not_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            xdg_log = state_dir / "xdg-open.log"
            wmctrl_log = state_dir / "wmctrl.log"
            browser_log = state_dir / "browser.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$XDG_LOG\"\n",
                encoding="utf-8",
            )
            (bin_dir / "wmctrl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$WMCTRL_LOG\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "firefox").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$BROWSER_LOG\"\n"
                "sleep 1\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "xdg-open", "wmctrl", "firefox"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "XDG_LOG": str(xdg_log),
                    "WMCTRL_LOG": str(wmctrl_log),
                    "BROWSER_LOG": str(browser_log),
                    "AGENT_LAUNCHER_WAIT_SECONDS": "3",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                    "AGENT_LAUNCHER_WINDOW_CHECK_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertEqual(0, proc.returncode, proc.stderr)
            self.assertIn("Opening Personal Agent UI at http://127.0.0.1:8765", proc.stderr)
            self.assertTrue(xdg_log.is_file())
            self.assertIn("http://127.0.0.1:8765/", xdg_log.read_text(encoding="utf-8"))
            self.assertTrue(browser_log.is_file())
            self.assertIn("--new-window http://127.0.0.1:8765/", browser_log.read_text(encoding="utf-8"))
            self.assertIn("xdg-open did not surface a visible window", proc.stderr)

    def test_launcher_starts_service_when_it_is_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            systemctl_log = state_dir / "systemctl.log"
            curl_count = state_dir / "curl-count.txt"
            open_log = state_dir / "open.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 3; fi\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-enabled\" ]; then exit 3; fi\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"enable\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "count=0\n"
                "if [ -f \"$CURL_COUNT\" ]; then count=$(cat \"$CURL_COUNT\"); fi\n"
                "count=$((count + 1))\n"
                "printf '%s' \"$count\" > \"$CURL_COUNT\"\n"
                "if [ \"$count\" -lt 2 ]; then exit 1; fi\n"
                "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$OPEN_LOG\"\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "xdg-open"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "SYSTEMCTL_LOG": str(systemctl_log),
                    "CURL_COUNT": str(curl_count),
                    "OPEN_LOG": str(open_log),
                    "AGENT_LAUNCHER_WAIT_SECONDS": "3",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertEqual(0, proc.returncode, proc.stderr)
            rendered = systemctl_log.read_text(encoding="utf-8")
            self.assertIn("enable --now personal-agent-api.service", rendered)
            self.assertTrue(open_log.is_file())

    def test_launcher_starts_service_when_it_is_enabled_but_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            systemctl_log = state_dir / "systemctl.log"
            curl_count = state_dir / "curl-count.txt"
            open_log = state_dir / "open.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 3; fi\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-enabled\" ]; then exit 0; fi\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"start\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "count=0\n"
                "if [ -f \"$CURL_COUNT\" ]; then count=$(cat \"$CURL_COUNT\"); fi\n"
                "count=$((count + 1))\n"
                "printf '%s' \"$count\" > \"$CURL_COUNT\"\n"
                "if [ \"$count\" -lt 2 ]; then exit 1; fi\n"
                "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$OPEN_LOG\"\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "xdg-open"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "SYSTEMCTL_LOG": str(systemctl_log),
                    "CURL_COUNT": str(curl_count),
                    "OPEN_LOG": str(open_log),
                    "AGENT_LAUNCHER_WAIT_SECONDS": "3",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertEqual(0, proc.returncode, proc.stderr)
            rendered = systemctl_log.read_text(encoding="utf-8")
            self.assertIn("start personal-agent-api.service", rendered)
            self.assertNotIn("enable --now personal-agent-api.service", rendered)
            self.assertTrue(open_log.is_file())

    def test_launcher_reports_timeout_and_does_not_open_when_readiness_never_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            open_log = state_dir / "open.log"

            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "exit 1\n",
                encoding="utf-8",
            )
            (bin_dir / "xdg-open").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$OPEN_LOG\"\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "xdg-open"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "OPEN_LOG": str(open_log),
                    "AGENT_LAUNCHER_WAIT_SECONDS": "1",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("did not become ready", proc.stderr.lower())
            self.assertFalse(open_log.exists())

    def test_launcher_reports_missing_xdg_open_when_browser_opening_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            home = root / "home"
            bin_dir = root / "bin"
            state_dir = root / "state"
            home.mkdir(parents=True, exist_ok=True)
            bin_dir.mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)

            browser_probe = bin_dir / "browser-never-visible"
            (bin_dir / "systemctl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [ \"${1-}\" = \"--user\" ] && [ \"${2-}\" = \"is-active\" ]; then exit 0; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            (bin_dir / "curl").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '{\"ready\": true, \"summary\": \"Ready.\"}'\n",
                encoding="utf-8",
            )
            browser_probe.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "exit 0\n",
                encoding="utf-8",
            )
            for item in ("systemctl", "curl", "browser-never-visible"):
                path = bin_dir / item
                path.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:/bin:/usr/bin",
                    "AGENT_LAUNCHER_BROWSER_BIN": "browser-never-visible",
                    "AGENT_LAUNCHER_XDG_OPEN": "missing-xdg-open",
                    "AGENT_LAUNCHER_WAIT_SECONDS": "1",
                    "AGENT_LAUNCHER_POLL_SECONDS": "0",
                }
            )
            proc = _run_script(REPO_ROOT / "scripts" / "launch_webui.sh", env=env)

            self.assertNotEqual(0, proc.returncode)
            self.assertIn("Could not open", proc.stderr)
            self.assertIn("firefox --new-window http://127.0.0.1:8765/", proc.stderr)
            self.assertIn("open it manually", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
