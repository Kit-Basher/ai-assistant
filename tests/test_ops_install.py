import os
import unittest

from ops import env_templates


class TestOpsEnvTemplates(unittest.TestCase):
    def test_agent_env_has_keys(self) -> None:
        content = env_templates.render_agent_env()
        for key in (
            "TELEGRAM_BOT_TOKEN=",
            "SUPERVISOR_SOCKET_PATH=",
            "SUPERVISOR_HMAC_KEY=",
            "AGENT_UNIT_NAME=",
            "OPS_CONFIG_PATH=",
        ):
            self.assertIn(key, content)

    def test_supervisor_env_has_keys(self) -> None:
        content = env_templates.render_supervisor_env()
        for key in (
            "SUPERVISOR_SOCKET_PATH=",
            "SUPERVISOR_HMAC_KEY=",
            "AGENT_UNIT_NAME=",
            "SUPERVISOR_LOG_LINES_MAX=",
            "SUPERVISOR_SYSTEMCTL_MODE=",
        ):
            self.assertIn(key, content)


class TestOpsInstallScript(unittest.TestCase):
    def test_install_script_contains_required_strings(self) -> None:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ops", "install.sh"))
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        for snippet in (
            "set -euo pipefail",
            "--dry-run",
            "daemon-reload",
            "enable --now",
            "personal-agent-supervisor.service",
            "personal-agent.service",
            "personal-agent-daily-brief.service",
            "personal-agent-daily-brief.timer",
            "install -m 600",
        ):
            self.assertIn(snippet, content)


if __name__ == "__main__":
    unittest.main()
