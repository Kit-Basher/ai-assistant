from __future__ import annotations


def render_agent_env() -> str:
    return "\n".join(
        [
            "TELEGRAM_BOT_TOKEN=",
            "SUPERVISOR_SOCKET_PATH=/run/personal-agent/supervisor.sock",
            "SUPERVISOR_HMAC_KEY=",
            "AGENT_UNIT_NAME=personal-agent.service",
            "OPS_CONFIG_PATH=/etc/personal-agent/ops_config.json",
            "",
        ]
    )


def render_supervisor_env() -> str:
    return "\n".join(
        [
            "SUPERVISOR_SOCKET_PATH=/run/personal-agent/supervisor.sock",
            "SUPERVISOR_HMAC_KEY=",
            "AGENT_UNIT_NAME=personal-agent.service",
            "SUPERVISOR_LOG_LINES_MAX=200",
            "SUPERVISOR_SYSTEMCTL_MODE=system",
            "",
        ]
    )
