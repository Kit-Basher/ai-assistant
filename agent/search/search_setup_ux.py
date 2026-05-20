from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SEARCH_ENV_VARS = {
    "SEARCH_ENABLED": "1",
    "SEARCH_PROVIDER": "searxng",
    "SEARXNG_BASE_URL": "http://127.0.0.1:8080",
    "SEARCH_TIMEOUT_SECONDS": "5",
    "SEARCH_MAX_RESULTS": "5",
}


@dataclass(frozen=True)
class SearchSetupUX:
    ok: bool
    available: bool
    status: str
    message: str
    missing_requirement: str | None
    next_safe_setup_step: str
    exact_env_vars: dict[str, str]
    safety_reminder: str
    provider: str
    endpoint_configured: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_search_setup_ux(status_payload: dict[str, Any] | None) -> SearchSetupUX:
    payload = dict(status_payload or {})
    enabled = bool(payload.get("enabled", False))
    available = bool(payload.get("available", False))
    provider = str(payload.get("provider") or "searxng").strip().lower() or "searxng"
    endpoint_configured = bool(payload.get("endpoint_configured", False))
    reason = str(payload.get("reason") or "").strip().lower()
    safety_reminder = (
        "Safe web search returns metadata only. Results are untrusted, and the agent does not fetch pages, "
        "run browser automation, download files, or import/install packs from search results."
    )

    if available:
        message = (
            "Web search is configured and available according to /search/status. It uses the configured SearXNG "
            "endpoint and returns untrusted metadata only."
        )
        return SearchSetupUX(
            ok=True,
            available=True,
            status="available",
            message=message,
            missing_requirement=None,
            next_safe_setup_step="Use an explicit prompt such as: search the web for <topic>.",
            exact_env_vars=dict(SEARCH_ENV_VARS),
            safety_reminder=safety_reminder,
            provider=provider,
            endpoint_configured=endpoint_configured,
        )

    if not enabled or reason == "search_disabled":
        missing = "SEARCH_ENABLED=1"
        next_step = "Set SEARCH_ENABLED=1 and set SEARXNG_BASE_URL to a SearXNG instance before using web search."
        message = "Web search is disabled. I cannot search the internet until it is enabled and configured."
    elif provider != "searxng" or reason == "unsupported_provider":
        missing = "SEARCH_PROVIDER=searxng"
        next_step = "Set SEARCH_PROVIDER=searxng; no other search provider is supported by this runtime yet."
        message = "Web search is misconfigured. This runtime only supports SearXNG; set SEARCH_PROVIDER=searxng."
    elif not endpoint_configured or reason == "endpoint_missing":
        missing = "SEARXNG_BASE_URL"
        next_step = "Set SEARXNG_BASE_URL to your SearXNG instance, for example http://127.0.0.1:8080."
        message = "Web search is enabled, but the SearXNG endpoint is missing."
    else:
        missing = reason or "search_runtime_unavailable"
        next_step = "Check /search/status and configure SEARCH_ENABLED=1 with a valid SEARXNG_BASE_URL."
        message = "Web search is not available right now."

    return SearchSetupUX(
        ok=False,
        available=False,
        status="unavailable",
        message=message,
        missing_requirement=missing,
        next_safe_setup_step=next_step,
        exact_env_vars=dict(SEARCH_ENV_VARS),
        safety_reminder=safety_reminder,
        provider=provider,
        endpoint_configured=endpoint_configured,
    )


def render_search_setup_ux(status_payload: dict[str, Any] | None) -> str:
    if isinstance(status_payload, dict) and isinstance(status_payload.get("exact_env_vars"), dict):
        env_vars = dict(SEARCH_ENV_VARS)
        env_vars.update({str(k): str(v) for k, v in dict(status_payload.get("exact_env_vars") or {}).items()})
        ux = SearchSetupUX(
            ok=bool(status_payload.get("ok", False)),
            available=bool(status_payload.get("available", False)),
            status=str(status_payload.get("status") or "unavailable"),
            message=str(status_payload.get("message") or "Web search is not available right now."),
            missing_requirement=(
                str(status_payload.get("missing_requirement")).strip()
                if status_payload.get("missing_requirement") is not None
                else None
            ),
            next_safe_setup_step=str(status_payload.get("next_safe_setup_step") or "Configure safe web search."),
            exact_env_vars=env_vars,
            safety_reminder=str(status_payload.get("safety_reminder") or build_search_setup_ux({}).safety_reminder),
            provider=str(status_payload.get("provider") or "searxng"),
            endpoint_configured=bool(status_payload.get("endpoint_configured", False)),
        )
    else:
        ux = build_search_setup_ux(status_payload)
    lines = [
        ux.message,
    ]
    if ux.missing_requirement:
        lines.append(f"Missing requirement: {ux.missing_requirement}.")
    lines.append(f"Next safe setup step: {ux.next_safe_setup_step}")
    lines.append("Environment variables:")
    lines.append(f"- SEARCH_ENABLED={ux.exact_env_vars['SEARCH_ENABLED']}")
    lines.append(f"- SEARCH_PROVIDER={ux.exact_env_vars['SEARCH_PROVIDER']}")
    lines.append(f"- SEARXNG_BASE_URL={ux.exact_env_vars['SEARXNG_BASE_URL']}")
    lines.append(f"- SEARCH_TIMEOUT_SECONDS={ux.exact_env_vars['SEARCH_TIMEOUT_SECONDS']}")
    lines.append(f"- SEARCH_MAX_RESULTS={ux.exact_env_vars['SEARCH_MAX_RESULTS']}")
    lines.append(ux.safety_reminder)
    return "\n".join(lines)


def setup_hint_for_search_failure(search_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = dict(search_payload or {})
    error_kind = str(payload.get("error_kind") or "").strip().lower()
    if error_kind not in {"search_disabled", "endpoint_missing", "unsupported_provider"}:
        return None
    status_payload = {
        "enabled": bool(payload.get("enabled", False)),
        "provider": str(payload.get("provider") or "searxng").strip().lower() or "searxng",
        "available": False,
        "endpoint_configured": error_kind != "endpoint_missing",
        "reason": error_kind,
    }
    return build_search_setup_ux(status_payload).to_dict()
