from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
import socket
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, build_opener
from urllib.error import HTTPError, URLError

from agent.search.search_setup_ux import setup_hint_for_search_failure


SUPPORTED_PROVIDER = "searxng"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_RESULTS = 5
HARD_MAX_RESULTS = 10


_TOKEN_QUERY_RE = re.compile(
    r"(?i)\b(token|api[_-]?key|access[_-]?token|auth|signature|sig|key)=([^\s&]+)"
)
_LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")
_LOCAL_PATH_RE = re.compile(r"(?<!\w)(?:/home|/Users|/data|/tmp)/[^\s]+")


def redact_search_query(query: str | None) -> str:
    text = " ".join(str(query or "").split())
    text = _TOKEN_QUERY_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _LOCAL_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _LONG_SECRET_RE.sub("[REDACTED_TOKEN]", text)
    if len(text) > 160:
        return text[:157].rstrip() + "..."
    return text


@dataclass(frozen=True)
class SafeWebSearchConfig:
    enabled: bool = False
    provider: str = SUPPORTED_PROVIDER
    searxng_base_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_results: int = DEFAULT_MAX_RESULTS

    @classmethod
    def from_runtime_config(cls, config: Any) -> "SafeWebSearchConfig":
        return cls(
            enabled=bool(getattr(config, "search_enabled", False)),
            provider=str(getattr(config, "search_provider", SUPPORTED_PROVIDER) or SUPPORTED_PROVIDER)
            .strip()
            .lower(),
            searxng_base_url=str(getattr(config, "searxng_base_url", "") or "").strip() or None,
            timeout_seconds=float(getattr(config, "search_timeout_seconds", DEFAULT_TIMEOUT_SECONDS) or DEFAULT_TIMEOUT_SECONDS),
            max_results=int(getattr(config, "search_max_results", DEFAULT_MAX_RESULTS) or DEFAULT_MAX_RESULTS),
        )


@dataclass(frozen=True)
class SafeSearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str | None = None
    engine: str | None = None
    untrusted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafeWebSearchResponse:
    ok: bool
    enabled: bool
    provider: str
    status: str
    message: str
    results: tuple[SafeSearchResult, ...] = ()
    query_redacted: str | None = None
    error_kind: str | None = None
    untrusted: bool = True
    redactions_applied: bool = False
    setup_hint: dict[str, Any] | None = None
    safety: dict[str, Any] = field(
        default_factory=lambda: {
            "results_are_untrusted": True,
            "page_fetching": False,
            "browser_automation": False,
            "downloads": False,
            "pack_install_import": False,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [item.to_dict() for item in self.results]
        return payload


class SafeWebSearchClient:
    """Bounded metadata-only search through a configured SearXNG JSON endpoint."""

    def __init__(self, config: SafeWebSearchConfig, *, opener: Any | None = None) -> None:
        self.config = config
        self._opener = opener or build_opener()

    def status(self) -> dict[str, Any]:
        enabled = bool(self.config.enabled)
        provider = str(self.config.provider or "").strip().lower() or SUPPORTED_PROVIDER
        endpoint_configured = bool(str(self.config.searxng_base_url or "").strip())
        available = enabled and provider == SUPPORTED_PROVIDER and endpoint_configured
        reason = None
        if not enabled:
            reason = "search_disabled"
        elif provider != SUPPORTED_PROVIDER:
            reason = "unsupported_provider"
        elif not endpoint_configured:
            reason = "endpoint_missing"
        return {
            "ok": True,
            "enabled": enabled,
            "provider": provider,
            "available": available,
            "endpoint_configured": endpoint_configured,
            "max_results": min(HARD_MAX_RESULTS, max(1, int(self.config.max_results or DEFAULT_MAX_RESULTS))),
            "timeout_seconds": max(0.1, float(self.config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS)),
            "reason": reason,
            "safety": {
                "metadata_only": True,
                "results_are_untrusted": True,
                "page_fetching": False,
                "browser_automation": False,
                "downloads": False,
                "pack_install_import": False,
            },
        }

    def search(self, query: str, *, max_results: int | None = None) -> SafeWebSearchResponse:
        query_clean = " ".join(str(query or "").split())
        redacted_query = redact_search_query(query_clean)
        redacted = bool(redacted_query != query_clean)
        provider = str(self.config.provider or "").strip().lower() or SUPPORTED_PROVIDER
        if not self.config.enabled:
            return self._failure(
                "search_disabled",
                "Web search is disabled. Set SEARCH_ENABLED=1 and configure SEARXNG_BASE_URL to use it.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        if provider != SUPPORTED_PROVIDER:
            return self._failure(
                "unsupported_provider",
                "Web search is configured with an unsupported provider. This runtime only supports SearXNG.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        base_url = str(self.config.searxng_base_url or "").strip()
        if not base_url:
            return self._failure(
                "endpoint_missing",
                "Web search is enabled, but SEARXNG_BASE_URL is not configured.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        if not query_clean:
            return self._failure(
                "empty_query",
                "Search needs a query.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        limit = self._bounded_max_results(max_results)
        url = self._search_url(base_url, query_clean)
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "personal-agent-safe-web-search/1"})
        try:
            with self._opener.open(request, timeout=self._timeout()) as response:
                raw = response.read(2 * 1024 * 1024)
        except TimeoutError:
            return self._failure(
                "search_timeout",
                "Web search timed out before returning metadata.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        except (HTTPError, URLError, OSError, socket.timeout) as exc:
            return self._failure(
                "search_error",
                f"Web search failed safely: {exc.__class__.__name__}.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return self._failure(
                "bad_response",
                "Web search returned malformed JSON.",
                query_redacted=redacted_query,
                redactions_applied=redacted,
            )
        results = tuple(self._normalize_results(payload, limit=limit))
        message = (
            f"Search returned {len(results)} untrusted metadata result"
            f"{'' if len(results) == 1 else 's'}. I did not open pages, run JavaScript, download files, or import packs."
        )
        return SafeWebSearchResponse(
            ok=True,
            enabled=True,
            provider=provider,
            status="ok",
            message=message,
            results=results,
            query_redacted=redacted_query,
            redactions_applied=redacted,
        )

    def _failure(
        self,
        error_kind: str,
        message: str,
        *,
        query_redacted: str | None = None,
        redactions_applied: bool = False,
    ) -> SafeWebSearchResponse:
        return SafeWebSearchResponse(
            ok=False,
            enabled=bool(self.config.enabled),
            provider=str(self.config.provider or SUPPORTED_PROVIDER).strip().lower() or SUPPORTED_PROVIDER,
            status="blocked",
            message=message,
            error_kind=error_kind,
            query_redacted=query_redacted,
            redactions_applied=redactions_applied,
            setup_hint=setup_hint_for_search_failure(
                {
                    "error_kind": error_kind,
                    "enabled": bool(self.config.enabled),
                    "provider": str(self.config.provider or SUPPORTED_PROVIDER).strip().lower() or SUPPORTED_PROVIDER,
                }
            ),
        )

    def _bounded_max_results(self, requested: int | None) -> int:
        value = int(requested or self.config.max_results or DEFAULT_MAX_RESULTS)
        return min(HARD_MAX_RESULTS, max(1, value))

    def _timeout(self) -> float:
        return max(0.1, float(self.config.timeout_seconds or DEFAULT_TIMEOUT_SECONDS))

    @staticmethod
    def _search_url(base_url: str, query: str) -> str:
        base = base_url.rstrip("/") + "/"
        params = urlencode({"q": query, "format": "json", "safesearch": "1"})
        return urljoin(base, "search") + "?" + params

    @staticmethod
    def _normalize_results(payload: Any, *, limit: int) -> list[SafeSearchResult]:
        if not isinstance(payload, dict):
            return []
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            return []
        results: list[SafeSearchResult] = []
        for row in raw_results:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            if not title or not SafeWebSearchClient._is_http_url(url):
                continue
            snippet = str(row.get("content") or row.get("snippet") or "").strip()
            engine = SafeWebSearchClient._engine_name(row)
            source = str(row.get("source") or row.get("category") or "").strip() or None
            results.append(
                SafeSearchResult(
                    title=title[:300],
                    url=url,
                    snippet=snippet[:800],
                    source=source,
                    engine=engine,
                    untrusted=True,
                )
            )
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _engine_name(row: dict[str, Any]) -> str | None:
        engine = str(row.get("engine") or "").strip()
        if engine:
            return engine
        engines = row.get("engines")
        if isinstance(engines, list):
            joined = ", ".join(str(item).strip() for item in engines if str(item).strip())
            return joined[:200] or None
        return None

    @staticmethod
    def _is_http_url(url: str) -> bool:
        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
