from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


SECRET_QUERY_KEYS = {"token", "key", "api_key", "access_token", "auth", "signature", "sig"}
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass(frozen=True)
class SourceLead:
    title: str
    url: str
    snippet: str = ""
    source_engine: str | None = None
    suspected_source_kind: str = "generic_web_result"
    untrusted: bool = True
    requires_source_approval: bool = True
    blocked_from_fetch: bool = True
    reason: str = "safe_web_search_metadata_only_untrusted_lead"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceLeadSearchResult:
    ok: bool
    leads: tuple[SourceLead, ...]
    source: str = "safe_web_search"
    searched: bool = False
    reason: str | None = None
    search_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "leads": [lead.to_dict() for lead in self.leads],
            "source": self.source,
            "searched": bool(self.searched),
            "reason": self.reason,
            "search_status": dict(self.search_status) if isinstance(self.search_status, dict) else None,
        }


def build_source_leads_from_safe_search(search_payload: dict[str, Any] | None, *, limit: int = 5) -> SourceLeadSearchResult:
    payload = dict(search_payload or {})
    if not bool(payload.get("ok", False)):
        return SourceLeadSearchResult(
            ok=False,
            leads=(),
            searched=False,
            reason=str(payload.get("error_kind") or "search_unavailable"),
            search_status=payload,
        )
    raw_results = payload.get("results") if isinstance(payload.get("results"), list) else []
    leads: list[SourceLead] = []
    seen: set[str] = set()
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        raw_url = str(row.get("url") or "").strip()
        normalized_key = _normalization_key(raw_url)
        if not normalized_key or normalized_key in seen:
            continue
        display_url = sanitize_lead_url(raw_url)
        if not display_url:
            continue
        seen.add(normalized_key)
        title = " ".join(str(row.get("title") or display_url).split())[:300]
        snippet = " ".join(str(row.get("snippet") or row.get("content") or "").split())[:500]
        engine = str(row.get("engine") or row.get("source") or "").strip() or None
        leads.append(
            SourceLead(
                title=title,
                url=display_url,
                snippet=snippet,
                source_engine=engine,
                suspected_source_kind=infer_suspected_source_kind(display_url),
            )
        )
        if len(leads) >= max(1, int(limit or 5)):
            break
    return SourceLeadSearchResult(
        ok=bool(leads),
        leads=tuple(leads),
        searched=True,
        reason=None if leads else "no_http_leads",
        search_status=payload,
    )


def sanitize_lead_url(url: str) -> str | None:
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    safe_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_clean = str(key or "").strip()
        if key_clean.lower() in SECRET_QUERY_KEYS:
            safe_query.append((key_clean, "[REDACTED]"))
        else:
            safe_query.append((key_clean, value))
    clean_path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunparse((parsed.scheme, parsed.netloc, clean_path, "", urlencode(safe_query), ""))


def infer_suspected_source_kind(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if any(path.endswith(ext) for ext in ARCHIVE_EXTENSIONS):
        if "github.com" in host or "githubusercontent.com" in host:
            return "github_archive"
        return "generic_archive_url"
    if host == "github.com" or host.endswith(".github.com"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return "github_repo"
    return "generic_web_result"


def _normalization_key(url: str) -> str | None:
    sanitized = sanitize_lead_url(url)
    if not sanitized:
        return None
    parsed = urlparse(sanitized)
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    path = (parsed.path or "/").rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))

