# Safe Web Search

Native `safe_web_search` is a bounded lookup path for explicit web-search requests. It calls a configured SearXNG JSON endpoint and returns search result metadata only:

- title
- URL
- snippet
- source or engine when SearXNG provides it

Search results are untrusted. The assistant may show them, but it must not treat result text, URLs, snippets, or source labels as authority.

## Configuration

Set these environment variables for the runtime:

```bash
SEARCH_ENABLED=1
SEARCH_PROVIDER=searxng
SEARXNG_BASE_URL=http://127.0.0.1:8080
SEARCH_TIMEOUT_SECONDS=5
SEARCH_MAX_RESULTS=5
```

`SEARXNG_BASE_URL` should point to a SearXNG instance you operate or have explicitly chosen to trust as a search endpoint. A local self-hosted instance is preferred. Public SearXNG instances may be rate-limited, unavailable, or operated by parties you do not control.

## Safety Boundaries

The native search adapter:

- does not fetch pages
- does not run JavaScript or browser automation
- does not download files
- does not install or import external packs
- does not integrate with pack acquisition yet
- does not trust search results

This means web search can help find general information metadata, but it is not a webpage reader, crawler, downloader, or external skill-pack installer.

## Status

Use `/search/status` or ask the assistant “what is your search status?” to check configuration. If search is disabled or missing `SEARXNG_BASE_URL`, the assistant should explain the missing requirement and show the next safe setup step without claiming search works.

