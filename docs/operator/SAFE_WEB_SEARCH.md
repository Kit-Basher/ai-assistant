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

## Managed Setup

Personal Agent can preview and apply a bounded SearXNG setup as the first
managed local service:

- `POST /search/setup/plan` returns a confirmation-gated plan.
- `POST /search/setup/apply` requires the confirmation token before mutation.
- User-provided setup accepts loopback SearXNG URLs only.
- Managed container setup uses the approved `personal-agent-searxng` container
  and binds only to `127.0.0.1`.
- On Linux, rootless Podman is the preferred managed-service engine. Docker is
  shown only as an explicit fallback when rootless Podman is unavailable or not
  confirmed.
- If Podman is missing, the default setup plan is a Podman prerequisite plan,
  not a Docker plan. It previews the package action, requires confirmation,
  uses the system package manager for the `podman` package only, does not store
  sudo passwords, verifies Podman/rootless usability, and then asks the operator
  to preview SearXNG setup again.
- Docker fallback plans include `preferred_engine=podman`,
  `selected_engine=docker`, a `fallback_reason`, `rootless_expected=false` or
  unknown, and a Docker fallback warning before confirmation.
- Setup updates the running Personal Agent search configuration after a
  successful SearXNG JSON probe.
- Persistent managed-action journals record planned/running/verified,
  rolled_back, recovery_needed, or failed status without secrets or private
  paths.

The current managed setup does not silently install Podman, Docker, SearXNG, or
system packages. Podman prerequisite setup is explicit, confirmation-gated, and
narrowly allowlisted for SearXNG. If no supported runtime or trusted endpoint is
available and prerequisite setup cannot be prepared, setup is blocked and
reports the next operator action. To keep search enabled after restarting
Personal Agent, set the environment variables above in the service environment.

## Safety Boundaries

The native search adapter:

- does not fetch pages
- does not run JavaScript or browser automation
- does not download files
- does not install or import external packs
- does not trust search results

This means web search can help find general information metadata, but it is not a webpage reader, crawler, downloader, or external skill-pack installer.

## External Pack Source Leads

When approved/trusted external-pack catalogs have no candidate, pack acquisition may use `safe_web_search` to find possible source leads. These leads are still only untrusted metadata. Lead discovery:

- does not fetch pages
- does not download files or archives
- does not install or import external packs
- does not approve sources
- does not trust GitHub or any other domain by name

A lead can only point to a later source approval review. Source approval is still required before any future fetch/import path, and fetched content must still go through quarantine and review.

## Status

Use `/search/status` or ask the assistant “what is your search status?” to check configuration. The status reports whether search is enabled, whether the SearXNG endpoint is configured, whether the JSON endpoint is reachable, a redacted base URL, setup source, blocked reason, and one next action. If search is disabled or missing `SEARXNG_BASE_URL`, the assistant should explain the missing requirement and offer the confirmation-gated local setup path without claiming search works.
