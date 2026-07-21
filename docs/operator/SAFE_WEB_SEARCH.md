# Safe Web Search

Current checkpoint truth lives in `docs/operator/PROJECT_STATE.md`.

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
- `POST /search/setup/apply` consumes the durable, scoped approval associated
  with the current preview before mutation.
- User-provided setup accepts loopback SearXNG URLs only.
- Managed container setup uses the approved `personal-agent-searxng` container
  and binds only to `127.0.0.1`.
- The configured SearXNG image is currently tag-based. Preview/apply bind that
  exact tag plus local image/container/config state, and local drift invalidates
  approval. Upstream tag movement cannot be ruled out; immutable digest pinning
  remains release hardening and is not claimed.
- The first managed SearXNG install seeds and validates an owned minimal
  `settings.yml` before bind-mounting `/etc/searxng`. Empty config mounts and
  arbitrary settings content are rejected.
- The seeded config preserves default SearXNG behavior and enables JSON output
  for Personal Agent's metadata-only safe search provider check. It also
  creates or preserves a non-default `server.secret_key`; the inherited
  `ultrasecretkey` value is rejected and replaced, and the key is redacted from
  journals, diagnostics, and support output.
- If rootless Podman/SearXNG changes ownership on the owned config directory,
  setup stops before pull/run with `managed_service_config_dir_not_writable`
  and returns a bounded visible-terminal handoff:
  `sudo chown -R "$USER:$USER" ~/.local/share/personal-agent/memory/local_services/searxng`
  followed by
  `chmod -R u+rwX ~/.local/share/personal-agent/memory/local_services/searxng`.
  The API service must not run hidden sudo.
- If `personal-agent-searxng` already exists, setup inspects it. It reuses a
  running container only when image, loopback bind, and config mount exactly
  match the approved plan and JSON search verifies. It may restart a stopped
  matching container as the approved repair path. Mismatched containers remain
  blocked for manual inspection and are not removed automatically.
- On Linux, rootless Podman is the preferred managed-service engine. Docker is
  shown only as an explicit fallback when rootless Podman is unavailable or not
  confirmed.
- If Podman is missing, the default setup plan is a Podman prerequisite plan,
  not a Docker plan. It previews the package action, requires confirmation,
  and is allowlisted for the `podman` package only. When privilege is required,
  the background API service does not run hidden `sudo`; apply returns a
  visible elevated terminal handoff for `sudo apt-get install -y podman`, with
  no sudo password storage. After the handoff runs, the operator retries setup;
  Personal Agent verifies Podman/rootless usability and then asks the operator
  to preview SearXNG setup again.
- Docker fallback plans include `preferred_engine=podman`,
  `selected_engine=docker`, a `fallback_reason`, `rootless_expected=false` or
  unknown, and a Docker fallback warning before confirmation.
- Podman detection checks the service `PATH` and approved absolute paths such as
  `/usr/bin/podman`. Setup previews include `podman_found`, `podman_path`,
  `podman_version`, rootless status, and detection source so operators can
  distinguish “search not configured” from “Podman missing.”
- Setup updates the running Personal Agent search configuration after a
  successful SearXNG JSON probe. Managed setup waits up to 30 seconds for
  first boot, treats HTTP 200 as healthy, and retries with `GET` if `HEAD`
  does not prove readiness.
- After verified setup, Personal Agent persists a small local
  `search_runtime_config.json` next to the runtime database so stable restarts
  and promotions can reuse the same loopback SearXNG endpoint. Explicit service
  environment variables (`SEARCH_ENABLED`, `SEARCH_PROVIDER`,
  `SEARXNG_BASE_URL`) still take precedence. If the persisted file is tampered
  to a non-loopback or unsupported endpoint, the runtime refuses to load it and
  `/search/status` reports `invalid_persisted_search_config`.
- The managed SearXNG container is not silently installed or started for the
  first time. After setup has been verified, future API restarts and promotions
  preserve the trusted loopback search config. A PC reboot may leave the
  managed container stopped unless the user's container runtime is configured
  separately to auto-start it. In that case `/search/status` reports
  `configured_stopped`, and the next public lookup offers an inline Plan Mode
  start/repair preview instead of telling the user to set up search from
  scratch.
- If health still fails, setup captures redacted diagnostics for the owned
  `personal-agent-searxng` container, including `ps -a` state and the last
  container logs, before rolling back only that container.
- Persistent managed-action journals record planned/running/verified,
  rolled_back, recovery_needed, or failed status without secrets or private
  paths.

The current managed setup does not silently install Podman, Docker, SearXNG, or
system packages. Podman prerequisite setup is explicit, confirmation-gated, and
narrowly allowlisted for SearXNG; when it needs OS privilege, it prepares an
operator-visible handoff instead of pretending the background service installed
Podman. If no supported runtime or trusted endpoint is available and
prerequisite setup cannot be prepared, setup is blocked and reports the next
operator action. If you want operator-managed configuration rather than the
runtime state file, set the environment variables above in the service
environment.

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

A lead can only point to source/catalog metadata review. Source policy may permit
metadata queries, but remote pack acquisition is unavailable. Any future fetch
would require a new, separately authorized quarantine stage and content-bound
review; current approval does not pre-authorize it.

## Status

Use `/search/status` or ask the assistant “what is your search status?” to check configuration. The status reports whether search is enabled, whether the SearXNG endpoint is configured, whether the JSON endpoint is reachable, a redacted base URL, setup source, persisted-config state, blocked reason, `search_state`, and one next action.

Lifecycle states:

- `never_configured`: no trusted endpoint is configured. First public lookup
  offers local SearXNG setup and requires confirmation.
- `configured_running`: trusted endpoint is configured and JSON search works.
  Public lookups search immediately.
- `configured_stopped`: trusted endpoint is configured but unreachable. Public
  lookups offer inline managed start/repair confirmation and do not ask for a
  full new setup ritual. After approval, Personal Agent starts or repairs only
  the known trusted managed endpoint, rechecks `/search/status`, and continues
  the original lookup when search becomes available.
- `invalid_or_untrusted_config`: persisted or configured endpoint failed trust
  checks. The runtime refuses to use it and offers safe reconfiguration.

`POST /search/setup/plan` is idempotent when search is already running. In that
case it returns `setup_mode=already_configured`, `requires_confirmation=false`,
and `mutated=false` instead of trying to create a new container or allocate a
port.

## v2F managed setup authorization

Status and prerequisite checks remain read-only. Setup, repair, reuse, and the
bounded Podman prerequisite use Universal Mutation Plans and durable scoped
confirmation; SAFE MODE blocks apply. The executor fixes service identity,
image allowlist, loopback bind, container name, configuration location, and
package. Arbitrary image/command/mount/port/URL selection is rejected. The
allowlisted image is not yet registry-digest pinned, so immutable digest binding
remains a release warning.
