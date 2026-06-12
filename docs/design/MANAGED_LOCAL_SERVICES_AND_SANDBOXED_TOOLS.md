# Managed Local Services And Sandboxed Tools

Status: design boundary plus the first SearXNG managed-service implementation
notes. This document does not implement Podman/Docker setup, MCP execution, or
app control.

## Purpose

Personal Agent needs a clear boundary for capabilities that go beyond safe text
skill packs. SearXNG is the first managed local service and must set the pattern
for future Ollama, llama.cpp server, MCP servers, Blender, Godot, PDF tools,
image tools, browser helpers, and other executable integrations.

The product should support three capability tiers:

| Tier | Name | Runtime authority |
| --- | --- | --- |
| 1 | Safe text skill packs | No code execution. Imported content is guidance only and must pass quarantine, normalization, review, approval, and enablement before use. |
| 2 | Managed local services | Core-owned service setup/start/stop/status for approved services after explicit preview and confirmation. |
| 3 | Sandboxed tool/MCP runtimes | Executable tool servers with narrow manifests, sandboxing, per-project grants, health checks, and explicit stop/remove paths. |

## Boundaries

### External Text Packs

External text packs are Tier 1. They may provide instructions, examples, and
metadata, but they do not execute code, install dependencies, start processes,
open browsers, mount filesystems, call OAuth flows, or run MCP servers.

Text-pack lifecycle remains:

1. discover from approved sources
2. preview candidate metadata
3. import into quarantine
4. normalize and scan
5. review
6. approve content
7. enable
8. grant narrow managed-adapter permissions if needed
9. invoke only core-owned managed adapters when usable

Source trust is not content trust. Review approval is not enablement. Enablement
is not permission grant. Permission grant is not execution authority.

### Native/Core Adapters

Native/core adapters are runtime-owned code paths compiled into Personal Agent.
They are the only path a text pack can use for bounded actions. A pack may
request a native adapter, but the core runtime owns validation, confirmation,
permission grants, audit records, and execution.

Examples:

- metadata-only local file import dry-run
- approved pack source search/preview
- future safe PDF metadata inspection if implemented as core code

Native adapters must not become arbitrary shell, filesystem write, Docker,
Podman, MCP, browser, or package-install access.

### Managed Local Services

Managed local services are Tier 2. They are optional local services that the
assistant may offer to set up, start, stop, or remove only through approved
bounded plans.

Examples:

- SearXNG for safe web search
- Ollama if future setup management is added
- llama.cpp server if future setup management is added

Managed services are not text packs and not arbitrary containers. They require a
core service definition with approved image/binary/source, fixed ownership
markers, localhost-only network policy by default, health checks, persistent
managed-action journals, and rollback for owned resources.

### Sandboxed Tool/MCP Servers

Sandboxed tool/MCP runtimes are Tier 3. They are executable integrations that
can perform real work through a server process. They require stricter controls
than managed local services because they expose tools, project access, and
often app-specific operations.

Examples:

- Blender MCP
- Godot MCP
- PDF extraction/OCR services
- image processing services
- browser automation helpers

The preferred Linux runtime for optional sandboxed services is rootless Podman
where useful. Docker may be supported only as an optional fallback under the
same bounded service policy. Podman/Docker is not a base Personal Agent
dependency.

Sandboxed runtimes must declare:

- executable/process model
- tool list
- project folder mounts
- read/write paths
- network policy
- secret requirements
- resource limits
- health checks
- stop/remove path
- audit fields
- proof tests

Default policy is deny-all: no broad filesystem access, no arbitrary shell, no
network unless approved, no secret access unless mapped, and no project write
access unless granted.

### App Bridges And Plugins

Some tools cannot be safely controlled only through a sidecar service. Blender,
Godot, browser helpers, and other live GUI apps may need an app plugin or bridge
running inside the application.

App bridges are not text packs. They are reviewed executable integrations with a
host app boundary:

- bridge listens on `127.0.0.1` by default
- user chooses the project/workspace
- per-project grants are required
- dangerous operations require preview/confirm
- bridge exposes a narrow command surface
- health/status is read-only unless mutation is confirmed
- uninstall/disable path is documented

For Blender/Godot-like integrations, the safe direction is usually a
containerized or supervised sidecar where possible plus an app plugin/bridge
only where live editor control is required.

## User-Facing Policy

The assistant may offer to set up missing fundamental services, but it must not
silently install or start them.

Required user-facing behavior:

- explain what capability is missing
- show exactly what will be installed, pulled, started, configured, or removed
- bind local services to `127.0.0.1` unless the user explicitly approves a
  broader bind and the service definition allows it
- require confirmation before install/start/config mutation
- verify readiness after setup
- roll back owned resources if setup fails
- report `recovery_needed` when rollback cannot fully complete
- give one safe next step on failure
- keep secrets, tokens, private paths, hostile pack text, prompts, and provider
  response bodies out of logs and journals

The assistant must never silently install Docker, Podman, SearXNG, Ollama,
llama.cpp, MCP servers, Blender plugins, Godot plugins, browser helpers, or
system packages.

## Implementation Requirements

Every future managed service or sandboxed runtime needs:

1. **Service/runtime definition**
   - stable id
   - tier
   - approved image/binary/source
   - ownership markers
   - allowed ports/binds
   - allowed volumes/mounts
   - allowed environment keys
   - health checks
   - stop/remove plan

2. **Preflight detection**
   - existing user-provided URL or binary
   - existing Personal-Agent-owned resource
   - port conflicts
   - runtime availability
   - resource limits
   - unsupported platform
   - user-owned resource collision

3. **Setup plan**
   - deterministic plan id
   - exact command shape or API operation
   - image/binary name and version/ref
   - ports and binds
   - owned volume/cache paths
   - expected health endpoint
   - rollback plan
   - redacted audit metadata

4. **Confirmation token**
   - bound to the exact plan hash
   - short expiration
   - one-time use
   - stale/consumed confirmations must not replay

5. **Apply path**
   - revalidate preflight
   - compare plan hash
   - execute only approved operations
   - no shell interpolation
   - no arbitrary command strings
   - no unapproved images, mounts, ports, env vars, or package installs

6. **Persistent managed-action journal**
   - `planned`
   - `running`
   - `verified`
   - `rolled_back`
   - `recovery_needed`
   - `failed`
   - redacted metadata only

7. **Health/status endpoint**
   - read-only status
   - configured vs available distinction
   - owned vs user-provided distinction
   - readiness reason
   - next safe action

8. **Stop/remove path**
   - only owned resources
   - preserve audit/tombstone metadata where useful
   - never delete unrelated containers, models, files, projects, packs, or
     source entries

9. **Rollback/recovery behavior**
   - rollback only resources created by the failed action
   - restore previous config when config mutation was part of the plan
   - leave user-owned resources untouched
   - persist `recovery_needed` if cleanup cannot complete

10. **Audit log redaction**
    - no secrets/tokens/API keys
    - no prompts or raw chat text
    - no hostile imported content
    - no raw provider response bodies
    - no broad private filesystem paths unless redacted or hashed

11. **Docs and proof checks**
    - operator setup guidance
    - known limits
    - release gate or proof workflow
    - external pack safety regression when relevant

## Runtime Strategy

Personal Agent itself remains a Python application running as a user systemd
service. The base install must not require Podman or Docker.

Preferred strategy:

- user-provided URL remains supported for services like SearXNG
- managed container setup is optional and confirmation-gated
- rootless Podman is preferred for optional sandboxed services on Linux
- Docker may be supported only as an optional fallback under the same policy
- Python/systemd direct service paths are allowed only when safer and simpler
  than a container
- every managed path remains localhost-bound by default

No managed service may turn into general system administration. If the system
needs Podman, Docker, Ollama, llama.cpp, or an app plugin installed and it is not
already present, the assistant may show the user the manual setup requirement or
offer a future approved managed setup plan. It must not install silently.

## SearXNG First

SearXNG is the first implemented managed local service under this boundary.

Current state:

- search is disabled unless explicitly configured/enabled
- a user-provided loopback `SEARXNG_BASE_URL` works when configured and
  reachable
- `/search/status` distinguishes disabled, configured, reachable, and not
  reachable; it redacts the base URL and gives one next action
- `/search/setup/plan` and `/search/setup/apply` provide a confirmation-gated
  setup path for either a user-provided loopback URL or the approved local
  SearXNG container plan
- when rootless Podman is missing, `/search/setup/plan` returns a
  confirmation-gated Podman prerequisite plan instead of silently selecting
  Docker
- the Podman prerequisite path is narrow: it previews the package action, runs
  only the allowlisted Podman package install command when it can do so without
  hidden interactive privilege, stores no sudo password, verifies Podman,
  verifies rootless Podman usability, and does not start SearXNG or enable
  search
- when Podman install needs interactive privilege, apply returns a bounded
  elevated terminal handoff for `sudo apt-get install -y podman` plus
  verification/retry steps instead of running hidden sudo from the background
  API service
- managed setup uses only the approved SearXNG image, container name, bind, and
  owned state directory; it binds to `127.0.0.1` only
- on Linux, rootless Podman is the preferred managed-service engine; Docker is
  an explicit fallback only when rootless Podman is unavailable or unconfirmed
- Docker fallback plans must say that Podman was not found or rootless Podman
  was not confirmed, set `preferred_engine=podman`, set
  `selected_engine=docker`, include `fallback_reason`, include
  `rootless_expected=false` or unknown, and require explicit Docker fallback
  confirmation
- setup writes persistent managed-action journal status rows and restores the
  previous runtime search settings if verification fails
- the core workflow proof keeps internet/search `BLOCKED` when no trusted
  backend is configured
- mocked search must not be called proof of real internet/search

Later implementation may add:

- optional Docker policy controls if operators want to disable Docker fallback
- owned container/volume markers
- stop/remove path
- persistent restart-safe search environment configuration; the current managed
  setup updates the running Personal Agent process and tells operators how to
  keep the setting after restart

Search must stay disabled unless explicitly configured/enabled. The assistant
must not claim search works until a real configured runtime path works.

## Future MCP, Blender, And Godot Direction

Future executable tool integrations should use Tier 3, not text-pack execution.

Direction:

- containerized sidecar services where possible
- rootless Podman preferred on Linux for optional sandboxed services
- app plugin/bridge where live editor control is required
- localhost-only bridge by default
- per-project permission grants
- narrow project-folder mounts
- network disabled unless approved
- CPU/memory/time limits where available
- dangerous operations previewed and confirmed
- read-only status and health checks before mutation
- stop/remove/disable path before public use

Examples:

- Blender: app plugin for live scene control plus localhost bridge; project file
  access granted per project; destructive scene operations previewed.
- Godot: editor plugin or sidecar bridge; project folder grant required; export,
  filesystem writes, and script execution require explicit approval.
- PDF/image tools: prefer sidecar service with narrow input/output folder
  mounts; no broad home-directory access.
- Browser helpers: no hidden browser automation. Require an explicit bridge,
  localhost binding, profile isolation, and per-site/network policy.

## Non-Goals For This Pass

- no SearXNG setup implementation
- no Podman/Docker base dependency
- no MCP execution
- no arbitrary shell execution
- no broad filesystem writes
- no relaxation of external pack safety gates
- no fake proof of search, direct llama.cpp support, or executable tool support
