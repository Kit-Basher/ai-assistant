# Managed Local Services

Managed local services are a native runtime lane for optional dependencies that make Personal Agent more useful without turning the assistant into a general system administrator.

## Purpose

Managed local service support should let the assistant help set up approved local services such as SearXNG. The goal is to make optional dependencies easy for normal users while keeping all system, container, and configuration changes previewed, explicit, bounded, and reversible.

The first target service is SearXNG for native safe web search.

## Native Service Boundary

Managed service actions are native runtime actions. They are not external pack actions, not arbitrary shell access, and not a way for imported content to run Docker commands.

Allowed native service actions are limited to:

- detect Docker or Podman availability
- detect whether an approved service container exists
- preview a setup plan
- pull an approved image only
- run an approved image only
- bind service ports to `127.0.0.1` only
- use approved fallback ports only after preflight detects a conflict
- use fixed container names, fixed ports, and fixed volume paths
- health check a local endpoint
- stop, restart, or remove Personal-Agent-managed service containers
- update Personal Agent config or env only after confirmation, if an existing config flow safely supports it

For SearXNG, the runtime uses a fixed Personal-Agent-managed container name, loopback-only bind, approved image reference, and bounded persistent volume location. Execution is confirm-gated and validates the approved plan again before any Docker or Podman command runs.

Mutating service actions follow the managed-action recovery pattern in [MANAGED_ACTION_RECOVERY.md](MANAGED_ACTION_RECOVERY.md): journal the attempt, verify success, roll back only owned changes on failure, and never silently mutate pre-existing user resources.

## Blocked Actions

Managed local services must block:

- arbitrary Docker commands
- arbitrary images
- host networking
- privileged containers
- mounting random host paths
- exposing ports to the LAN or internet
- container actions requested by imported or external packs
- Dockerfile builds from untrusted packs
- Docker Compose files from untrusted content
- silent install of Docker, Podman, or system packages
- using containers as an escape hatch for external pack code execution

Approved source, approved service, or approved image does not mean arbitrary content trust. The runtime owns the service contract and must reject anything outside that contract.

## Confirmation Chain

The intended user flow is one explicit step at a time:

1. user asks for web search
2. assistant detects search unavailable
3. assistant detects Docker or Podman availability
4. assistant shows a setup preview
5. user explicitly confirms
6. runtime revalidates the approved plan
7. runtime runs only the bounded approved pull/run commands with `shell=False`
8. runtime health checks the local endpoint
9. runtime rolls back only owned setup resources if verification fails
10. runtime saves or checks config only if supported and separately confirmed
11. assistant reports the result and the next safe step

No step should silently perform the next one. Confirmation for a setup preview does not grant future arbitrary Docker control.

## Normal User UX

Normal user copy should be short and specific:

> Web search is not set up. I can help set up a local SearXNG service using Docker. It will run only on this computer. Say yes to see the setup plan.

Avoid raw env dumps, random Docker documentation, giant technical checklists, or claims that the assistant can install Docker or Podman automatically.

If Docker or Podman is missing, prefer:

- "Show terminal command"
- "I installed it, check again"

Do not present silent or automatic system package installation.

## Relationship To Packs

External or generated packs must not request container execution directly. If a future pack needs a local service, it can ask for a managed native service action. The core runtime still owns detection, preview, confirmation, approved image selection, container command construction, health checks, config writes, and audit records.

Managed local services complement managed adapters. Both are core-owned safety boundaries. Neither allows arbitrary external code execution.

## Implemented SearXNG Setup Contract

The first mutating service action is confirm-gated SearXNG setup. It may only:

- use detected `docker` or `podman`
- pull `docker.io/searxng/searxng:latest`
- run container `personal-agent-searxng`
- prefer `127.0.0.1:8080:8080`
- preflight port 8080 before pull/run
- if 8080 is busy, offer only approved fallback `127.0.0.1:8888:8080`
- if both 8080 and 8888 are busy, do not pull or run
- mount only the Personal-Agent-managed SearXNG volume
- run detached
- health check `http://127.0.0.1:8080`

It must not update search config automatically. If SearXNG starts successfully but search is not configured, the assistant should tell the user to set `SEARCH_ENABLED=1` and `SEARXNG_BASE_URL` to the selected approved local URL, either `http://127.0.0.1:8080` or `http://127.0.0.1:8888`, or use a future separately confirmed config path.

If the approved container name already exists, including a failed `Created` container from an earlier attempt, the conservative default is to stop and require manual inspection or a separate confirmed cleanup action. The runtime must not silently delete or recreate containers.

If health check fails after a setup action created and started the approved container, the runtime stops and removes only that container as owned rollback. If rollback cannot finish, the assistant reports what remains. Users can also ask to stop web search; that cleanup is a separate confirmed action targeting only `personal-agent-searxng`.

## Required Tests

Service action tests must prove:

- unknown services are rejected
- unknown images are rejected
- non-loopback port binds are rejected
- 8080 conflicts select only approved fallback 8888
- both approved ports busy blocks pull/run
- host networking is rejected
- privileged mode is rejected
- random host mounts are rejected
- external packs cannot trigger Docker actions
- Dockerfile or Compose content from packs is rejected
- confirmation is one step only
- missing Docker/Podman produces setup guidance, not auto-install
- SearXNG setup uses only the approved image/name/port/volume contract
