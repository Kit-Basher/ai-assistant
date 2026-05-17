# Managed Adapter Lane

Status: historical design proposal for executable/process-backed adapters.

For current implemented external/generated pack adapter behavior, use
[`docs/design/MANAGED_PACK_ADAPTERS.md`](/home/c/personal-agent/docs/design/MANAGED_PACK_ADAPTERS.md).
For product intent, use
[`docs/product/PROJECT_INTENT.md`](/home/c/personal-agent/docs/product/PROJECT_INTENT.md).

This document is retained as a future-design sketch for MCP servers, native
integrations, plugin packs, dependency-backed tools, and other adapters that
would execute code or start processes. It does not describe current implemented
text-pack behavior.

## Purpose

MCP servers, native integrations, plugin packs, and dependency-backed tools are not text skill packs. They must not be imported through `/packs/install` or the Capability Rescue text-pack flow.

The managed adapter lane is a future review and enablement path for integrations that can execute code, start processes, call external services, expose tools, or require dependency installation.

## Architecture

The lane should be separate from pack discovery/import:

- Discovery may mention managed adapters, but only as blocked candidates with preview and risk metadata.
- Preview fetches metadata and manifest details without starting servers, installing dependencies, or executing code.
- Import creates a quarantined adapter record only. It does not install dependencies, start processes, enable tools, or grant permissions.
- Review normalizes the manifest into a canonical adapter descriptor, including requested tools, process model, network requirements, filesystem scope, secrets, and health checks.
- Approval is per adapter and per tool. Enabling one tool must not enable every exposed tool.
- Runtime execution goes through a sandboxed adapter supervisor, not through text skill loading.

## Endpoint Proposal

- `GET /adapter_sources`
  Lists approved managed-adapter discovery sources and policy status.
- `GET /adapter_sources/{source_id}/adapters/search?q=...`
  Searches only enabled, allowlisted sources. Metadata remains untrusted.
- `GET /adapter_sources/{source_id}/adapters/{remote_id}/preview`
  Returns manifest preview, requested permissions, tool list, health-check plan, and blocker reasons. No execution.
- `POST /adapters/import`
  Imports a previewed adapter into quarantine for review only. Requires loopback/operator surface and a preview token or source/remote id pair from an approved source.
- `POST /adapters/{adapter_id}/approve`
  Approves normalized metadata after review. Does not enable tools.
- `POST /adapters/{adapter_id}/tools/{tool_id}/enable`
  Enables one reviewed tool with explicit permission grants.
- `POST /adapters/{adapter_id}/health-check`
  Runs a sandboxed, non-mutating health check with timeout and audit log.

## Permission Model

Default state is deny-all:

- No filesystem access unless a reviewed scope is granted.
- No network access unless destination classes are granted.
- No secrets unless the operator maps a named secret to the adapter.
- No subprocess startup except through the adapter supervisor.
- No dependency install by default.
- No tool is enabled by adapter-level approval alone.

Permissions should be recorded as explicit grants:

- `filesystem.read:{path}`
- `filesystem.write:{path}`
- `network:{host_or_policy}`
- `secret:{secret_id}`
- `process.start:{adapter_id}`
- `tool.invoke:{adapter_id}:{tool_id}`

## Threat Model

Primary risks:

- Arbitrary code execution from dependency install, server startup, postinstall hooks, or plugin handlers.
- Prompt injection in adapter descriptions, manifests, READMEs, or tool docs.
- Secret exfiltration through network tools or logs.
- Lateral movement through broad filesystem or shell access.
- Confused-deputy behavior where a text pack import enables a server-like integration.
- Supply-chain drift when refs are not pinned to immutable commits or verified artifacts.

Required mitigations:

- Quarantine first, normalize second, approve later.
- Treat all discovery metadata as untrusted.
- Require loopback/operator guard for mutating endpoints.
- Require pinned refs or record unpinned-ref warnings.
- Strip or isolate docs from executable manifests.
- Enforce process, network, filesystem, and timeout sandboxing before health checks.
- Audit every preview, import, approval, permission grant, health check, and tool invocation.

## Future Test Plan

- MCP/native/plugin candidates never produce `/packs/install` handoffs.
- Search refuses disabled, denied, or non-allowlisted adapter sources.
- Preview does not start servers or install dependencies.
- Import creates quarantined review records with no enabled tools.
- Approval does not grant permissions.
- Per-tool enablement requires explicit permission grants.
- Health checks run only through the adapter supervisor with timeout and audit logs.
- Unpinned refs, dependency manifests, postinstall hooks, and broad permissions produce blockers or warnings.
- Chat confirmations cannot collapse preview, import, approval, permission grant, and execution into one click.
