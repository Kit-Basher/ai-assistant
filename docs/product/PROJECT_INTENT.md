# Project Intent

This is the product-intent source of truth for Personal Agent. If implementation notes, status handoffs, or older design docs conflict with this file, this file defines what the runtime is supposed to become.

## Product Truth

Personal Agent is a local personal AI assistant runtime. The user talks to one assistant through the local API, web UI, CLI, or optional Telegram adapter; routing, models, tools, and pack mechanics are implementation details underneath that assistant.

The assistant layer is the user-facing layer. It interprets the user’s intent, decides when grounded runtime or tool facts are needed, asks the agent layer for those facts or bounded action results, and explains the result back to the user.

The agent layer is the grounded runtime/computer/tool layer. It validates requests against allowed capabilities, reads runtime truth or native skill output, performs only approved bounded actions, and returns structured facts/results. It should not invent state or speak to the user as a separate persona.

Core deterministic/native reports are factual agent outputs. Direct report commands such as storage, resource, network, runtime, and model status must stay raw and deterministic, independent of LLM availability. Optional presentation, narration, rewrite, or style transforms are not hidden core behavior; they belong in bounded text-only skill packs or presentation adapters when explicitly added and gated.

Native skills ship with the agent. They are built-in, bounded runtime abilities such as runtime/model status, safe filesystem inspection, safe shell/controller paths, memory/status surfaces, and operational diagnostics.

External skill packs are not bundled built-in abilities. They are optional capability packages that the assistant can discover, preview, import, review, configure, permission, enable, and use only after the user asks for a capability and completes the required safety gates.

External packs use the documented external pack format and remain untrusted until they pass the acquisition lifecycle.

Online external pack sources are hostile by default. GitHub, public registries, remote archives, README files, manifests, and catalog metadata are not trusted merely because of where they are hosted. Remote discovery, preview, fetch, or import requires an explicit trusted source policy or one-time source approval before quarantine can begin. Source approval means only "allowed to inspect or fetch into quarantine"; catalog metadata and archive contents remain untrusted and must pass strict schema, containment, extraction, normalization, and scan gates.

Imported external pack documents are untrusted guidance. They must not become system, developer, or runtime instructions, and prompt-injection attempts in primary instruction files must block import or force a manual rewrite path.

Support and diagnostics must retain only minimal external-pack audit metadata. Removed pack records should preserve identity, hashes, risk state, and lifecycle history without raw imported documents, private local paths, or credential-bearing source URLs.

Codex, development agents, and the repo control plane are not part of the normal runtime user workflow. They may help build or verify this project, but a normal user should not need a developer to manually build every new skill.

## External Pack Lifecycle Contract

When a user asks for a capability that is not already available, the assistant must not dead-end or pretend the capability exists. It should truthfully explain what is missing and guide the user through the next safe step.

The intended lifecycle is:

1. discover approved external pack sources
2. preview a candidate pack or scaffold
3. scaffold/create a local candidate when no approved pack exists
4. quarantine imported or generated content
5. inspect normalized content and safety metadata
6. approve reviewed content
7. configure required settings
8. request and record explicit permissions
9. enable the pack or specific managed capability
10. use it only after approval, enablement, configuration, and permissions are complete

Starter catalogs are discoverable sources only. A starter catalog entry is not an installed capability, not an enabled pack, and not a bundled native ability.

The bundled local starter catalog is discovery-only and does not create trust for remote URLs. Any online source named by a starter entry, catalog entry, or user prompt still needs explicit source approval before fetch/import.

Generated and external packs must not run arbitrary generated code. Text-only pack content may guide behavior, but executable behavior must go through managed adapters implemented in the core runtime.

## Managed Adapters

Managed adapters are the safety boundary for external and generated packs. A pack can request an adapter, but the adapter implementation lives in the trusted core runtime and is guarded by explicit validation, confirmation, grants, and logging.

The current implemented adapter contract starts with `local_file_import`: user-selected-file-only, allowed extensions only, no directory scanning, no network, no dependency install, no browser scraping, and no arbitrary `handler.py` execution.

Future adapters must be explicitly implemented, documented, reviewed, tested, and gated before any external or generated pack can use them.

Managed local services are a future native-runtime direction for approved optional dependencies such as SearXNG. Docker helper packs must remain separate from native Docker control: packs may guide or request approved managed-service actions, but only core runtime code may preview, confirm, and run bounded container operations.

## User Experience Rule

The assistant should say what is true now, what is missing, what is blocked, and what the next safe step is. It should never describe external packs as active abilities until the relevant pack is imported, approved, configured, permissioned, enabled, and usable.
