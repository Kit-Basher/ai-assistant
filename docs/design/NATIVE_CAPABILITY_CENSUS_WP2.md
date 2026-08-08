# Native Capability Census and Proof Architecture (WP2)

## Scope and ownership

This document is the concise architecture and reconciliation record for Work
Package 2. The protected machine-readable inventory is
`config/native_capabilities.json`; the live `CapabilityRegistry` remains the
runtime authority. The inventory prevents accidental deletion from redefining
product truth, while the registry provides contracts, health, invocation,
verification, self-test, policy, and current availability.

The taxonomy is based on user goals, not HTTP methods. One capability may own
several API, CLI, UI, Telegram, native-skill, controller, and compatibility
surfaces. Transport and operator endpoints are mapped beneath a parent instead
of being inflated into separate capabilities.

The production ownership chain is:

1. `/chat` builds one `RequestUnderstanding` from the live chat-selectable
   registry.
2. Deterministic pending approval, denial, cancellation, expiry, and safety
   guards run at their established boundaries.
3. A selected registry entry receives validated structured inputs and calls its
   canonical implementation directly.
4. Compatibility classification is eligible only after understanding declines
   to select a registered capability. It cannot replace a registry selection.
5. Direct API/CLI/UI surfaces call the same runtime truth, controller, pack,
   memory, filesystem, or central executor implementations.

## Census method

The census inspected the complete repository and promoted runtime, including:

- all 127 statically discoverable HTTP path literals plus dynamic path
  families in `APIServerHandler`;
- 19 operator CLI commands;
- nine normal-user UI areas and the developer/operator panels;
- ordinary Telegram text, status, and service-control boundaries;
- 15 native skill manifests;
- the frozen executor registry and provider/model, organization/memory, and
  pack/search authorization domains;
- compatibility route families, setup/doctor/lifecycle scripts, release
  nodes, and current documentation claims.

Every surface is assigned to a registered parent or one of these explicit
dispositions: `supporting_internal`, `optional_unavailable`, `operator_only`,
`compatibility_only`, or `unsupported`. Prefix rules are validated against the
actual API handler by the proof runner; a new unmatched path fails the release
gate.

## Registered native capabilities

The 22 user-goal capabilities are:

- Assistant: `assistant.presence`, `assistant.capabilities`
- Conversation: `conversation.history`
- Filesystem: `filesystem.list`, `filesystem.search`, `filesystem.read`,
  `filesystem.create_directory`
- System: `system.status`, `system.shell.inspect`, `system.package.install`
- Models: `models.inventory`, `models.switch`, `models.scout`
- Packs: `packs.use`, `packs.manage`
- Memory: `memory.status`, `memory.manage`
- Search: `search.web`
- Telegram: `telegram.status`, `telegram.manage`
- Lifecycle: `operator.status`, `operator.lifecycle`

Granularity separates read-only status/use from mutation whenever approval
policy differs. It deliberately groups controller sub-actions such as model
test, temporary switch, default switch, switch-back, and acquisition beneath
the model-control capability rather than pretending each endpoint is a new
ability.

## Proof contract

Every definition declares typed input/output contracts, read-only or mutating
mode, approval policy, permission/mode requirements, a side-effect-free health
hook, a direct invocation hook, a result verifier, a deterministic fixture
self-test, an unavailable message, and proof requirements.

Proof profiles are protected in the inventory:

- deterministic read: health, self-test, verifier, chat, policy, and
  model-unavailable behavior;
- dependency read: the same plus degraded/failure behavior;
- persistent read: dependency proof plus restart persistence;
- optional provider: healthy/unavailable, timeout, malformed/failure, and
  redaction behavior;
- mutation: preview, approval, denial, cancellation, expiry, thread binding,
  replay/idempotency, verification, and indeterminate outcome.

`scripts/native_capability_proof.py` constructs the actual registry with a
temporary database, enumerates the protected inventory, validates contracts
and policy, runs health and self-test hooks, reconciles all mapped surfaces,
executes every declared proof node once, associates pass/fail results with each
capability and proof category, and emits JSON and text reports tied to the current commit plus dirty-diff
fingerprint. Reports omit secrets, tokens, private URLs, raw imported pack
content, and model prompts/responses. A stale report is never input to the
gate.

## Reconciliation findings and resolutions

| Finding | Resolution |
|---|---|
| Eleven WP1 entries described only the conversational migration subset | Expanded the same registry to 22 user-goal capabilities and protected the expected set. |
| Public snapshot exposed only `verification: hook` | Added structured health, self-test presence, permission/mode, chat-selectability, and proof metadata. |
| Release gate named WP1 tests statically | Added registry-driven proof execution and WP2 sensitivity/production-path tests to main and extended validation. |
| README claimed filesystem stat/content search, bounded shell, mutations, memory, lifecycle, search, Telegram, and pack management beyond the registry | Mapped supporting surfaces and registered distinct user goals where policy or natural invocation is material. |
| Runtime spec claimed remote pack ingress while runtime denied URLs | Corrected current runtime truth to local-directory text-pack ingestion only; remote catalog metadata remains non-installing. |
| `docs/operator/PROJECT_STATE.md` presented v0.2.5 as current | Replaced the top-level current snapshot with the WP2 candidate and labeled older material historical. |
| Optional search/Telegram could look green by endpoint existence | Capability status derives from live health and reports precise unavailable/degraded reasons. |
| Capability answers could be replaced by legacy prose | Capability answers now render directly from the live registry snapshot. |
| No normal-user capability status view | Added `/capabilities` and a Web UI Capabilities view, with advanced details opt-in. |
| Four npm advisories | Non-breaking updates removed `nanoid` and `postcss` findings. Remaining Vite/esbuild findings affect the development server/build tool, not served static production assets; the only automatic fix is a Vite 8 major upgrade and is deferred with explicit residual risk. |

## Explicit boundaries

- Remote pack acquisition and executable/plugin packs are unsupported. Local
  `SKILL.md`-centered text-pack ingestion remains quarantine/review gated.
- Safe web search returns SearXNG result metadata only; it is not a page fetcher
  or browser. Missing SearXNG is an expected optional-unavailable state.
- Managed SearXNG/Podman setup remains an existing operator adapter. WP2 does
  not install it or add managed services.
- Git push stays preview-only without configured proof. Arbitrary shell,
  Docker/Podman/systemctl, broad filesystem mutation, and purge uninstall stay
  unsupported.
- Operator lifecycle and provider configuration remain loopback, policy, and
  confirmation governed.
- Generic chat is a grounded fallback, not a registered tool capability.
- Executable packs, general task planning/execution, automatic acquisition,
  and assistant-created packs remain WP3-WP5 work.
