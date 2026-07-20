# Architecture and Safety Audit v2

Date: 2026-07-20
Baseline: `d911b9b95ce5df9b71cd9c78e28fa13e8767e79a`
Release claim: none; fresh Debian VM testing remains deferred.

## Canonical Request Paths

Web chat follows `POST /chat` -> `AgentRuntime.chat` ->
`AgentOrchestrator.handle_message`. Telegram follows `telegram_bridge` ->
`AgentOrchestrator.handle_message`. Both reach the same assistant entity.
`route_inference()` has one definition in `agent/llm/inference_router.py`;
product inference calls originate in the orchestrator. Telegram remains
transport glue and owns no second model, policy, or executor path.

Read-only runtime status follows API/CLI/Telegram adapters ->
`RuntimeTruthService` -> derived serializers. `/ready` and `/runtime` use this
authority. Compatibility payloads remain a risk where they aggregate or cache
older fields; they are not a second authority.

Migrated mutations follow assistant preview -> canonical Plan Mode object ->
Universal Mutation Plan -> user/thread confirmation -> capability policy ->
Executor Registry -> capability-specific executor -> redacted journal and
receipt. The registry now refuses to synthesize a missing Universal Mutation
Plan and requires scope-bound confirmation metadata before issuing trusted
invocation context.

## Confirmed Findings

### High

- The Executor Registry previously synthesized a Universal Mutation Plan when
  none was supplied and unconditionally told capability policy that the action
  was confirmed. A direct caller could fabricate a canonical-looking plan and
  reach a migrated executor without front-door confirmation metadata.
- Public mutation coverage is incomplete. The machine inventory currently
  classifies 48 API/CLI/background surfaces as `legacy_unmigrated` and seven
  more as `plan_gated_legacy`. Examples include provider/config/default writes,
  model switching and maintenance, semantic-memory maintenance, bootstrap,
  notification maintenance, and legacy built-in skill writes.

### Medium

- The generic mutation-bypass audit treated every file below `agent/` and
  `scripts/` as reviewed merely because of its directory. This produced a
  false-green result. It now warns for each mutation-bearing file without an
  explicit reviewed-path entry.
- The legacy built-in skill policy only recognized delete, overwrite, and
  restart as confirmable. Unknown/write-like action types now default to
  confirmation, while explicit observe/report/read actions remain read-only.
- The managed skill-pack invocation broker has permission/grant enforcement
  but no per-invocation Plan confirmation handoff. The strengthened registry
  now blocks that path with `mutation_confirmation_missing`; Audit 3 must give
  it a preview/confirm continuation before enabling mutation.

### Low

- Compatibility and operator endpoints are numerous and can obscure which
  fields are derived from Runtime Truth. No representative contradiction was
  reproduced in `/ready`, `/runtime`, provider/model status, or optional
  Telegram status, but serializer consolidation remains worthwhile.
- Arbitrary malicious Python already executing inside the trusted process is
  not isolated. Supported platform APIs fail closed, but process isolation is
  not claimed.

## Mutation Inventory

The authoritative machine-readable inventory is
`docs/operator/MUTATION_SURFACE_INVENTORY_V2.json`. It is regenerated and
checked by `python scripts/architecture_safety_audit_v2.py --json`.

Current inventory: 91 surfaces: 2 canonical front doors, 2 central-authorized
surface groups, 7 domain Plan/confirm paths, 7 Plan-gated legacy paths, 14
read-only previews, 9 read-only endpoints, 1 internal state writer, 1
unimplemented-denied foreign-code surface, and 48 legacy-unmigrated surfaces.

`legacy_unmigrated` means exactly that: it is not evidence of an exploitable
remote vulnerability by itself, but it is not compliant with the requested
single central authorization contract and must not be described as migrated.

## Changes in Audit v2

- Added exact mutation confirmation validation bound to plan, fingerprint,
  capability, executor, actor, thread, session, activation fingerprint,
  affirmative phrase class, and expiry.
- Added single-use confirmation consumption at the Executor Registry boundary.
- Removed automatic Universal Mutation Plan synthesis.
- Made legacy unknown skill operations confirmation-required and explicitly
  classified observe/report operations as read-only.
- Repaired the generic bypass audit's directory-wide reviewed assumption.
- Added machine-readable route/surface inventory and architecture invariants.
- Extended adversarial proof expectations for missing confirmation and replay.

## Remaining Migration Work

Do not turn the core into a general admin console. Prioritize public assistant
capabilities, and keep email, notes, tasks, calendar, communications, and
mobile integrations behind bounded native/external packs. Migrate legacy
surfaces by capability family, remove aliases after product-path replacement,
and preserve SAFE MODE blocks for provider/model installs, downloads, imports,
and remote switches.

Audit 3 should be an end-user behavior and UX audit. It should exercise normal
language from Web and disabled/enabled-fixture Telegram through the same
assistant, verify honest runtime-truth wording, and validate preview, confirm,
cancel, expiry, replay, and blocked-legacy behavior. It should not be the fresh
Debian VM acceptance audit.

## Audit v2B working addendum — 2026-07-20

The scanner has explicit evidence records for all 150 current mutation-bearing
files: 132 newly flagged plus 18 previously reviewed. Counts are 78 offline
tools, 24 internal-only writers, 24 read-only files, eight central-boundary
files, and 16 supported files pending central migration.

Managed skill-pack mutation dispatch now uses persisted preview/confirm/cancel
with pack/grant/argument/target/scope drift checks. The 91-surface inventory now
has three central-authorized groups and 47 legacy-unmigrated groups.

This is not v2B closure: seven Plan-gated legacy surfaces and 47 legacy
surfaces remain, and the 24 internal writers do not yet all use one enforced
scheduler/service identity contract. Universal authorization cannot honestly
be claimed.
