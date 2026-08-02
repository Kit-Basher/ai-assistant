# Unified conversation understanding and capability routing (WP1)

## Production path before this change

All first-party clients ultimately use `POST /chat`: the Web UI posts directly,
the CLI posts directly, and Telegram normally proxies to it. `POST /ask` still
has compatibility code, but is rewritten to `/chat` after its legacy intent
assessment. The public `/chat` path is:

1. `APIServerHandler.do_POST` validates JSON and request shape. It also owns
   legacy `/ask` clarification and intent-assessment compatibility.
2. `AgentRuntime.chat` normalizes messages, derives user/thread/surface IDs,
   uses `classify_trivial_social_turn` only as a no-model-bootstrap performance
   hint, calls `chat_route_decision` for compatibility bootstrap choice, creates
   tracing metadata, and invokes the
   orchestrator. It serializes, persists the transcript, and writes runtime and
   route audit events.
3. `chat_route_decision` calls `setup_chat_flow.classify_runtime_chat_route`.
   That classifier currently overlaps with the orchestrator: setup, model,
   filesystem, system, search, pack, capability, and runtime phrases are all
   recognized there, but its result is mostly a hint.
4. `Orchestrator.handle_message` wraps `_handle_message_impl` with response
   guards, epistemic checks, continuity writes, and timing.
5. `_handle_message_impl` owns deterministic approval/denial/cancellation and
   confirmation expiry/thread binding. It then has separate social phrase
   handling, capability questions, vague clarification, context reset, command
   routing, runtime/setup routing, native tool routing, pack routing,
   `MemoryRuntime.resolve_followup`, `nl_router`, the older `intent_router`, and
   finally generic chat. This is the principal ownership overlap.
6. `_handle_runtime_truth_chat` calls `classify_runtime_chat_route` again and
   dispatches to grounded runtime, filesystem, model, Model Scout, search,
   lifecycle, and setup response methods.
7. `_llm_chat` checks deterministic front-door routes yet again, retrieves
   selective, working, v2, and semantic memory, prepares the selected local
   model, performs inference, interprets bounded tool directives, and shapes a
   response. The generic system prompt has partial runtime guardrails but no
   complete live capability/runtime contract.
8. `serialize_orchestrator_chat_response` builds the stable response envelope;
   `AgentRuntime.chat` adds timing, writes audit/runtime events, and records both
   sides of the conversation in SQLite.

Tests exercise both direct classifiers and the production handler. Web UI,
Telegram, and CLI converge on `/chat`; direct `Orchestrator.handle_message`
tests are an additional internal surface rather than a different product path.

## Ownership problems

Ordinary intent is presently decided by overlapping phrase lists in
`setup_chat_flow`, `assistant_ux`, `public_chat`, `nl_router`, `intent_router`,
and many orchestrator `_looks_like_*` methods. Pending state is split between
`ConfirmationStore`, `MemoryRuntime`, setup state, onboarding state, compare
state, and pack adapter state. Safety checks are intentionally repeated at
executor boundaries, but ordinary capability selection is also repeated. The
result is uneven language coverage and allows an unrecognized product request
to reach an unconstrained generic model response.

## Target ownership and integration plan

`request_understanding` becomes the single owner for non-command user meaning.
It returns one typed, auditable `RequestUnderstanding` containing the original
text, non-destructive normalization, context reference, candidates, selected
capability, confidence/ambiguity, validated inputs, mutation/approval state,
clarification, and fallback category. It uses a dynamic `CapabilityRegistry`.

The registry holds only callable, health-checked capabilities with JSON-like
input/output contracts, policy, provenance, invocation, and verification. A
fast offline feature-vector matcher (word and subword features over capability
descriptions/examples) handles robust ordinary selection without a second LLM
generation. Deterministic state transitions remain authoritative for approval,
denial, cancellation, expiry, and thread binding. Low-margin materially
different actions produce one clarification; no-capability turns become either
grounded casual chat, ordinary model chat with authoritative runtime context, or
an honest unavailable result.

The production migration now starts at the API boundary. `POST /chat` computes a
unified preview before asking the setup classifier for compatibility hints. The
preview controls bootstrap choice whenever it selects a capability, asks a
clarification, or selects grounded casual/unavailable fallback. Inside the
orchestrator, approval/cancellation and explicit slash-command handlers remain
first; unified selection then dispatches immediately through registry hooks to
the already-proven native implementations. A registry hook receives its
validated structured inputs and cannot call the setup classifier to select,
veto, or replace the chosen capability.

The legacy setup classifier remains callable for deterministic safety/setup and
capability families outside WP1. A denylist of its historical WP1 result kinds
is enforced after unified understanding: if unified understanding declined to
select a WP1 capability, one of those legacy results is discarded instead of
becoming a second vote. Capability answers are rendered from the live registry.
Generic model output receives verified runtime context and is checked for
contradictory sandbox/access/identity assertions.

WP1 migrates presence/identity, live capability listing, bounded filesystem
list/search/read, system/runtime information, model inventory/current-model and
switching, Model Scout, pack discovery/use, and conversation/history/memory
orientation. A complete native capability census, executable packs, automated
acquisition, and plan/act/verify automation remain WP2+ work.

## Final ownership map

| Decision | Sole production owner | Retained lower-level responsibility |
| --- | --- | --- |
| Ordinary meaning and WP1 capability ID | `RequestUnderstandingService` | none |
| Capability existence, health, schemas, policy | `CapabilityRegistry` | native implementation performs the operation only |
| Approval, denial, cancellation, expiry, thread binding | deterministic orchestrator state machine | executor rechecks authorization at the mutation boundary |
| Structured path/query/model/pack/history arguments | post-selection schema-bound extractors | native implementation validates resolved resources |
| Ambiguity | unified confidence/material-group decision | response envelope persists one concise question |
| Casual/presence/runtime facts | unified fallback or registered capability plus `RuntimeTruthService` | response guard rejects contradictory generated claims |
| Generic conversation | one local-model inference with authoritative runtime context | a second bounded generation is allowed only to repair a detected bad draft |
| Serialization, transcript persistence, audit | `AgentRuntime.chat` | SQLite and event sinks preserve the public envelope |

## Remaining compatibility-classifier calls

`classify_runtime_chat_route` intentionally remains in production, but not in a
WP1 registry invocation and not as an ordinary WP1 intent owner:

- `AgentRuntime.chat_route_decision` supplies setup/bootstrap hints only after
  the unified preview; it cannot override a selected capability, clarification,
  grounded casual turn, or unavailable result.
- The orchestrator uses it before unified dispatch only for deterministic
  safety-bypass, shell containment, explicit setup, and pending-state guards.
- After unified dispatch it serves non-WP1 compatibility families such as safe
  web search, Telegram service control, operator lifecycle, managed adapters,
  and setup flows. Historical WP1 kinds are explicitly rejected there.
- Generic-chat preparation uses it only for containment of safe web search,
  package/shell execution, and setup—not to invoke a registered capability.
- Helper predicates for provided-text transforms, fresh-intent cancellation,
  and deterministic-route detection use it to protect state boundaries. They
  do not choose or invoke a WP1 capability.
- `_handle_runtime_truth_chat` and old pack/capability-gap adapters remain for
  non-WP1 callers and public compatibility. The production `/chat` path reaches
  them only after unified understanding produced no WP1 selection; registry
  hooks never call them to reclassify their input.

The older classifier and response adapters can be physically removed only when
their remaining non-WP1 families are migrated. That census is WP2+ scope; this
package prevents them from silently owning or reclassifying WP1 traffic.
