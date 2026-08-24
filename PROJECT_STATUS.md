# PROJECT_STATUS

This is the current-state handover doc. Product intent lives in
`docs/product/PROJECT_INTENT.md`; runtime contract lives in
`PRODUCT_RUNTIME_SPEC.md`. Treat this file as a status snapshot, not the sole
source of truth.

## WP6 pre-reinstall candidate truth (v0.2.28)

- WP5 is closed at commit `71a9b54bd9ecbf0b7dca03fec9b5daac40224c12`.
  WP6 adds a layered, registry-reconciled release proof, a separate blind
  messy-language corpus, complete journey accounting, and actual canonical-gate
  sensitivity checks.
- A normal-user diagnostics export is available in the Web UI under
  Diagnostics & recovery. It reports bounded runtime/capability/model/pack
  health and excludes secrets, conversations, raw pack documents, prompts,
  environment values, and private file contents.
- The existing confirmation-gated backup action now also produces a validated
  `personal-agent.portable-backup.v2` archive using SQLite online snapshots.
  Machine-bound secrets and model artifacts are intentionally excluded.
- The Web UI build toolchain is Vite 8 with zero known npm advisories at the
  candidate audit. Ubuntu 24.04 preflight and the human recovery runbook are
  ready, but the physical fresh-host journey has not occurred and WP6 must not
  be called complete.
- The only valid successful interim status is `WP6 PRE-REINSTALL GATE COMPLETE
  - READY FOR UBUNTU RECOVERY TEST` after exact-candidate and live gates pass.

## Completed WP5 truth (v0.2.23)

- Missing-capability rescue can search enabled configured metadata sources, but
  never fetches automatically. Exact-confirmed supported HTTPS/GitHub artifacts
  stream only to quarantine through DNS/peer/redirect/SSRF and archive gates.
- Supported assistant-created drafts are deterministic and quarantine-only.
  Review, exact broker grants, enablement, activation, invocation, update,
  rollback, revocation, and removal remain separate authority transitions.
- Core-owned brokers provide one selected local text/JSON/CSV/HTML file with a
  bounded private index, namespaced structured storage, exact public HTTPS
  GET/HEAD without private-data exfiltration, and a core-rendered PNG visualizer.
- `scripts/pack_acquisition_broker_proof.py` is commit/diff-bound and runs from
  the canonical release gate. See
  `docs/design/SAFE_PACK_ACQUISITION_BROKERS_WP5.md` and
  `docs/releases/v0.2.21.md`.

## Completed WP4.5 truth (v0.2.20)

- Structured Ollama observation, registry configuration, Model Manager history,
  and Scout advice are reconciled without presenting remote/history rows as
  installed. Nine physical artifacts were observed for this release host;
  eight are chat-capable and one is embedding-only.
- Every eligible installed chat model has the same bounded production-adapter
  evaluation. The result is advisory and does not change the selected model.
- Deterministic chat routing no longer performs synchronous provider readiness
  probes. Runtime status uses an aged observed snapshot; explicit refresh keeps
  bounded live probing available.
- See `docs/design/MODEL_TRUTH_AND_LATENCY_WP4_5.md` and
  `docs/releases/v0.2.20.md`.

## Completed WP4 truth (v0.2.16)

- WP4 preserves the protected 22-capability native inventory and allows an
  explicitly supplied, reviewed local pack to add exact digest-bound dynamic
  `pack.*` capabilities through the same registry, chat path, and WP3 task loop.
- Portable text remains non-executable. Declarative capabilities can only map
  validated inputs into a fixed registered native contract. Executable packs
  are pure-computation Wasm under a short-lived Bubblewrap/no-WASI Wasmtime
  worker with no host-effect broker.
- Import, review, grants, enablement, disable/revoke, update, and removal are
  separate exact preview/confirmation mutations. Content changes invalidate
  old authority and stored tasks fail closed rather than substituting versions.
- Automatic remote acquisition, automatic lifecycle advancement, and
  assistant-created packs remain WP5 exclusions.

## Completed WP3 truth (v0.2.14)

- WP3 adds a bounded general task coordinator above the protected 22-capability
  WP2 registry. It is not a second tool layer: every step invokes a registered,
  typed canonical implementation and its verifier.
- Direct casual and one-capability requests remain on the existing fast path.
  Multi-capability goals receive durable plans, evidence, task control, exact
  approval binding for mutations, and restart reconciliation.
- Task APIs and the responsive Web UI task card use the same canonical SQLite
  records. Progress and completion are evidence-based and redact content,
  prompts, tokens, and secrets.
- Missing capabilities use a structured no-action handoff. Executable packs,
  automatic acquisition, assistant-created packs/tools, and general pack
  workers remain deferred to WP4/WP5.
- `scripts/task_loop_proof.py` and the canonical/extended gates protect task
  schemas, composability metadata, approval, cancellation, restart,
  verification, invariants, redaction, and all 16 required scenario categories.

## Completed WP2 baseline (v0.2.11)

- The protected inventory and live registry reconcile 22 native user-goal
  capabilities across assistant, conversation, filesystem, system, models,
  packs, memory, optional search, optional Telegram, and operator lifecycle.
- `GET /capabilities` and the Web UI Capabilities view report live health and
  confirmation requirements. Optional dependencies are shown as unavailable
  or degraded with a reason; endpoint existence is not treated as health.
- `scripts/native_capability_proof.py` reconciles the registry with API, CLI,
  UI, Telegram, native-skill, compatibility-route, and documentation mappings
  and is release-blocking.
- Text-only pack ingress is local-directory only. Arbitrary remote archives,
  executable packs, automatic acquisition, and assistant-created packs remain
  unsupported.
- Live verification of v0.2.10 exposed a capability-status health mismatch for
  stopped SearXNG. The exact v0.2.11 release corrected semantic-domain and
  dependency-truth reconciliation before promotion.

The older narrative below is retained as historical context; where it differs,
the WP3 truth above and the higher-priority product/runtime documents
govern.

## Current State

- The core system foundations are largely in place:
  - API surface
  - web UI surface
  - Telegram surface
  - shared orchestrator
  - shared runtime truth service
  - shared router/provider layer
- Deterministic runtime-truth/status routes are now fast across API, Telegram, and web UI.
- Telegram perceived latency improved with a short grace-window placeholder policy.
- Web/UI perceived latency improved with a short grace-window spinner policy.
- The assistant is now useful for:
  - deterministic status and runtime-truth answers
  - covered troubleshooting flows
  - uncovered device/system troubleshooting via generic fallback diagnostics
  - coding help
  - lightweight planning/task organization
- Robustness on messy user input is materially better:
  - typo-prone status questions route correctly
  - natural confirmations are accepted more reliably
  - vague system trouble now asks for one symptom instead of overreaching
  - obvious coding prompts stay out of disk-pressure troubleshooting paths
- The remaining weakness is coverage breadth and polish, not core viability.

## Proven Working

- Deterministic runtime-truth/status turns bypass the LLM when safe.
- Fast status turns no longer pay the expensive post-response guard on the read-only paths.
- API chat responses expose timing metadata for debugging.
- Telegram logs transport timing and placeholder timing for the live path.
- Web UI logs request/placeholder/response/render timing for the live path.
- Confirmation and mutation flows still use the normal approval boundary.
- Confirmed diagnostics flows work for generic, bluetooth_audio, storage_disk, printer_cups, and generic device fallback cases.

## Partially Working

- Generic `chat` still carries the most variability and depends on model quality when a deterministic path is not available.
- Memory, continuity, and clarification handling exist, but they still need more real-world polish on awkward follow-up turns.
- Coverage is still incomplete outside the current troubleshooting presets and generic fallback.

## Not Yet Proven

- Broad coverage across many troubleshooting domains without adding more focused presets or heuristics.
- A smoother, more polished generic-chat experience for tasks that genuinely need model reasoning.
- That the assistant is ready for every common user workflow without some gaps.

## Known Issues

- Generic `chat` can still feel slow or generic when it genuinely needs model work.
- Some troubleshooting domains still rely on the generic fallback instead of a domain-specific preset.
- Historical docs elsewhere in the repo may still contain stale claims; trust this file first.

## Current Focus

- Polish the real user workflows that are now working.
- Prioritize recurring gaps that show up in actual use.
- Add new presets or features only when repeated real use justifies them.
- Keep the assistant grounded in runtime truth, confirmations, and cautious analysis.

## Last Meaningful Changes

- Deterministic status routes were fast-pathed in the backend.
- Telegram got a grace-window placeholder policy and transport timing instrumentation.
- Web UI got the same perceived-latency treatment and timing instrumentation.
- Read-only runtime-truth routes now skip the expensive assistant response guard.
- API responses now carry structured timing metadata for chat requests.
- Confirmed diagnostics coverage was expanded to bluetooth/audio, storage/disk, printer/CUPS, and a generic device fallback path.
- A recent robustness barrage tightened routing for messy direct prompts, confirmations, and vague system-trouble phrasing.
- Degraded external provider/model health is no longer hard-blocked in the router; it is now a penalty signal.
- Runtime truth no longer force-upgrades provider health to `ok` when model health is `ok`.
- Inference routing now exposes adapter downgrade metadata when fallback call signatures are used.
- Semantic memory context can now inject even when deterministic memory-v2 is disabled.
- Multi-message chat handling now preserves existing compacted warm/cold memory layers instead of replacing them.
