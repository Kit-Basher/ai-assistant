# PROJECT_STATUS

This is the current-state handover doc. Product intent lives in
`docs/product/PROJECT_INTENT.md`; runtime contract lives in
`PRODUCT_RUNTIME_SPEC.md`. Treat this file as a status snapshot, not the sole
source of truth.

## WP2 candidate truth (v0.2.10)

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
- The v0.2.9 promotion exposed two live-language validation gaps before final
  acceptance. The exact v0.2.10 candidate includes their semantic-domain fixes
  and remains a candidate until its complete gates and live verification pass.

The older narrative below is retained as historical context; where it differs,
the WP2 candidate truth above and the higher-priority product/runtime documents
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
