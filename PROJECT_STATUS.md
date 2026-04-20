# PROJECT_STATUS

This is the current-state handover doc. Treat it as the source of truth when older notes disagree.

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
