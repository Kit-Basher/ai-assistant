# Known Issues

## Product-Level Issues

- Coverage is still uneven outside the current diagnostics presets and generic fallback.
- Some common troubleshooting domains still need either a targeted preset or a better generic analysis path.
- Short, messy follow-up phrasing can still expose router edge cases, especially when the thread changes topic abruptly.
- Optimization profiles are now introduced for arbitration, but they still need broader endpoint-level controls and UI exposure.

## Infra / Deployment Issues

- The web UI is shipped as a built asset, so front-end changes require a rebuild before release.
- Some historical docs and release artifacts still reflect older states; this document and `PROJECT_STATUS.md` should be treated as the current baseline.

## UX / Perceived-Latency Issues

- Fast deterministic status turns are now improved on API, Telegram, and web UI.
- Slower `generic_chat` turns can still feel slow when they genuinely need model work.
- The UI still uses a spinner/placeholder for slower turns by design.
- Multi-turn generic-chat reasoning can still feel less polished than the deterministic flows.

## Stale Issues Removed

- Deterministic runtime-truth/status turns being blocked by guard work: fixed.
- Telegram first-turn placeholder slowness on fast status turns: fixed.
- Web/UI spinner showing immediately on fast deterministic status turns: fixed.
- `generic_chat` being the main unresolved question: no longer the primary blocker.
- Broadly obvious coding prompts being routed into disk-pressure troubleshooting: fixed.
- Vague “system is weird” prompts being overcommitted into broad observe: fixed.
- Semantic memory being blocked whenever deterministic memory-v2 is disabled: fixed.
- External degraded health always being hard-blocked in router candidate selection: fixed.
