# Assistant Interaction Hardening

This file is the canonical tracked matrix for user-facing assistant behavior hardening.

Goal:
- Reach zero obviously bad agent-style interactions on normal user-facing chat surfaces.
- Keep replies helpful, correct when grounded, and consistently in assistant voice.
- Prevent the user from seeing control-plane chatter, chooser prompts, transport placeholders, or model self-disclaimer sludge.

Status legend:
- `PASS`: recently verified by the barrage or a focused regression.
- `FAIL`: reproduced and needs a code fix.
- `IN PROGRESS`: being actively worked.
- `NOT RUN`: added to the matrix but not recently exercised.

## Quality Bar

Every user-facing reply should satisfy all of these unless the scenario explicitly requires a confirmation step:

- Sounds like one assistant, not a router, shell wrapper, or support bot.
- Avoids internal fields, trace data, route jargon, or policy/debug metadata.
- Avoids placeholder/busy messages when the runtime is healthy.
- Avoids forced chooser UX for ordinary chat.
- Avoids generic model self-disclaimers such as "I am an AI language model" or "I have no sensory perception".
- Answers grounded local-system questions from real runtime/tool state.
- Answers ordinary non-live knowledge questions directly, or fails gracefully without meta boilerplate.
- Keeps continuity across short follow-ups and topic returns.
- Keeps Telegram parity with WebUI/API on equivalent prompts.

## Barrage Matrix

| ID | Category | Surface | Scenario | Status | Last run notes |
|---|---|---|---|---|---|
| AIR-001 | Social | WebUI | `hi` then capability follow-up | PASS | Friendly reply, no chooser prompt. |
| AIR-002 | Social | Telegram | `hi` then capability follow-up | PASS | Parity with WebUI. |
| AIR-003 | Social | WebUI | typo greeting like `herllo` | PASS | Fast-path social handling added. |
| AIR-004 | Social | Telegram | presence check like `are you really there?` | PASS | Parity smoke verified. |
| AIR-005 | Terse | WebUI | `say hi` | PASS | No ambiguity chooser. |
| AIR-006 | Terse | Telegram | `say hi` | PASS | Live barrage clean. |
| AIR-007 | Knowledge | WebUI | ordinary factual question like `What colour is a bluejay?` | PASS | Now degrades consistently without model boilerplate. |
| AIR-008 | Knowledge | Telegram | same factual question | PASS | Parity smoke clean after stable promotion. |
| AIR-009 | Runtime truth | WebUI | runtime status and readiness follow-up | PASS | Covered by release and real-world smoke. |
| AIR-010 | Runtime truth | Telegram | runtime status and readiness follow-up | PASS | Live barrage clean. |
| AIR-011 | Hardware truth | WebUI | RAM/VRAM inspection | PASS | Real-world smoke clean. |
| AIR-012 | Hardware truth | Telegram | RAM/VRAM inspection | PASS | Live barrage clean. |
| AIR-013 | Filesystem | WebUI | list/read a canary file | PASS | Real-world smoke clean. |
| AIR-014 | Filesystem | Telegram | list/read a canary file | PASS | Live barrage clean. |
| AIR-015 | Memory | WebUI | preference recall | PASS | Real-world smoke clean. |
| AIR-016 | Memory | WebUI | resume prior task after interruption | PASS | Existing viability smoke covers. |
| AIR-017 | Memory | Telegram | resume prior task after interruption | PASS | Live barrage clean after proxy-safe memory fallback patch. |
| AIR-018 | Topic shift | WebUI | plan, joke, then return to plan | PASS | Existing viability smoke covers. |
| AIR-019 | Topic shift | Telegram | plan, joke, then return to plan | PASS | Live barrage clean. |
| AIR-020 | Confirmation | WebUI | temporary model switch with explicit confirm | PASS | Existing viability smoke covers. |
| AIR-021 | Confirmation | Telegram | temporary model switch with explicit confirm | PASS | Covered indirectly by continuity parity and command-path hardening; no current barrage failure. |
| AIR-022 | Capability routing | WebUI | skill/capability pack question | PASS | Real-world smoke clean. |
| AIR-023 | Capability routing | WebUI | direct capability ask like `Talk to me out loud` | PASS | Live barrage now uses cleaner install-details wording instead of stiffer preview phrasing. |
| AIR-024 | Capability routing | Telegram | direct capability ask like `Talk to me out loud` | PASS | Telegram parity now uses the same cleaner install-details wording. |
| AIR-025 | Coding | WebUI | `Help me code` style request | PASS | Live barrage now returns one concise coding-helper reply with no duplicated intro text. |
| AIR-026 | Coding | Telegram | `Help me code` style request | PASS | Telegram parity now uses the same concise coding-helper wording. |
| AIR-027 | Politeness | WebUI | `thanks` / `ok` acknowledgements | PASS | Live barrage clean. |
| AIR-028 | Politeness | Telegram | `thanks` / `ok` acknowledgements | PASS | Live barrage clean. |
| AIR-029 | Anti-slop | WebUI | ensure no `AI language model` disclaimer leak | PASS | New guards and fallback added. |
| AIR-030 | Anti-slop | Telegram | ensure no `AI language model` disclaimer leak | PASS | Parity smoke clean. |
| AIR-031 | Anti-control-plane | WebUI | ensure no unsolicited health/update chatter in normal chat | PASS | Notification suppression and clarify fixes landed. |
| AIR-032 | Anti-control-plane | Telegram | ensure no unsolicited health/update chatter in normal chat | PASS | Notification path fixed; barrage and parity checks clean. |
| AIR-033 | Continuity polish | WebUI | rewind prompt like `what are we doing?` after task setup | PASS | Live barrage now returns natural recap phrasing. |
| AIR-034 | Continuity polish | Telegram | correction/rewind prompt like `no, go back and explain the larger task` | PASS | Telegram parity verified in the live barrage. |
| AIR-035 | Mixed-session soak | WebUI | long mixed session across planning, runtime, memory, files, capability, and recap | PASS | Live barrage reached stable actionable next-step phrasing. |
| AIR-036 | Mixed-session soak | Telegram | same long mixed session through the proxy bridge | PASS | Live barrage and Telegram viability scenario both passed. |
| AIR-037 | Typo resilience | WebUI | malformed model query like `what model are you uding` | PASS | Live barrage now routes to model status instead of runtime fallback. |
| AIR-038 | Typo resilience | Telegram | malformed runtime query like `r u heathy right now` | PASS | Live sweep now routes to runtime status with no model boilerplate leak. |
| AIR-039 | Typo resilience | WebUI | malformed continuity prompts like `go bak to the day plan` and `what shoud we do next` | PASS | Live barrage now stays on deterministic memory/context paths. |
| AIR-040 | Continuity probe | Telegram | `I'm testing whether you can stay coherent through a long chat.` | PASS | Live `/chat` now returns a deterministic assistant reply instead of `/brief` drift or `internal_error`. |
| AIR-041 | Planning front-door | WebUI | `help me plan my day` | PASS | Live `/chat` now returns a real planning reply instead of runtime-state fallback. |
| AIR-044 | Generic-chat tone | WebUI | `say what you do in one sentence, but keep it natural` | PASS | Now routes to assistant capabilities instead of falling into confused generic chat. |
| AIR-045 | Generic-chat tone | WebUI | `i need help thinking through something messy, but keep it simple` | PASS | Now returns a short assistant-owned collaboration reply instead of canned coaching boilerplate. |
| AIR-046 | Capability tone | WebUI | custom helper proposal for a new assistant workflow | PASS | Now uses collaborative wording with a clean task summary instead of template-like prompt echo. |

## Current Worklist

1. Keep expanding long mixed-session cases that try to force recap drift, tool drift, or context loss after many turns.
2. Tighten robotic generic-chat openings that are correct but still read too much like one-shot model narration.
3. Keep probing for shared-state failures under realistic overlapping surfaces, but run the release barrage sequentially by default.
4. Promote every newly found live miss into a deterministic regression or barrage scenario before moving on.

## Latest Notes

- 2026-04-23/24: `scripts/assistant_interaction_barrage.py --base-url http://127.0.0.1:8765 --timeout 15` reached `27/27 PASS` on the live stable runtime.
- 2026-04-24: `scripts/assistant_interaction_barrage.py --base-url http://127.0.0.1:8765 --timeout 15` reached `29/29 PASS` on the live stable runtime after adding mixed-session soak coverage.
- 2026-04-24: typo-routing hardening closed a real Telegram miss for `what model are you uding`; the live barrage reached `32/32 PASS`, and an additional typo-heavy Telegram sweep no longer leaks generic-chat boilerplate for malformed model/runtime/memory prompts.
- 2026-04-24: `scripts/assistant_viability_smoke.py --base-url http://127.0.0.1:8765 --surface telegram --scenario long_human_like_session_telegram --timeout 180 --retry-attempts 2` passed live after recap prompts were rerouted away from correction phrasing.
- 2026-04-24: live `/chat` probes closed two fresh regressions:
  - `help me plan my day` now stays on `plan_day` and returns a planning reply instead of `runtime_state_unavailable`
  - `I'm testing whether you can stay coherent through a long chat.` now returns a deterministic assistant reply instead of drifting into `/brief` behavior or leaking `internal_error`
- 2026-04-24: fresh post-fix live barrage on stable runtime is clean again:
  - `python scripts/assistant_interaction_barrage.py --base-url http://127.0.0.1:8765 --timeout 25`
  - result: `32/32 PASS`
  - previously failing WebUI typo-memory and file-read cases now pass in the suite, not just as isolated repros
- 2026-04-24: overlapping live validation exposed a real shared-state failure class in `/chat`:
  - concurrent live suites could trigger `OperationalError` during early generic-chat turns even when the same turns passed individually
  - `/chat` now serializes the orchestrator-backed path through the runtime lock, with a new concurrency regression in `tests/test_runtime_concurrency.py`
- 2026-04-24: recap phrasing gap closed for `summarize where we left this in one sentence`:
  - the phrase now routes to working-context memory instead of drifting into a generic plan summary
  - fresh live barrage on stable runtime reached `34/34 PASS`
  - fresh Telegram long-session viability also passed again after the fix
- 2026-04-24: generic-chat tone pass closed a straightforward conversational miss:
  - `Say what you do in one sentence, but keep it natural.` no longer falls into confused generic chat
  - the phrase now routes to `assistant_capabilities` on stable, with a new barrage scenario and route regression
- 2026-04-24: another tone pass closed a collaboration-style generic-chat miss:
  - `I need help thinking through something messy, but keep it simple.` no longer falls into long canned coaching text
  - the phrase now routes to a short deterministic assistant reply, with a new barrage scenario and focused regressions
- 2026-04-24: capability-gap reply cleanup closed a repetitive coding response:
  - `Help me code` no longer repeats its own partial-help intro on WebUI or Telegram
  - the capability renderer now owns that intro once, with new regression coverage and a clean `36/36 PASS` live barrage
- 2026-04-24: custom-helper proposal wording was tightened in two passes:
  - the reply now says `The simplest way to add it...` and `I can sketch that with you` instead of the stiffer old template wording
  - prompt scaffolding like `make an assistant that ...` is stripped before rendering the helper summary
  - fresh live barrage on stable runtime reached `37/37 PASS`
- 2026-04-24: install-preview capability wording was tightened too:
  - single-pack recommendations now say `most practical option here` and `install details` instead of the stiffer `best fit for this machine` and `install preview`
  - lighter-weight tradeoff text now reads more naturally
  - fresh live barrage on stable runtime stayed clean at `37/37 PASS`
- Continuity voice polish is now live:
  - working-context rewind replies use natural assistant phrasing
  - generic `we are testing ...` seed turns now get a fast deterministic acknowledgement instead of a slow empty-prone generic chat path
- Mixed-session soak is now live:
  - WebUI and Telegram both preserve continuity across planning, runtime, memory, file-read, capability, and recap turns
  - `what should we do next?` now uses a deterministic working-context next-step response instead of timing out through generic chat
  - `summarize the work` style prompts now stay on the working-context path instead of leaking correction phrasing
- Next slice is tone polish for robotic generic-chat openings that still sound more like model narration than an assistant.
- The barrage found and closed three concrete issues:
  - capability-recognition miss for direct asks like `read something to me`
  - Telegram proxy fallback failure for `/memory`-style phrases when the harness runs without a bound runtime object
  - overly narrow barrage expectations for polite acknowledgements and working-context recap phrasing
- Short-turn ambiguity regressions were fixed.
- Low-signal autopilot notification chatter was suppressed from Telegram delivery.
- Social-turn fast path now handles more real-world variants.
- Generic chat disclaimer leakage is now contained and normalized instead of surfacing raw model boilerplate.
- Capability recommendation detection was tightened after the extended release suite found a real regression.
