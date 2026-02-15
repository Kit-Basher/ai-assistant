# Good Assistant Foundation Handoff

## v0.2.5 — Project Mode (Per-Thread Preset)

Purpose: deterministic per-thread behavior preset for focused project work.

Command syntax:
- `/project_mode`
- `/project_mode on`
- `/project_mode off`

Runtime behavior when ON:
- `show_summary` forced off
- `show_next_action` forced on
- `terse_mode` forced off
- `commands_in_codeblock` forced on
- Plan threshold lowered (`>=1` imperative sentence)
- Options shown when `>=2` distinct imperative first verbs

Clarifications:
- overrides are runtime-only and do not mutate stored prefs
- intercept replies are unaffected
- no new LLM usage introduced

Verification:

```bash
pytest -q
python3 -m agent.epistemics.canary
python3 -m agent.friction.canary
```

## v0.2.4 — Thread Workflow Template (/thread_new)

Purpose: explicit thread lifecycle initialization for intentional setup.

Command syntax:
- `/thread_new "<label>"`
- `/thread_new "<label>" --terse on --summary off`
- `/thread_new "<label>"`
  - `- bullet`
  - `Open: next step`

Behavior:
- deterministic thread id generation
- sets active thread
- persists label
- optional per-thread prefs via flags
- optional initial anchor from message body
- no LLM usage
- no question marks in output
- `skip_friction_formatting` applied

Verification:

```bash
pytest -q
python3 -m agent.epistemics.canary
python3 -m agent.friction.canary
```

## v0.2.2 — Path B Continuity (Plan/Options/Anchors/Resume)

Since `v0.2.1-good-assistant-foundation`, the following deterministic continuity features were added:

- Deterministic `Plan:` section on pass-path replies when planning-oriented (`agent/friction/plan.py`).
- Deterministic `Options:` section for safe decision forks without `?` (`agent/friction/options.py`).
- Per-thread anchors/checkpoints:
  - `/anchor <title>` (and `/checkpoint` alias) to save checkpoints
  - `/anchors` to list recent checkpoints
  - `/resume` for anchor-driven kickoff
- `/anchors` continuity header now includes:
  - `Current focus: <latest title>`
  - `Next: <latest open text>` when available
- `/resume` is fully deterministic and anchor-derived:
  - no LLM call
  - no new claims
  - no cross-thread blending
  - no `?` in output

Verification:

```bash
pytest -q
python3 -m agent.epistemics.canary
python3 -m agent.friction.canary
```

## v0.2.3 — Thread Navigation (threads index, switch, labels)

Thread navigation now supports explicit, deterministic context control:

- `/threads` lists recent threads with:
  - `thread_id`
  - `last_ts`
  - `Label: <label|none>`
  - `Focus: <latest anchor title|none>`
  - line format: `1) <thread_id>  <last_ts>  Label: <label|none>  Focus: <title|none>`
- `/thread_use <thread_id>` explicitly switches active thread without cross-thread blending.
- `/thread_label <label>` sets a persisted per-thread label.
- `/thread_unlabel` clears the current thread label.
- Determinism/safety:
  - outputs are normalized to remove `?`
  - command replies use `skip_friction_formatting`
  - no LLM calls or inferred labels

Verification:

```bash
pytest -q
python3 -m agent.epistemics.canary
python3 -m agent.friction.canary
```

## 1) Overview

This checkpoint captures the stable "Good Assistant foundation" on branch `brief-v0.2-clean`.

Scope included:
- Epistemic Integrity Phase 1 (gate, thread scope, provenance, monitoring, canaries)
- Friction Reduction Phase 2 (summary + next-action, deterministic guards)
- Explicit formatting preferences (global + per-thread)

## 2) Epistemic Integrity (Phase 1)

- Runtime gate: `agent/epistemics/gate.py::apply_epistemic_gate` validates candidate contract and detector output before user-visible response.
- Intercept spec is locked to exact 3-line format:
  1. `I’m not sure.`
  2. blank line
  3. exactly one clarifying question line with exactly one `?`
- Thread scope and memory boundaries are enforced in `agent/epistemics/detectors.py` and context construction in `agent/orchestrator.py`.
- Claim provenance is enforced in `agent/epistemics/contract.py` and `agent/epistemics/types.py` (`user_turn_id`, `memory_id`, `tool_event_id`).
- Internal monitoring/reporting:
  - monitor: `agent/epistemics/monitor.py`
  - report command implementation: `agent/epistemics/report.py`
  - orchestrator command: `/epistemics_report`
- Epistemics canary suite:
  - cases: `agent/epistemics/canary_cases.py`
  - runner: `agent/epistemics/canary.py`
  - run: `python3 -m agent.epistemics.canary`

## 3) Friction Reduction (Phase 2)

- Next-action line (`Next: ...`):
  - logic: `agent/friction/next_action.py`
  - applied only on epistemic pass path
  - env toggle: `FRICTION_NEXT_ACTION` (set `0`/`off` to disable)
- Summary header (`In short: ...`):
  - logic: `agent/friction/summary.py`
  - applied only on epistemic pass path when body length conditions are met
  - env toggle: `FRICTION_SUMMARY` (set `0`/`off` to disable)
- Preferences:
  - global prefs table: `user_prefs`
  - per-thread overrides table: `thread_prefs`
  - data access and resolution: `agent/prefs.py`
- Friction canary:
  - cases: `agent/friction/canary_cases.py`
  - runner: `agent/friction/canary.py`
  - run: `python3 -m agent.friction.canary`

## 4) Key Commands (user-facing)

- `/prefs`
- `/prefs_set <key> <on|off>`
- `/prefs_reset`
- `/prefs_thread`
- `/prefs_thread_set <key> <on|off>`
- `/prefs_thread_reset`
- `/epistemics_report`

## 5) Operational Notes

Environment variables currently supported by this foundation:
- `PASS_SCORE_THRESHOLD`
- `ROLLING_WINDOW_SIZE`
- `SPIKE_THRESHOLD`
- `SOFT_CROSS_THREAD_PHRASES`
- `FRICTION_NEXT_ACTION`
- `FRICTION_SUMMARY`

DB tables added for preferences:
- `user_prefs` (global formatting prefs)
- `thread_prefs` (per-thread overrides)

Schema location: `memory/schema.sql`.

## 6) Verification Checklist

Run all:

```bash
pytest -q
python3 -m agent.epistemics.canary
python3 -m agent.friction.canary
```

## 7) Recent Commits

- `b42e4bc` feat(epistemics): enforce uncertainty gate in orchestrator
- `239db04` fix(epistemics): lock intercept format and clarify leakage
- `fe877c8` feat(epistemics): add tunable thresholds and internal report
- `39f7653` feat(epistemics): enforce explicit thread scope and memory boundaries
- `19314e5` feat(epistemics): add internal claim provenance tagging
- `4d50151` test(epistemics): add regression canary suite
- `148f152` feat(friction): add safe next-action line on epistemic pass
- `7fd508e` feat(friction): add deterministic intent compression summary
- `4deab3e` feat(epistemics): improve clarifying question templates
- `f69ec4a` test(friction): add regression canary suite
- `eec330f` feat(prefs): add explicit formatting preferences
- `e9ce4c8` feat(prefs): add per-thread preference overrides
