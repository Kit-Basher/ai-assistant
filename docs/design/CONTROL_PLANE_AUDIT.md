# Control Plane Audit

Date: 2026-03-14

## Executive Summary

The current control plane does not have one fully unified authority for "what model/provider is in use, whether it is healthy, and whether the system should recommend or apply a change."

What is authoritative today depends on the question:

- Configured defaults are persisted in `AgentRuntime.registry_document["defaults"]` and read through `AgentRuntime.get_defaults()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py).
- Provider and model health are primarily held in `self._health_monitor.state`, overlaid into status surfaces through `_canonical_runtime_router_snapshot()` and read deterministically through `RuntimeTruthService` in [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py).
- Runtime readiness is exposed through `ready_status()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py).
- Ordinary chat routing uses the canonical selector stack and chooses a model per request.
- Background recommendation systems (`model_watch`, `model_scout`) are not fully grounded in the same live health/auth truth. In particular, Model Watch currently evaluates proposals against synthetic `"ok"` health and `require_auth=False`.

That means the following can all be true at once:

- configured default says provider/model `A`
- health says `A` is degraded or down
- a given chat request may still attempt or report `A`
- model watch/model scout may recommend `C`
- background notifications may proactively report a recommendation or automated change that does not match the runtime-truth view the user gets from `/ready` or deterministic status chat routes

The main operational risk is not one broken function. It is that multiple subsystems mutate or describe overlapping state with different authority and different safety assumptions.

## Architecture Map of State Authority

```text
Persisted registry defaults
  AgentRuntime.registry_document["defaults"]
        |
        v
  AgentRuntime.get_defaults()
        |
        +--------------------------+
        |                          |
        v                          v
RuntimeTruthService         Default-switch policy
current_chat_target_*       choose_best_default_chat_candidate()
provider_status()           autoconfig / self-heal / bootstrap / model watch
runtime_status()
        |
        v
User-facing deterministic status
(/ready, provider/model/runtime chat answers)


Health monitor state
  self._health_monitor.state
        |
        v
  _canonical_runtime_router_snapshot()
        |
        +--------------------------+
        |                          |
        v                          v
  /llm/status                RuntimeTruthService provider/model reads


Recommendation systems
  Model Scout                Model Watch
  run_model_scout()          _build_model_watch_proposal()
        |                          |
        v                          v
Proactive Telegram          Proactive Telegram
notifications               notifications / proposal state
```

### Source of truth by topic

#### Configured default provider/model

Primary authority:

- `AgentRuntime.registry_document["defaults"]`
- read by `AgentRuntime.get_defaults()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- mutated by `AgentRuntime.update_defaults()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Notes:

- `default_provider` and `chat_model` / `default_model` are persisted control-plane state.
- `resolved_default_model` is derived, not separately authoritative.

#### Effective active chat target

There is no single persisted authority.

Current canonical read adapter:

- `RuntimeTruthService.current_chat_target_status()` in [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)

Fallback synthesis path:

- `RuntimeTruthService.current_chat_target()`
- `AgentRuntime._current_chat_target()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Notes:

- The "effective target" is synthesized from defaults, registry model rows, provider/model health, and runtime/router snapshots.
- This is the first major drift point: configured default and effective healthy target are not stored as one reconciled object.

#### Provider health / model health

Primary authority:

- `self._health_monitor.state`
- `LLMHealthMonitor` in [agent/llm/health.py](/home/c/personal-agent/agent/llm/health.py)

Important overlays:

- `_canonical_runtime_router_snapshot()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `RuntimeTruthService._provider_health_row()` / `_model_health_row()` in [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)

Important direct writes:

- `_record_authoritative_provider_success()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- called by `test_provider()` and `configure_openrouter()`

Notes:

- This is the authoritative live-health layer for status surfaces.
- Not every recommendation system uses it faithfully.

#### Current runtime status

Fast status authority:

- `ready_status()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Heavier status authority:

- `llm_status()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Deterministic chat authority:

- `RuntimeTruthService.runtime_status()` in [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)

Notes:

- `/ready` is intended to be fast and cached.
- `/llm/status` is a heavier control-plane snapshot.
- Runtime-truth chat answers mostly sit on top of these.

#### Recommendations / model watch / model scout

There is no single recommendation authority.

Relevant paths:

- `choose_best_default_chat_candidate()` in [agent/llm/default_model_policy.py](/home/c/personal-agent/agent/llm/default_model_policy.py)
- `run_model_scout()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `_build_model_watch_proposal()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Notes:

- Autoconfig, self-heal, and bootstrap now share the unified default-switch selector.
- Model Watch proposal generation does not use canonical live health truth; it uses synthetic all-`ok` health and `require_auth=False`.
- Model Scout change detection is based on registry/router/inventory deltas and suggestion scores, not canonical runtime truth.

## Mutation Path Inventory

### Defaults / configured provider-model mutations

| Path | File / function | Trigger | User or background | Notifications |
|---|---|---|---|---|
| Update defaults | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `AgentRuntime.update_defaults()` | direct API/internal call | both | emits `provider_switch` and `default_model_change` runtime events |
| Set default chat model | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `AgentRuntime.set_default_chat_model()` | setup flow, manual switch, automation callers | both | indirect via `update_defaults()` |
| Rollback defaults | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `AgentRuntime.rollback_defaults()` | manual/admin flow | user | indirect status/notification side effects |
| Configure local chat model | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `AgentRuntime.configure_local_chat_model()` | setup flow | user | may update default after provider test |
| Configure OpenRouter | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `AgentRuntime.configure_openrouter()` | setup flow | user | may update provider entry, secret, health, and optionally default |

### Background automatic default mutations

| Path | File / function | Trigger | User or background | Notifications |
|---|---|---|---|---|
| Autoconfig apply | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `llm_autoconfig_apply()` | scheduler/manual | background or user | yes, via autopilot notification cycle |
| Self-heal apply | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `llm_self_heal_apply()` | scheduler/manual | background or user | yes, via autopilot notification cycle |
| Bootstrap apply | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `llm_autopilot_bootstrap()` | scheduler/manual | background or user | yes, via autopilot notification cycle |
| Capabilities reconcile apply | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `llm_capabilities_reconcile_apply()` | scheduler/manual | background or user | yes, via autopilot notification cycle |
| Cleanup / hygiene apply | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `llm_cleanup_apply()` / `llm_hygiene_apply()` | scheduler/manual | background or user | yes, via autopilot notification cycle |

### Health-state mutations

| Path | File / function | Trigger | User or background | Notifications |
|---|---|---|---|---|
| Health monitor probe cycle | [agent/llm/health.py](/home/c/personal-agent/agent/llm/health.py) `LLMHealthMonitor.run_once()` | scheduler / polling | background | indirect, through status and autopilot notifications |
| Wrapped runtime health run | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `run_llm_health()` | scheduler/manual | background or user | indirect |
| Authoritative provider success write | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `_record_authoritative_provider_success()` | successful provider test/setup | user-triggered | no direct Telegram send, but affects later status |
| Provider test | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `test_provider()` | manual/setup/automation | both | updates authoritative health |

### Recommendation-state mutations

| Path | File / function | Trigger | User or background | Notifications |
|---|---|---|---|---|
| Model Scout run | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `run_model_scout()` | scheduler/manual | background or user | yes, proactive Telegram if change detected |
| Model Watch run | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `run_model_watch_once()` | scheduler/manual | background or user | yes, proactive Telegram proposal |
| Persist Model Watch proposal | [agent/api_server.py](/home/c/personal-agent/agent/api_server.py) `_persist_model_watch_proposal_state()` | model watch proposal created | background or user | stored pending state, may later prompt user |

## Notification Path Inventory

### Proactive Telegram/user-facing notifications

#### Autopilot / automation notifications

Primary path:

- `_process_scheduler_notification_cycle()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- delivery via `_deliver_autopilot_notification()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Message shaping:

- `build_notification_from_state_diff()` in [agent/llm/notifications.py](/home/c/personal-agent/agent/llm/notifications.py)
- `summarize_notification_message()` in [agent/ux/llm_fixit_wizard.py](/home/c/personal-agent/agent/ux/llm_fixit_wizard.py)

Observed user-facing fallback:

- `"I updated LLM settings in the background and the service is still running."`

Why it appears:

- the notification diff was real enough to send
- the summarizer did not find a more specific friendly summary

#### Model Scout proactive notifications

Primary path:

- `run_model_scout()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `_model_scout_change_events()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `_notify_model_scout_change()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Observed user-facing message family:

- `"Model Scout update: better default candidate..."`

Important audit point:

- these notifications are driven by suggestion deltas and registry/router observations
- they are not gated by canonical runtime-truth health in the same way status answers are

#### Model Watch proactive notifications

Primary path:

- `_build_model_watch_proposal()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `_notify_model_watch_proposal()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Observed user-facing message family:

- `"Model Watch found a better default candidate..."`

Important audit point:

- proposal generation currently assumes watched providers/models are `"ok"` and does not require auth
- the proposal can therefore disagree with live health and with runtime-truth surfaces

#### Setup / completion notices

These are mostly reactive chat replies, not unsolicited notifications.

Relevant user-triggered flows:

- `configure_openrouter()`
- `configure_local_chat_model()`
- deterministic setup/status replies through `RuntimeTruthService` and `Orchestrator`

## Drift / Inconsistency Findings

### Finding 1: There is no single reconciled "effective control-plane truth"

Configured defaults, health truth, effective target, default-switch policy, model-watch proposals, and model-scout suggestions are all computed in different places.

Impact:

- different surfaces can be individually correct according to their own source
- but still disagree with each other

### Finding 2: Model Watch proposals are not grounded in live health or auth

Exact code path:

- `_build_model_watch_proposal()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Current behavior:

- builds `watch_health_summary` with every watched provider/model forced to `status: "ok"`
- calls `choose_best_default_chat_candidate(... require_auth=False, health_summary=watch_health_summary)`

Impact:

- the system can recommend a provider/model that is actually unhealthy, unreachable, or not currently usable
- this directly explains scenarios like "OpenRouter was down earlier, but Model Watch later recommended it"

### Finding 3: Model Scout change notifications are advisory, not runtime-truth grounded

Exact code path:

- `run_model_scout()`
- `_model_scout_change_events()`
- `_notify_model_scout_change()`

Current behavior:

- looks at defaults, router snapshot, usage stats, and scout suggestion scores
- emits proactive Telegram notices if the top suggestion differs from the current default

Impact:

- a "better default candidate" notification can appear even when live health says the provider is degraded or down

### Finding 4: Status surfaces and recommendation surfaces do not share one precedence rule

Today:

- runtime-truth status surfaces prefer configured defaults over a separately persisted "effective fallback" state because no such authoritative persisted fallback state exists
- provider/model health overlays try to qualify the configured default
- recommendation systems may propose alternatives that are not consistent with that health view

Impact:

- user can see:
  - configured default `A`
  - provider `A` down
  - model watch recommends `C`
  - model scout recommends `C`
  - autopilot background notification says it changed something

There is no single "winner" explained consistently across all channels.

### Finding 5: Background apply/send policies are permissive on loopback

Exact code paths:

- `compute_notification_send_policy()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `compute_self_heal_apply_policy()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- `compute_autopilot_bootstrap_apply_policy()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

Current behavior:

- if explicit config is unset and the service is loopback-bound, apply/send defaults become `loopback_auto`

Impact:

- background changes and proactive Telegram notifications can happen without an explicit operator approval step
- this is operationally surprising even if it is intentional by current policy

### Finding 6: Autopilot notification summaries describe diffs, not authoritative semantic truth

Exact code path:

- `_process_scheduler_notification_cycle()`
- `build_notification_from_state_diff()`
- `summarize_notification_message()`

Current behavior:

- state diffs produce messages
- summarizer maps some diff lines to human phrases
- unrecognized diffs fall back to a vague operational message

Impact:

- users can receive unsolicited messages like:
  - `"I updated LLM settings in the background and the service is still running."`
- without being told which subsystem changed what and why

## Consistency Scenarios That Can Happen Today

### Scenario A: Provider is down, but still reported as configured/current

How it happens:

1. `registry_document["defaults"]` still points to provider/model `A`
2. health monitor marks provider `A` down
3. runtime-truth status answers still describe `A` as the configured/current target, but with degraded qualification
4. some user-facing surfaces may still say "currently using A" because they are describing the configured target, not a persisted effective fallback

This is partially mitigated in `RuntimeTruthService.current_chat_target_status()` and `provider_status()`, but the concept itself is still split.

### Scenario B: Provider is down, but Model Watch recommends it anyway

How it happens:

1. provider/model has changed catalog or delta rows
2. `_build_model_watch_proposal()` evaluates only the watch set
3. it forces watched provider/model health to `"ok"`
4. it may recommend the down provider/model as the new default

This is a real control-plane inconsistency, not just wording drift.

### Scenario C: Provider is down, but Model Scout recommends it later

How it happens:

1. `ModelScout.run()` generates a new suggestion with a better score
2. `_model_scout_change_events()` sees the top suggestion differs from the current default
3. `_notify_model_scout_change()` proactively sends a Telegram recommendation

Because this path is not driven by canonical runtime-truth health, it can recommend a model/provider the runtime is not currently happy with.

### Scenario D: Background automation applies changes and user only sees a vague notice

How it happens:

1. autoconfig/self-heal/bootstrap/capabilities reconcile makes a state mutation
2. `_process_scheduler_notification_cycle()` detects a diff
3. `summarize_notification_message()` cannot map it to a specific user phrase
4. user receives `"I updated LLM settings in the background and the service is still running."`

This is not necessarily wrong, but it is poor observability and easy to misinterpret.

## Arbitration / Precedence Today

There is no single explicit arbitration layer for all control-plane disagreements.

In practice, precedence is distributed:

1. Configured defaults
   - `registry_document["defaults"]`
   - wins for "what is configured"

2. Health truth
   - `self._health_monitor.state`
   - wins for "is that provider/model healthy"

3. Runtime-truth chat status
   - `RuntimeTruthService`
   - tries to explain configured defaults plus health overlay

4. Per-request routing
   - inference router + selector stack
   - wins for "what model this one generic chat request will try"

5. Recommendations
   - model watch / model scout
   - advisory or proposal-producing, but currently not fully subordinated to runtime-truth health

If these disagree, user-facing answers vary by surface:

- `/ready` and deterministic runtime/provider/model questions: mostly configured default plus health qualification
- `/llm/status`: heavier snapshot of defaults, visible models, and health
- generic chat telemetry: may expose the actual selected model for that specific request
- model scout/watch notifications: recommendation or proposal state, not reconciled truth

## Immediate Safe Mitigations

These are safe operational mitigations based on the current code, before any deeper control-plane unification work.

### Should Model Scout be disabled for now?

Recommended: **disable proactive Model Scout notifications now**.

Reason:

- `run_model_scout()` and `_notify_model_scout_change()` can recommend defaults that are not grounded in canonical live health
- the feature is advisory, not required for runtime correctness

Safe interim stance:

- keep manual Model Scout inspection if desired
- disable proactive Telegram delivery until recommendation authority is reconciled

### Should autopilot default changes be disabled for now?

Recommended: **disable automatic apply for self-heal and bootstrap until control-plane authority is unified**.

Reason:

- defaults can change in the background under loopback-auto policy
- proactive messages about those changes are not always clearly attributable
- even with the unified default-switch selector, background apply still operates in a control plane where status, health, and recommendation systems are not fully reconciled

### Should background Telegram notifications be suppressed for now?

Recommended: **yes**.

Reason:

- unsolicited messages currently come from multiple subsystems
- they are not all grounded in one canonical truth layer
- they create more confusion than observability when state is drifting

Recommended immediate suppressions:

- `AUTOPILOT_NOTIFY_ENABLED=0`
- disable model scout proactive sends
- disable model watch proactive sends

### Additional safe immediate mitigations

- Disable `model_watch_enabled` for automatic proposal generation if operator experience matters more than advisory discovery right now.
- Set `LLM_SELF_HEAL_ALLOW_APPLY=0` and `LLM_AUTOPILOT_BOOTSTRAP_ALLOW_APPLY=0` until apply authority is intentionally re-enabled.
- Keep deterministic runtime/provider/model answers as the operator-facing truth channel.

## Concrete Recommendations

### Recommendation 1: Split "configured default" from "effective target" more explicitly

The runtime already partly does this implicitly, but not as a single authoritative object.

Need:

- one canonical read helper that returns:
  - configured default
  - effective routable target
  - health qualification
  - reason if configured default is not currently viable

This should become the basis for:

- `RuntimeTruthService.current_chat_target_*`
- `/ready`
- `/llm/status` active target fields
- user-facing "what model are you using?" answers

### Recommendation 2: Re-ground Model Watch in canonical live health and auth

Minimum requirement:

- stop forcing watched providers/models to `"ok"`
- stop using `require_auth=False` for recommendation quality that is shown as operational advice

If catalog-only recommendation behavior is still wanted, label it explicitly as:

- catalog advisory
- not current healthy runtime candidate

### Recommendation 3: Gate Model Scout notifications on canonical runtime health

Before sending a proactive "better default candidate" message:

- reconcile the candidate against live health/auth and current defaults
- suppress proactive delivery if the candidate is not presently healthy and usable

### Recommendation 4: Require explicit opt-in for background apply and proactive notifications

Loopback-auto is convenient, but it is the wrong default for a system where multiple overlapping control-plane actors still exist.

Safer near-term stance:

- explicit opt-in for apply
- explicit opt-in for proactive Telegram notifications

### Recommendation 5: Make notification summaries name the subsystem and exact mutation

Current notification summaries are often too vague.

At minimum, notifications should identify:

- which subsystem acted (`self_heal`, `bootstrap`, `autoconfig`, `model_watch`, `model_scout`)
- whether it only recommended or actually applied
- which default/provider/model changed

## Minimal Next Patch Plan

This is the smallest safe patch sequence suggested by this audit. It does not require a broad redesign.

1. Suppress proactive Telegram notifications from `model_scout` and `model_watch`.
   - keep state and audit records
   - stop sending unsolicited recommendation messages

2. Disable automatic default changes from self-heal/bootstrap by default on loopback.
   - require explicit config opt-in instead of `loopback_auto`

3. Make `_build_model_watch_proposal()` use canonical live health/auth.
   - remove synthetic all-`ok` health
   - remove `require_auth=False` for runtime-facing proposals

4. Add one canonical "effective control-plane state" helper.
   - configured default
   - effective healthy target
   - current health qualification
   - candidate recommendation, if any

5. Repoint proactive notifications and user-facing current-target answers to that helper.

## Immediate Safe Mitigations

Current recommended operational baseline:

- run with `AGENT_SAFE_MODE=1`
- keep proactive Scout/Watch Telegram notifications suppressed
- keep self-heal/bootstrap auto-apply disabled unless explicitly re-enabled

Why:

- safe mode pins one local chat target
- background control-plane mutation is suppressed
- Telegram and API ordinary chat stay aligned on the same effective target
- deterministic runtime/provider/setup/doctor/system-observe prompts still work

## Files / Functions Involved

### State authority and runtime truth

- [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)
  - `current_chat_target_status()`
  - `current_chat_target()`
  - `provider_status()`
  - `providers_status()`
  - `runtime_status()`

- [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `get_defaults()`
  - `update_defaults()`
  - `ready_status()`
  - `llm_status()`
  - `_canonical_runtime_router_snapshot()`
  - `_record_authoritative_provider_success()`
  - `test_provider()`
  - `set_default_chat_model()`
  - `configure_local_chat_model()`
  - `configure_openrouter()`

### Default switching and recommendations

- [agent/llm/default_model_policy.py](/home/c/personal-agent/agent/llm/default_model_policy.py)
  - `choose_best_default_chat_candidate()`

- [agent/llm/autoconfig.py](/home/c/personal-agent/agent/llm/autoconfig.py)
- [agent/llm/self_heal.py](/home/c/personal-agent/agent/llm/self_heal.py)

- [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `llm_autoconfig_apply()`
  - `llm_self_heal_apply()`
  - `llm_autopilot_bootstrap()`
  - `run_model_scout()`
  - `_model_scout_change_events()`
  - `_notify_model_scout_change()`
  - `_build_model_watch_proposal()`
  - `_notify_model_watch_proposal()`

### Notifications and summaries

- [agent/llm/notifications.py](/home/c/personal-agent/agent/llm/notifications.py)
- [agent/llm/notify_delivery.py](/home/c/personal-agent/agent/llm/notify_delivery.py)
- [agent/ux/llm_fixit_wizard.py](/home/c/personal-agent/agent/ux/llm_fixit_wizard.py)
  - `summarize_notification_message()`

### Policy defaults

- [agent/config.py](/home/c/personal-agent/agent/config.py)
- [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `compute_notification_send_policy()`
  - `compute_self_heal_apply_policy()`
  - `compute_autopilot_bootstrap_apply_policy()`
