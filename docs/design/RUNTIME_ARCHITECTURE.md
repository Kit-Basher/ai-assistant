# Runtime Architecture

This document describes the current runtime architecture of Personal Agent.

Source of truth is the code, not older handoff notes. The most relevant code
paths are:

- `agent/api_server.py`
- `agent/control_plane.py`
- `agent/orchestrator.py`
- `agent/runtime_truth_service.py`
- `agent/setup_chat_flow.py`
- `agent/chat_response_serializer.py`
- `agent/llm/inference_router.py`
- `agent/llm/model_selector.py`
- `agent/llm/model_state.py`
- `agent/llm/value_policy.py`
- `agent/llm/default_model_policy.py`
- `agent/telegram_bridge.py`
- `telegram_adapter/bot.py`
- `agent/cli.py`

## Runtime Topology

The main execution topology is:

```text
HTTP / CLI / Telegram
        |
        v
    API Server
        |
        v
    AgentRuntime
        |
        v
Orchestrator.handle_message()
        |
        v
  Inference Router
        |
        v
Model Selection Policy
        |
        v
     Provider
```

The runtime is constructed in `build_runtime()` and served in `run_server()`
in `agent/api_server.py`.

Current execution ownership:

- HTTP API is served by `ThreadingHTTPServer` in `agent/api_server.py`.
- `AgentRuntime` in `agent/api_server.py` is the shared runtime host.
- `Orchestrator.handle_message()` in `agent/orchestrator.py` is the canonical
  top-level chat brain.
- LLM execution for ordinary chat is delegated through
  `InferenceRouter.execute()` in `agent/llm/inference_router.py`.
- Model selection for ordinary chat is delegated through
  `select_model_for_task()` in `agent/llm/model_selector.py`.
- Automatic/default model switching is delegated through
  `choose_best_default_chat_candidate()` in
  `agent/llm/default_model_policy.py`.

## Runtime Truth Authority

`RuntimeTruthService` in `agent/runtime_truth_service.py` is the canonical
adapter for deterministic runtime state.

It is responsible for structured, non-LLM answers about:

- provider state
- model state
- runtime status
- setup state
- governance state
- model policy state

It is deliberately not an LLM surface.

RuntimeTruthService rules:

- It reads runtime state from `AgentRuntime`.
- It may call runtime helpers and shared selector/policy helpers.
- It must not call the LLM.
- It should return structured data, not freeform guessed explanations.

Current key runtime-truth entrypoints include:

- `chat_target_truth()`
- `current_chat_target_status()`
- `provider_status()`
- `model_status()`
- `runtime_status()`
- `skill_governance_status()`
- `model_policy_status()`
- `model_policy_candidate()`
- `model_policy_provider_candidate()`

In the current SAFE MODE baseline, these deterministic capabilities sit beneath
an assistant-first chat front door rather than a visible chooser/menu layer.

The current deterministic-capability dispatch lives in
`Orchestrator._handle_runtime_truth_chat()`. That is the internal layer the
assistant uses for runtime/provider/model/setup/governance/system answers when
the LLM should not be involved.

## Chat Routing

### Unified chat flow

The unified API chat flow is:

```text
POST /chat
  -> AgentRuntime.chat()
  -> Orchestrator.handle_message()
  -> deterministic runtime route OR generic chat path
  -> serialized HTTP/UI response
```

Relevant code:

- `APIServerHandler.do_POST()` in `agent/api_server.py`
- `AgentRuntime.chat()` in `agent/api_server.py`
- `Orchestrator.handle_message()` in `agent/orchestrator.py`
- `serialize_orchestrator_chat_response()` in
  `agent/chat_response_serializer.py`

`AgentRuntime.chat()` is now an adapter only. It still owns:

- request normalization
- session/thread/user derivation
- compatibility hooks such as local bootstrap gating
- serializer invocation
- API-surface logging

It should not own independent chat-brain decisions.

In the stabilized SAFE MODE baseline, `AgentRuntime.should_use_assistant_frontdoor()`
is the shared gate for ordinary user chat. When it returns true, the system
should keep the assistant in front and suppress router-era chooser/menu UX.

### Clarification preflight

The HTTP `/chat` boundary still owns a small clarification layer before
ordinary chat runs.

Current preflight clarification types:

- low-confidence/underspecified payload clarification
- binary ambiguity clarification (`A` / `B`)
- intent-choice clarification (`chat` / `ask` / `model check-switch`)

Important behavior:

- deterministic runtime/setup/provider/model/governance routes bypass this layer
- clarification replies are consumed at the `/chat` boundary before ambiguity is
  re-evaluated
- clarification state is keyed by source/user and is cleared once consumed or
  expired

This logic currently lives in `APIServerHandler.do_POST()` plus the prompt-state
helpers in `AgentRuntime`, both in `agent/api_server.py`.

### Telegram routing

Telegram ordinary text is intentionally aligned with the same canonical path:

```text
Telegram text
  -> telegram_adapter
  -> telegram_bridge
  -> local API /chat
  -> AgentRuntime.chat()
  -> Orchestrator.handle_message()
```

Relevant code:

- `_handle_message()` in `telegram_adapter/bot.py`
- `handle_telegram_text()` in `agent/telegram_bridge.py`
- async local API `/chat` proxy helpers in `telegram_adapter/bot.py`

Design rule:

- Ordinary Telegram text must share the same path as UI/API chat.
- Telegram may keep transport-local behavior for commands, delivery, retries,
  reminders, and chat overlap handling.
- Telegram should not host a parallel runtime brain for ordinary chat.
- In safe mode, Telegram ordinary text and API `/chat` share the same pinned
  local chat target.
- In safe mode with a healthy pinned local model, ordinary user messages stay on
  the assistant path and do not surface router-era chooser/meta prompts.

### Deterministic route classification

The pre-LLM classifier is `classify_runtime_chat_route()` in
`agent/setup_chat_flow.py`.

It routes requests into categories such as:

- `setup_flow`
- `provider_status`
- `model_status`
- `runtime_status`
- `governance_status`
- `model_policy_status`
- `generic_chat`

Generic chat is still agent-mediated. It does not mean direct raw-model access.

## Status and Health System

### `/ready`

`/ready` is implemented by `ready_status()` in `agent/api_server.py`.

Current design:

- fast snapshot endpoint
- uses `_canonical_llm_ready_context()`
- reads current known defaults and health state
- includes lifecycle `phase`, internal `startup_phase`, and Telegram state
- avoids synchronous provider probing before replying

`/ready` is intended to answer quickly with current known readiness, even when
background health or catalog work is slow.

In SAFE MODE, `/ready` also reports `safe_mode_target`, which distinguishes the
configured pin, the effective active target, and any invalid-pin fallback
reason.

### `/runtime`

`/runtime` is implemented by `runtime_snapshot()` in `agent/api_server.py`.

Current design:

- lightweight introspection endpoint for CLI/operator polling
- uses cached readiness/runtime truth plus registry and health-state snapshots
- reports lifecycle phase, default chat model, provider states, router counts,
  health summary, and Telegram status
- includes the same `safe_mode_target` snapshot used by `/ready`
- avoids provider probes and heavy recomputation

### `/runtime/history`

`/runtime/history` is implemented by `runtime_event_history()` in
`agent/api_server.py`.

Current design:

- lightweight snapshot of recent runtime events
- backed by an in-memory circular buffer
- intended for CLI/operator introspection
- never triggers provider probes

### `/health`

`/health` is implemented by `health()` in `agent/api_server.py`.

Current design:

- lightweight service metadata endpoint
- reports router mode, configured providers, registry path, and safe mode state
- not a full LLM readiness report

### `/llm/status`

`/llm/status` is implemented by `llm_status()` in `agent/api_server.py`.

Current design:

- heavy status endpoint
- builds a canonical runtime router snapshot
- computes a full health summary
- overlays active provider and model health
- returns detailed provider/model/default visibility

### Health monitor role

The health monitor is `self._health_monitor` inside `AgentRuntime`.

It is the main background health authority for provider/model probe results.

Important related paths:

- `_probe_llm_candidate()` in `agent/api_server.py`
- `llm_health_summary()` in `agent/api_server.py`
- `_canonical_runtime_router_snapshot()` in `agent/api_server.py`

### Authoritative success writes

Successful provider test/setup flows can immediately update health truth through
`_record_authoritative_provider_success()` in `agent/api_server.py`.

This is important because it ensures that successful setup/switch work is
visible right away to:

- `/ready`
- `/llm/status`
- CLI status
- provider/model status chat answers

## Safe Mode Baseline

Safe mode is the current trusted stabilization baseline.

When `AGENT_SAFE_MODE=1`:

- one local chat model is pinned through `AgentRuntime.get_defaults()` and
  `prepare_orchestrator_chat_request()`
- remote fallback is disabled
- local auto-bootstrap is disabled
- background scheduler startup is suppressed
- Model Scout / Model Watch scheduler runs are skipped
- self-heal / bootstrap / notification apply policies are denied with
  `allow_reason=safe_mode`
- deterministic runtime/setup/provider/model/doctor/system-observe prompts still
  work
- when the pinned local model is healthy, normal user messages stay on the
  assistant path instead of exposing router/menu clarification UX
- raw recovery/setup guidance appears only when there is no usable pinned local
  chat target
- explicit user-confirmed model switches can temporarily override the normal
  safe-mode pin for that one switch action
- `/ready` and `/runtime` report both the configured safe-mode pin and the
  effective active target when they differ

The intent is to keep one fixed local chat target and remove background
control-plane mutation while integrated chat/status behavior is verified.
- model-policy explainability answers

### Fast snapshot endpoints vs heavy recomputation endpoints

Fast snapshot endpoints:

- `/ready`
- `/runtime`
- many RuntimeTruthService-backed conversational status routes

Heavy recomputation endpoints:

- `/llm/status`
- `/llm/health`
- some catalog/model-watch/admin surfaces

Rule of thumb:

- readiness and conversational runtime truth should usually read current known
  cached state
- detailed operator status may do heavier recomputation

## Runtime Lifecycle

Lifecycle state is derived in `agent/runtime_lifecycle.py`.

Authoritative inputs are existing runtime signals:

- `startup_phase`
- startup warmup flags and remaining warmup tasks
- cached readiness/runtime mode from `/ready` logic
- cached active provider/model health

Lifecycle phases:

- `boot`
  - runtime object exists, but startup warmup has not really begun yet
- `warmup`
  - runtime is listening or still completing startup warmup work
- `ready`
  - runtime is operational with a usable chat model
- `degraded`
  - runtime is up, but current known readiness or cached health is degraded
- `recovering`
  - runtime was degraded and is now transitioning back to healthy

Important properties:

- lifecycle phase is observational only
- it does not trigger probes
- ambiguous healthy states prefer `ready`
- internal startup sequencing is still preserved separately as `startup_phase`

Lifecycle diagram:

```text
BOOT
  |
  v
WARMUP
  |
  v
READY
  |
  v
DEGRADED
  \
   v
RECOVERING
   \
   v
   READY
```

## Runtime Event Logging

Structured runtime events are implemented in `agent/runtime_events.py`.

Current event families:

- `runtime_phase_change`
- `provider_switch`
- `default_model_change`
- `provider_health_transition`
- `chat_request_start`
- `chat_request_end`
- `telegram_latency_guard`
- `telegram_latency_fallback`

Current behavior:

- events are emitted through the standard logging stack as structured JSON
- recent events are also stored in a small in-memory circular buffer
- chat requests carry a lightweight `request_id` for correlation across API and
  Telegram transport logs

Current hook points:

- lifecycle changes inside `AgentRuntime` lifecycle observation
- default/provider changes inside default-update paths
- authoritative provider health success writes and health-run transitions
- API `/chat` adapter around orchestrator invocation

`/runtime/history` exposes the recent in-memory event buffer for fast
inspection without touching providers.

## Provider Routing

### Ordinary chat selection stack

The ordinary chat selection stack is:

```text
InferenceRouter.execute()
  -> build_model_inventory()
  -> select_model_for_task()
  -> build_effective_model_state()
  -> score_candidate_utility()
```

Relevant code:

- `agent/llm/inference_router.py`
- `agent/llm/model_selector.py`
- `agent/llm/model_state.py`
- `agent/llm/value_policy.py`

Current behavior:

- local-first by default
- health/approval/capability/context/policy gating
- remote use allowed only when policy allows it
- premium escalation is separately guarded

### Default-switch policy

Automatic/default switching is unified through
`choose_best_default_chat_candidate()` in
`agent/llm/default_model_policy.py`.

Current tier order:

1. strongest healthy approved local candidate
2. strongest healthy approved free remote candidate
3. strongest healthy approved cheap remote candidate under the strict cheap cap
4. otherwise no switch

Current anti-churn behavior:

- hysteresis lives in `_switch_decision()`
- do not switch unless the challenger is materially better by tier, quality,
  context headroom, cost, or utility threshold

### Cheap remote cap

Automatic/default switching uses a stricter cheap-remote cap than ordinary
routing.

Current config field:

- `default_switch_cheap_remote_cap_per_1m` in `agent/config.py`

Ordinary routing still uses the broader general value-policy cap.

## Latency-Aware Routing

Latency-sensitive routing is currently channel-aware for ordinary chat without
changing the core runtime pipeline.

Current design:

- `AgentRuntime.chat()` carries `source_surface` into `chat_context`.
- `Orchestrator._llm_chat()` normalizes that into a `channel` hint.
- `route_inference()` passes the hint through `metadata`.
- `InferenceRouter.execute()` forwards the hint into
  `select_model_for_task()`.
- `score_candidate_utility()` applies Telegram-only latency penalties using
  `agent/llm/model_latency.py`.

Current channel behavior:

- `api` and `cli`
  - unchanged canonical routing behavior
- `telegram`
  - prefer smaller, faster viable models
  - penalize medium/slow models in utility scoring
  - keep the same health/approval/capability/cost gates

Current Telegram latency guard:

- ordinary generic chat keeps the existing orchestrator path
- the first Telegram inference attempt uses a short internal timeout budget
- if that attempt times out, the orchestrator retries once with
  `latency_fallback=true`
- the retry uses the same selector stack, but with stronger Telegram latency
  penalties so the fastest viable model is preferred

Current runtime events for this path:

- `telegram_latency_guard`
- `telegram_latency_fallback`

These events are recorded through the runtime event logger and include request
correlation ids, selected model details, elapsed time, and whether fallback was
used.

### Current automatic/default-switch callers

These all route through the same shared default-model policy helper:

- `agent/llm/autoconfig.py`
- `agent/llm/self_heal.py`
- autopilot bootstrap in `agent/api_server.py`
- model-watch proposal generation in `agent/api_server.py`

## External Interfaces

### HTTP API

Primary HTTP surface lives in `APIServerHandler` in `agent/api_server.py`.

Most relevant endpoints:

- `GET /ready`
- `GET /runtime`
- `GET /runtime/history`
- `GET /health`
- `GET /version`
- `GET /telegram/status`
- `GET /llm/health`
- `GET /llm/status`
- `GET /model`
- `GET /models`
- `GET /providers`
- `GET /defaults`
- `GET /permissions`
- `GET /packs`
- `GET /skill-governance/status`
- `GET /skill-governance/adapters`
- `GET /skill-governance/background-tasks`
- `POST /chat`
- `POST /ask`

### CLI

CLI status surface lives in `agent/cli.py`.

Current behavior:

- `python -m agent status` reads `/ready`
- `python -m agent status` also reads `/runtime` when available for lifecycle
  and lightweight provider-health summary
- `python -m agent status` also reads `/runtime/history` when available for
  recent lifecycle/runtime event lines
- `python -m agent health` reads `/llm/status`

There is also `agent/runtime_status.py`, which builds a service/systemd/database
report. That is a diagnostics surface, not the canonical conversational runtime
truth interface.

### Telegram

Telegram transport lives in:

- `telegram_adapter/bot.py`
- `agent/telegram_bridge.py`

Current split:

- ordinary text goes through local API `/chat`
- slash-command-like command handling may stay local
- polling, delivery, retries, and same-chat overlap handling stay local

## Concurrency Model

The server uses `ThreadingHTTPServer`, so multiple request threads can access
the same `AgentRuntime`.

`AgentRuntime` is mutable.

Shared mutable state includes:

- `registry_document`
- health monitor state
- runtime lifecycle phase cache
- startup phase and warmup fields
- scheduler state
- autopilot/model-watch/notification cached state
- clarify/setup state
- embedded Telegram runner state

The system does not attempt full transactional locking across the whole runtime.

Instead, it relies on snapshot consistency:

- a request reads a coherent snapshot of runtime state at the time it accesses
  that state
- later updates affect future requests
- the current in-flight decision path should continue using its own captured
  state rather than mixing partial old/new truth

Important locking note:

- registry persistence uses `_registry_lock`
- status and routing reads generally operate on current snapshots/cached dicts
- many reads are intentionally lock-light for responsiveness

This means the design is:

- shared mutable runtime
- snapshot-style reads
- localized locking for writes/persistence

not:

- globally serialized actor model
- fully transactional runtime reads

## System Invariants

These are the current architectural invariants and should remain true unless the
runtime design is intentionally changed.

1. All natural-language chat enters through `Orchestrator.handle_message()`.
   - HTTP/API path: `AgentRuntime.chat()` -> `Orchestrator.handle_message()`
   - Telegram ordinary text path: Telegram adapter -> local `/chat` -> same orchestrator

2. Telegram ordinary text must route through the API `/chat` path.
   - Telegram must not host a parallel ordinary-chat brain.

3. `RuntimeTruthService` is the canonical read interface for deterministic
   runtime state.
   - It must not call the LLM.

4. `/ready` must never perform blocking provider probes before responding.
   - It should use cached/current known runtime truth.

5. `/runtime` must stay on the same fast snapshot side as `/ready`.
   - It must not synchronously probe providers before replying.

6. Successful provider setup/test must immediately update authoritative health
   state.
   - This is what keeps `/ready`, `/llm/status`, CLI, and conversational status
     answers aligned.

7. Default model switching must use `choose_best_default_chat_candidate()`.
   - autoconfig
   - self-heal
   - autopilot bootstrap
   - model-watch proposal generation

8. Snapshot-based status endpoints must not mix unrelated stale sources when a
   single canonical runtime source exists.
   - Current known health/default/runtime state should be preferred over older
     parallel status logic.

9. Generic chat is still agent-mediated.
   - `generic_chat` is not direct raw-model fallback.

10. Pending `/chat` clarification replies must advance or clear the active
    clarification state instead of repeating the same chooser forever.
    - intent-choice replies such as `chat` / `ask` / `model switch` are
      consumed at the API boundary
    - binary clarification replies such as `A` / `B` are consumed at the API
      boundary

## Architecture Diagram

```text
                         +----------------------+
                         |   CLI / Telegram /   |
                         |     HTTP clients     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |      API Server      |
                         |  APIServerHandler    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |     AgentRuntime     |
                         | registry, health,    |
                         | scheduler, secrets,  |
                         | embedded telegram    |
                         +-----+-----------+----+
                               |           |
             runtime truth ----+           +---- chat adapter
                               |                 (`chat()`)
                               v
                   +--------------------------+
                   |   RuntimeTruthService    |
                   | deterministic status     |
                   | provider/model/runtime   |
                   | setup/governance/policy  |
                   +-----------+--------------+
                               |
                               | reads
                               v
                 +-------------------------------+
                 | registry_document             |
                 | health_monitor.state          |
                 | router snapshots              |
                 | defaults / setup state        |
                 +-------------------------------+

                                    |
                                    v
                         +----------------------+
                         |     Orchestrator     |
                         | handle_message()     |
                         +----------+-----------+
                                    |
                           deterministic route
                           or generic chat path
                                    |
                                    v
                         +----------------------+
                         |   Inference Router   |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Model Selection      |
                         | task selector /      |
                         | default policy       |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |      Provider        |
                         | ollama/openrouter/...|
                         +----------------------+
```

## Related Documents

- `docs/design/MODEL_SELECTION_POLICY.md`
- `docs/design/RUNTIME_TRUTH_SERVICE.md`
- `docs/design/TELEGRAM_API_BACKEND_HANDOFF.md`
- `docs/design/SYSTEM_ARCHITECTURE_MAP.md`
