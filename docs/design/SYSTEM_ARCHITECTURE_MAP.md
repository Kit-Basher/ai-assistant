# System Architecture Map

Superseded by `docs/design/RUNTIME_ARCHITECTURE.md`, which is now the
authoritative architecture document. This file is retained as a shorter field
map.

## 1. Runtime Architecture

- Shared runtime host: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `build_runtime()` builds one `AgentRuntime`.
  - `run_server()` serves it through `ThreadingHTTPServer`.
- Main runtime object: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `AgentRuntime` owns registry state, router, orchestrator, health monitor, scheduler, audit log, secret store, and embedded Telegram runner.
- Canonical chat brain: [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `Orchestrator.handle_message()` is the top-level agent boundary for ordinary chat.
  - It owns runtime-query routing, setup flows, governance queries, memory writes, tool use, and final response assembly.
- Runtime truth source: [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)
  - `RuntimeTruthService` is the deterministic adapter over canonical runtime state.
  - It must not call the LLM.
- Chat request classifier: [agent/setup_chat_flow.py](/home/c/personal-agent/agent/setup_chat_flow.py)
  - `classify_runtime_chat_route()` determines whether a request is setup, provider status, model status, runtime status, governance status, model-policy explainability, or generic chat.

## 2. Chat Request Flow

### UI / API chat

1. `POST /chat` enters [agent/api_server.py](/home/c/personal-agent/agent/api_server.py).
2. `AgentRuntime.chat()` normalizes the payload, derives `user_id` and `thread_id`, runs lightweight compatibility hooks, and delegates.
3. `Orchestrator.handle_message()` becomes the only chat decision-maker.
4. `serialize_orchestrator_chat_response()` in [agent/chat_response_serializer.py](/home/c/personal-agent/agent/chat_response_serializer.py) maps the orchestrator response into the HTTP/UI response envelope.

### Telegram ordinary text

1. Incoming Telegram text enters [telegram_adapter/bot.py](/home/c/personal-agent/telegram_adapter/bot.py).
2. `handle_telegram_text()` in [agent/telegram_bridge.py](/home/c/personal-agent/agent/telegram_bridge.py) classifies commands vs ordinary text.
3. Ordinary text is proxied to local API `/chat` through the async local HTTP path in [telegram_adapter/bot.py](/home/c/personal-agent/telegram_adapter/bot.py).
4. Telegram formats the API/orchestrator response for Telegram output.

### Canonical routing rule

- UI and Telegram share the same ordinary chat brain through the orchestrator.
- Runtime/provider/model/setup/governance/model-policy queries are intended to resolve from runtime truth before generic chat.

## 3. Status and Health System

### `/ready`

- Endpoint: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- Implementation: `ready_status()`
- Fast path: `_canonical_llm_ready_context()`
- Current design:
  - Uses current cached runtime truth, not fresh expensive probes.
  - Reads defaults, health monitor state, startup phase, and Telegram state.
  - Produces current known readiness plus normalized runtime status.

### `/llm/status`

- Endpoint: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- Implementation: `llm_status()`
- Heavier path than `/ready`.
- Builds:
  - canonical router snapshot
  - health summary
  - drift/autopilot metadata
  - active provider and model health
  - visible model/provider status payload

### `/llm/health`

- Endpoint: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- Implementation: `llm_health_summary()`
- Aggregates:
  - provider/model health monitor output
  - scheduler state
  - drift report
  - catalog status
  - capabilities reconcile status
  - notifications state
  - autopilot safety state

### Health truth update paths

- Background health probing: `_probe_llm_candidate()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- Authoritative immediate success write: `_record_authoritative_provider_success()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- Canonical snapshot overlay: `_canonical_runtime_router_snapshot()` in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

This means successful setup/test/switch actions should update the same health state later consumed by `/ready`, `/llm/status`, runtime-truth answers, and CLI status.

## 4. Provider Routing and Model Selection

### Ordinary chat selection

- Entry: [agent/llm/inference_router.py](/home/c/personal-agent/agent/llm/inference_router.py)
- Selector: `select_model_for_task()` in [agent/llm/model_selector.py](/home/c/personal-agent/agent/llm/model_selector.py)
- State builder: `build_effective_model_state()` in [agent/llm/model_state.py](/home/c/personal-agent/agent/llm/model_state.py)
- Value policy: [agent/llm/value_policy.py](/home/c/personal-agent/agent/llm/value_policy.py)

Current ordinary-chat behavior:
- local-first by default
- gates on availability, health, approval, capability, context, and policy
- remote use must pass the general policy cost cap
- premium escalation remains separately guarded

### Default-switch and autopilot selection

- Shared selector: `choose_best_default_chat_candidate()` in [agent/llm/default_model_policy.py](/home/c/personal-agent/agent/llm/default_model_policy.py)

Current tier order:
1. healthy approved local
2. healthy approved free remote
3. healthy approved cheap remote under the strict cheap cap
4. otherwise no switch

Shared switch hysteresis:
- keep current default unless the challenger is materially better by tier, quality, context headroom, cost, or utility threshold

Current callers using the shared selector:
- autoconfig: [agent/llm/autoconfig.py](/home/c/personal-agent/agent/llm/autoconfig.py)
- self-heal: [agent/llm/self_heal.py](/home/c/personal-agent/agent/llm/self_heal.py)
- autopilot bootstrap: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- model-watch proposal evaluation: [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)

### Runtime setup and switching actions

- best local candidate: `choose_best_local_chat_model()`
- switch default: `set_default_chat_model()`
- configure local model: `configure_local_chat_model()`
- configure OpenRouter: `configure_openrouter()`
- opportunistic local bootstrap: `_auto_bootstrap_local_chat_model()`

All of these live in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py).

## 5. External Interfaces

### HTTP API

Primary endpoints in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py):
- `GET /ready`
- `GET /health`
- `GET /version`
- `GET /telegram/status`
- `GET /llm/health`
- `GET /llm/status`
- `GET /model`, `GET /llm/model`
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
- provider/defaults/permissions/pack mutation endpoints

### CLI

- File: [agent/cli.py](/home/c/personal-agent/agent/cli.py)
- `python -m agent status`
  - reads `/ready`
  - uses direct Python loopback HTTP
  - normalizes `localhost` to `127.0.0.1`
- `python -m agent health`
  - reads `/llm/status`

### Telegram

- Bot transport: [telegram_adapter/bot.py](/home/c/personal-agent/telegram_adapter/bot.py)
- Text bridge: [agent/telegram_bridge.py](/home/c/personal-agent/agent/telegram_bridge.py)
- Current rule:
  - commands may stay local to the Telegram bridge
  - ordinary natural-language text should go through API-backed `/chat`

## 6. Concurrency Assumptions

- The API server uses `ThreadingHTTPServer`, so multiple requests can read and mutate one shared `AgentRuntime`.
- `AgentRuntime` is mutable in memory.
- Important mutable state includes:
  - `registry_document`
  - `self._health_monitor.state`
  - startup phase and warmup fields
  - scheduler next-run fields
  - notification/model-watch/autopilot cached status
  - clarify/setup state
  - embedded Telegram runner state
- Registry updates are partially serialized through `_registry_lock`, but runtime truth is not globally transactional.
- Many read surfaces are snapshot-style:
  - `/ready` reads current known cached truth
  - runtime-truth answers read current snapshots/defaults/health state
  - `/llm/status` rebuilds a heavier current snapshot

### Race-condition shape to keep in mind

- provider/model health writes and router snapshot reads can be slightly out of sync
- startup phase can change while readiness is being read
- Telegram status merges external system state and embedded runner state
- selection/explainability answers depend on current known health/default snapshots, not fresh live probes

## 7. Transport Reliability Notes

- HTTP response framing is handled in [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `_send_json()`
  - `_send_bytes()`
- Current behavior:
  - `Connection: close`
  - explicit `close_connection = True`
  - flush after headers and after body write

This is important because `/ready` and CLI/Telegram loopback clients depend on prompt response framing.

## 8. Files Most Relevant

- [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
- [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
- [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)
- [agent/setup_chat_flow.py](/home/c/personal-agent/agent/setup_chat_flow.py)
- [agent/chat_response_serializer.py](/home/c/personal-agent/agent/chat_response_serializer.py)
- [agent/llm/inference_router.py](/home/c/personal-agent/agent/llm/inference_router.py)
- [agent/llm/model_selector.py](/home/c/personal-agent/agent/llm/model_selector.py)
- [agent/llm/model_state.py](/home/c/personal-agent/agent/llm/model_state.py)
- [agent/llm/value_policy.py](/home/c/personal-agent/agent/llm/value_policy.py)
- [agent/llm/default_model_policy.py](/home/c/personal-agent/agent/llm/default_model_policy.py)
- [agent/cli.py](/home/c/personal-agent/agent/cli.py)
- [agent/telegram_bridge.py](/home/c/personal-agent/agent/telegram_bridge.py)
- [telegram_adapter/bot.py](/home/c/personal-agent/telegram_adapter/bot.py)
- [agent/runtime_status.py](/home/c/personal-agent/agent/runtime_status.py)
- [docs/design/MODEL_SELECTION_POLICY.md](/home/c/personal-agent/docs/design/MODEL_SELECTION_POLICY.md)

## 9. Authoritative Summary

- Shared runtime host: `AgentRuntime`
- Shared chat brain: `Orchestrator.handle_message()`
- Shared deterministic truth: `RuntimeTruthService`
- Fast readiness endpoint: `/ready`
- Heavy status endpoint: `/llm/status`
- Ordinary Telegram text: API-backed `/chat`
- Ordinary chat model routing: `InferenceRouter + select_model_for_task()`
- Automatic default switching: `choose_best_default_chat_candidate()`
