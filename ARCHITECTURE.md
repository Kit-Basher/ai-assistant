# Architecture

Short orientation only. This is the system shape that matters for day-to-day work.

## Major Surfaces

- API: `agent/api_server.py`
  - Hosts `/chat`, `/ready`, status endpoints, and operator/admin surfaces.
- Web UI: `agent/webui/dist`
  - Served by the same API server.
  - Uses the API chat surface for assistant turns.
- Telegram: `telegram_adapter/bot.py`
  - Transport adapter only.
  - Forwards user messages to the local API and handles perceived-latency behavior.

## Core Roles

- Orchestrator: `agent/orchestrator.py`
  - Classifies the turn.
  - Chooses deterministic runtime-truth handling vs LLM-backed chat.
  - Owns memory/continuity/clarification/confirmation behavior.
  - Skips expensive post-response guard work on safe read-only fast paths.
- `RuntimeTruthService`: `agent/runtime_truth_service.py`
  - Single runtime-truth source for model status, inventory, readiness, provider health, operational status, and other deterministic facts.
- Router/provider layer: `agent/llm/router.py` and provider adapters
  - Handles model/provider selection and provider transport.
  - Should not own assistant policy or turn classification.

## Boundary Rules

- Read-only deterministic status questions should use runtime truth directly when safe.
- Confirmation/mutation flows stay behind an explicit approval boundary.
- Generic `chat` can use the LLM path, but it should not leak internal state or bypass safety rules.

## Simple Request Flow

1. User sends a message from API, Web UI, or Telegram.
2. Transport forwards the message into the API chat surface.
3. API/orchestrator classifies the turn.
4. If the turn is deterministic runtime truth, the orchestrator reads `RuntimeTruthService` and returns a grounded answer.
5. If the turn is generic assistant work, the orchestrator routes through the LLM/controller path.
6. The response is serialized with timing/meta fields and rendered back to the surface.

## What Not To Assume

- A fast status answer does not mean the assistant is broadly useful yet.
- A working LLM path does not mean the user experience is good.
- The next milestone is proving one useful assistant interaction, not adding more layers.
