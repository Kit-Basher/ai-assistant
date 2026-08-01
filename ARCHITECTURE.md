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
  - Runs deterministic approval/cancellation state before ordinary understanding.
  - Uses `RequestUnderstandingService` and the live `CapabilityRegistry` to classify ordinary turns once.
  - Chooses deterministic runtime-truth handling vs LLM-backed chat.
  - Owns memory/continuity/clarification/confirmation behavior.
  - Skips expensive post-response guard work on safe read-only fast paths.
- `RuntimeTruthService`: `agent/runtime_truth_service.py`
  - Single runtime-truth source for model status, inventory, readiness, provider health, operational status, and other deterministic facts.
- Router/provider layer: `agent/llm/router.py` and provider adapters
  - Handles model/provider selection and provider transport.
  - Should not own assistant policy or turn classification.
- Conversation capability registry: `agent/capability_registry.py`
  - Declares stable IDs, contracts, availability/health, approval policy, invocation, verification, and provenance.
- Request understanding: `agent/request_understanding.py`
  - Preserves original text while producing normalized meaning, ranked candidates, ambiguity, validated inputs, approval classification, fallback category, and concise audit metadata.

## Boundary Rules

- Read-only deterministic status questions should use runtime truth directly when safe.
- Confirmation/mutation flows stay behind an explicit approval boundary.
- Generic `chat` can use the LLM path, but it should not leak internal state or bypass safety rules.

## Simple Request Flow

1. User sends a message from API, Web UI, or Telegram.
2. Transport forwards the message into the API chat surface.
3. Deterministic approval, denial, cancellation, expiry, and thread-binding checks run first.
4. The unified understanding layer selects only a healthy registered capability, asks one clarification, or chooses a grounded fallback.
5. Registered capabilities invoke the existing native runtime/files/model/pack/memory implementations through their hooks.
6. Generic assistant work receives authoritative live runtime context before the single LLM generation, and its response is checked for contradictory runtime/access claims.
7. The response is serialized with timing/meta fields and rendered back to the surface. Registered turns expose non-sensitive understanding diagnostics in the existing setup/runtime payload for UI debugging.

## What Not To Assume

- A fast status answer does not mean the assistant is broadly useful yet.
- A working LLM path does not mean the user experience is good.
- The next milestone is proving one useful assistant interaction, not adding more layers.
