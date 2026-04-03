# LLM Setup Handoff

## What Changed

- Added canonical runtime setup actions in `agent/api_server.py`:
  - `choose_best_local_chat_model()`
  - `configure_local_chat_model(model_id)`
  - `configure_openrouter(api_key, payload=None)`
  - `set_default_chat_model(model_id)`
  - existing `test_provider(provider_id, payload)` remains the probe action
- Chat now intercepts setup and model-switch requests before generic LLM routing.
- `/chat` returns a first-class `setup` payload for conversational setup states such as:
  - `request_secret`
  - `confirm_switch_model`
  - `provider_test_result`
  - `setup_complete`
  - `action_required`
- `llm_status()` now attempts safe local auto-bootstrap when no working chat model is configured and a usable local Ollama chat model is available.
- Telegram now routes setup-related messages through the same runtime chat/setup path instead of generic setup help or hallucinated chat fallback.
- The chat UI now reads structured setup payloads for approval and clarification UI instead of depending only on text heuristics.

## New Canonical Setup Actions

- `choose_best_local_chat_model()`
  - Ranks installed local chat-capable models with a visible policy:
    - local provider only
    - enabled + available + chat-capable
    - prefer routable/healthy
    - prefer higher `quality_rank`
    - prefer larger model size over tiny defaults
    - prefer stronger model families and larger context
- `configure_local_chat_model(model_id)`
  - Refreshes local model inventory, verifies the model with a real Ollama provider probe, and sets it as the default chat model.
- `configure_openrouter(api_key, payload=None)`
  - Normalizes the OpenRouter provider config, stores the API key in the secret store, runs a real provider test, ensures a usable chat model exists in registry inventory, and optionally switches the default chat model.
- `set_default_chat_model(model_id)`
  - Single runtime path for changing the active chat model.

## Known Follow-Ups

- The setup state is currently in-memory per surface (`api`, `telegram`), not persisted across process restarts.
- The existing fix-it wizard still exists for operator/recovery flows; it now shares some canonical actions, but it is not fully retired.
- One unrelated existing API warmup test still fails in `tests/test_api_server.py::test_ready_responds_quickly_while_warmup_pending_then_ready_after_completion`. The provider/setup subset passes when that pre-existing warmup case is excluded.

## OpenRouter Intent Follow-Up

- The earlier implementation only partially intercepted OpenRouter traffic:
  - the parser recognized some setup phrases
  - but the runtime had no first-class `provider_status` response branch
  - so exact questions like `what model do we have set up for openrouter?` still fell through into generic chat
- The setup parser also needed stronger product-specific precedence:
  - provider-specific status checks now run before generic `what model are you using` handling
  - `openrouter` is treated as the LLM provider by default in this product unless the user explicitly makes a networking-hardware question clear
- The conversational OpenRouter flow now reuses a stored OpenRouter key when the provider is already configured, so `use openrouter` can switch deterministically instead of asking for the key again.

## Hard Routing Boundary

- `/chat` now applies a mandatory pre-LLM route classifier before any generic completion:
  - `setup_flow`
  - `provider_status`
  - `model_status`
  - `runtime_status`
  - `generic_chat`
- Product/runtime questions no longer fall through to generic chat by accident.
- If a message looks product-specific but the runtime router cannot answer it from canonical state, the reply is now:
  - `I couldn't read that from the runtime state.`
- Telegram uses the same route classifier before it ever forwards text to the orchestrator, and both runtime chat and Telegram now log:
  - selected route
  - whether generic fallback happened
  - why generic fallback was allowed
