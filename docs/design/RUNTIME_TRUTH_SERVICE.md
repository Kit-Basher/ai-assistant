# Runtime Truth Service

`agent/runtime_truth_service.py` introduces `RuntimeTruthService`, a deterministic adapter over canonical runtime state and setup actions.

Rules:
- no LLM calls
- no guessed system state
- structured data only

Current interface:
- `current_chat_target()`
  - current default provider/model plus readiness of the active chat target
- `provider_status(provider_id)`
  - structured provider configuration/health/default-model state
- `providers_status()`
  - structured list of configured providers and the active provider/model
- `model_status()`
  - structured model/runtime inventory snapshot from canonical runtime state
- `runtime_status(kind="runtime_status")`
  - structured ready/runtime or Telegram status
- `choose_best_local_chat_model(payload=None)`
- `configure_local_chat_model(model_id)`
- `configure_openrouter(api_key, payload=None)`
- `set_default_chat_model(model_id)`

Intended use:
- Orchestrator calls this service for runtime truth
- chat/UI layers format user-facing text from these structured results
- runtime/setup/provider/model questions should never bypass this service and ask the LLM to infer state

Current orchestrator contract:
- runtime/provider/model/setup answers should return response metadata with:
  - `route`
  - `used_runtime_state`
  - `used_llm`
  - `used_memory`
  - `used_tools`
