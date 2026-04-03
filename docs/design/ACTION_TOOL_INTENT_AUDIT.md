# Action + Tool Intent Audit

This note summarizes the action-oriented capabilities that the assistant can invoke
today from the shared assistant frontdoor.

## Current deterministic/action capabilities

### Local time and date
- Entrypoint:
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `Orchestrator._handle_action_tool_intent()`
  - `Orchestrator._local_time_response()`
- What it returns:
  - grounded local time or date from the runtime timezone
- Trigger phrasing:
  - `what time is it?`
  - `what time is it right now?`
  - `what day is it?`
  - `what's today's date?`
- Status:
  - present

### Model Scout manual evaluation
- Entrypoint:
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `Orchestrator._model_scout_action_response()`
  - [agent/api_server.py](/home/c/personal-agent/agent/api_server.py)
  - `AgentRuntime.run_model_scout()`
- What it returns:
  - grounded Model Scout suggestions from the real runtime/model-scout store
  - optional filtering against explicit model terms or the most recent model context
- Trigger phrasing:
  - `run the model scout`
  - `check if those models are any good`
  - `evaluate those nanbeige models`
  - `check them` after a recent model-status/model-scout turn
- Status:
  - present, but scoped to the most recent relevant model context

### Runtime / provider / model truth
- Entrypoint:
  - [agent/setup_chat_flow.py](/home/c/personal-agent/agent/setup_chat_flow.py)
  - `classify_runtime_chat_route()`
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `Orchestrator._handle_runtime_truth_chat()`
  - [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py)
- What it returns:
  - grounded runtime/provider/model/setup answers from canonical runtime truth
- Trigger phrasing:
  - `what model are you using?`
  - `runtime`
  - `openrouter health`
  - `configure ollama`
- Status:
  - present

### Machine inspection
- Entrypoint:
  - [agent/setup_chat_flow.py](/home/c/personal-agent/agent/setup_chat_flow.py)
  - `classify_runtime_chat_route()`
  - [agent/nl_router.py](/home/c/personal-agent/agent/nl_router.py)
  - `classify_free_text()` / `select_observe_skills()`
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `Orchestrator._handle_nl_observe()`
- What it returns:
  - grounded RAM/storage/hardware/system inspection from native skills
- Trigger phrasing:
  - `how much memory am I using?`
  - `how is my storage?`
  - `what CPU and GPU do I have?`
  - `what other PC stats can you find?`
- Status:
  - present

### Deeper machine inspection
- Entrypoint:
  - [agent/nl_router.py](/home/c/personal-agent/agent/nl_router.py)
  - deeper machine/system phrases map to `OBSERVE_PC`
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `Orchestrator._deep_system_followup_response()`
- What it returns:
  - broader hardware + resource + storage inspection
- Trigger phrasing:
  - `can you dig deeper into my system?`
  - `run a system check`
  - `run a check and see what else you can find`
  - `dig deeper` after a recent machine-inspection turn
- Status:
  - present

### Runtime/provider repair and setup
- Entrypoint:
  - [agent/setup_chat_flow.py](/home/c/personal-agent/agent/setup_chat_flow.py)
  - `classify_setup_intent()`
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `_configure_ollama_response()`
  - `_configure_openrouter_response()`
  - `_confirm_pending_setup_response()`
- What it returns:
  - deterministic provider repair / setup / switch flows
- Trigger phrasing:
  - `configure ollama`
  - `configure openrouter`
  - `yes` after a specific switch confirmation
- Status:
  - present

### Planning / task summarization
- Entrypoint:
  - [agent/nl_router.py](/home/c/personal-agent/agent/nl_router.py)
  - `PLAN_DAY`
  - [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)
  - `_today_cards_payload()`
- What it returns:
  - user-facing daily priorities from open loops + active tasks
- Trigger phrasing:
  - `can you help me plan my day?`
  - `what should I work on today?`
- Status:
  - present, but still intentionally lightweight

## Missing or partial capabilities

### Rich model comparison beyond Model Scout
- Current state:
  - Model Scout can surface grounded suggestions, but the assistant still does not have
    a richer assistant-ready comparison workflow for arbitrary model sets.
- Status:
  - partial

### Long-horizon action follow-up resolution
- Current state:
  - short follow-ups like `check them` reuse only the most recent relevant deterministic
    model context.
- Status:
  - partial

### Broader explicit tool verbs outside grounded capability families
- Current state:
  - action verbs now map well for time, model scout, and machine inspection, but not
    every possible native skill has an assistant-facing action-intent layer yet.
- Status:
  - partial

## Misroutes that existed before this pass

- Time/date prompts fell through to generic LLM chat instead of a deterministic local-time capability.
- Action-oriented model prompts such as `run the model scout` were not strongly mapped to the real Model Scout runtime action.
- Machine-inspection verbs such as `run a system check` and `dig deeper into my system` were too easy to miss unless the prompt looked like a static status query.
- Short model-action follow-ups such as `check them` did not reuse the most recent model context.

## Current routing rule of thumb

- If the user is asking for facts the runtime can already know deterministically, prefer
  the deterministic capability.
- If the user is asking the assistant to run or check a known native skill, route to the
  real capability instead of generic chat.
- If the user uses a short follow-up like `check them` or `dig deeper`, reuse the most
  recent relevant deterministic context when it is clear enough.
- If a capability truly does not exist, answer honestly rather than letting the LLM invent one.
