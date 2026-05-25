# Assistant-Agent Planning

Personal Agent should use the assistant/persona LLM as the normal-language interpretation layer for ordinary user messages. Deterministic code remains the safety and execution layer.

## Flow

1. User sends a normal message.
2. The assistant planner LLM interprets what the user is trying to do.
3. The planner emits a small structured request.
4. The agent layer validates the request against an allowlist of capabilities and actions.
5. The agent performs only bounded work through existing safe runtime handlers.
6. The agent returns structured facts/results.
7. The assistant explains the result to the user.

## Planner Schema

```json
{
  "intent": "answer_directly | ask_agent | clarify",
  "agent_request": {
    "capability": "web_search | telegram | chat_model | local_models | external_skills | runtime_status | file_work | none",
    "action": "status | setup | query | preview | diagnose | none",
    "goal": "plain user goal"
  },
  "confidence": 0.0,
  "user_facing_summary": "short internal summary"
}
```

The planner may request capabilities, but it cannot execute them. Planner output is untrusted until schema validation passes.

## Deterministic Code Is For

- schema validation
- safety validation
- permission gates
- confirmation state
- truth/status checks
- lifecycle enforcement
- tool execution boundaries
- model-unavailable fallback
- explicit operator commands

## Deterministic Code Is Not For

- understanding arbitrary human language as the main path
- predicting every possible phrasing
- expanding giant phrase lists
- treating semantic keyword routing as the assistant brain

## Allowed Capability Dispatch

Validated planner requests route only to existing safe capability handlers:

- `web_search`: safe web-search status/setup/query; search results remain untrusted metadata only.
- `telegram`: Telegram status/setup truth.
- `chat_model`: current model and setup truth.
- `local_models`: local model status/setup guidance.
- `external_skills`: external pack acquisition lifecycle with source, quarantine, review, approval, enablement, permission, and managed-adapter gates.
- `runtime_status`: runtime status or bounded diagnostics.
- `file_work`: bounded file actions only after a specific safe path/action is known.

Unknown capabilities or actions are rejected. Requests for Docker commands, shell, package installs, OAuth, browser automation, pack approval, pack enablement, permission grants, or arbitrary tool execution are not valid planner actions.

## Ordering

The planner runs only for normal messages. It does not replace deterministic handling for:

- yes/no pending confirmations
- safe mode and controlled mode enforcement
- hard runtime-status fallbacks
- emergency model-unavailable fallback
- explicit slash/operator commands

## Failure Behavior

If the planner output is invalid JSON or the model is unavailable, the runtime falls back to existing deterministic handling. If the planner emits a schema-valid but unsafe request, the agent layer rejects it with a safe clarification and no tool execution.
