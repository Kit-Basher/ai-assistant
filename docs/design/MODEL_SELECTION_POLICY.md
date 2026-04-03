# Model Selection Policy

## Authority

The authoritative ordinary-chat selector is:

- `agent/llm/inference_router.py`
- `agent/llm/model_selector.py`
- `agent/llm/model_state.py`
- `agent/llm/value_policy.py`

The authoritative default-switch selector is:

- `agent/llm/default_model_policy.py`

Automatic/default-switch callers delegate to that shared policy from:

- `agent/llm/autoconfig.py`
- `agent/llm/self_heal.py`
- `agent/api_server.py` autopilot bootstrap
- `agent/api_server.py` model-watch proposal generation

## Ordinary Chat Selection

Ordinary chat builds a canonical model inventory, classifies the task, computes effective model
state, and selects from suitable candidates.

Selection properties:

- local-first by default
- health, approval, routability, capability, context, and value-policy gates
- remote candidates are blocked when they exceed the configured hard cap
- premium escalation still uses the existing preflight confirmation guard for over-cap upgrades

## Default-Switch / Autopilot Selection

Automatic default switching uses the same inventory/state/value-policy foundations as ordinary chat,
but applies an explicit tier order for default chat models:

1. `local`
   Strongest healthy approved local chat model that is policy-allowed.
2. `free_remote`
   Strongest healthy approved remote chat model with effective cost `0`.
3. `cheap_remote`
   Strongest healthy approved remote chat model under the stricter automatic cheap cap.
4. no switch

Within a tier, the selector prefers stronger candidates first:

- higher `quality_rank`
- larger context window
- higher utility score
- then lower effective cost as a later tiebreaker

## Hard Cost Caps

Remote cost gating is authoritative in `agent/llm/value_policy.py`.

- default policy cap: `default_policy.cost_cap_per_1m`
- premium policy cap: `premium_policy.cost_cap_per_1m`
- automatic default-switch cheap cap: `default_switch_cheap_remote_cap_per_1m` (default `0.5`)

Remote candidates above the relevant cap are rejected and are never silently selected.
Automatic/default-switch selection never broadens the general routing cap: the cheap-remote tier
uses the lower of `default_switch_cheap_remote_cap_per_1m` and `default_policy.cost_cap_per_1m`.

## Hysteresis / Anti-Flap

The shared default-switch selector only recommends a switch when one of these is true:

- the current default is not suitable
- the candidate is in a higher-priority tier
- the candidate is a material same-tier upgrade in quality, context headroom, lower expected cost, or utility

The selector also uses `model_watch_min_improvement` as the default utility threshold for
same-tier changes. Existing churn detection and autopilot safe-mode protections remain in place in
`agent/llm/autopilot_safety.py`.

## Deterministic Explainability

Runtime-truth explainability for model policy lives behind the orchestrator/runtime routes, not
generic chat. The agent can answer these deterministically from the shared selector state:

- `what is my model selection policy?`
- `what is my cheap remote cap?`
- `why are you using this model?`
- `why didn’t you switch to openrouter?`
- `what model would you switch to right now?`
- `what free remote model would you choose?`
- `what cheap remote model would you choose?`

Those answers are grounded in:

- the current active/default chat model
- current health and approval state
- the shared default-switch selector result
- tier candidates (`local`, `free_remote`, `cheap_remote`)
- the strict cheap-remote cap

## Notes

- The legacy `LLMRouter` still exists for compatibility and snapshot plumbing, but it is not the
  authoritative chat-selection policy.
- Model-watch proposals use the shared selector but intentionally skip provider-auth enforcement at
  proposal time so catalog comparisons can still be evaluated before an operator confirms a switch.
