# Model Scout v2

Model Scout v2 is the assistant-facing advisory layer for model strategy.

It is intentionally separate from:

- inventory truth
- readiness truth
- controller actions such as test, trial switch, promote default, and rollback

It is also intentionally separate from the older background `model_scout.run()` path.
In SAFE MODE, the background scout remains disabled because background mutation
and proactive notifications are suppressed. The assistant-facing scout still works
because it is deterministic, grounded in current runtime truth, and advisory-only.

## Runtime-available scouting

For normal assistant prompts such as:

- `run the model scout`
- `is there a better model I should use?`
- `should we switch to a better model?`
- `try a better model`
- `what better local models could I try?`

the assistant uses live runtime truth, not the LLM, to inspect:

- inventory truth: what models are known / installed / registered
- readiness truth: what models are usable right now
- policy truth: whether SAFE MODE, remote fallback, or install/pull are allowed
- quality / context / utility metadata when available

Recommendation rules are conservative:

- unusable, unhealthy, unapproved, or auth-missing models are excluded
- SAFE MODE keeps the scout local-only and advisory-only
- expensive remote models are excluded from the v2 recommendation set
- the scout recommends a switch only when a candidate is meaningfully better than
  the current active model
- if nothing looks materially better, the assistant says to keep the current model

The scout does not directly switch or mutate configuration in SAFE MODE.
Instead it can recommend one or two candidates and tell the user what controller
actions are available next.

## Controller workflow

Controller actions are explicit and auditable:

1. test a model without adopting it
2. switch temporarily
3. make a target the default
4. switch back to the previous trial target

The explicit target remains authoritative for those actions. Automatic policy
selection is not allowed to rewrite the approved target during the controller step.

The assistant stores the prior exact target for a trial switch. That allows a later
`switch back` or `go back` to restore the previous model deterministically instead
of re-running automatic selection.

## External discovery

If the Hugging Face discovery substrate is enabled, the assistant can also handle
prompts such as:

- `check for a better model we should download`
- `look on hugging face for a better model`

Current behavior is intentionally narrow:

- use the real HF discovery status/scan substrate
- suggest promising download candidates truthfully
- clearly say the candidate is not installed yet
- do not download, install, or switch anything implicitly

If HF discovery is disabled or unavailable, the assistant says so plainly and falls
back to comparing the models already available in the runtime.

Local/downloaded model inventory is separate from HF discovery. Questions about
installed, downloaded, local, or Ollama models use grounded runtime inventory only
and must not imply that HF discoveries are already installed.

## Scope limits

Model Scout v2 currently does not:

- run benchmarks
- do LLM-driven model evaluation
- auto-switch in the background
- auto-install or auto-pull models in SAFE MODE
- claim that external discoveries are installed before they actually are
