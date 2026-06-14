# Plan Mode Policy

Plan Mode is the central runtime policy layer that classifies operations before
tool or runtime execution.

## Classification

- Read-only operations may run without confirmation. This includes read, list,
  search, status, preview, query, diagnose, compare, check, and health
  operations.
- Mutating operations require a plan, preview, and explicit confirmation. This
  includes setup/apply, install/import, approve, enable/disable, grant, create,
  update, delete/remove, reset/clear, pull/run/start/stop/restart, and write.
- Unknown operations default to mutating unless explicitly declared read-only.

The central helper is `agent.policy.classify_operation()`. Current explicit
reference classifications include safe web search status/query, managed local
service status/setup plan/apply, and external pack install/approve/enable/grant
or remove lifecycle actions.

## Mutator Plan Contract

Every mutating apply path must carry a plan with:

- `action_type`
- resources that may be created, changed, or deleted
- rollback scope
- whether rollback is supported
- confirmation token
- expiration

`agent.policy.build_mutator_plan()` creates this shape and
`agent.policy.validate_mutator_apply()` rejects missing, expired, wrong-token,
or tampered plans before the runtime executor runs.

## Reference Implementation

Managed SearXNG setup is the reference implementation. `/search/setup/plan`
embeds a Plan Mode `mutation_plan`; `/search/setup/apply` validates that plan
before managed local service execution, then journals executed steps and changed
resources through the existing managed-action journal.

The policy layer does not permit hidden sudo, arbitrary shell execution, public
SearXNG binds, or external pack install/import from search results.
