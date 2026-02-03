# NOTES / Parking Lot

- Future features to consider:
  - Versioned provenance UI to let reviewers replay agent reasoning.
  - Fine-grained simulation environment for agents using synthetic data.
  - Policy-as-code module for writing and testing safety rules.

- Integrations (deferred):
  - Read-only knowledge connectors (wiki, docs).
  - Secure audit storage backend (immutable ledger).
  - Human approval channel (email/slack/ui) with cryptographic confirmation.

- TODO items for later design phases:
  - Define schema for the Audit Log entries.
  - Threat model document and red-team plan.
  - Cost modeling for agent orchestration.
  - UX spec for human-in-the-loop confirmation flows.

- Open heuristics:
  - TTL defaults for ephemeral memory (e.g., 1 hour vs 24 hours).
  - What qualifies as an "external action" that must be human-approved.

- Meeting notes / quick thoughts:
  - Keep the initial pilot as read-only and synthetic-data-only.
  - Prefer simple, auditable building blocks over clever but opaque automation.
