# Organization and Memory UX

Audit 3 user contract:

- ordinary requests such as “remember this,” “remind me tomorrow,” and “mark
  that task done” enter the same assistant path as Web and Telegram;
- conversation context is not described as durable saved memory;
- user-directed durable writes show a preview and require approval;
- model-inferred facts, preferences, tasks, and reminders are suggestions, not
  authorization and not silently trusted durable records;
- create retries use durable idempotency identity; update/complete/delete bind
  the exact record and version; `/done` clarifies when more than one target is
  plausible;
- reminder previews include the interpreted date, time, and timezone;
- stale targets require a fresh preview; cancellation changes nothing;
- memory reset is destructive and explains its scope, derived-index effects,
  rollback limits, and restoration policy without repeating private content;
- unsupported archive, merge, restore, or broad CRUD variants are explained as
  unavailable instead of being simulated.

Canonical authorization details remain in
`docs/operator/ARCHITECTURE_SAFETY_AUDIT_V2E.md`. Behavioral evidence is in
`docs/operator/END_USER_BEHAVIOR_UX_AUDIT_3.md` and the deterministic Audit 3
inventory.
