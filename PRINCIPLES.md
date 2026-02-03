# PRINCIPLES — Non-negotiables and Safety Rules

Status: Design-only / Do Not Implement Yet

## Core non-negotiables
- Human approval for outward-facing or write actions: No external side-effects without explicit human consent.
- Least privilege: Every capability must be scoped and time-limited.
- Auditability: All decisions, prompts, and data accesses must be logged immutably with provenance metadata.
- Data minimization: Avoid retaining sensitive information; prefer ephemeral memory with enforced TTLs.
- Transparency: Agents must provide traceable chains of reasoning and references where possible.
- No secret hoarding: Secrets must never be stored in plaintext in repo or ephemeral memory. Use secure secret stores (out of scope for this design).

## Anti-goals (explicitly forbidden)
- Autonomous actuators (deploying code, financial transactions, account changes).
- Collecting or monetizing user data without consent.
- Uncontrolled model fine-tuning with private or sensitive datasets.
- Hiding logs, redactions, or audit trails.

## Safety rules & operational controls
- Input hygiene: canonicalize and sanitize all external inputs before use.
- Prompt-injection defenses: treat any external text as untrusted; apply layers of validation.
- Rate limiting: per-agent and per-component rate caps to limit runaway costs/behavior.
- Verification loop: at least one independent verification step for high-risk outputs.
- Red-team and continuous testing: scheduled adversarial testing against the runtime assumptions.
- Incident response: clear rollback and quarantine procedures.

## Escalation and accountability
- Every deployment or pilot must have named safety owner(s) and an on-call incident path.
- Post-incident reviews and remediation plans are mandatory.

## Ethical & legal guardrails
- Comply with applicable data protection laws and obtain necessary consents.
- Respect user autonomy: provide opt-out and data-deletion processes.

If you only remember one thing: do not implement this design until the governance and runtime safety primitives exist.