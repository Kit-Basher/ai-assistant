# moltbook-v2 — Design Document

Status: Design-only / Do Not Implement Yet

## 1. Executive summary
moltbook-v2 is a conceptual design for an "AI-only workspace": a structured environment where AI agents, models, memory, and carefully scoped integrations collaborate to solve complex, multi-step tasks under strict safety controls. The design prioritizes safety, auditability, human oversight, and a staged path to implementation rather than immediate execution.

## 2. Goals
- Define a minimal, coherent architecture for coordinated AI agents and shared memory.
- Specify non-negotiable safety constraints, operational boundaries, and acceptance criteria for any implementation.
- Provide a clear handoff for future engineering once secure runtime primitives and governance are in place.

## 3. Non-goals
- Delivering runnable deployments, infra manifests, or model training code.
- Building external integrations or autonomous actuators.
- Storing real user data or secrets in this repo.

## 4. High-level architecture
Components (conceptual):
- Orchestrator (coordination layer): routes tasks, enforces policies, logs decisions.
- Agent layer: specialized AI roles (analysis agent, summarizer, verifier). Agents are stateless conversational actors unless explicitly given controlled memory access.
- Ephemeral Memory: short-lived structured memory with TTL and strict redaction rules.
- Persistent Knowledge Store: vetted, sanitized documents with versioned read-only access.
- Tool Adapters: strictly isolated, audited connectors (e.g., read-only document fetcher). No write-capable actuator unless human authorized.
- Audit Log: immutable append-only log of agent inputs, outputs, policy decisions, and human approvals.
- Governance & Policy Engine: enforces anti-goals, safety checks, and rollbacks.

Data flow (conceptual)
1. Human or system submits task to Orchestrator.
2. Orchestrator breaks task into subtasks, assigns to Agents per policy.
3. Agents request data from Ephemeral Memory or Knowledge Store via Tool Adapters.
4. Agent outputs are verified by a Verifier Agent and recorded in Audit Log.
5. Human-in-the-loop confirmation required for any external actions.

## 5. Safety & operational constraints
- No autonomous external actions. All write or external effects require explicit human authorization.
- Principle of least privilege for every tool or adapter.
- Prompt injection mitigations: canonicalization, provenance checks, strict escaping, and token limits.
- Mandatory differential logging and redaction for PII.
- Rate limiting and throttling for model usage.

## 6. Privacy & compliance
- Data minimization and retention policies for Ephemeral Memory (short TTL).
- Persistent Knowledge Store contains only reviewed, consented, and anonymized material.
- Interfaces designed to support auditability for legal/regulatory review.

## 7. Acceptance criteria to move from design → implementation
- Verified secure runtime primitives exist: sandboxed tool adapters, immutable audit log, human-approval workflow.
- Threat model assessed and mitigations tested via red-team exercises.
- Clear ownership & escalation flows defined for incidents.
- Formal sign-off from safety, legal, and product stakeholders.

## 8. Staged implementation roadmap (high level)
1. Build governance primitives (policy engine, audit log, approval workflow).
2. Implement read-only tooling and a test-only sandbox with synthetic data.
3. Add ephemeral memory with strict TTL + redaction.
4. Run closed pilot with human oversight and monitoring.
5. Gradual expansion only after meeting acceptance criteria.

## 9. Open questions / risks
- Dependency on model behavior: how to detect and mitigate emergent undesired behaviors?
- Economic cost model for long-running agent orchestration.
- Human UX: how to present agent reasoning and provenance clearly for reviewers.

## 10. References & further reading
(kept intentionally minimal here; add links to company policies, threat models, or legal guidance when moving to implementation)