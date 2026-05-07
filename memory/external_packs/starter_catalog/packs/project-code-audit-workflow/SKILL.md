---
name: Project and Code Audit Workflow
description: Structure project and code audits around risks, tests, maintainability, and user-facing behavior.
---

# Project and Code Audit Workflow

Use this pack when the user wants a project audit, code audit, repository review, or implementation review plan.

Rules:
- Stay text-only. Do not edit code, run tests, install dependencies, or execute tooling from this pack.
- Prioritize bugs, regressions, security risks, missing tests, and operational hazards.
- Keep findings grounded in concrete files, behavior, or evidence when available.

Workflow:
1. Define audit scope and risk areas.
2. Inventory architecture, entry points, tests, configuration, and deployment surfaces.
3. Review behavior before style.
4. Report findings by severity with evidence and suggested remediation.
5. Identify test gaps and residual risk.
