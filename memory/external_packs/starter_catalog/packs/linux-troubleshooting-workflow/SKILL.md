---
name: Linux Troubleshooting Workflow
description: Structure Linux diagnostics and repair planning before any command execution.
---

# Linux Troubleshooting Workflow

Use this pack when the user asks for Linux debugging, troubleshooting, or repair guidance.

Rules:
- Stay text-only. Do not run commands, edit system files, install packages, restart services, or escalate privileges from this pack.
- Separate read-only diagnostics from mutating repair actions.
- Prefer reversible checks and clearly state risk before changes.
- Treat destructive commands, privilege escalation, package installs, and service restarts as separate confirmation-gated actions.

Workflow:
1. Capture symptoms, recent changes, distribution, environment, and impact.
2. Build a read-only evidence plan.
3. Interpret findings conservatively.
4. Propose the least risky repair first.
5. Define rollback and success checks before any mutation.
