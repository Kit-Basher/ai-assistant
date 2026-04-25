# Master Plan

This file is the canonical high-level plan for the local control plane.

Current objective:
- Make the assistant release-ready for normal human use on one machine.

Current priorities:
- Keep the local-first, file-backed, loopback-only philosophy intact.
- Make the model onboarding path obvious for Ollama and online API users.
- Make skill and pack safety visible before anything is imported or enabled.
- Keep the control plane deterministic and easy for local agents to follow.
- Keep the release gate green and document the exact operator flow.
- Run and maintain a tracked barrage of user-facing assistant interactions until bad agent-style replies are gone.
- Polish long-session continuity so rewind, recap, and correction turns sound like one steady assistant voice.

Operating model:
- ChatGPT manager updates this file and the task list.
- Codex implements the smallest safe code changes.
- Kimi verifies the result and reports blind spots.

Canonical storage:
- `control/master_plan.md`
- `control/DEVELOPMENT_TASKS.md`
- `control/agent_events.jsonl`
