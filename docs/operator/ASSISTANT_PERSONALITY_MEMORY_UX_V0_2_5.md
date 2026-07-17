# Assistant Personality, Memory, and Capability UX v0.2.5

## Reported Issue

After v0.2.4, the live Telegram prompt “what can you help me do?” returned an
internal architecture explanation about the assistant and agent layer. That was
technically accurate but not appropriate for normal users.

## User-Facing Capability Answer

Capability questions now receive a concise assistant-facing summary:

- everyday questions;
- project planning;
- coding and debugging;
- writing, summarizing, and rewriting;
- research with sources when available;
- local system checks;
- useful long-term memory;
- approval-gated changes for files, messages, settings, and system actions.

Internal terms such as “agent layer,” “control plane,” “orchestrator,” and
“tool registry” are reserved for explicit architecture questions.

## Memory Policy

The assistant distinguishes memory requests before storing durable state.

Durable examples:

- “remember that my main PC has an RTX 2060”;
- “remember that I prefer Debian instructions”;
- “remember that my project is called Personal Agent”;
- “remember that I like concise shell commands.”

Temporary or low-value examples are not stored by default:

- “remember that I like pizza today”;
- “remember that I am sitting in my chair”;
- “remember that this random number is 48192.”

Sensitive examples are refused and should go to the secret store or a password
manager:

- Telegram bot tokens;
- passwords;
- API keys;
- payment card details.

Uncertain examples ask for confirmation rather than becoming hard facts.

## Clarifying Questions

The assistant asks one concrete follow-up when required details are missing,
for example recipient/channel for sending a message or the kind of project the
user wants to build. It does not ask for confirmation before harmless read-only
actions such as local memory or system inspection.

## Mutation Boundary

This patch does not weaken Plan Mode. File cleanup, message sending, service
changes, model changes, and other mutations still need a preview and explicit
confirmation through existing authorization gates.

## Verification

Run:

```bash
python scripts/assistant_personality_memory_smoke.py
python -m pytest -q tests/test_assistant_personality_memory_ux.py
```

The smoke verifies friendly capability answers, memory classification,
retrieval, forget behavior, clarification quality, mutating-action boundaries,
and internal-term leakage.
