# Development Tasks

This file is the canonical machine-readable task list for the local control plane.

```json
[
  {
    "task_id": "PAI-401",
    "title": "Document model onboarding and pack safety",
    "owner": "manager",
    "status": "DONE",
    "kind": "docs",
    "priority": 1,
    "depends_on": [],
    "summary": "Make the Ollama, online API, and skill safety paths obvious for a normal human.",
    "files_expected": ["README.md", "docs/operator/SETUP.md", "docs/control_plane.md"],
    "acceptance_criteria": [
      "Docs explain the canonical model setup path.",
      "Docs mention how to inspect pack and skill safety.",
      "Docs point to the control plane workflow without ambiguity."
    ]
  },
  {
    "task_id": "PAI-402",
    "title": "Verify runtime-aware CLI defaults",
    "owner": "codex",
    "status": "DONE",
    "kind": "test",
    "priority": 2,
    "depends_on": [],
    "summary": "Keep the CLI pointed at the right local API for dev and stable installs.",
    "files_expected": ["agent/cli.py", "tests/test_agent_cli.py"],
    "acceptance_criteria": [
      "Dev checkout resolves the dev API base URL by default.",
      "Pack safety summary is available from the CLI.",
      "Tests cover the default and the summary command."
    ]
  },
  {
    "task_id": "PAI-403",
    "title": "Run release gate and close regressions",
    "owner": "kimi",
    "status": "DONE",
    "kind": "test",
    "priority": 3,
    "depends_on": ["PAI-402"],
    "summary": "Re-run the release gate after the CLI and control-plane fixes.",
    "files_expected": ["tests/test_publishability_smoke.py", "scripts/release_gate.py"],
    "acceptance_criteria": [
      "Release gate passes on the current branch.",
      "Any timing or setup regressions are either fixed or explicitly documented."
    ]
  },
  {
    "task_id": "PAI-404",
    "title": "Track and harden user-facing assistant interaction barrage",
    "owner": "codex",
    "status": "DONE",
    "kind": "test",
    "priority": 1,
    "depends_on": ["PAI-403"],
    "summary": "Create a living interaction matrix and repeatedly run a broad barrage until user-facing chat stops leaking agent/control-plane behavior.",
    "files_expected": [
      "control/ASSISTANT_INTERACTION_HARDENING.md",
      "scripts/assistant_interaction_barrage.py",
      "agent/orchestrator.py",
      "agent/public_chat.py",
      "agent/api_server.py"
    ],
    "acceptance_criteria": [
      "A tracked matrix exists for real-world user-facing assistant scenarios.",
      "A runnable barrage script exercises both WebUI and Telegram surfaces.",
      "Barrage failures are turned into concrete code fixes and rerun until the active matrix is clean."
    ]
  },
  {
    "task_id": "PAI-405",
    "title": "Polish long-session continuity voice",
    "owner": "codex",
    "status": "DONE",
    "kind": "behavior",
    "priority": 1,
    "depends_on": ["PAI-404"],
    "summary": "Tighten resume, rewind, and correction replies so they read like one helpful assistant instead of a memory subsystem.",
    "files_expected": [
      "agent/orchestrator.py",
      "scripts/assistant_interaction_barrage.py",
      "control/ASSISTANT_INTERACTION_HARDENING.md",
      "tests/test_orchestrator.py"
    ],
    "acceptance_criteria": [
      "Working-context recap replies use natural assistant phrasing.",
      "The barrage explicitly covers rewind and correction-style continuity prompts.",
      "Live barrage remains clean after the wording changes."
    ]
  },
  {
    "task_id": "PAI-406",
    "title": "Expand release soak coverage for mixed conversations",
    "owner": "codex",
    "status": "DONE",
    "kind": "test",
    "priority": 2,
    "depends_on": ["PAI-405"],
    "summary": "Add longer mixed-topic conversations that try to drag the assistant back into control-plane or brittle persona behavior.",
    "files_expected": [
      "scripts/assistant_interaction_barrage.py",
      "scripts/assistant_viability_smoke.py",
      "control/ASSISTANT_INTERACTION_HARDENING.md"
    ],
    "acceptance_criteria": [
      "At least one longer mixed-topic soak path is tracked and runnable.",
      "The soak path checks both continuity and personality stability.",
      "Any failures are promoted into deterministic regressions."
    ]
  },
  {
    "task_id": "PAI-407",
    "title": "Polish generic chat tone in long sessions",
    "owner": "codex",
    "status": "IN_PROGRESS",
    "kind": "behavior",
    "priority": 2,
    "depends_on": ["PAI-406"],
    "summary": "Trim robotic or self-referential generic-chat phrasing that still shows up in some long conversational openings and summary turns.",
    "files_expected": [
      "agent/orchestrator.py",
      "agent/public_chat.py",
      "scripts/assistant_viability_smoke.py",
      "control/ASSISTANT_INTERACTION_HARDENING.md"
    ],
    "acceptance_criteria": [
      "Long-session generic chat avoids robotic self-description when a simpler assistant reply would do.",
      "Recap and summary turns stay in assistant voice across WebUI and Telegram.",
      "Focused live long-session checks remain green after tone tightening."
    ]
  }
]
```
