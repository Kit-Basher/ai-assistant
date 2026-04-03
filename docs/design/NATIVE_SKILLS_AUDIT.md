# Native Skills Audit

This audit reflects the current assistant-first SAFE MODE baseline as implemented in
code. It focuses on the native deterministic skills and adjacent internal handlers
that the assistant can call underneath normal conversation.

## 1. Native skills currently implemented

### User-facing machine / system inspection

| Skill | Entrypoint | What it returns today | Deterministic/tool-backed | Example prompts |
| --- | --- | --- | --- | --- |
| `hardware_report` | [skills/hardware_report/handler.py](/home/c/personal-agent/skills/hardware_report/handler.py) `hardware_report()` | Local hardware and machine inventory: CPU model, GPU visibility, total RAM, storage mounts, OS/kernel/arch, uptime | Yes | `what CPU do I have?`, `can you see the GPU?`, `what are my PC specs?` |
| `resource_governor` | [skills/resource_governor/handler.py](/home/c/personal-agent/skills/resource_governor/handler.py) `resource_report()` | CPU load, memory, swap, and top process/resource snapshot from stored observe data | Yes | `how much memory am I using?`, `ram usage`, `how is cpu and memory?` |
| `storage_governor` | [skills/storage_governor/handler.py](/home/c/personal-agent/skills/storage_governor/handler.py) `storage_report()` | Disk usage, mount totals, top directories, and growth deltas from stored observe data | Yes | `how is my storage?`, `disk usage`, `what changed on my disk?` |
| `disk_pressure_report` | [skills/disk_pressure_report/handler.py](/home/c/personal-agent/skills/disk_pressure_report/handler.py) `disk_pressure_report()` | Focused disk-pressure view with largest files / growth culprits | Yes | `show top growing paths`, `largest files`, `disk pressure` |
| `network_governor` | [skills/network_governor/handler.py](/home/c/personal-agent/skills/network_governor/handler.py) `network_report()` | DNS, route, and interface/network health snapshot | Yes | `network health`, `dns status`, `latency` |
| `service_health_report` | [skills/service_health_report/handler.py](/home/c/personal-agent/skills/service_health_report/handler.py) `service_health_report()` | Personal-agent service state and recent logs | Yes | `service health`, `agent service status` |

### User-facing memory / planning / recall

| Skill or handler | Entrypoint | What it returns today | Deterministic/tool-backed | Example prompts |
| --- | --- | --- | --- | --- |
| Agent memory summary | [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) `_assistant_memory_overview_response()` | Preferences, open loops, saved context, anchors, and remembered system context | Yes | `what do you remember?`, `what are we working on?`, `show my open loops` |
| Day planning | [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) `_today_cards_payload()` | User-facing priorities built from open loops plus active tasks/tasks.md | Yes | `plan my day`, `help me plan today`, `what should I work on today?` |
| `core` | [skills/core/handler.py](/home/c/personal-agent/skills/core/handler.py) `daily_plan()`, `next_best_task()`, task/project helpers | Structured DB-backed task/project actions; mostly internal or command-style today | Yes | internal/task flows, not the main assistant surface |
| `recall` | [skills/recall/handler.py](/home/c/personal-agent/skills/recall/handler.py) `ask_query()` | DB-backed recall/search answers | Yes | internal memory-query paths |

### Runtime / control-plane truth

These are deterministic but not manifest-backed skills. They are important native
capabilities in the current product surface.

| Capability | Entrypoint | What it returns today | Deterministic/tool-backed | Example prompts |
| --- | --- | --- | --- | --- |
| Runtime/provider/model truth | [agent/runtime_truth_service.py](/home/c/personal-agent/agent/runtime_truth_service.py) via [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) `_handle_runtime_truth_chat()` | Active model/provider, provider status, runtime readiness, model availability, setup truth | Yes | `what model are you using?`, `runtime`, `openrouter health` |
| Agent doctor | [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) `_tool_handler_doctor()` | Deterministic doctor/repair summary | Yes | `agent doctor` |
| Agent/service status | [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) `_tool_handler_status()` | Bot/scheduler/DB style agent status | Yes | `bot status`, `service status` |
| Local system health fallback | [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) `_tool_handler_observe_system_health()` | Machine/service/network snapshot from `collect_system_health()` | Yes | internal fallback when no narrower observe skill is selected |

### Other loaded native skills

These are loaded today but are not the primary surface in the current SAFE MODE
assistant-first baseline:

- `observe_now` in [skills/observe_now/handler.py](/home/c/personal-agent/skills/observe_now/handler.py)
- `knowledge_query` in [skills/knowledge_query/handler.py](/home/c/personal-agent/skills/knowledge_query/handler.py)
- `reflection` in [skills/reflection/handler.py](/home/c/personal-agent/skills/reflection/handler.py)
- `opinion` in [skills/opinion/handler.py](/home/c/personal-agent/skills/opinion/handler.py)
- `opinion_on_report` in [skills/opinion_on_report/handler.py](/home/c/personal-agent/skills/opinion_on_report/handler.py)
- `ops_supervisor` in [skills/ops_supervisor/handler.py](/home/c/personal-agent/skills/ops_supervisor/handler.py)
- `git` in [skills/git/handler.py](/home/c/personal-agent/skills/git/handler.py)

These remain useful, but they are not the main explanation path for the user-facing
machine/runtime prompts covered in this pass.

## 2. Native skills expected but previously missing or partial

- Hardware inventory was only partially reconnected. The manifest-backed
  `hardware_report` skill existed, but machine/hardware prompts were not strongly
  shaped toward it and broad machine-inspection prompts were not composed well.
- Broader machine inspection was partial. Prompts like `what other PC stats can you
  find?` needed a combined machine view, not just coarse observe fallback.
- Deeper follow-up inspection was missing. After a system-inspection result, prompts
  like `can you learn more?`, `show me more`, or `dig deeper` did not reliably stay
  on the deterministic machine path.
- Day planning existed, but the current response shape was too task-storage oriented
  and did not consistently summarize open loops / priorities in a user-facing way.

## 3. Misroutes seen before this pass

- Machine-status prompts could produce the wrong flavor of answer:
  - expected: machine stats / hardware / storage / RAM
  - observed risk: agent-service / bot / scheduler / DB flavored status
- Hardware inventory prompts could include the wrong companion skills:
  - expected: hardware-first inventory
  - observed risk: resource snapshot mixed in too aggressively for `what CPU/GPU do I have?`
- Follow-ups like `can you run a check and see if you can learn more?` were not
  anchored to the previous machine-inspection result.
- Day-planning prompts like `can you help me plan my day?` matched the planning
  intent internally but still produced a storage-flavored task listing instead of a
  concise prioritization answer.

## 4. Narrow fixes implemented in this pass

- Broadened machine/hardware prompt matching in
  [agent/nl_router.py](/home/c/personal-agent/agent/nl_router.py):
  - hardware-only prompts now select `hardware_report` first
  - broader machine-inspection prompts now select a combined deterministic set:
    `hardware_report`, `resource_report`, `storage_report`
- Added hardware-report summary integration in
  [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) so the
  assistant summary/follow-ups for hardware inventory are grounded and specific.
- Added deterministic deeper machine follow-up handling in
  [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py), so
  `learn more` / `show me more` / `dig deeper` can extend the last machine
  inspection without falling into generic chat.
- Refined day-planning output in
  [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py) so it now
  summarizes open loops and active tasks as user-facing priorities instead of
  referring to internal task storage.

## 5. Agent-status vs machine-status distinction

The intended split is now:

- Agent / assistant runtime status:
  - runtime truth, readiness, provider/model state, bot/service/scheduler health
  - examples: `runtime`, `is everything working with the agent?`, `agent health`
- Machine / PC status:
  - CPU, GPU, RAM, storage, live machine stats, deeper local inspection
  - examples: `what other pc stats can you find?`, `what CPU and GPU do I have?`,
    `how is my storage?`

Code boundaries:

- Agent/runtime status classification:
  [agent/setup_chat_flow.py](/home/c/personal-agent/agent/setup_chat_flow.py)
- Machine-status NL routing:
  [agent/nl_router.py](/home/c/personal-agent/agent/nl_router.py)
- Deterministic machine execution:
  [agent/orchestrator.py](/home/c/personal-agent/agent/orchestrator.py)

## 6. What remains intentionally deferred

- A fuller machine-health “deep inspect” family beyond hardware/resource/storage
  composition
- Richer user-facing planning beyond open loops plus active tasks
- Re-exposing dormant/older native skills that are not part of the current
  assistant-first SAFE MODE baseline
