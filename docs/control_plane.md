# Local Control Plane

The local control plane is the shared coordination surface for one machine and
one project. It is intentionally small: loopback-only HTTP, file-backed state,
atomic writes, no auth, no database, no queue, no websocket layer.

## Canonical Storage

These are the documented defaults on this machine:

- `/home/c/personal-agent/control/master_plan.md`
- `/home/c/personal-agent/control/DEVELOPMENT_TASKS.md`
- `/home/c/personal-agent/control/agent_events.jsonl`

Environment variables can override the directory or individual file paths, but
the project-local `control/` folder is the supported default.

## Actors

- `manager`: ChatGPT writes the plan, keeps priorities aligned, and can reset a
  task to `READY`, mark it `BLOCKED`, or close it with `DONE`.
- `codex`: Codex CLI claims implementation work, moves it to `IMPLEMENTED`, and
  can block its own task if needed.
- `kimi`: Kimi/Cascade claims verification work, moves an `IMPLEMENTED` task to
  `VERIFIED`, and can block it if a test fails.
- `unassigned`: task owner state before claim.

## Task Format

`DEVELOPMENT_TASKS.md` is markdown with one canonical fenced JSON block:

```json
[
  {
    "task_id": "T-001",
    "title": "Implement control plane",
    "owner": "unassigned",
    "status": "READY",
    "kind": "feature",
    "priority": 1,
    "depends_on": [],
    "summary": "Short human summary.",
    "files_expected": ["agent/control_plane.py"],
    "acceptance_criteria": ["Task index parses cleanly."]
  }
]
```

Field rules:

- `task_id`: stable string identifier.
- `title`: human-readable name.
- `owner`: `manager`, `codex`, `kimi`, or `unassigned`.
- `status`: `READY`, `CLAIMED`, `IMPLEMENTED`, `VERIFIED`, `BLOCKED`, `DONE`.
- `kind`: short task type such as `feature`, `bug`, `test`, or `docs`.
- `priority`: integer, lower values run first.
- `depends_on`: list of task ids that must be `DONE` first.
- `summary`: short progress note.
- `files_expected`: list of files expected to change.
- `acceptance_criteria`: list of concrete checks.

## Status Meaning

- `READY`: eligible to be claimed.
- `CLAIMED`: owned by one actor, work is in progress.
- `IMPLEMENTED`: code or docs are done, verification pending.
- `VERIFIED`: tester has validated the work.
- `BLOCKED`: work cannot continue yet.
- `DONE`: manager has closed it out.

## Claim / Update Flow

The workflow is deliberately narrow:

1. Manager keeps the task file current.
2. Codex or Kimi asks for `GET /control/tasks/next?owner=<owner>` or sends a JSON body with `owner`.
3. The actor claims the task with `POST /control/tasks/claim`.
4. The actor updates state with `POST /control/tasks/update`.
5. The tester records verification with `POST /control/tasks/update`.
6. Manager closes the task with `POST /control/tasks/update` to `DONE`.

Allowed transitions:

- `READY -> CLAIMED`
- `CLAIMED -> IMPLEMENTED` by the claimed owner
- `CLAIMED -> BLOCKED` by the claimed owner
- `IMPLEMENTED -> VERIFIED` by `kimi`
- `IMPLEMENTED -> BLOCKED` by `kimi`
- any task -> `READY`, `BLOCKED`, or `DONE` by `manager`
- `VERIFIED -> DONE` by `manager`

Invalid transitions fail closed.

## Event Log

`agent_events.jsonl` is the shared audit trail. Every appended event has:

- `ts`
- `actor`
- `type`
- `task_id`
- `status_before`
- `status_after`
- `message`
- `extra`

The file is JSONL, one JSON object per line. Blank lines are ignored on read.
Malformed lines fail closed with a clear error.

Common event types:

- `claim`
- `update`
- `comment`
- `progress`
- `failure`

## HTTP API

Read/write surfaces:

- `GET /control/master_plan`
- `PUT /control/master_plan`
- `GET /control/tasks`
- `PUT /control/tasks`
- `GET /control/events`
- `POST /control/events`

Task workflow surfaces:

- `GET /control/tasks/index`
- `GET /control/tasks/next?owner=codex`
- `GET /control/tasks/next` with a JSON body like `{"owner":"codex"}`
- `POST /control/tasks/claim`
- `POST /control/tasks/update`
- `POST /control/tasks/comment`

## Running

Start the service:

```bash
personal-agent-control
```

Or directly:

```bash
python -m agent.control_plane --port 18888
```

The service binds only to `127.0.0.1`.

## Bootstrap

Initialize missing files:

```bash
personal-agent-control --init
```

Or directly:

```bash
python -m agent.control_plane --init
```

Bootstrap creates missing versions of:

- `master_plan.md`
- `DEVELOPMENT_TASKS.md`
- `agent_events.jsonl`

## Configuration

Environment variables:

- `AGENT_CONTROL_PORT`
- `AGENT_CONTROL_DIR`
- `AGENT_CONTROL_MASTER_PLAN_PATH`
- `AGENT_CONTROL_TASKS_PATH`
- `AGENT_CONTROL_EVENTS_PATH`

If individual file paths are unset, they default under `AGENT_CONTROL_DIR`.
