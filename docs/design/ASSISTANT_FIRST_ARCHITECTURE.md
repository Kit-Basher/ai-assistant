# Assistant-First Architecture

This document defines the intended product shape for the stabilized baseline.

## 1. Assistant vs Agent split

- Assistant:
  - the single conversational entity the user talks to
  - owns continuity, memory-backed conversation, and normal user-facing chat
  - may call deterministic runtime capabilities and agent tools underneath
- Agent:
  - the runtime/control-plane substrate
  - owns setup, model/provider management, tool execution, skills, health, and recovery

The user should normally experience one assistant conversation, not a router or menu.

Assistant identity is authoritative. Models and providers are internal components of
the runtime, not the entity speaking to the user. Assistant replies should not claim
to be DeepSeek, GPT, OpenAI, or any other model/vendor identity.

Assistant tone should stay friendly, calm, competent, concise, and practical. The
assistant should feel like one consistent personal assistant, not a benchmark model,
shell wrapper, or canned support bot.

## 2. Normal interaction model

- Ordinary user messages go through the assistant flow.
- The assistant decides internally when to:
  - answer directly with the LLM
  - call deterministic runtime truth
  - call tool-backed operational capabilities
- Product routing remains internal implementation detail.
- Visible chooser/menu/meta-choice UX is not part of the normal SAFE MODE baseline.
- Runtime/provider/model state answers come from canonical runtime truth, not free-form
  LLM generation.
- Capability answers such as “what can you do?” are grounded in real agent
  subsystems like system inspection, runtime/model/provider truth, provider
  repair flows, local memory, and scheduler status rather than generic model traits.
- Model-availability answers are grounded in the live registry/runtime state, so the
  assistant describes only models/providers that actually exist and are usable or
  blocked right now.
- Model inventory distinguishes local/downloaded models from broader available models.
  Local/install prompts emphasize installed Ollama or other local chat models, while
  broader availability prompts may also mention usable remote models that are ready now.
- “Memory” is disambiguated: system-memory questions use the machine-inspection path,
  while agent-memory questions use the local memory store for preferences, anchors,
  open loops, and saved assistant state.
- Internal failures should be translated into assistant UX rather than exposing raw
  recovery/setup strings during normal conversation.
- Tool-backed facts may be interpreted on follow-up turns. The assistant can explain
  a prior deterministic/tool result in plain English, and may temporarily use a
  stronger available model for that explanation step while keeping the facts grounded
  in the original tool output or runtime truth.
- Native machine/system skills remain grounded underneath the assistant. Machine
  prompts such as hardware inventory, PC stats, RAM, and storage inspection use
  deterministic local skill/tool paths rather than generic LLM answers.
- Agent/runtime status and machine/computer status are distinct capability families.
  `runtime`, provider/model health, and scheduler/bot status come from control-plane
  truth, while PC/system/hardware prompts come from machine-inspection skills.
- Action/tool-intent routing is also deterministic when the runtime already has a
  real capability. Time/date prompts use local runtime time, model-scout/evaluation
  prompts use the real Model Scout action, and deeper machine-inspection verbs stay on
  the grounded machine-inspection path.
- Model Scout v2 is assistant-facing and deterministic. It ranks real runtime-available
  models, recommends only meaningful upgrades, asks before switching, and can roll
  back to the previous model after a trial switch. Optional external discovery may
  suggest downloadable candidates, but it must not claim a model is installed unless
  the runtime really has it.
- Short action follow-ups like `check them` or `dig deeper` should reuse the most
  recent relevant deterministic context when it is clear enough. If more specificity is
  needed, the assistant should ask one concrete clarification instead of falling back
  to generic router-like prompts.
- Follow-up interpretation requests such as worry/concern/normality questions should
  also reuse the most recent grounded tool result rather than falling back to generic
  chat or raw error text.
- Memory is used selectively. Ordinary chat may receive a small amount of relevant
  remembered context when it helps continuity, but memory should not be dumped into
  every reply.
- “What do you remember?” style answers are summarized from real local memory content
  such as preferences, open loops, thread anchors, and active work context rather
  than implementation details about the database.
- Agent-memory summaries and system-context summaries stay grounded in the real local
  memory store plus canonical runtime truth underneath the assistant persona.

In the current codebase this means:

- UI/API `POST /chat`
  -> `AgentRuntime.chat()`
  -> `Orchestrator.handle_message()`
- Telegram ordinary text
  -> local API `/chat`
  -> same `AgentRuntime.chat()`
  -> same `Orchestrator.handle_message()`

The guardrail for this baseline is `AgentRuntime.should_use_assistant_frontdoor()`.
In SAFE MODE, ordinary user chat should stay on the assistant frontdoor whenever a
usable chat target exists. Router/menu clarification UX is not supposed to surface
for normal user prompts in that state.

For grounded repair/help follow-ups after a recent unhealthy provider/model reply,
the assistant should also stay in assistant mode instead of surfacing chooser UX.
Those turns are expected to reuse the recent runtime context and respond with a
concise repair/status or rollback-oriented answer.

Legacy clarification and thread-integrity prompts are compatibility behavior for
non-frontdoor paths only. When the assistant frontdoor owns a SAFE MODE turn, stale
clarification state should be cleared before deterministic handling continues.

## 3. Failure / no-LLM mode

- When a usable chat model exists, the assistant stays in front.
- When no usable LLM exists, the system may expose raw recovery/setup guidance instead of pretending the assistant can answer normally.
- Recovery/setup mode is the exception path, not the main product surface.

For the current SAFE MODE baseline:

- a healthy pinned local model keeps the assistant in front
- if that pinned local model is unavailable, generic chat falls back to recovery/setup guidance

## 4. Transport model

- UI and Telegram are thin transports.
- They should not contain major product logic.
- They should forward ordinary text into the same assistant service.
- Deterministic transport-specific commands may still exist, but ordinary conversation should converge on the same assistant-first path.

## 5. First-run flow

- First-run and repair focus on local Ollama first.
- Detect a usable local Ollama/model.
- If missing or unhealthy, surface repair/setup guidance.
- Once a usable model exists, the assistant takes over and remains the normal front door.

## 6. SAFE MODE baseline

SAFE MODE is the current trustworthy assistant-first baseline:

- one pinned local model
- no remote fallback
- no background default mutation
- no proactive control-plane notifications
- deterministic runtime/provider/setup/system capabilities remain available underneath the assistant
- deeper follow-ups after a machine inspection, such as `show me more` or `learn more`,
  stay on the deterministic machine-inspection path
- day-planning answers stay user-facing and summarize open loops plus active tasks
- `Orchestrator._handle_runtime_truth_chat()` is the shared deterministic-capability layer the assistant can use for runtime/provider/model/setup answers without calling the LLM
- chooser/meta routing is suppressed for normal chat while the pinned model is healthy
- ordinary SAFE MODE prompts do not surface visible thread/new-thread disambiguation
- repair/help follow-ups tied to a recent unhealthy provider/model state reuse that
  grounded context instead of surfacing router/menu chooser text
- SAFE MODE keeps one containment gate ahead of generic chat/bootstrap fallthrough,
  so grounded runtime/setup/repair turns do not escape into legacy help or bootstrap
  behavior when canonical truth is already available
- if the assistant asks the user to confirm a specific model switch, that exact confirmed target overrides automatic selection for that one switch action
- safe-mode target reporting distinguishes the configured pin from the effective active target when the configured pin is invalid or when an explicit confirmed switch override is active
- assistant-driven model trials keep the previous exact target so `switch back` can
  restore it deterministically after a confirmed switch
- when the assistant frontdoor is active, raw internal errors are translated into assistant-facing responses instead of surfacing `Chat LLM is unavailable` or direct setup commands mid-conversation
- system/model/runtime questions have a final grounded safety net, so missed routing
  should still return deterministic runtime truth or a truthful runtime-state
  limitation instead of raw LLM/environment boilerplate
- model-management truth is split by responsibility:
  inventory (`model_inventory_status()`), readiness (`model_readiness_status()`),
  controller policy, explicit controller actions, and scout advice. The older
  merged availability view remains compatibility-only and should not be the primary
  assistant/runtime truth source.

## 7. Deferred work

This pass does not attempt to solve the full dynamic/autonomous system. It intentionally defers:

- re-enabling background autonomy
- model scout/watch/autopilot recovery as a primary UX
- richer multi-provider assistant arbitration outside SAFE MODE
- a full replacement of all router-era internals
