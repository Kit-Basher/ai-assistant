# Safe Mode

Safe mode is a minimal, reversible runtime baseline for debugging and trust restoration.

## What it does

When `AGENT_SAFE_MODE=1` is set:

- chat is pinned to one local chat model
- remote fallback is disabled
- API chat and Telegram chat use the same pinned target
- background automation scheduler is disabled
- model scout / model watch background runs are skipped
- autopilot notifications are suppressed
- self-heal and bootstrap auto-apply are disabled
- deterministic runtime/setup/status answers still use runtime state and do not require the LLM

## Recommended env

```bash
export AGENT_SAFE_MODE=1
export AGENT_SAFE_MODE_CHAT_MODEL=ollama:qwen3.5:4b
```

If `AGENT_SAFE_MODE_CHAT_MODEL` is not set, the runtime falls back to:

1. `OLLAMA_MODEL`
2. the current configured local chat model
3. the best available local chat-capable model already present in the registry

## How to run

Example:

```bash
AGENT_SAFE_MODE=1 \
AGENT_SAFE_MODE_CHAT_MODEL=ollama:qwen3.5:4b \
python -m agent.api_server
```

Or for the existing service environment, set the same variables in the unit environment and restart the service.

## What to expect

- `what model are you using?` should report the pinned Ollama model
- `tell me a joke` and other generic chat requests should stay on that same model
- `runtime` / health / setup questions should still resolve deterministically
- `agent doctor`, memory, and storage questions should stay deterministic/tool-backed
- API `/chat` and Telegram ordinary text should agree on the same effective pinned target
- `/ready` and `/runtime` should report the same effective SAFE MODE target, plus safe-mode target details when the configured pin is invalid or an explicit confirmed override is active
- short clarification replies are consumed once and cleared instead of repeating the same prompt forever
- unsolicited background model-change notifications should stop while safe mode is enabled

## Current baseline

Safe mode is the trusted stabilization baseline for integrated testing.

In safe mode:

- one local chat model is pinned and used by both API and Telegram generic chat
- normal user messages stay in one assistant conversation while that pinned model is healthy
- deterministic runtime/provider/setup/model/doctor/system-observe prompts bypass the LLM
- SAFE MODE assistant containment and generic chat availability are separate:
  deterministic grounded turns stay on the assistant frontdoor even if the current
  chat target is unhealthy, while generic freeform chat still requires a ready chat target
- background control-plane mutation and proactive control-plane notifications stay suppressed
- visible chooser/meta routing is suppressed for normal chat while the pinned model is healthy
- ordinary user messages do not surface visible thread/new-thread disambiguation while the pinned model is healthy
- repair/help follow-ups after a recent unhealthy provider/model answer stay assistant-facing
  and reuse that runtime context instead of surfacing chooser/menu UX
- assistant-frontdoor turns clear stale legacy clarification/thread state before
  deterministic SAFE MODE handling continues
- raw recovery/setup guidance appears only when there is no usable pinned local chat target
- an explicit user-confirmed model switch applies that exact offered target for that action instead of being replaced by automatic selection

## Safe-mode target truth

- The configured safe-mode pin comes from `AGENT_SAFE_MODE_CHAT_MODEL` when set.
- The effective safe-mode target is what the runtime can actually use right now.
- If the configured pin is valid, it becomes the effective target and is what `/ready`,
  `/runtime`, API chat, and Telegram chat should report.
- If the configured pin is invalid or unavailable, the runtime reports that clearly and
  only falls back to another local chat target when one is actually usable.
- If the assistant asks for explicit confirmation to switch to a specific model and the
  user confirms, that exact confirmed target temporarily overrides the normal safe-mode
  pinning logic for that one switch action.
