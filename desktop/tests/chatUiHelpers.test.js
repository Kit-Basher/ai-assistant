import assert from "node:assert/strict";
import test from "node:test";

import {
  buildAssistantMessage,
  buildComposerPlaceholder,
  buildStatusSummary,
  buildStarterPrompts
} from "../src/lib/chatUiHelpers.js";

test("buildStatusSummary treats chat_usable as ready even when ready is false", () => {
  const status = buildStatusSummary({
    ready: false,
    chat_usable: true,
    phase: "ready",
    runtime_mode: "DEGRADED",
    onboarding: { summary: "Setup is not fully ready yet." },
    recovery: { summary: "Something needs attention." }
  });

  assert.equal(status.label, "Ready");
  assert.equal(status.tone, "ready");
  assert.equal(status.ready, true);
  assert.equal(status.description, "Ready to help.");
});

test("buildStarterPrompts and composer placeholder follow chat usability rather than stale setup state", () => {
  const readyState = {
    ready: false,
    chat_usable: true,
    phase: "ready",
    runtime_mode: "DEGRADED"
  };

  assert.deepEqual(buildStarterPrompts(readyState), [
    "Help me plan today",
    "Summarize something for me",
    "Draft a reply",
    "Explain an error message"
  ]);
  assert.equal(buildComposerPlaceholder(readyState), "Message Personal Agent");
});

test("buildAssistantMessage prefers assistant content and strips internal fallback payloads", () => {
  const message = buildAssistantMessage({
    ok: false,
    error_kind: "internal_error",
    assistant: { content: "" },
    message: "{\"trace_id\":\"abc123\",\"route_reason\":\"generic_chat\"}",
    error: "trace_id: abc123"
  });

  assert.equal(message.role, "assistant");
  assert.equal(message.content, "I couldn't complete that request.");
  assert.equal(message.tone, "error");
});

test("buildAssistantMessage keeps normal assistant-facing setup text", () => {
  const message = buildAssistantMessage({
    ok: true,
    assistant: { content: "I can switch chat to ollama:qwen2.5:7b-instruct now." },
    setup: {
      type: "confirm_switch_model",
      title: "Use this model?",
      prompt: "I can switch chat to ollama:qwen2.5:7b-instruct now."
    }
  });

  assert.equal(message.content, "I can switch chat to ollama:qwen2.5:7b-instruct now.");
  assert.equal(message.ui.confirmation?.title, "Use this model?");
});

test("buildAssistantMessage presents canonical Plan approval in plain language", () => {
  const message = buildAssistantMessage({
    ok: true,
    assistant: { content: "I can remove the selected local pack. Reply confirm to proceed." },
    setup: {
      type: "pack_remove_plan",
      plan_mode: {
        action_type: "pack.remove",
        rollback_supported: false,
        resources: { created: [], changed: [], deleted: ["registered pack version"] }
      }
    }
  });

  assert.equal(message.ui.operationState, "waiting_for_confirmation");
  assert.equal(message.ui.confirmation?.title, "Review this destructive change");
  assert.equal(message.ui.confirmation?.approveLabel, "Approve destructive change");
  assert.match(message.ui.confirmation?.riskSummary || "", /no automatic rollback/i);
});

test("capability rescue states that catalog discovery cannot remotely install", () => {
  const message = buildAssistantMessage({
    ok: true,
    assistant: { content: "I found untrusted catalog metadata." },
    setup: {
      capability_gap_rescue: {
        type: "capability_gap_rescue",
        candidate_packs: [{ name: "Example", source_id: "catalog", remote_id: "example" }]
      }
    }
  });

  assert.equal(message.ui.capability?.type, "rescue");
  assert.equal(message.ui.capability?.installAllowedInitially, false);
});
