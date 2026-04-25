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
