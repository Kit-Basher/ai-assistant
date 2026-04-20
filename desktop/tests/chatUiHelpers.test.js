import assert from "node:assert/strict";
import test from "node:test";

import { buildComposerPlaceholder, buildStatusSummary, buildStarterPrompts } from "../src/lib/chatUiHelpers.js";

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
