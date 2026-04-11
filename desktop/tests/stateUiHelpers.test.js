import assert from "node:assert/strict";
import test from "node:test";

import { buildStateView } from "../src/lib/stateUiHelpers.js";

test("buildStateView keeps the state snapshot compact and assistant-shaped", () => {
  const view = buildStateView({
    updated_at: "2026-04-08T12:34:56+00:00",
    runtime: {
      status: "ready",
      summary: "Ready. Using ollama / ollama:qwen2.5:3b-instruct.",
      next_action: "No action needed."
    },
    model: {
      provider: "ollama",
      model: "ollama:qwen2.5:3b-instruct",
      path: "ollama / ollama:qwen2.5:3b-instruct",
      routing_mode: "auto",
      health: "up"
    },
    conversation: {
      topic: null,
      recent_request: null,
      open_loop: null
    },
    action: {
      pending_approval: false,
      blocked_reason: null,
      last_action: null
    },
    signals: {
      response_style: "concise",
      confidence_visible: false
    }
  });

  assert.equal(view.summary, "Ready. Using ollama / ollama:qwen2.5:3b-instruct.");
  assert.equal(view.updatedAt, "2026-04-08T12:34:56+00:00");
  assert.deepEqual(
    view.cards.map((card) => card.title),
    ["Runtime", "Model Path", "Conversation", "Action State", "Signals"]
  );
  assert.match(view.cards[0].lines.join("\n"), /status: ready/);
  assert.match(view.cards[0].lines.join("\n"), /next action: No action needed\./);
  assert.match(view.cards[2].lines.join("\n"), /topic: n\/a/);
  assert.match(view.cards[3].lines.join("\n"), /pending approval: no/);
  assert.match(view.cards[4].lines.join("\n"), /response style: concise/);
  assert.match(view.cards[4].lines.join("\n"), /confidence visible: no/);
  assert.ok(!view.summary.includes("Agent is"));
  assert.ok(!view.summary.includes("OpenAI"));
});

test("buildStateView degrades gracefully when state is unavailable", () => {
  const view = buildStateView(null);

  assert.equal(view.summary, "State snapshot unavailable.");
  assert.equal(view.updatedAt, "n/a");
  assert.equal(view.cards[0].lines[1], "summary: n/a");
  assert.equal(view.cards[1].lines[1], "model: n/a");
  assert.equal(view.cards[2].lines[0], "topic: n/a");
  assert.equal(view.cards[3].lines[1], "blocked reason: n/a");
  assert.equal(view.cards[4].lines[0], "response style: n/a");
});
