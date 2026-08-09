import assert from "node:assert/strict";
import test from "node:test";

import { buildAssistantMessage, extractTaskUi } from "../src/lib/chatUiHelpers.js";


const fixture = {
  setup: {
    task: {
      schema_version: "personal-agent.task.v1",
      task_id: "task-123",
      goal: "Inspect system and model state",
      state: "running",
      revision: 4,
      plan_version: 1,
      current_step: 1,
      steps: [
        { step_id: "step-1", capability_id: "system.status", status: "completed", mode: "read_only", verifier_status: "pass" },
        { step_id: "step-2", capability_id: "models.inventory", status: "running", mode: "read_only" }
      ]
    }
  },
  assistant: { content: "Task progress is current." }
};


test("task UI is derived from canonical response state", () => {
  const task = extractTaskUi(fixture);
  assert.equal(task.state, "running");
  assert.equal(task.steps.length, 2);
  assert.equal(task.steps[0].verifier_status, "pass");
});

test("assistant messages retain a compact task card model", () => {
  const message = buildAssistantMessage(fixture);
  assert.equal(message.ui.task.task_id, "task-123");
  assert.equal(message.ui.task.goal, "Inspect system and model state");
});

test("unknown task states fail closed instead of appearing healthy", () => {
  assert.equal(extractTaskUi({ setup: { task: { state: "model_says_done" } } }), null);
});

test("task UI excludes raw evidence, prompts, secrets, and plan hashes", () => {
  const payload = structuredClone(fixture);
  payload.setup.task.raw_prompt = "hidden reasoning";
  payload.setup.task.secret = "secret-token";
  payload.setup.task.plan_hash = "private-hash";
  payload.setup.task.steps[0].evidence = { content: "sensitive file body" };
  const serialized = JSON.stringify(extractTaskUi(payload));
  assert.equal(serialized.includes("hidden reasoning"), false);
  assert.equal(serialized.includes("secret-token"), false);
  assert.equal(serialized.includes("private-hash"), false);
  assert.equal(serialized.includes("sensitive file body"), false);
});

test("normal task UI ignores nested terminal outcome evidence", () => {
  const task = extractTaskUi({ setup: { task: {
    task_id: "task-safe", goal: "Inspect two things", state: "succeeded",
    outcome: { status: "succeeded", evidence: [{ raw: "secret-token-value" }] },
    steps: [],
  } } });
  assert.equal(JSON.stringify(task).includes("secret-token-value"), false);
  assert.equal(Object.hasOwn(task, "outcome"), false);
});
