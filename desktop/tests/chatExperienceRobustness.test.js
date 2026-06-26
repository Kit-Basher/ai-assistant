import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const chatExperience = readFileSync(new URL("../src/components/ChatExperience.jsx", import.meta.url), "utf8");
const app = readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");

test("ChatExperience pins the actual transcript container without scrollIntoView", () => {
  assert.match(chatExperience, /const transcriptRef = useRef\(null\)/);
  assert.match(chatExperience, /function isNearBottom\(container\)/);
  assert.match(chatExperience, /function scrollToBottom\(container\)/);
  assert.match(chatExperience, /container\.scrollTop = container\.scrollHeight/);
  assert.doesNotMatch(chatExperience, /\.scrollIntoView\s*\(/);
  assert.match(chatExperience, /className="chat-transcript" onScroll=\{updateStickiness\} ref=\{transcriptRef\}/);
});

test("ChatExperience force-scrolls after user send and preserves passive reading", () => {
  assert.match(chatExperience, /forceNextScrollRef\.current = true/);
  assert.match(chatExperience, /userTurnInProgressRef\.current = true/);
  assert.match(chatExperience, /maybeScrollToBottom\(\{ force: true \}\)/);
  assert.match(chatExperience, /if \(!force && !shouldStickToBottomRef\.current\) return/);
  assert.match(chatExperience, /requestAnimationFrame/);
});

test("ChatExperience exposes busy and approval states without enabling duplicate sends", () => {
  assert.match(chatExperience, /function ThinkingBubble\(\)/);
  assert.match(chatExperience, /disabled=\{chatBusy \|\| !draft\.trim\(\)\}/);
  assert.match(chatExperience, /disabled=\{disabled\} onClick=\{\(\) => onReply\(confirmation\.approveCommand\)\}/);
  assert.match(chatExperience, /disabled=\{disabled\} onClick=\{\(\) => onReply\(confirmation\.cancelCommand\)\}/);
});

test("App surfaces send failure and supports transcript export", () => {
  assert.match(app, /catch \(error\) \{/);
  assert.match(app, /I ran into a problem:/);
  assert.match(app, /const exportConversation = \(\) => \{/);
  assert.match(app, /personal-agent-chat-\$\{Date\.now\(\)\}\.json/);
  assert.doesNotMatch(app, /importConversation/);
});
