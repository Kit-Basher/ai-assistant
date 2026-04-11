import assert from "node:assert/strict";
import test from "node:test";

import { buildPacksView } from "../src/lib/packStateUiHelpers.js";

test("buildPacksView renders installed, available, and blocked pack state compactly", () => {
  const view = buildPacksView({
    updated_at: "2026-04-09T00:00:00+00:00",
    summary: {
      total: 4,
      installed: 2,
      enabled: 0,
      healthy: 1,
      machine_usable: 1,
      task_unconfirmed: 1,
      usable: 0,
      blocked: 1,
      available: 1
    },
    packs: [
      {
        id: "pack.voice.local_fast",
        name: "Local Voice",
        capabilities: ["voice_output"],
        installed: true,
        enabled: null,
        healthy: true,
        machine_usable: true,
        usable: false,
        state: "installed_healthy",
        state_label: "Installed · Healthy",
        status_note: "Installed and healthy, but task usability is not confirmed.",
        blocker: null,
        next_action: "Open the pack preview before relying on it.",
        source_label: "Local Registry",
        type: "skill",
        severity: "ready",
        normalized_state: {
          discovery_state: "discovered",
          install_state: "installed",
          activation_state: "unknown",
          health_state: "healthy",
          compatibility_state: "compatible",
          usability_state: "task_unconfirmed",
          machine_usable: true,
          task_usable: false
        }
      },
      {
        id: "pack.avatar.basic",
        name: "Basic Avatar",
        capabilities: ["avatar_visual"],
        installed: true,
        enabled: false,
        healthy: true,
        machine_usable: false,
        usable: false,
        state: "installed_disabled",
        state_label: "Installed · Disabled",
        status_note: "Installed, but disabled.",
        blocker: "not enabled as a live capability",
        next_action: "Enable it before using it.",
        source_label: "Local Registry",
        type: "skill",
        severity: "degraded",
        normalized_state: {
          discovery_state: "discovered",
          install_state: "installed",
          activation_state: "disabled",
          health_state: "healthy",
          compatibility_state: "compatible",
          usability_state: "unusable",
          machine_usable: false,
          task_usable: false
        }
      }
    ],
    available_packs: [
      {
        id: "text-speech",
        name: "Local Voice",
        capabilities: ["voice_output"],
        installed: false,
        enabled: false,
        healthy: null,
        machine_usable: false,
        usable: false,
        state: "available",
        state_label: "Available",
        status_note: "Available to preview.",
        blocker: null,
        next_action: "Open the preview before installing.",
        source_label: "Local Registry",
        type: "portable_text_skill",
        severity: "ready",
        normalized_state: {
          discovery_state: "previewable",
          install_state: "installable",
          activation_state: "unknown",
          health_state: "unknown",
          compatibility_state: "unconfirmed",
          usability_state: "unknown",
          machine_usable: false,
          task_usable: false
        }
      },
      {
        id: "robot-camera",
        name: "Robot Camera",
        capabilities: ["camera_feed"],
        installed: false,
        enabled: false,
        healthy: false,
        machine_usable: false,
        usable: false,
        state: "blocked",
        state_label: "Blocked",
        status_note: "Blocked by current policy.",
        blocker: "missing GPU acceleration",
        next_action: "Review the blocker before installing.",
        source_label: "Local Registry",
        type: "native_code_pack",
        severity: "blocked",
        normalized_state: {
          discovery_state: "discovered",
          install_state: "not_installed",
          activation_state: "unknown",
          health_state: "failing",
          compatibility_state: "blocked",
          usability_state: "unusable",
          machine_usable: false,
          task_usable: false
        }
      }
    ],
    source_warnings: []
  });

  assert.equal(view.readOnly, true);
  assert.equal(view.summaryLine, "total 4 · installed 2 · enabled 0 · healthy 1 · blocked 1 · available 1");
  assert.equal(view.updatedAt, "2026-04-09T00:00:00+00:00");
  assert.equal(view.installedCards.length, 2);
  assert.equal(view.availableCards.length, 2);
  assert.match(view.installedCards[0].lines.join("\n"), /provides: voice_output/);
  assert.match(view.installedCards[0].lines.join("\n"), /enabled: n\/a/);
  assert.match(view.installedCards[0].lines.join("\n"), /health: healthy/);
  assert.match(view.installedCards[0].lines.join("\n"), /machine usability: yes/);
  assert.match(view.installedCards[0].lines.join("\n"), /compatibility: compatible/);
  assert.match(view.installedCards[0].lines.join("\n"), /usability: task_unconfirmed/);
  assert.match(view.installedCards[1].lines.join("\n"), /enabled: no/);
  assert.match(view.installedCards[1].lines.join("\n"), /machine usability: no/);
  assert.match(view.installedCards[0].lines.join("\n"), /status: Installed and healthy, but task usability is not confirmed\./);
  assert.match(view.installedCards[1].lines.join("\n"), /status: Installed, but disabled\./);
  assert.match(view.installedCards[1].lines.join("\n"), /blocker: not enabled as a live capability/);
  assert.match(view.availableCards[0].lines.join("\n"), /installed: no/);
  assert.match(view.availableCards[0].lines.join("\n"), /compatibility: unconfirmed/);
  assert.match(view.availableCards[0].lines.join("\n"), /status: Available to preview\./);
  assert.match(view.availableCards[1].lines.join("\n"), /compatibility: blocked/);
  assert.match(view.availableCards[1].lines.join("\n"), /blocker: missing GPU acceleration/);
  const installedText = view.installedCards[0].lines.join("\n");
  const availableText = view.availableCards[0].lines.join("\n");
  assert.ok(!/install now|show the install preview|enable it before using it/i.test(installedText));
  assert.ok(!/install now|show the install preview|enable it before using it/i.test(availableText));
});

test("buildPacksView renders empty state without guessing", () => {
  const view = buildPacksView({
    updated_at: null,
    summary: {}
  });

  assert.equal(view.summaryLine, "total 0 · installed 0 · enabled 0 · healthy 0 · blocked 0 · available 0");
  assert.equal(view.installedEmpty, true);
  assert.equal(view.availableEmpty, true);
  assert.equal(view.updatedAt, "n/a");
  assert.equal(view.installedCards.length, 0);
  assert.equal(view.availableCards.length, 0);
});
