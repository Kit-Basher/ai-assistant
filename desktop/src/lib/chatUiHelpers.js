const CONFIRM_TO_PROCEED_RE = /reply\s+["']?(\/?confirm)["']?\s+to proceed/i;
const CONFIRM_TOKEN_RE = /\b(\/?confirm)\b/i;
const INTERNAL_TEXT_MARKERS = [
  "trace_id:",
  "component:",
  "failure_code:",
  "next_action:",
  "local_observations",
  "route_reason:",
  "selection_policy",
  "runtime_payload",
  "runtime_state_failure_reason",
  "setup_type:",
  "generic_fallback_reason:",
  "autopilot:",
  "operator_only:",
  "thread_id:",
  "user_id:",
  "source_surface:"
];
const INTERNAL_JSON_KEYS = new Set([
  "trace_id",
  "component",
  "failure_code",
  "next_action",
  "local_observations",
  "route_reason",
  "selection_policy",
  "runtime_payload",
  "runtime_state_failure_reason",
  "setup_type",
  "generic_fallback_reason",
  "autopilot",
  "operator_only",
  "thread_id",
  "user_id",
  "source_surface"
]);

function looksLikeInternalJson(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed || !/^[\[{][\s\S]*[\]}]$/.test(trimmed)) {
    return false;
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return Object.keys(parsed).some((key) => INTERNAL_JSON_KEYS.has(String(key || "").trim().toLowerCase()));
    }
    if (Array.isArray(parsed)) {
      return parsed.some(
        (item) => item && typeof item === "object" && !Array.isArray(item)
          && Object.keys(item).some((key) => INTERNAL_JSON_KEYS.has(String(key || "").trim().toLowerCase()))
      );
    }
  } catch (_error) {
    return false;
  }
  return false;
}

function looksLikeInternalText(text) {
  const lowered = String(text || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  if (looksLikeInternalJson(text)) {
    return true;
  }
  return INTERNAL_TEXT_MARKERS.some((marker) => lowered.includes(marker));
}

function sanitizeAssistantText(text, fallback = "I couldn't complete that request.") {
  const cleaned = String(text || "").trim();
  if (!cleaned) {
    return fallback;
  }
  if (looksLikeInternalText(cleaned)) {
    return fallback;
  }
  return cleaned;
}

function extractAssistantText(payload) {
  const assistantContent = String(payload?.assistant?.content || "").trim();
  if (assistantContent) {
    return sanitizeAssistantText(assistantContent);
  }
  const messageText = String(payload?.message || "").trim();
  if (messageText) {
    return sanitizeAssistantText(messageText);
  }
  return "I couldn't complete that request.";
}

export function buildStatusSummary(readyState) {
  const phase = String(readyState?.phase || "").trim().toLowerCase();
  const runtimeMode = String(readyState?.runtime_mode || "").trim().toUpperCase();
  const onboardingSummary = String(readyState?.onboarding?.summary || "").trim();
  const recoverySummary = String(readyState?.recovery?.summary || "").trim();
  const chatUsable =
    readyState?.chat_usable === true
    || readyState?.ready === true
    || runtimeMode === "READY";

  if (phase && ["starting", "listening", "warming"].includes(phase)) {
    return {
      label: "Starting up",
      tone: "waiting",
      ready: false,
      description: "Getting things ready."
    };
  }

  if (runtimeMode === "FAILED") {
    return {
      label: "Needs attention",
      tone: "danger",
      ready: false,
      description: recoverySummary || "Something needs attention. Ask me to take a look."
    };
  }

  if (chatUsable) {
    return {
      label: "Ready",
      tone: "ready",
      ready: true,
      description: "Ready to help."
    };
  }

  return {
    label: "Needs attention",
    tone: "attention",
    ready: false,
    description:
      onboardingSummary
      || recoverySummary
      || "Setup is not fully ready yet, but I can help you sort it out."
  };
}

export function buildStarterPrompts(readyState) {
  const status = buildStatusSummary(readyState);
  if (!status.ready) {
    return [
      "Help me get this working",
      "What needs attention?",
      "Check setup and explain what's wrong"
    ];
  }
  return [
    "Help me plan today",
    "Summarize something for me",
    "Draft a reply",
    "Explain an error message"
  ];
}

export function buildComposerPlaceholder(readyState) {
  const status = buildStatusSummary(readyState);
  if (!status.ready) {
    return "Ask for help getting things working";
  }
  return "Message Personal Agent";
}

function extractConfirmationUi(text, payload) {
  const sourceText = [text, payload?.next_question, payload?.message]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join("\n");
  if (!sourceText) return null;

  const explicitMatch = sourceText.match(CONFIRM_TO_PROCEED_RE);
  const genericMatch = sourceText.match(CONFIRM_TOKEN_RE);
  const looksLikeApproval =
    /approve|confirmation|confirm|to proceed|continue/i.test(sourceText)
    && Boolean(explicitMatch || genericMatch);

  if (!looksLikeApproval) {
    return null;
  }

  const approveCommand = explicitMatch?.[1] || genericMatch?.[1] || "confirm";
  return {
    title: "Approve this step?",
    description: text || "The agent needs your approval before it continues.",
    approveLabel: "Approve",
    approveCommand,
    cancelLabel: "Cancel",
    cancelCommand: "cancel"
  };
}

function extractClarificationUi(text, payload) {
  const setup = payload?.setup && typeof payload.setup === "object" ? payload.setup : null;
  if (setup) {
    const prompt = String(setup.prompt || setup.summary || payload?.next_question || text || payload?.message || "").trim();
    if (setup.type === "request_secret") {
      return {
        title: String(setup.title || "Secret needed").trim() || "Secret needed",
        prompt,
        hints: [String(setup.submit_hint || "").trim()].filter(Boolean)
      };
    }
    if (["action_required", "provider_test_result", "provider_status", "providers_status"].includes(String(setup.type || "").trim())) {
      return {
        title: String(setup.title || "Needs attention").trim() || "Needs attention",
        prompt,
        hints: []
      };
    }
  }

  const envelope = payload?.envelope && typeof payload.envelope === "object" ? payload.envelope : {};
  const clarification =
    envelope?.clarification && typeof envelope.clarification === "object"
      ? envelope.clarification
      : null;
  const hints = Array.isArray(clarification?.hints)
    ? clarification.hints
        .map((hint) => String(hint || "").trim())
        .filter(Boolean)
        .slice(0, 3)
    : [];
  const prompt = String(payload?.next_question || clarification?.next_question || text || payload?.message || "").trim();

  if (!clarification && String(payload?.error_kind || "").trim() !== "needs_clarification") {
    return null;
  }

  return {
    title: "A bit more detail will help",
    prompt: prompt || "Tell me a little more.",
    hints
  };
}

function extractCapabilityUi(payload) {
  const setup = payload?.setup && typeof payload.setup === "object" ? payload.setup : null;
  if (!setup) return null;

  const rescue =
    setup?.capability_gap_rescue && typeof setup.capability_gap_rescue === "object"
      ? setup.capability_gap_rescue
      : null;
  if (rescue?.type === "capability_gap_rescue") {
    const candidates = Array.isArray(rescue.candidate_packs)
      ? rescue.candidate_packs
          .filter((candidate) => candidate && typeof candidate === "object")
          .slice(0, 3)
          .map((candidate) => ({
            name: String(candidate.name || "Skill pack").trim() || "Skill pack",
            source: String(candidate.source_name || candidate.source_id || "Approved source").trim() || "Approved source",
            sourceId: String(candidate.source_id || "").trim(),
            remoteId: String(candidate.remote_id || "").trim(),
            summary: String(candidate.summary || "").trim(),
            artifactType: String(candidate.artifact_type_hint || "unknown").trim() || "unknown",
            recommended: candidate.recommended === true,
            installable: candidate.installable_by_current_policy === true,
            blocker: String(candidate.blocker || "").trim(),
            warning: String(candidate.status_note || "Discovery metadata is untrusted until previewed.").trim()
          }))
      : [];
    return {
      type: "rescue",
      title: String(rescue.missing_capability || rescue.capability_label || "Missing capability").trim(),
      goal: String(rescue.user_goal || "").trim(),
      searchQuery: String(rescue.search_query || "").trim(),
      sourceScope: String(rescue.source_scope || "approved_pack_sources_only").trim(),
      previewRequired: rescue.preview_required === true,
      installAllowedInitially: rescue.install_allowed_initially === true,
      warnings: Array.isArray(rescue.trust_warnings)
        ? rescue.trust_warnings.map((warning) => String(warning || "").trim()).filter(Boolean).slice(0, 3)
        : [],
      candidates,
      previewCommand: "yes",
      cancelCommand: "no"
    };
  }

  if (setup.type === "capability_gap_preview") {
    const preview = setup.preview && typeof setup.preview === "object" ? setup.preview : {};
    const listing = preview.listing && typeof preview.listing === "object" ? preview.listing : {};
    return {
      type: "preview",
      ok: setup.ok !== false,
      title: String(listing.name || "Pack preview").trim() || "Pack preview",
      summary: String(preview.summary || listing.summary || setup.summary || "").trim(),
      artifactType: String(preview.artifact_type_hint || listing.artifact_type_hint || "unknown").trim() || "unknown",
      sourceId: String(setup.source_id || "").trim(),
      remoteId: String(setup.remote_id || listing.remote_id || "").trim(),
      policyHint: String(preview.policy_hint || "Discovery metadata is untrusted until fetched and normalized.").trim(),
      importOffered: setup.import_offered === true
    };
  }

  if (setup.type === "capability_gap_import") {
    return {
      type: "import_result",
      ok: setup.ok === true,
      title: setup.ok === true ? "Imported for review" : "Import not completed",
      summary: String(setup.summary || "").trim(),
      blockedReason: String(setup.blocked_reason || "").trim()
    };
  }

  return null;
}

export function buildAssistantMessage(payload) {
  const text = extractAssistantText(payload);
  const setup = payload?.setup && typeof payload.setup === "object" ? payload.setup : null;
  const capability = extractCapabilityUi(payload);
  const capabilityImportConfirmation =
    capability?.type === "preview" && capability.importOffered
      ? {
          title: "Import for review?",
          description: "This imports the pack into review only. It will not be enabled, approved, granted permissions, or executed.",
          approveLabel: "Import for review",
          approveCommand: "yes",
          cancelLabel: "Cancel",
          cancelCommand: "no"
        }
      : null;
  const setupConfirmation =
    setup?.type === "confirm_switch_model" || setup?.type === "confirm_reuse_secret"
      ? {
          title: String(setup.title || "Approve this step?").trim() || "Approve this step?",
          description: String(setup.prompt || text || payload?.message || "").trim(),
          approveLabel: String(setup.approve_label || "Approve").trim() || "Approve",
          approveCommand: String(setup.approve_command || "yes").trim() || "yes",
          cancelLabel: String(setup.cancel_label || "Cancel").trim() || "Cancel",
          cancelCommand: String(setup.cancel_command || "no").trim() || "no"
        }
      : null;
  const confirmation = capabilityImportConfirmation || setupConfirmation || extractConfirmationUi(text, payload);
  const clarification = confirmation ? null : extractClarificationUi(text, payload);
  const errorKind = String(payload?.error_kind || "").trim();
  const setupFailed = setup?.type === "provider_test_result" && setup?.ok === false;

  return {
    role: "assistant",
    content: text,
    tone: (!payload?.ok && errorKind && errorKind !== "needs_clarification") || setupFailed ? "error" : "default",
    ui: {
      confirmation,
      clarification,
      capability
    }
  };
}
