const CONFIRM_TO_PROCEED_RE = /reply\s+["']?(\/?confirm)["']?\s+to proceed/i;
const CONFIRM_TOKEN_RE = /\b(\/?confirm)\b/i;

export function buildStatusSummary(readyState) {
  const phase = String(readyState?.phase || "").trim().toLowerCase();
  const runtimeMode = String(readyState?.runtime_mode || "").trim().toUpperCase();
  const onboardingSummary = String(readyState?.onboarding?.summary || "").trim();
  const recoverySummary = String(readyState?.recovery?.summary || "").trim();

  if (phase && ["starting", "listening", "warming"].includes(phase)) {
    return {
      label: "Starting up",
      tone: "waiting",
      ready: false,
      description: "Getting things ready."
    };
  }

  if (readyState?.ready === true || runtimeMode === "READY") {
    return {
      label: "Ready",
      tone: "ready",
      ready: true,
      description: "Ready to help."
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

export function buildAssistantMessage(payload) {
  const text = String(payload?.assistant?.content || payload?.message || payload?.error || "").trim();
  const setup = payload?.setup && typeof payload.setup === "object" ? payload.setup : null;
  const confirmation =
    setup?.type === "confirm_switch_model" || setup?.type === "confirm_reuse_secret"
      ? {
          title: String(setup.title || "Approve this step?").trim() || "Approve this step?",
          description: String(setup.prompt || text || payload?.message || "").trim(),
          approveLabel: String(setup.approve_label || "Approve").trim() || "Approve",
          approveCommand: String(setup.approve_command || "yes").trim() || "yes",
          cancelLabel: String(setup.cancel_label || "Cancel").trim() || "Cancel",
          cancelCommand: String(setup.cancel_command || "no").trim() || "no"
        }
      : extractConfirmationUi(text, payload);
  const clarification = confirmation ? null : extractClarificationUi(text, payload);
  const errorKind = String(payload?.error_kind || "").trim();
  const setupFailed = setup?.type === "provider_test_result" && setup?.ok === false;

  return {
    role: "assistant",
    content: text || "I couldn't complete that request.",
    tone: (!payload?.ok && errorKind && errorKind !== "needs_clarification") || setupFailed ? "error" : "default",
    ui: {
      confirmation,
      clarification
    }
  };
}
