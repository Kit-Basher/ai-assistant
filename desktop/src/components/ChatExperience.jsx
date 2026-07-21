import { useCallback, useLayoutEffect, useRef } from "react";

const BOTTOM_STICKINESS_PX = 96;

function isNearBottom(container) {
  if (!container) return true;
  return container.scrollHeight - container.scrollTop - container.clientHeight <= BOTTOM_STICKINESS_PX;
}

function scrollToBottom(container) {
  if (!container) return;
  container.scrollTop = container.scrollHeight;
}

function ApprovalCard({ confirmation, disabled, onReply }) {
  if (!confirmation) return null;

  return (
    <div aria-label="Proposed change awaiting approval" className="inline-action-card" role="group">
      <div className="inline-action-copy">
        <p className="inline-action-eyebrow">{confirmation.statusLabel || "Waiting for your approval"}</p>
        <p className="inline-action-title">{confirmation.title}</p>
        {confirmation.actionSummary ? <p className="inline-action-text">{confirmation.actionSummary}</p> : null}
        <p className="inline-action-text">{confirmation.description || "I will wait for your okay before continuing."}</p>
        {confirmation.riskSummary ? <p className="inline-action-text">{confirmation.riskSummary}</p> : null}
      </div>
      <div className="inline-action-buttons">
        <button aria-label={confirmation.approveLabel} className="button-primary" disabled={disabled} onClick={() => onReply(confirmation.approveCommand)} type="button">
          {confirmation.approveLabel}
        </button>
        <button aria-label="Cancel this proposed change" disabled={disabled} onClick={() => onReply(confirmation.cancelCommand)} type="button">
          {confirmation.cancelLabel}
        </button>
      </div>
    </div>
  );
}

function ClarificationCard({ clarification }) {
  if (!clarification) return null;

  return (
    <div className="inline-note-card">
      <p className="inline-note-title">{clarification.title}</p>
      <p className="inline-note-text">{clarification.prompt}</p>
      {clarification.hints.length > 0 ? (
        <ul className="inline-note-list">
          {clarification.hints.map((hint) => (
            <li key={hint}>{hint}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function CapabilityCard({ capability, disabled, onReply }) {
  if (!capability) return null;

  if (capability.type === "rescue") {
    return (
      <div className="capability-card">
        <div className="capability-card-header">
          <p className="inline-action-eyebrow">Capability rescue</p>
          <p className="inline-action-title">{capability.title}</p>
          <p className="inline-action-text">
            These are untrusted catalog descriptions only. Discovery is not installation, and remote pack download is unavailable.
          </p>
        </div>
        {capability.candidates.length > 0 ? (
          <div className="capability-candidates">
            {capability.candidates.map((candidate) => (
              <div className="capability-candidate" key={`${candidate.sourceId}:${candidate.remoteId}:${candidate.name}`}>
                <div>
                  <p className="capability-candidate-name">{candidate.name}</p>
                  <p className="capability-candidate-meta">{candidate.source} - {candidate.artifactType}</p>
                </div>
                {candidate.summary ? <p className="capability-candidate-summary">{candidate.summary}</p> : null}
                <p className="capability-candidate-warning">{candidate.warning}</p>
                {candidate.recommended ? <span className="capability-tag">Recommended</span> : null}
              </div>
            ))}
          </div>
        ) : (
          <p className="inline-action-text">No candidate pack was found from the approved sources.</p>
        )}
        {capability.warnings.length > 0 ? (
          <ul className="capability-warning-list">
            {capability.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        ) : null}
        {capability.candidates.length > 0 ? (
          <div className="inline-action-buttons">
            <button className="button-primary" disabled={disabled} onClick={() => onReply(capability.previewCommand)} type="button">
              Inspect metadata
            </button>
            <button disabled={disabled} onClick={() => onReply(capability.cancelCommand)} type="button">
              Cancel
            </button>
          </div>
        ) : null}
      </div>
    );
  }

  if (capability.type === "preview") {
    return (
      <div className="capability-card capability-preview-card">
        <p className="inline-action-eyebrow">Pack preview</p>
        <p className="inline-action-title">{capability.title}</p>
        {capability.summary ? <p className="inline-action-text">{capability.summary}</p> : null}
        <div className="capability-preview-grid">
          <span>Source</span>
          <strong>{capability.sourceId || "approved source"}</strong>
          <span>Remote id</span>
          <strong>{capability.remoteId || "unknown"}</strong>
          <span>Artifact</span>
          <strong>{capability.artifactType}</strong>
        </div>
        <p className="capability-candidate-warning">{capability.policyHint}</p>
      </div>
    );
  }

  if (capability.type === "import_result") {
    return (
      <div className={`capability-card ${capability.ok ? "capability-import-ok" : "capability-import-blocked"}`}>
        <p className="inline-action-eyebrow">Pack import</p>
        <p className="inline-action-title">{capability.title}</p>
        <p className="inline-action-text">{capability.summary}</p>
      </div>
    );
  }

  return null;
}

function MessageBubble({ busy, message, onReply }) {
  const bubbleClass = [
    "chat-bubble",
    message.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant",
    message.tone === "error" ? "chat-bubble-error" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`chat-message-row ${message.role === "user" ? "chat-message-user" : "chat-message-assistant"}`}>
      <div aria-label={message.role === "user" ? "You" : `Assistant response: ${String(message.ui?.operationState || "complete").replaceAll("_", " ")}`} className={bubbleClass}>
        <p className="chat-message-copy">{message.content}</p>
        <ClarificationCard clarification={message.ui?.clarification} />
        <CapabilityCard capability={message.ui?.capability} disabled={busy} onReply={onReply} />
        <ApprovalCard confirmation={message.ui?.confirmation} disabled={busy} onReply={onReply} />
      </div>
    </div>
  );
}

function ThinkingBubble() {
  return (
    <div className="chat-message-row chat-message-assistant">
      <div className="chat-bubble chat-bubble-assistant chat-bubble-thinking">
        <span>Working on it</span>
        <span className="thinking-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
      </div>
    </div>
  );
}

export default function ChatExperience({
  chatBusy,
  chatPlaceholderVisible,
  composerPlaceholder,
  draft,
  messages,
  onDraftChange,
  onExportConversation,
  onOpenAdmin,
  onResetConversation,
  onSendMessage,
  onStarterPrompt,
  onToggleTheme,
  status,
  starterPrompts,
  theme
}) {
  const transcriptRef = useRef(null);
  const shouldStickToBottomRef = useRef(true);
  const forceNextScrollRef = useRef(false);
  const userTurnInProgressRef = useRef(false);
  const previousMessageCountRef = useRef(messages.length);
  const previousBusyRef = useRef(chatBusy);
  const scrollFrameRef = useRef(null);
  const settleFrameRef = useRef(null);

  const maybeScrollToBottom = useCallback(({ force = false } = {}) => {
    const transcript = transcriptRef.current;
    if (!transcript) return;

    if (!force && !shouldStickToBottomRef.current) return;

    // The transcript div is the only scroll container. Setting scrollTop
    // directly avoids child scrollIntoView alignment jumps in nested layouts.
    scrollToBottom(transcript);
    if (scrollFrameRef.current) window.cancelAnimationFrame(scrollFrameRef.current);
    if (settleFrameRef.current) window.cancelAnimationFrame(settleFrameRef.current);
    scrollFrameRef.current = window.requestAnimationFrame(() => {
      scrollToBottom(transcript);
      settleFrameRef.current = window.requestAnimationFrame(() => {
        scrollToBottom(transcript);
        shouldStickToBottomRef.current = true;
        settleFrameRef.current = null;
      });
      scrollFrameRef.current = null;
    });
  }, []);

  const updateStickiness = useCallback(() => {
    const transcript = transcriptRef.current;
    if (!transcript) return;
    shouldStickToBottomRef.current = isNearBottom(transcript);
  }, []);

  const handleSendMessage = useCallback(
    (message) => {
      forceNextScrollRef.current = true;
      userTurnInProgressRef.current = true;
      shouldStickToBottomRef.current = true;
      onSendMessage(message);
      maybeScrollToBottom({ force: true });
    },
    [maybeScrollToBottom, onSendMessage]
  );

  useLayoutEffect(() => {
    const messageCountChanged = previousMessageCountRef.current !== messages.length;
    previousMessageCountRef.current = messages.length;

    const assistantCompleted = previousBusyRef.current && !chatBusy;
    previousBusyRef.current = chatBusy;

    const forceScroll =
      forceNextScrollRef.current ||
      (userTurnInProgressRef.current && (messageCountChanged || assistantCompleted));
    forceNextScrollRef.current = false;
    if (assistantCompleted) userTurnInProgressRef.current = false;

    maybeScrollToBottom({ force: forceScroll });
  }, [chatBusy, chatPlaceholderVisible, maybeScrollToBottom, messages]);

  useLayoutEffect(() => {
    return () => {
      if (scrollFrameRef.current) window.cancelAnimationFrame(scrollFrameRef.current);
      if (settleFrameRef.current) window.cancelAnimationFrame(settleFrameRef.current);
    };
  }, []);

  return (
    <div className="chat-product-shell">
      <header className="product-topbar">
        <div>
          <p className="product-kicker">Personal Agent</p>
          <h1>Ask for help naturally.</h1>
        </div>
        <div className="product-topbar-actions">
          <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
          <button
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            className="theme-toggle"
            onClick={onToggleTheme}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            type="button"
          >
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <button className="admin-entry" onClick={onOpenAdmin} type="button">
            Advanced
          </button>
        </div>
      </header>

      <main className="chat-product-main">
        <section className="chat-surface">
          {messages.length === 0 ? (
            <div className="chat-empty-state">
              <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
              <h2>What can I help you with?</h2>
              <p>{status.description}</p>
              <div className="starter-prompts">
                {starterPrompts.map((prompt) => (
                  <button key={prompt} onClick={() => handleSendMessage(prompt)} type="button">
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div aria-label="Conversation" aria-live="polite" aria-relevant="additions" className="chat-transcript" onScroll={updateStickiness} ref={transcriptRef} role="log">
              {!status.ready ? <p className="chat-surface-note">{status.description}</p> : null}
              {messages.map((message, index) => (
                <MessageBubble busy={chatBusy} key={`${message.role}-${index}-${message.content.slice(0, 24)}`} message={message} onReply={handleSendMessage} />
              ))}
              {chatPlaceholderVisible ? <ThinkingBubble /> : null}
            </div>
          )}

          {messages.length > 0 ? (
            <div className="conversation-tools">
              <button onClick={onResetConversation} type="button">
                New chat
              </button>
              <button onClick={onExportConversation} type="button">
                Export
              </button>
            </div>
          ) : null}

          <form
            className="chat-composer"
            onSubmit={(event) => {
              event.preventDefault();
              handleSendMessage();
            }}
          >
            <button aria-label="Attachments are not available" className="attachment-button" disabled type="button">
              +
            </button>
            <textarea
              onChange={(event) => onDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSendMessage();
                }
              }}
              placeholder={composerPlaceholder}
              rows={1}
              value={draft}
            />
            <button className="button-primary" disabled={chatBusy || !draft.trim()} type="submit">
              Send
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}
