import { useEffect, useRef } from "react";

function ApprovalCard({ confirmation, disabled, onReply }) {
  if (!confirmation) return null;

  return (
    <div className="inline-action-card">
      <div className="inline-action-copy">
        <p className="inline-action-eyebrow">Approval needed</p>
        <p className="inline-action-title">{confirmation.title}</p>
        <p className="inline-action-text">{confirmation.description || "I will wait for your okay before continuing."}</p>
      </div>
      <div className="inline-action-buttons">
        <button className="button-primary" disabled={disabled} onClick={() => onReply(confirmation.approveCommand)} type="button">
          {confirmation.approveLabel}
        </button>
        <button disabled={disabled} onClick={() => onReply(confirmation.cancelCommand)} type="button">
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
      <div className={bubbleClass}>
        <p className="chat-message-copy">{message.content}</p>
        <ClarificationCard clarification={message.ui?.clarification} />
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
  status,
  starterPrompts
}) {
  const transcriptEndRef = useRef(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, chatBusy]);

  return (
    <div className="chat-product-shell">
      <header className="product-topbar">
        <div>
          <p className="product-kicker">Personal Agent</p>
          <h1>Ask for help naturally.</h1>
        </div>
        <div className="product-topbar-actions">
          <span className={`status-pill status-pill-${status.tone}`}>{status.label}</span>
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
                  <button key={prompt} onClick={() => onStarterPrompt(prompt)} type="button">
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="chat-transcript">
              {!status.ready ? <p className="chat-surface-note">{status.description}</p> : null}
              {messages.map((message, index) => (
                <MessageBubble busy={chatBusy} key={`${message.role}-${index}-${message.content.slice(0, 24)}`} message={message} onReply={onSendMessage} />
              ))}
              {chatPlaceholderVisible ? <ThinkingBubble /> : null}
              <div ref={transcriptEndRef} />
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
              onSendMessage();
            }}
          >
            <button className="attachment-button" disabled type="button">
              +
            </button>
            <textarea
              onChange={(event) => onDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  onSendMessage();
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
