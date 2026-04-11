const VALUE_UNKNOWN = "n/a";

export function formatStateValue(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  if (value === null || value === undefined || value === "") return VALUE_UNKNOWN;
  if (Array.isArray(value)) {
    return value.length > 0 ? value.map((item) => formatStateValue(item)).join(", ") : VALUE_UNKNOWN;
  }
  return String(value);
}

export function stateSeverityForStatus(status) {
  const normalized = String(status || "").trim().toLowerCase();
  if (["ready", "ok", "up", "active"].includes(normalized)) return "health-ok";
  if (["degraded", "bootstrap_required", "waiting", "attention"].includes(normalized)) return "health-degraded";
  if (["failed", "down", "error", "blocked"].includes(normalized)) return "health-down";
  return "";
}

function buildCard({ key, title, badge, lines, severity }) {
  return {
    key,
    title,
    badge: formatStateValue(badge),
    badgeClassName: stateSeverityForStatus(severity || badge),
    lines: Array.isArray(lines) ? lines.filter(Boolean) : []
  };
}

export function buildStateView(stateSnapshot) {
  const snapshot = stateSnapshot && typeof stateSnapshot === "object" ? stateSnapshot : {};
  const runtime = snapshot.runtime && typeof snapshot.runtime === "object" ? snapshot.runtime : {};
  const model = snapshot.model && typeof snapshot.model === "object" ? snapshot.model : {};
  const conversation = snapshot.conversation && typeof snapshot.conversation === "object" ? snapshot.conversation : {};
  const action = snapshot.action && typeof snapshot.action === "object" ? snapshot.action : {};
  const signals = snapshot.signals && typeof snapshot.signals === "object" ? snapshot.signals : {};

  const summary = formatStateValue(runtime.summary) !== VALUE_UNKNOWN ? formatStateValue(runtime.summary) : "State snapshot unavailable.";
  const updatedAt = formatStateValue(snapshot.updated_at) !== VALUE_UNKNOWN ? formatStateValue(snapshot.updated_at) : VALUE_UNKNOWN;

  return {
    summary,
    updatedAt,
    cards: [
      buildCard({
        key: "runtime",
        title: "Runtime",
        badge: runtime.status,
        severity: runtime.status,
        lines: [
          `status: ${formatStateValue(runtime.status)}`,
          `summary: ${formatStateValue(runtime.summary)}`,
          `next action: ${formatStateValue(runtime.next_action)}`
        ]
      }),
      buildCard({
        key: "model",
        title: "Model Path",
        badge: model.health,
        severity: model.health,
        lines: [
          `provider: ${formatStateValue(model.provider)}`,
          `model: ${formatStateValue(model.model)}`,
          `path: ${formatStateValue(model.path)}`,
          `routing mode: ${formatStateValue(model.routing_mode)}`,
          `health: ${formatStateValue(model.health)}`
        ]
      }),
      buildCard({
        key: "conversation",
        title: "Conversation",
        badge: "read-only",
        severity: "active",
        lines: [
          `topic: ${formatStateValue(conversation.topic)}`,
          `recent request: ${formatStateValue(conversation.recent_request)}`,
          `open loop: ${formatStateValue(conversation.open_loop)}`
        ]
      }),
      buildCard({
        key: "action",
        title: "Action State",
        badge: action.blocked_reason ? "blocked" : "clear",
        severity: action.blocked_reason ? "blocked" : "ready",
        lines: [
          `pending approval: ${formatStateValue(action.pending_approval)}`,
          `blocked reason: ${formatStateValue(action.blocked_reason)}`,
          `last action: ${formatStateValue(action.last_action)}`
        ]
      }),
      buildCard({
        key: "signals",
        title: "Signals",
        badge: signals.response_style || "signals",
        severity: signals.response_style,
        lines: [
          `response style: ${formatStateValue(signals.response_style)}`,
          `confidence visible: ${formatStateValue(signals.confidence_visible)}`
        ]
      })
    ]
  };
}
