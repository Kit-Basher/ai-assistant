import { formatStateValue, stateSeverityForStatus } from "./stateUiHelpers.js";

const VALUE_UNKNOWN = "n/a";

function packLabel(row) {
  const value = String(row?.name || row?.id || "").trim();
  return value || "Imported pack";
}

function packProvides(row) {
  const capabilities = Array.isArray(row?.capabilities) ? row.capabilities.map((item) => formatStateValue(item)).filter((item) => item !== VALUE_UNKNOWN) : [];
  if (capabilities.length > 0) {
    return capabilities.join(", ");
  }
  const sourceLabel = String(row?.source_label || "").trim();
  const type = String(row?.type || "").trim();
  if (sourceLabel && type) {
    return `${sourceLabel} · ${type}`;
  }
  if (sourceLabel) {
    return sourceLabel;
  }
  if (type) {
    return type;
  }
  return VALUE_UNKNOWN;
}

function packStateBadge(row) {
  return String(row?.state_label || row?.state || "unknown").trim() || "unknown";
}

function buildPackRow(row) {
  const state = String(row?.state || "").trim().toLowerCase();
  const installed = row?.installed === true;
  const enabled = row?.enabled === true ? true : row?.enabled === false ? false : null;
  const usable = row?.usable === true;
  const title = packLabel(row);
  const metaLines = [
    `id: ${formatStateValue(row?.id)}`,
    `provides: ${packProvides(row)}`,
    `source: ${formatStateValue(row?.source_label || row?.source?.name || row?.source?.kind || row?.source?.id)}`,
    `installed: ${formatStateValue(installed)}`,
    `enabled: ${formatStateValue(enabled)}`,
    `health: ${formatStateValue(row?.healthy === true ? "healthy" : row?.healthy === false ? row?.normalized_state?.health_state || "unhealthy" : row?.normalized_state?.health_state || "unknown")}`,
    `machine usability: ${formatStateValue(row?.normalized_state?.machine_usable === true)}`,
    `compatibility: ${formatStateValue(row?.normalized_state?.compatibility_state || row?.normalized_state?.compatibility || "unknown")}`,
    `usability: ${formatStateValue(row?.normalized_state?.usability_state || (usable ? "usable" : "unconfirmed"))}`,
    `status: ${formatStateValue(row?.status_note)}`,
    row?.blocker ? `blocker: ${formatStateValue(row?.blocker)}` : null,
    row?.next_action ? `next: ${formatStateValue(row?.next_action)}` : null
  ].filter(Boolean);

  return {
    key: `${state || "pack"}:${String(row?.id || row?.name || "").trim().toLowerCase()}`,
    title,
    badge: packStateBadge(row),
    badgeClassName: stateSeverityForStatus(row?.severity || state),
    lines: metaLines
  };
}

export function buildPacksView(packsSnapshot) {
  const snapshot = packsSnapshot && typeof packsSnapshot === "object" ? packsSnapshot : {};
  const summary = snapshot.summary && typeof snapshot.summary === "object" ? snapshot.summary : {};
  const installedRows = Array.isArray(snapshot.packs) ? snapshot.packs : [];
  const availableRows = Array.isArray(snapshot.available_packs) ? snapshot.available_packs : [];
  const warnings = Array.isArray(snapshot.source_warnings) ? snapshot.source_warnings : [];
  const updatedAt = formatStateValue(snapshot.updated_at) !== VALUE_UNKNOWN ? formatStateValue(snapshot.updated_at) : VALUE_UNKNOWN;

  const summaryLine = [
    `total ${Number(summary.total || 0)}`,
    `installed ${Number(summary.installed || 0)}`,
    `enabled ${Number(summary.enabled || 0)}`,
    `healthy ${Number(summary.healthy || 0)}`,
    `blocked ${Number(summary.blocked || 0)}`,
    `available ${Number(summary.available || 0)}`
  ].join(" · ");

  return {
    readOnly: true,
    summaryLine,
    updatedAt,
    installedCards: installedRows.map((row) => buildPackRow(row)),
    availableCards: availableRows.map((row) => buildPackRow(row)),
    installedEmpty: installedRows.length === 0,
    availableEmpty: availableRows.length === 0,
    warnings: warnings
      .map((row) => {
        const sourceId = String(row?.source_id || "").trim();
        const error = String(row?.error || "").trim();
        if (!sourceId && !error) return "";
        return `${sourceId || "source"}: ${error || "unavailable"}`;
      })
      .filter(Boolean)
  };
}
