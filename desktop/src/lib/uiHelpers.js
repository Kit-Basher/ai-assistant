export function healthStatus(entity) {
  return entity?.health?.status || "ok";
}

export function healthLabel(entity) {
  const status = healthStatus(entity);
  if (status === "down") return "down";
  if (status === "degraded") return "degraded";
  return "ok";
}

export function asErrorText(error) {
  if (!error) return "Unknown error";
  if (typeof error === "string") return error;
  if (error.message) return error.message;
  return JSON.stringify(error);
}

export function parseJsonObject(rawText, fieldLabel) {
  const value = String(rawText || "").trim();
  if (!value) return { ok: true, value: {} };
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, error: `${fieldLabel} must be a JSON object` };
    }
    return { ok: true, value: parsed };
  } catch (_error) {
    return { ok: false, error: `${fieldLabel} must be valid JSON` };
  }
}

export function formatNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function formatEpoch(epochSeconds) {
  if (!epochSeconds) return "n/a";
  const asNumber = Number(epochSeconds);
  if (!Number.isFinite(asNumber) || asNumber <= 0) return "n/a";
  return new Date(asNumber * 1000).toLocaleString();
}

export function newestNotificationHash(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return "";
  const first = rows[0] || {};
  return String(first.dedupe_hash || "").trim();
}

export function normalizeSupportTarget(target) {
  const value = String(target || "").trim();
  if (!value) return "";
  if (value.startsWith("provider:")) return value.slice("provider:".length);
  if (value.startsWith("model:")) return value.slice("model:".length);
  return value;
}
