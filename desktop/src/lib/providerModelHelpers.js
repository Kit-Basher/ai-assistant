import { healthStatus } from "./uiHelpers";

export function matchesProviderModelFilter(model, view) {
  const query = String(view?.query || "").trim().toLowerCase();
  const filter = String(view?.filter || "all");
  const status = healthStatus(model);
  const capabilities = Array.isArray(model?.capabilities) ? model.capabilities : [];
  const matchesQuery = !query
    || String(model?.id || "").toLowerCase().includes(query)
    || String(model?.model || "").toLowerCase().includes(query)
    || capabilities.some((capability) => String(capability || "").toLowerCase().includes(query));

  if (!matchesQuery) return false;
  if (filter === "available") return model?.available === true;
  if (filter === "routable") return model?.routable === true;
  if (filter === "issues") return status !== "ok" || model?.available === false || model?.routable === false;
  return true;
}

export function summarizeProviderCapabilities(modelRows) {
  const counts = new Map();
  modelRows.forEach((model) => {
    const capabilities = Array.isArray(model?.capabilities) ? model.capabilities : [];
    capabilities.forEach((capability) => {
      const key = String(capability || "").trim();
      if (!key) return;
      counts.set(key, (counts.get(key) || 0) + 1);
    });
  });
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, 4)
    .map(([name, count]) => ({ name, count }));
}
