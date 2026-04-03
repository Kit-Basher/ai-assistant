import DetailRow from "./DetailRow";
import { healthLabel } from "../lib/uiHelpers";

export default function ModelStatusRow({
  capabilityLimit = 0,
  model,
  showCapabilities = false,
  showLastError = false,
  showProvider = false
}) {
  const health = healthLabel(model);
  const summaryBits = [];
  if (showProvider && model.provider) summaryBits.push(model.provider);
  summaryBits.push(model.available ? "available" : "unavailable");
  summaryBits.push(model.routable ? "routable" : "not routable");

  return (
    <DetailRow
      badge={<span className={`badge health-${health}`}>{health}</span>}
      metaLines={[
        summaryBits.join(" · "),
        showLastError && model.health?.last_error_kind
          ? `Last error: ${model.health.last_error_kind}${model.health.status_code ? ` (${model.health.status_code})` : ""}`
          : null
      ]}
      title={model.id}
    >
      {showCapabilities && Array.isArray(model.capabilities) && model.capabilities.length > 0 ? (
        <div className="provider-capability-badges">
          {model.capabilities.slice(0, capabilityLimit || model.capabilities.length).map((capability) => (
            <span key={`${model.id}-${capability}`} className="badge">
              {capability}
            </span>
          ))}
        </div>
      ) : null}
    </DetailRow>
  );
}
