import DetailRow from "./DetailRow";

export default function ModelScoutTab({
  modelScoutMessage,
  modelScoutRunning,
  modelScoutStatus,
  modelScoutSuggestions,
  runModelScout
}) {
  const lifecycleCounts = modelScoutStatus?.lifecycleCounts || {};
  const providerCounts = modelScoutStatus?.providerCounts || {};
  const providerSummary = Object.entries(providerCounts)
    .map(([providerId, count]) => `${providerId} ${count}`)
    .join(" · ");
  const lifecycleSummary = [
    `ready ${Number(lifecycleCounts.ready || 0)}`,
    `queued ${Number(lifecycleCounts.queued || 0)}`,
    `downloading ${Number(lifecycleCounts.downloading || 0)}`,
    `failed ${Number(lifecycleCounts.failed || 0)}`
  ].join(" · ");
  const warnings = Array.isArray(modelScoutStatus?.warnings) ? modelScoutStatus.warnings : [];

  return (
    <section className="grid">
      <div className="card">
        <h2>Model Recommendations</h2>
        <p className="help-text">
          Mode: {modelScoutStatus?.mode || "unknown"} · Current: {modelScoutStatus?.currentModel || "unavailable"}
        </p>
        <p className="help-text">
          Recommendations: {modelScoutStatus?.recommendationCount || 0} shown · {modelScoutStatus?.availableCount || 0} eligible
        </p>
        <p className="help-text">
          Providers: {providerSummary || "none"}
        </p>
        <p className="help-text">
          Lifecycle: {lifecycleSummary}
        </p>
        <div className="row-actions">
          <button className="button-primary" disabled={modelScoutRunning} onClick={runModelScout}>
            {modelScoutRunning ? "Refreshing..." : "Refresh Recommendations"}
          </button>
        </div>
        <p className="status-line">
          {modelScoutMessage || "Canonical operator view only. Assistant recommendations use the same selector and runtime truth."}
        </p>
        {warnings.length > 0 ? <p className="meta-line">Warnings: {warnings.join(", ")}</p> : null}
      </div>

      <div className="card">
        <h2>Suggestions</h2>
        <div className="model-list">
          {modelScoutSuggestions.length === 0 ? <p className="empty">No suggestions yet.</p> : null}
          {modelScoutSuggestions.map((item) => (
            <DetailRow
              key={item.id}
              badge={<span className="badge">{item.purposeLabel}</span>}
              metaLines={[
                `${item.local ? "local" : "remote"} · score ${Number(item.score || 0).toFixed(3)}${item.tier ? ` · tier ${item.tier}` : ""}`,
                item.reason,
                item.whyBetter.length > 0 ? item.whyBetter.join(" ") : null,
                item.tradeoffs.length > 0 ? `tradeoffs: ${item.tradeoffs.join(", ")}` : null
              ]}
              title={item.canonical_model_id}
            />
          ))}
        </div>
        <p className="status-line">Recommendations are canonical and advisory here. Use assistant/controller flows for explicit test, switch, or default changes.</p>
      </div>
    </section>
  );
}
