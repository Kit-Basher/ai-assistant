function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return "size unknown";
  return `${(bytes / (1024 ** 3)).toFixed(1)} GB`;
}

function ModelRow({ row }) {
  const benchmark = row?.benchmark || {};
  const score = benchmark?.score || {};
  return (
    <article className="list-item">
      <div className="setup-card-head">
        <strong>{row.provider_native_id || row.canonical_id}</strong>
        <span className={`badge ${row.ready ? "health-ok" : "health-degraded"}`}>
          {row.effective ? "using now" : row.ready ? "installed + ready" : "installed, unavailable"}
        </span>
      </div>
      <p className="help-text">
        {formatBytes(row.size_bytes)} · {(row.roles || []).join(", ") || "role unknown"}
        {row.parameter_size ? ` · ${row.parameter_size}` : ""}
        {row.quantization ? ` · ${row.quantization}` : ""}
      </p>
      <p className="help-text">
        Benchmark: {benchmark.status || "not evaluated"}
        {Number.isFinite(score.passed) ? ` · ${score.passed}/${score.total}` : ""}
        {benchmark?.latency?.median_ms ? ` · median ${Math.round(benchmark.latency.median_ms)} ms` : ""}
      </p>
      <details>
        <summary>Evidence</summary>
        <p className="help-text">{row.canonical_id} · digest {row.digest || "unknown"}</p>
        <p className="help-text">Aliases: {(row.aliases || []).join(", ") || "none"}</p>
      </details>
    </article>
  );
}

export default function ModelTruthTab({ snapshot, onRefresh, refreshing }) {
  const observation = snapshot?.observation || {};
  const selection = snapshot?.selection || {};
  const recommendation = snapshot?.recommendation || {};
  const installed = Array.isArray(snapshot?.installed) ? snapshot.installed : [];
  const unavailable = installed.filter((row) => !row.ready);
  const ready = installed.filter((row) => row.ready);
  return (
    <section className="grid">
      <div className="card">
        <h2>Models on this machine</h2>
        <p className="status-line">Using now: {selection.effective_model || "not verified"}</p>
        <p className="status-line">
          Recommended default: {recommendation.default_general_assistant || "no current benchmark recommendation"}
        </p>
        <p className="help-text">
          Physical observation: {observation.status || "unknown"} · {observation.observed_at || "never"}
          {observation.stale ? " · stale" : ""}
        </p>
        <button type="button" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? "Refreshing…" : "Refresh installed models"}
        </button>
      </div>
      <div className="card">
        <h3>Installed and ready</h3>
        <div className="model-list">
          {ready.length ? ready.map((row) => <ModelRow key={row.canonical_id} row={row} />) : <p className="empty">No ready chat model was observed.</p>}
        </div>
      </div>
      <div className="card">
        <h3>Installed but unavailable or non-chat</h3>
        <div className="model-list">
          {unavailable.length ? unavailable.map((row) => <ModelRow key={row.canonical_id} row={row} />) : <p className="empty">None.</p>}
        </div>
      </div>
      <details className="card operator-details">
        <summary>Advanced registry and history evidence</summary>
        <p className="help-text">Registered but not physically observed: {(snapshot?.registered_not_observed || []).length}</p>
        <p className="help-text">Remote catalog/registry candidates: {(snapshot?.remote_registered || []).length}</p>
        <p className="help-text">Manager history only: {(snapshot?.history_only || []).length}</p>
        <p className="help-text">Evidence code commit: {snapshot?.evaluation?.code_commit || "not available"}</p>
      </details>
    </section>
  );
}
