import { useState } from "react";

export default function CapabilityStatusTab({ snapshot, onRefresh }) {
  const [advanced, setAdvanced] = useState(false);
  const rows = Array.isArray(snapshot?.capabilities) ? snapshot.capabilities : [];
  const visible = advanced ? rows : rows.filter((row) => row.available || row.status !== "available");

  return (
    <section className="panel-stack" aria-label="Capability status">
      <header className="section-heading">
        <div>
          <h2>What Personal Agent can do</h2>
          <p>{snapshot?.summary || "Capability status is not available yet."}</p>
        </div>
        <button type="button" onClick={onRefresh}>Refresh</button>
      </header>
      <label className="checkbox-row">
        <input type="checkbox" checked={advanced} onChange={(event) => setAdvanced(event.target.checked)} />
        Show advanced runtime details
      </label>
      <div className="card-grid">
        {visible.map((row, index) => (
          <article className="status-card" key={row.id || `${row.description}-${index}`}>
            <h3>{row.description}</h3>
            <p><strong>{row.available ? "Available" : "Needs setup or is unavailable"}</strong></p>
            {row.reason ? <p>{String(row.reason).replaceAll("_", " ")}</p> : null}
            {row.requires_confirmation ? <p>Changes require your explicit confirmation.</p> : <p>Read-only; no change confirmation is needed.</p>}
            <p>{row.next_step}</p>
            {advanced ? <pre>{JSON.stringify(row, null, 2)}</pre> : null}
          </article>
        ))}
      </div>
    </section>
  );
}
