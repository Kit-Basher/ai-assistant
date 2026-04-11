import DetailRow from "./DetailRow";
import { buildStateView } from "../lib/stateUiHelpers";

export default function StateTab({ stateSnapshot }) {
  const view = buildStateView(stateSnapshot);

  return (
    <section className="grid">
      <div className="card">
        <h2>Live State</h2>
        <p className="help-text">Read-only snapshot of runtime truth, active model path, and current signals.</p>
        <p className="status-line">{view.summary}</p>
        <p className="help-text">Updated: {view.updatedAt}</p>
        <div className="model-list">
          {view.cards.map((card) => (
            <DetailRow
              key={card.key}
              badge={<span className={`badge ${card.badgeClassName || ""}`.trim()}>{card.badge}</span>}
              metaLines={card.lines}
              title={card.title}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
