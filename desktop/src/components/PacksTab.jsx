import DetailRow from "./DetailRow";
import { buildPacksView } from "../lib/packStateUiHelpers";

export default function PacksTab({ packsSnapshot }) {
  const view = buildPacksView(packsSnapshot);

  return (
    <section className="grid">
      <div className="card">
        <h2>Pack State</h2>
        <p className="help-text">Read-only snapshot of installed and discoverable packs.</p>
        <p className="status-line">{view.summaryLine}</p>
        <p className="help-text">Updated: {view.updatedAt}</p>
        {view.warnings.length > 0 ? <p className="meta-line">Warnings: {view.warnings.join(" · ")}</p> : null}
        <p className="help-text">
          This panel stays read-only. Preview and install happen in the pack flow; this section only explains state.
        </p>
      </div>

      <div className="grid two">
        <div className="card">
          <h2>Installed</h2>
          <div className="model-list">
            {view.installedEmpty ? <p className="empty">No installed packs yet.</p> : null}
            {view.installedCards.map((card) => (
              <DetailRow
                key={card.key}
                badge={<span className={`badge ${card.badgeClassName || ""}`.trim()}>{card.badge}</span>}
                metaLines={card.lines}
                title={card.title}
              />
            ))}
          </div>
        </div>

        <div className="card">
          <h2>Available</h2>
          <div className="model-list">
            {view.availableEmpty ? <p className="empty">No available packs discovered yet.</p> : null}
            {view.availableCards.map((card) => (
              <DetailRow
                key={card.key}
                badge={<span className={`badge ${card.badgeClassName || ""}`.trim()}>{card.badge}</span>}
                metaLines={card.lines}
                title={card.title}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
