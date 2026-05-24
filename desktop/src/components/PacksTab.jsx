import { useState } from "react";
import DetailRow from "./DetailRow";
import { buildPacksView } from "../lib/packStateUiHelpers";

function packName(row) {
  return String(row?.name || row?.title || row?.id || "Skill pack").trim() || "Skill pack";
}

function packPurpose(row) {
  const summary = String(row?.summary || row?.description || row?.status_note || "").trim();
  if (summary) return summary.length > 140 ? `${summary.slice(0, 137)}...` : summary;
  const capabilities = Array.isArray(row?.capabilities) ? row.capabilities.filter(Boolean).join(", ") : "";
  return capabilities ? `Helps with ${capabilities}.` : "Purpose not described yet.";
}

function isDiagnosticPack(row) {
  const haystack = [row?.id, row?.name, row?.title, row?.source_label, row?.state, row?.status_note]
    .map((item) => String(item || "").toLowerCase())
    .join(" ");
  return /smoke|test|diagnostic|fixture|blocked/.test(haystack) || row?.blocker || row?.severity === "blocked";
}

function stateBadge(label, className = "") {
  return <span className={`badge ${className}`.trim()}>{label}</span>;
}

function UserPackRow({ badge, children, row }) {
  return (
    <DetailRow badge={badge} metaLines={[packPurpose(row)]} title={packName(row)}>
      {children}
    </DetailRow>
  );
}

export default function PacksTab({ packsSnapshot }) {
  const [showDiagnosticPacks, setShowDiagnosticPacks] = useState(false);
  const view = buildPacksView(packsSnapshot);
  const snapshot = packsSnapshot && typeof packsSnapshot === "object" ? packsSnapshot : {};
  const installedRows = Array.isArray(snapshot.packs) ? snapshot.packs : [];
  const availableRows = Array.isArray(snapshot.available_packs) ? snapshot.available_packs : [];

  const installedUsable = installedRows.filter((row) => row?.usable === true && !isDiagnosticPack(row));
  const needsSetup = installedRows.filter((row) => row?.usable !== true && !isDiagnosticPack(row));
  const availableToPreview = availableRows.filter((row) => !isDiagnosticPack(row));
  const diagnosticRows = [...installedRows, ...availableRows].filter(isDiagnosticPack);

  return (
    <section className="grid">
      <div className="card">
        <h2>Skills</h2>
        <p className="help-text">External skills are not usable until they are previewed, reviewed, enabled, and given any needed permissions.</p>
        <p className="status-line">{view.summaryLine}</p>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={showDiagnosticPacks}
            onChange={(event) => setShowDiagnosticPacks(event.target.checked)}
          />
          Show diagnostic packs
        </label>
      </div>

      <div className="grid two">
        <div className="card">
          <h2>Installed and usable</h2>
          <div className="model-list">
            {installedUsable.length === 0 ? <p className="empty">No external skills are usable yet.</p> : null}
            {installedUsable.map((row) => (
              <UserPackRow key={`usable:${row.id || row.name}`} badge={stateBadge("Ready", "health-ok")} row={row} />
            ))}
          </div>
        </div>

        <div className="card">
          <h2>Needs review/setup</h2>
          <div className="model-list">
            {needsSetup.length === 0 ? <p className="empty">No installed skills need setup.</p> : null}
            {needsSetup.map((row) => (
              <UserPackRow key={`setup:${row.id || row.name}`} badge={stateBadge("Not usable yet", "health-degraded")} row={row}>
                <p className="help-text">Next step: review or finish setup in chat.</p>
              </UserPackRow>
            ))}
          </div>
        </div>

        <div className="card">
          <h2>Available to preview</h2>
          <div className="model-list">
            {availableToPreview.length === 0 ? <p className="empty">No catalog skills are available to preview right now.</p> : null}
            {availableToPreview.map((row) => (
              <UserPackRow key={`available:${row.id || row.name}`} badge={stateBadge("Not installed")} row={row}>
                <div className="row-actions">
                  <button disabled title="Ask the assistant to preview this skill." type="button">Preview</button>
                </div>
                <p className="help-text">Not installed. Not usable yet.</p>
              </UserPackRow>
            ))}
          </div>
        </div>

        <div className="card">
          <h2>Blocked/diagnostic</h2>
          {showDiagnosticPacks ? (
            <div className="model-list">
              {diagnosticRows.length === 0 ? <p className="empty">No diagnostic packs are visible.</p> : null}
              {diagnosticRows.map((row) => (
                <UserPackRow key={`diagnostic:${row.id || row.name}`} badge={stateBadge("Diagnostic", "health-down")} row={row} />
              ))}
            </div>
          ) : (
            <p className="help-text">Hidden by default. Turn on “Show diagnostic packs” to inspect smoke/test/blocked entries.</p>
          )}
        </div>
      </div>
    </section>
  );
}
