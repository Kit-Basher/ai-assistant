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

export default function PacksTab({ packsSnapshot, capabilitySnapshot, request, onRefresh }) {
  const [showDiagnosticPacks, setShowDiagnosticPacks] = useState(false);
  const [localPath, setLocalPath] = useState("");
  const [mutationPlan, setMutationPlan] = useState(null);
  const [mutationAction, setMutationAction] = useState("import");
  const [mutationStatus, setMutationStatus] = useState("");
  const view = buildPacksView(packsSnapshot);
  const snapshot = packsSnapshot && typeof packsSnapshot === "object" ? packsSnapshot : {};
  const installedRows = Array.isArray(snapshot.packs) ? snapshot.packs : [];
  const availableRows = Array.isArray(snapshot.available_packs) ? snapshot.available_packs : [];

  const installedUsable = installedRows.filter((row) => row?.usable === true && !isDiagnosticPack(row));
  const needsSetup = installedRows.filter((row) => row?.usable !== true && !isDiagnosticPack(row));
  const availableToPreview = availableRows.filter((row) => !isDiagnosticPack(row));
  const diagnosticRows = [...installedRows, ...availableRows].filter(isDiagnosticPack);
  const dynamicRows = Array.isArray(capabilitySnapshot?.packs) ? capabilitySnapshot.packs : [];

  const previewMutation = async (action, mutationPayload) => {
    try {
      const payload = await request("POST", `/packs/capabilities/${action}/plan`, { ...mutationPayload, actor_id: "webui", session_id: "webui", thread_id: "pack-admin" });
      setMutationPlan(payload.plan || null);
      setMutationAction(action);
      setMutationStatus(payload.message || "Preview ready.");
    } catch (error) {
      setMutationStatus(`Preview refused: ${String(error?.message || error)}`);
    }
  };

  const previewImport = async () => {
    const path = localPath.trim();
    if (!path) return setMutationStatus("Choose a local pack directory first.");
    return previewMutation("import", { path });
  };

  const applyPreview = async () => {
    if (!mutationPlan) return;
    try {
      await request("POST", `/packs/capabilities/${mutationAction}/apply`, { plan_id: mutationPlan.plan_id, binding_digest: mutationPlan.binding_digest, confirmed: true, actor_id: "webui", session_id: "webui", thread_id: "pack-admin" });
      setMutationStatus("Applied exactly the previewed gate. Any remaining gates are still required.");
      setMutationPlan(null);
      if (onRefresh) await onRefresh();
    } catch (error) {
      setMutationStatus(`Import refused: ${String(error?.message || error)}`);
    }
  };

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

      <div className="card">
        <h2>Local capability-pack runtime</h2>
        <p className="help-text">Portable text is never executable. Declarative packs may call only declared native contracts. Executable packs use an isolated pure-computation worker with no host files or network.</p>
        <p className="status-line">
          Executable isolation: {capabilitySnapshot?.isolation_runtime?.available ? "available" : `unavailable (${capabilitySnapshot?.isolation_runtime?.reason || "not verified"})`} · Wasmtime inside Bubblewrap · pure computation only
        </p>
        <label>
          Local pack directory
          <input value={localPath} onChange={(event) => setLocalPath(event.target.value)} placeholder="/path/to/local/pack" />
        </label>
        <div className="row-actions">
          <button type="button" onClick={previewImport}>Preview import</button>
          <button className="button-primary" type="button" disabled={!mutationPlan} onClick={applyPreview}>Confirm exact preview</button>
        </div>
        {mutationPlan ? <p className="help-text">{mutationPlan.preview} Expires in five minutes; approval and enablement remain separate.</p> : null}
        <p className="status-line" aria-live="polite">{mutationStatus || `${dynamicRows.length} local capability-pack version(s) recorded.`}</p>
        <div className="model-list">
          {dynamicRows.map((row) => (
            <DetailRow
              key={row.record_id}
              title={`${row.pack_id} ${row.version}`}
              badge={stateBadge(row.lifecycle?.usable ? "Usable" : "Not usable", row.lifecycle?.usable ? "health-ok" : "health-degraded")}
              metaLines={[`${row.pack_class} · ${row.capabilities?.length || 0} capability(s)`, `Next gate: ${row.lifecycle?.missing_gate || "none"}`, `Permissions: ${(row.lifecycle?.granted_permissions || []).join(", ") || "none"}`, `Last invocation: ${row.last_invocation?.outcome || "none"}`]}
            >
              {(row.capabilities || []).map((capability) => <p className="help-text" key={capability.id}>{capability.display_name}: {capability.description}</p>)}
              <div className="row-actions">
                {!row.review_approved ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "review_approved", value: true })}>Preview review approval</button> : null}
                {row.review_approved && row.lifecycle?.missing_gate === "permission" ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "grants", value: row.lifecycle?.requested_permissions || [] })}>Preview exact grants</button> : null}
                {row.review_approved && !row.enabled ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "enabled", value: true })}>Preview enable</button> : null}
                {row.enabled ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "enabled", value: false })}>Preview disable</button> : null}
                <button type="button" onClick={() => previewMutation("remove", { record_id: row.record_id })}>Preview removal</button>
              </div>
            </DetailRow>
          ))}
        </div>
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
