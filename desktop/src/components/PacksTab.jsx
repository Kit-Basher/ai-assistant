import { useEffect, useState } from "react";
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

function PackVisualizerPreview({ visualizer }) {
  const [state, setState] = useState("thinking");
  const [frameOffset, setFrameOffset] = useState(0);
  const declaration = visualizer?.declaration || {};
  const animation = declaration?.animations?.[state] || declaration?.animations?.idle;
  useEffect(() => {
    setFrameOffset(0);
    if (!animation || typeof window === "undefined" || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return undefined;
    const timer = window.setInterval(() => setFrameOffset((value) => (value + 1) % animation.frames.length), Math.max(16, Number(animation.frame_duration_ms || 250)));
    return () => window.clearInterval(timer);
  }, [state, animation]);
  if (!visualizer?.available || !animation) return <p className="help-text">Core default presence indicator is active. No pack visualizer is selected.</p>;
  const frame = animation.frames[Math.min(frameOffset, animation.frames.length - 1)] || 0;
  const column = frame % Number(declaration.columns || 1);
  const row = Math.floor(frame / Number(declaration.columns || 1));
  return <div className="pack-visualizer-preview">
    <div role="img" aria-label={`${state} presence animation preview`} className="pack-visualizer-frame" style={{ width: declaration.frame_width, height: declaration.frame_height, backgroundImage: `url(${visualizer.asset_url})`, backgroundPosition: `${-column * declaration.frame_width}px ${-row * declaration.frame_height}px` }} />
    <label>Preview state<select value={state} onChange={(event) => setState(event.target.value)}>{Object.keys(declaration.animations || {}).map((name) => <option key={name}>{name}</option>)}</select></label>
    <p className="help-text">Core-rendered raster only · reduced motion supported · no scripts or remote assets</p>
  </div>;
}

export default function PacksTab({ packsSnapshot, capabilitySnapshot, request, onRefresh }) {
  const [showDiagnosticPacks, setShowDiagnosticPacks] = useState(false);
  const [localPath, setLocalPath] = useState("");
  const [mutationPlan, setMutationPlan] = useState(null);
  const [mutationAction, setMutationAction] = useState("import");
  const [mutationStatus, setMutationStatus] = useState("");
  const [remoteUrl, setRemoteUrl] = useState("");
  const [draftName, setDraftName] = useState("");
  const [draftTemplate, setDraftTemplate] = useState("local_data_search");
  const [selectedFile, setSelectedFile] = useState("");
  const [authorityPlan, setAuthorityPlan] = useState(null);
  const [visualizer, setVisualizer] = useState(null);
  const [candidatePreview, setCandidatePreview] = useState(null);
  const [updateComparison, setUpdateComparison] = useState(null);
  useEffect(() => {
    request("GET", "/packs/visualizer").then(setVisualizer).catch(() => setVisualizer(null));
    // `request` is supplied by the parent render; capability refreshes are the
    // state boundary that should trigger another visualizer observation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capabilitySnapshot]);
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

  const previewAuthority = async (operation, payload) => {
    try {
      const result = await request("POST", `/packs/${operation}/plan`, { ...payload, actor_id: "webui", session_id: "webui", thread_id: "pack-admin" });
      setAuthorityPlan({ operation, payload, plan: result.plan });
      setMutationStatus(result.message || "Exact authorization preview ready.");
    } catch (error) { setMutationStatus(`Preview refused: ${String(error?.message || error)}`); }
  };
  const applyAuthority = async () => {
    if (!authorityPlan?.plan) return;
    const plan = authorityPlan.plan;
    const confirmation = { confirmation_id: `webui-${plan.plan_id}-${Date.now()}`, plan_id: plan.plan_id, plan_fingerprint: plan.plan_fingerprint, capability_id: plan.capability_id, executor_id: plan.executor_id, thread_id: plan.thread_id, session_id: plan.session_id, actor_id: plan.actor_id, confirmed_at: new Date().toISOString(), confirmation_phrase_class: "affirmative", activation_fingerprint: plan.activation_fingerprint || null };
    try {
      await request("POST", `/packs/${authorityPlan.operation}/apply`, { ...authorityPlan.payload, mutation_plan: plan, confirmation, actor_id: "webui", session_id: "webui", thread_id: "pack-admin" });
      setAuthorityPlan(null); setMutationStatus("Applied exactly one previewed pack gate. Later gates remain separate.");
      if (onRefresh) await onRefresh();
    } catch (error) { setMutationStatus(`Action refused: ${String(error?.message || error)}`); }
  };
  const previewCandidate = async (row) => {
    const sourceId = String(row?.source_id || "").trim();
    const remoteId = String(row?.remote_id || row?.id || "").trim();
    if (!sourceId || !remoteId) return setMutationStatus("This catalog result has no exact source binding, so it cannot be previewed.");
    try {
      const result = await request("GET", `/pack_sources/${encodeURIComponent(sourceId)}/packs/${encodeURIComponent(remoteId)}/preview`);
      setCandidatePreview(result);
      setMutationStatus("Loaded bounded untrusted metadata only. Nothing was fetched or enabled.");
    } catch (error) { setMutationStatus(`Candidate preview refused: ${String(error?.message || error)}`); }
  };
  const compareVersion = async (row) => {
    const active = dynamicRows.find((item) => item?.pack_id === row?.pack_id && item?.active);
    if (!active) return setMutationStatus("There is no active version to compare with this staged version.");
    try {
      const result = await request("GET", `/packs/capabilities/compare?from=${encodeURIComponent(active.record_id)}&to=${encodeURIComponent(row.record_id)}`);
      setUpdateComparison(result?.result || null);
      setMutationStatus("Loaded the exact bounded version comparison. Nothing was activated.");
    } catch (error) { setMutationStatus(`Version comparison refused: ${String(error?.message || error)}`); }
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
        <h2>Find, fetch, or create a skill</h2>
        <p className="help-text">Metadata search is read-only. A remote artifact is fetched only after exact confirmation and goes to quarantine; fetch is not approval or activation.</p>
        <label>Exact HTTPS or GitHub source<input value={remoteUrl} onChange={(event) => setRemoteUrl(event.target.value)} placeholder="https://github.com/owner/repo" /></label>
        <div className="row-actions"><button type="button" onClick={() => previewAuthority("fetch", { source: { url: remoteUrl.trim(), kind: /github\.com\/[^/]+\/[^/]+\/?$/.test(remoteUrl.trim()) ? "github_repo" : remoteUrl.includes("github.com") ? "github_archive" : "generic_archive_url" } })}>Preview quarantine fetch</button></div>
        <label>Skill name<input value={draftName} onChange={(event) => setDraftName(event.target.value)} placeholder="Library export search" /></label>
        <label>Supported template<select value={draftTemplate} onChange={(event) => setDraftTemplate(event.target.value)}><option value="local_data_search">Local data search</option><option value="portable_text">Portable guidance</option><option value="declarative_native">Native report wrapper</option><option value="presence_visualizer">Presence sprite visualizer</option></select></label>
        <div className="row-actions"><button type="button" onClick={() => previewAuthority("create", { template: draftTemplate, name: draftName.trim(), ...(draftTemplate === "presence_visualizer" ? { asset_path: selectedFile.trim() } : {}) })}>Preview assistant-created draft</button><button className="button-primary" type="button" disabled={!authorityPlan} onClick={applyAuthority}>Confirm exact action</button></div>
        {authorityPlan ? <p className="help-text">Pending: {authorityPlan.operation}. Confirmation applies this gate only.</p> : null}
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
        <label>Exact user-selected data or sprite file<input value={selectedFile} onChange={(event) => setSelectedFile(event.target.value)} placeholder="/home/you/export.json" /></label>
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
              metaLines={[`${row.pack_class} · ${row.capabilities?.length || 0} capability(s) · ${row.active ? "active" : "staged"}`, `Next gate: ${row.lifecycle?.missing_gate || "none"}`, `Permissions: ${(row.lifecycle?.granted_permissions || []).join(", ") || "none"}`, `Last invocation: ${row.last_invocation?.outcome || "none"}`]}
            >
              {(row.capabilities || []).map((capability) => <p className="help-text" key={capability.id}>{capability.display_name}: {capability.description}</p>)}
              <div className="row-actions">
                {!row.review_approved ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "review_approved", value: true })}>Preview review approval</button> : null}
                {row.review_approved && row.lifecycle?.missing_gate === "permission" ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "grants", value: row.lifecycle?.requested_permissions || [] })}>Preview exact grants</button> : null}
                {row.review_approved && !row.enabled ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "enabled", value: true })}>Preview enable</button> : null}
                {row.enabled && !row.active ? <button type="button" onClick={() => previewMutation("activate", { record_id: row.record_id })}>Preview activation</button> : null}
                {!row.active && dynamicRows.some((item) => item?.pack_id === row.pack_id && item?.active) ? <button type="button" onClick={() => compareVersion(row)}>Compare with active</button> : null}
                {row.enabled ? <button type="button" onClick={() => previewMutation("gate", { record_id: row.record_id, gate: "enabled", value: false })}>Preview disable</button> : null}
                {row.lifecycle?.requested_permissions?.includes("broker:selected_local_data") ? <button type="button" onClick={() => previewAuthority("grant", { pack_id: row.pack_id, adapter: { kind: "local_file_import", purpose: "build a bounded private search index", allowed_extensions: [".json", ".csv", ".html", ".htm", ".txt"], max_file_size_mb: 8, path_policy: "user_selected_file_only", stores_local_index: true, network_allowed: false }, requested_path: selectedFile.trim() })}>Preview exact-file grant</button> : null}
                {row.lifecycle?.requested_permissions?.includes("broker:selected_local_data") ? <button type="button" onClick={() => previewAuthority("index", { pack_id: row.pack_id, record_id: row.record_id })}>Preview private indexing</button> : null}
                {row.lifecycle?.requested_permissions?.includes("broker:selected_local_data") ? <button type="button" onClick={() => previewAuthority("revoke", { pack_id: row.pack_id })}>Preview revocation</button> : null}
                <button type="button" onClick={() => previewMutation("remove", { record_id: row.record_id, private_data: "retain" })}>Remove, retain derived data</button>
                <button type="button" onClick={() => previewMutation("remove", { record_id: row.record_id, private_data: "delete" })}>Remove and delete derived data</button>
              </div>
            </DetailRow>
          ))}
        </div>
      </div>

      <div className="card"><h2>Presence visualizer</h2><PackVisualizerPreview visualizer={visualizer} /></div>
      {updateComparison ? <div className="card"><h2>Update comparison</h2><p className="help-text">{updateComparison.from?.version} → {updateComparison.to?.version} · authority {updateComparison.authority_changed ? "changed" : "unchanged"}</p><p>Changed sections: {(updateComparison.changed_sections || []).join(", ") || "none"}. New permissions: {(updateComparison.new_permissions_required || []).join(", ") || "none"}. The active version stays active until a separately confirmed activation succeeds.</p></div> : null}

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
                  <button onClick={() => previewCandidate(row)} type="button">Preview metadata</button>
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
      {candidatePreview ? <div className="card"><h2>Catalog candidate preview</h2><p className="help-text">Untrusted metadata only · not fetched · not reviewed · not usable</p><p>{String(candidatePreview?.preview?.summary || candidatePreview?.listing?.summary || candidatePreview?.message || "No bounded summary was supplied.").slice(0, 500)}</p></div> : null}
    </section>
  );
}
