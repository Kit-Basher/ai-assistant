import DetailRow from "./DetailRow";
import { formatEpoch, normalizeSupportTarget } from "../lib/uiHelpers";

export default function OperationsTab({
  applyLlmAutoconfig,
  applyLlmCapabilitiesReconcile,
  applyLlmCleanup,
  applyLlmHygiene,
  autopilotBootstrapBusy,
  autopilotLastChange,
  autopilotLastChangeBusy,
  autopilotLastChangeStatus,
  autopilotLastReadHash,
  autopilotLedgerEntries,
  autopilotNotifications,
  autopilotNotificationsPolicy,
  autopilotNotifyBusy,
  autopilotNotifyStatus,
  autopilotToast,
  autopilotUndoBusy,
  autoconfigBusy,
  autoconfigPlan,
  autoconfigStatus,
  bootstrapAutopilotDefaults,
  canBootstrapAutopilot,
  canRollbackRegistry,
  canSendAutopilotTest,
  capabilitiesReconcileBusy,
  capabilitiesReconcilePlan,
  capabilitiesReconcileStatus,
  cleanupBusy,
  cleanupPlan,
  cleanupStatus,
  executeSupportRemediation,
  explainLastAutopilotChange,
  exportSupportBundle,
  hygieneBusy,
  hygienePlan,
  hygieneStatus,
  llmCatalogRows,
  llmCatalogStatus,
  llmHealth,
  llmHealthMessage,
  llmHealthRunning,
  markAutopilotRead,
  notificationStoreSummary,
  notifyStatusSummary,
  planLlmAutoconfig,
  planLlmCapabilitiesReconcile,
  planLlmCleanup,
  planLlmHygiene,
  planSupportRemediation,
  registrySnapshots,
  rollbackBusySnapshotId,
  rollbackRegistryToSnapshot,
  runLlmCatalogRefresh,
  runLlmHealthCheck,
  runSupportDiagnosis,
  safetyStatus,
  sendAutopilotTestNotification,
  setSupportDiagnoseIntent,
  setSupportDiagnoseTarget,
  supportBundlePreview,
  supportBusy,
  supportDiagnosis,
  supportDiagnoseIntent,
  supportDiagnoseTarget,
  supportRemediationPlan,
  supportRemediationResult,
  supportStatus,
  supportTargetOptions,
  undoLastAutopilotChange
}) {
  const autopilotPolicyBadge = (() => {
    const reason = String(autopilotNotificationsPolicy?.allow_reason || "");
    if (reason === "loopback_auto") return { label: "Dev Mode (Loopback Auto-Allow)", className: "health-ok" };
    if (reason === "permission_required") return { label: "Permission Required", className: "health-degraded" };
    if (reason === "explicit_true") return { label: "Explicitly Enabled", className: "policy-explicit-true" };
    if (reason === "explicit_false") return { label: "Explicitly Disabled", className: "health-down" };
    return null;
  })();

  return (
    <section className="grid">
      <div className="card">
        <h2>LLM Health + Automation</h2>
        <p className="help-text">
          Last health run: {llmHealth?.last_run_at_iso || "never"}
        </p>
        <p className="help-text">
          Status counts: ok {Number(llmHealth?.counts?.ok || 0)} · degraded {Number(llmHealth?.counts?.degraded || 0)} · down{" "}
          {Number(llmHealth?.counts?.down || 0)}
        </p>
        <p className="help-text">
          Scheduler: {llmHealth?.scheduler?.enabled ? "enabled" : "disabled"} · next health {formatEpoch(llmHealth?.scheduler?.next_health_run_at)}
        </p>
        <p className="help-text">
          Next catalog {formatEpoch(llmHealth?.scheduler?.next_catalog_run_at)} · next refresh {formatEpoch(llmHealth?.scheduler?.next_refresh_run_at)}
        </p>
        <p className="help-text">
          Next bootstrap {formatEpoch(llmHealth?.scheduler?.next_bootstrap_run_at)}
        </p>
        <p className="help-text">
          Next reconcile {formatEpoch(llmHealth?.scheduler?.next_capabilities_reconcile_run_at)}
        </p>
        <p className="help-text">
          Next hygiene {formatEpoch(llmHealth?.scheduler?.next_hygiene_run_at)}
        </p>
        <p className="help-text">
          Next cleanup {formatEpoch(llmHealth?.scheduler?.next_cleanup_run_at)}
        </p>
        <p className="help-text">
          Next self-heal {formatEpoch(llmHealth?.scheduler?.next_self_heal_run_at)} · next autoconfig{" "}
          {formatEpoch(llmHealth?.scheduler?.next_autoconfig_run_at)}
        </p>
        <p className="help-text">
          Next scout {formatEpoch(llmHealth?.scheduler?.next_model_scout_run_at)}
        </p>
        <div className="row-actions">
          <button className="button-primary" disabled={llmHealthRunning} onClick={runLlmHealthCheck}>
            {llmHealthRunning ? "Running..." : "Run Health Check"}
          </button>
          <button disabled={llmHealthRunning} onClick={runLlmCatalogRefresh}>
            {llmHealthRunning ? "Working..." : "Run Catalog Refresh"}
          </button>
        </div>
        <p className="status-line">{llmHealthMessage || "No health checks run from UI in this session."}</p>
        <div className="model-list">
          {Array.isArray(llmHealth?.last_actions) && llmHealth.last_actions.length > 0 ? null : (
            <p className="empty">No recent autopilot actions.</p>
          )}
          {(llmHealth?.last_actions || []).slice(0, 5).map((entry, index) => (
            <DetailRow
              key={`${entry.ts || "llm-action"}-${index}`}
              badge={<span className={`badge ${entry.outcome === "success" ? "health-ok" : "health-degraded"}`}>{entry.outcome || "unknown"}</span>}
              metaLines={[`${entry.ts || "n/a"} · ${entry.reason || "n/a"} · ${entry.duration_ms || 0}ms`]}
              title={entry.action}
            />
          ))}
        </div>
        <h3>Recent Notifications</h3>
        {autopilotToast ? <div className="toast-banner">{autopilotToast}</div> : null}
        {autopilotPolicyBadge ? (
          <p className="status-line">
            <span className={`badge ${autopilotPolicyBadge.className}`}>{autopilotPolicyBadge.label}</span>
          </p>
        ) : null}
        <p className="status-line">{notifyStatusSummary}</p>
        <p className="help-text">{notificationStoreSummary}</p>
        <div className="model-list">
          {autopilotNotifications.length === 0 ? <p className="empty">No notifications recorded.</p> : null}
          {autopilotNotifications.map((entry, index) => (
            <DetailRow
              key={`${entry.ts || "autopilot-note"}-${index}`}
              badge={<span className={`badge ${entry.outcome === "sent" ? "health-ok" : "health-degraded"}`}>{entry.outcome || "unknown"}</span>}
              metaLines={[
                `${entry.ts_iso || "n/a"} · ${entry.reason || "n/a"} · delivered_to ${entry.delivered_to || "none"} · ${entry.deferred ? "deferred" : "immediate"}`,
                String(entry.message || "")
                  .split("\n")
                  .slice(1, 3)
                  .join(" ")
                  .trim() || "(no body preview)",
                `hash: ${entry.dedupe_hash || "n/a"}`
              ]}
              title={String(entry.message || "").split("\n")[0] || "LLM Autopilot updated configuration"}
            />
          ))}
        </div>
        <div className="row-actions">
          <button disabled={!canSendAutopilotTest || autopilotNotifyBusy} onClick={sendAutopilotTestNotification}>
            {autopilotNotifyBusy ? "Sending..." : "Send Test Notification"}
          </button>
          <button disabled={autopilotLastChangeBusy} onClick={explainLastAutopilotChange}>
            {autopilotLastChangeBusy ? "Loading..." : "Explain last autopilot change"}
          </button>
          <button
            disabled={!autopilotNotifications[0]?.dedupe_hash}
            onClick={() => markAutopilotRead(autopilotNotifications[0]?.dedupe_hash)}
          >
            Mark Latest Read
          </button>
        </div>
        <p className="status-line">
          {autopilotNotifyStatus || (canSendAutopilotTest ? "Notification test is allowed by current policy." : "Enable llm.notifications.test in Permissions to send tests.")}
        </p>
        <p className="status-line">
          {autopilotLastChangeStatus || (autopilotLastReadHash ? `Last read hash: ${autopilotLastReadHash.slice(0, 12)}` : "No read acknowledgment stored yet.")}
        </p>
        {autopilotLastChange ? (
          <DetailRow
            badge={<span className="badge">{autopilotLastChange.snapshot_id_before || "no-snapshot"}</span>}
            metaLines={[
              `${formatEpoch(autopilotLastChange.ts)} · ${autopilotLastChange.reason || "n/a"} · hash ${String(autopilotLastChange.registry_hash_after || "").slice(0, 12) || "n/a"}`,
              `changed: ${(autopilotLastChange.changed_ids || []).join(", ") || "none"}`
            ]}
            title={autopilotLastChange.action || "llm.autopilot.apply"}
          >
            {(autopilotLastChange.rationale_lines || []).map((line, index) => (
              <div key={`${line}-${index}`} className="meta-line">
                {line}
              </div>
            ))}
          </DetailRow>
        ) : null}

        <div className="row-actions">
          <button disabled={autoconfigBusy} onClick={planLlmAutoconfig}>
            {autoconfigBusy ? "Working..." : "Plan Autoconfig"}
          </button>
          <button className="button-primary" disabled={autoconfigBusy || !autoconfigPlan} onClick={applyLlmAutoconfig}>
            Apply Autoconfig
          </button>
        </div>
        <p className="help-text">
          Autoconfig plan: {Number(autoconfigPlan?.impact?.changes_count || 0)} change(s)
        </p>
        <p className="status-line">{autoconfigStatus || "Autoconfig never runs without permissions."}</p>

        <div className="row-actions">
          <button disabled={hygieneBusy} onClick={planLlmHygiene}>
            {hygieneBusy ? "Working..." : "Plan Hygiene"}
          </button>
          <button className="button-primary" disabled={hygieneBusy || !hygienePlan} onClick={applyLlmHygiene}>
            Apply Hygiene
          </button>
        </div>
        <p className="help-text">Hygiene plan: {Number(hygienePlan?.impact?.changes_count || 0)} change(s)</p>
        <p className="status-line">{hygieneStatus || "Hygiene only touches registry entries."}</p>

        <h3>Catalog</h3>
        <p className="help-text">
          Last refresh: {llmCatalogStatus?.last_run_at_iso || "never"} · providers {Array.isArray(llmCatalogStatus?.providers) ? llmCatalogStatus.providers.length : 0}
        </p>
        <p className="help-text">
          Last errors:{" "}
          {(Array.isArray(llmCatalogStatus?.providers) ? llmCatalogStatus.providers : [])
            .filter((row) => row?.last_error_kind)
            .map((row) => `${row.provider_id}:${row.last_error_kind}`)
            .join(", ") || "none"}
        </p>
        <div className="model-list">
          {llmCatalogRows.length === 0 ? <p className="empty">No catalog rows available.</p> : null}
          {llmCatalogRows.slice(0, 8).map((row) => (
            <DetailRow
              key={row.id}
              badge={<span className="badge">{(row.capabilities || []).join(",") || "chat"}</span>}
              metaLines={[
                `ctx ${row.max_context_tokens || "?"} · in ${row.input_cost_per_million_tokens ?? "n/a"} · out ${row.output_cost_per_million_tokens ?? "n/a"} · ${row.source}`
              ]}
              title={row.id}
            />
          ))}
        </div>

        <h3>Cleanup</h3>
        <div className="row-actions">
          <button disabled={cleanupBusy} onClick={planLlmCleanup}>
            {cleanupBusy ? "Working..." : "Plan Cleanup"}
          </button>
          <button className="button-primary" disabled={cleanupBusy || !cleanupPlan} onClick={applyLlmCleanup}>
            Apply Cleanup
          </button>
        </div>
        <p className="help-text">
          Cleanup plan: {Number(cleanupPlan?.impact?.changes_count || 0)} change(s) · prune candidates {Number(cleanupPlan?.impact?.prune_candidates_count || 0)}
        </p>
        <p className="status-line">{cleanupStatus || "Cleanup marks stale entries and prunes only when policy allows."}</p>

        <h3>Capabilities</h3>
        <p className="help-text">
          Mismatch models: {Number(llmHealth?.capabilities_reconcile?.mismatch_count || 0)} · planned changes{" "}
          {Number(llmHealth?.capabilities_reconcile?.changes_count || 0)}
        </p>
        <div className="row-actions">
          <button disabled={capabilitiesReconcileBusy} onClick={planLlmCapabilitiesReconcile}>
            {capabilitiesReconcileBusy ? "Working..." : "Plan Reconcile"}
          </button>
          <button
            className="button-primary"
            disabled={capabilitiesReconcileBusy || !capabilitiesReconcilePlan}
            onClick={applyLlmCapabilitiesReconcile}
          >
            Apply Reconcile
          </button>
        </div>
        <p className="help-text">
          Reconcile plan: {Number(capabilitiesReconcilePlan?.impact?.changes_count || 0)} change(s)
        </p>
        <p className="status-line">
          {capabilitiesReconcileStatus || "Capability reconcile fixes chat/embedding drift from catalog inference."}
        </p>

        <h3>Safety</h3>
        <p className="help-text">
          Safe mode: {llmHealth?.autopilot?.safe_mode ? "enabled" : "disabled"} · last blocked reason{" "}
          {llmHealth?.autopilot?.last_blocked_reason || "none"}
        </p>
        <p className="help-text">
          Safe mode reason: {llmHealth?.autopilot?.safe_mode_reason || "none"} · last churn{" "}
          {llmHealth?.autopilot?.last_churn_event_ts_iso || "never"} ({llmHealth?.autopilot?.last_churn_reason || "none"})
        </p>
        <p className="help-text">
          Rollback policy: {llmHealth?.autopilot?.rollback_policy?.allow_reason || "unknown"} ·{" "}
          {llmHealth?.autopilot?.rollback_policy?.allow_rollback_effective ? "rollback allowed" : "permission required"}
        </p>
        <p className="help-text">
          Bootstrap policy: {llmHealth?.autopilot?.bootstrap_policy?.allow_reason || "unknown"} ·{" "}
          {llmHealth?.autopilot?.bootstrap_policy?.allow_apply_effective ? "apply allowed" : "permission required"}
        </p>
        <div className="row-actions">
          <button disabled={autopilotLastChangeBusy} onClick={explainLastAutopilotChange}>
            {autopilotLastChangeBusy ? "Loading..." : "Explain Last Change"}
          </button>
          <button className="button-danger" disabled={!canRollbackRegistry || autopilotUndoBusy} onClick={undoLastAutopilotChange}>
            {autopilotUndoBusy ? "Undoing..." : "Undo Last Autopilot Change"}
          </button>
          <button className="button-primary" disabled={!canBootstrapAutopilot || autopilotBootstrapBusy} onClick={bootstrapAutopilotDefaults}>
            {autopilotBootstrapBusy ? "Bootstrapping..." : "Bootstrap Defaults"}
          </button>
        </div>
        <p className="status-line">
          {safetyStatus || (canRollbackRegistry ? "Rollback is allowed by current policy." : "Enable llm.registry.rollback or use loopback auto policy.")}
        </p>
        <div className="model-list">
          {autopilotLedgerEntries.length === 0 ? <p className="empty">No autopilot ledger entries yet.</p> : null}
          {autopilotLedgerEntries.slice(0, 10).map((entry) => {
            const snapshotId = String(entry.snapshot_id || "").trim();
            return (
              <DetailRow
                key={entry.id}
                badge={<span className={`badge ${entry.outcome === "success" ? "health-ok" : "health-degraded"}`}>{entry.outcome || "unknown"}</span>}
                metaLines={[
                  `${entry.ts_iso || "n/a"} · ${entry.reason || "n/a"} · changed ${(entry.changed_ids || []).join(", ") || "none"}`,
                  `snapshot ${snapshotId || "n/a"} · hash ${entry.resulting_registry_hash || "n/a"}`
                ]}
                title={entry.action || "llm.apply"}
              >
                <div className="row-actions">
                  <button
                    className="button-danger"
                    disabled={!snapshotId || !canRollbackRegistry || rollbackBusySnapshotId === snapshotId}
                    onClick={() => rollbackRegistryToSnapshot(snapshotId)}
                  >
                    {rollbackBusySnapshotId === snapshotId ? "Rolling Back..." : "Rollback"}
                  </button>
                </div>
              </DetailRow>
            );
          })}
        </div>
        <div className="model-list">
          {registrySnapshots.length === 0 ? <p className="empty">No snapshots available.</p> : null}
          {registrySnapshots.slice(0, 10).map((row) => (
            <DetailRow
              key={row.snapshot_id}
              badge={<span className="badge">{row.size_bytes || 0} bytes</span>}
              metaLines={[`hash ${row.registry_hash || "n/a"}`]}
              title={row.snapshot_id}
            >
              <div className="row-actions">
                <button
                  className="button-danger"
                  disabled={!canRollbackRegistry || rollbackBusySnapshotId === row.snapshot_id}
                  onClick={() => rollbackRegistryToSnapshot(row.snapshot_id)}
                >
                  {rollbackBusySnapshotId === row.snapshot_id ? "Rolling Back..." : "Rollback"}
                </button>
              </div>
            </DetailRow>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Support</h2>
        <p className="help-text">
          Export a deterministic local support bundle, diagnose a provider/model, and generate a deterministic LLM remediation plan.
        </p>
        <div className="grid two">
          <label>
            Diagnose Target
            <select
              value={supportDiagnoseTarget}
              onChange={(event) => setSupportDiagnoseTarget(event.target.value)}
            >
              {supportTargetOptions.length === 0 ? <option value="">No targets available</option> : null}
              {supportTargetOptions.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Remediation Intent
            <select
              value={supportDiagnoseIntent}
              onChange={(event) => setSupportDiagnoseIntent(event.target.value)}
            >
              <option value="fix_routing">Fix Routing</option>
              <option value="reduce_churn">Reduce Churn</option>
              <option value="bootstrap">Bootstrap</option>
            </select>
          </label>
        </div>
        <div className="row-actions">
          <button disabled={supportBusy} onClick={exportSupportBundle}>
            {supportBusy ? "Working..." : "Export Support Bundle"}
          </button>
          <button
            disabled={supportBusy || !normalizeSupportTarget(supportDiagnoseTarget)}
            onClick={runSupportDiagnosis}
          >
            {supportBusy ? "Working..." : "Run Diagnosis"}
          </button>
          <button disabled={supportBusy} onClick={planSupportRemediation}>
            {supportBusy ? "Working..." : "Fix LLM setup"}
          </button>
          <button className="button-primary" disabled={supportBusy || !supportRemediationPlan} onClick={executeSupportRemediation}>
            {supportBusy ? "Working..." : "Execute safe steps"}
          </button>
        </div>
        <p className="status-line">{supportStatus || "No support actions run in this session."}</p>
        <p className="help-text">
          {supportBundlePreview
            ? `Bundle ready · registry ${String(supportBundlePreview.registry_hash || "").slice(0, 12)} · safe mode ${supportBundlePreview.safe_mode?.enabled ? "on" : "off"}`
            : "Support bundle not exported yet."}
        </p>
        <div className="model-list">
          {!supportDiagnosis ? <p className="empty">Run diagnosis to view root causes and suggested actions.</p> : null}
          {supportDiagnosis ? (
            <DetailRow
              badge={
                <span className={`badge ${String(supportDiagnosis.status || "") === "ok" ? "health-ok" : "health-degraded"}`}>
                  {supportDiagnosis.status || "unknown"}
                </span>
              }
              metaLines={[
                `error ${supportDiagnosis.last_error_kind || "none"} · code ${supportDiagnosis.status_code || "n/a"} · streak ${Number(supportDiagnosis.failure_streak || 0)}`,
                `root causes: ${(supportDiagnosis.root_causes || []).join(", ") || "none"}`
              ]}
              title="Status"
            >
              {(supportDiagnosis.recommended_actions || []).map((line, index) => (
                <div key={`${line}-${index}`} className="meta-line">
                  {line}
                </div>
              ))}
            </DetailRow>
          ) : null}
        </div>
        <div className="model-list">
          {!supportRemediationPlan ? <p className="empty">Run remediation plan to view next steps.</p> : null}
          {supportRemediationPlan ? (
            <DetailRow
              badge={<span className="badge">{supportRemediationPlan.plan_only ? "plan-only" : "apply"}</span>}
              metaLines={[`reasons: ${(supportRemediationPlan.reasons || []).join(" | ") || "n/a"}`]}
              title={`Remediation Plan (${supportRemediationPlan.intent || "fix_routing"})`}
            >
              {(supportRemediationPlan.steps || []).map((step) => (
                <div key={step.id || step.action} className="meta-line">
                  {(step.id || "step").replaceAll("_", " ")}: {step.action} ({step.reason})
                  {step.instructions ? ` · ${step.instructions}` : ""}
                </div>
              ))}
            </DetailRow>
          ) : null}
        </div>
        <div className="model-list">
          {!supportRemediationResult ? <p className="empty">Execute safe steps to apply registry/model changes.</p> : null}
          {supportRemediationResult ? (
            <DetailRow
              badge={
                <span className={`badge ${supportRemediationResult.ok ? "health-ok" : "health-degraded"}`}>
                  {supportRemediationResult.ok ? "ok" : "error"}
                </span>
              }
              metaLines={[
                `executed ${(supportRemediationResult.executed_steps || []).length} · blocked ${(supportRemediationResult.blocked_steps || []).length} · failed ${(supportRemediationResult.failed_steps || []).length}`,
                supportRemediationResult.message || "n/a"
              ]}
              title="Remediation Execute"
            />
          ) : null}
        </div>
      </div>
    </section>
  );
}
