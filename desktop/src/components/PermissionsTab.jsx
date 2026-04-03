import DetailRow from "./DetailRow";

export default function PermissionsTab({
  actions,
  auditEntries,
  permissionsConfig,
  permissionsStatus,
  savePermissions,
  setPermissionsConfig,
  updatePermissionAction,
  updatePermissionConstraint
}) {
  return (
    <section className="grid two">
      <div className="card">
        <h2>ModelOps Permissions</h2>
        <p className="help-text">
          Choose which automated actions may make changes. Raw ids are still shown for audit/debug work.
        </p>
        <label>
          Confirmation mode
          <select
            value={permissionsConfig?.mode || "manual_confirm"}
            onChange={(event) => setPermissionsConfig((prev) => ({ ...(prev || {}), mode: event.target.value }))}
          >
            <option value="manual_confirm">Require confirmation</option>
            <option value="auto">Auto-apply allowed actions</option>
          </select>
        </label>

        <div className="grid permission-actions">
          {actions.map((action) => (
            <label key={action.id} className="permission-option">
              <input
                type="checkbox"
                checked={permissionsConfig?.actions?.[action.id] === true}
                onChange={(event) => updatePermissionAction(action.id, event.target.checked)}
              />
              <span className="permission-option-copy">
                <span className="permission-option-title">{action.label}</span>
                <span className="permission-option-description">{action.description}</span>
                <span className="permission-option-id">{action.id}</span>
              </span>
            </label>
          ))}
        </div>

        <label>
          Max download budget (GB)
          <input
            type="number"
            min="0"
            step="0.1"
            value={permissionsConfig?.constraints?.max_download_gb ?? 5}
            onChange={(event) => updatePermissionConstraint("max_download_gb", Number(event.target.value || 0))}
          />
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={permissionsConfig?.constraints?.allow_install_ollama === true}
            onChange={(event) => updatePermissionConstraint("allow_install_ollama", event.target.checked)}
          />
          Allow installing Ollama automatically
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={permissionsConfig?.constraints?.allow_remote_models !== false}
            onChange={(event) => updatePermissionConstraint("allow_remote_models", event.target.checked)}
          />
          Allow remote models and providers
        </label>

        <label>
          Allowed providers (comma-separated ids)
          <input
            value={(permissionsConfig?.constraints?.allowed_providers || []).join(",")}
            onChange={(event) =>
              updatePermissionConstraint(
                "allowed_providers",
                event.target.value
                  .split(",")
                  .map((item) => item.trim().toLowerCase())
                  .filter(Boolean)
              )
            }
          />
        </label>

        <label>
          Allowed model patterns (comma-separated)
          <input
            value={(permissionsConfig?.constraints?.allowed_model_patterns || []).join(",")}
            onChange={(event) =>
              updatePermissionConstraint(
                "allowed_model_patterns",
                event.target.value
                  .split(",")
                  .map((item) => item.trim())
                  .filter(Boolean)
              )
            }
          />
        </label>

        <div className="row-actions">
          <button className="button-primary" onClick={savePermissions}>
            Save Permissions
          </button>
        </div>
        <p className="status-line">{permissionsStatus || "Default is deny for all ModelOps actions."}</p>
      </div>

      <div className="card">
        <h2>Recent Audit Events</h2>
        <div className="model-list">
          {auditEntries.length === 0 ? <p className="empty">No audit events yet.</p> : null}
          {auditEntries.map((entry, index) => (
            <DetailRow
              key={`${entry.ts || "entry"}-${index}`}
              badge={<span className={`badge ${entry.decision === "allow" ? "health-ok" : "health-down"}`}>{entry.decision}</span>}
              metaLines={[
                `${entry.ts} · outcome ${entry.outcome} · reason ${entry.reason}`,
                `dry_run ${entry.dry_run ? "yes" : "no"} · duration ${entry.duration_ms}ms`
              ]}
              title={entry.action}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
