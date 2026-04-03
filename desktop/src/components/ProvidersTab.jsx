import ModelStatusRow from "./ModelStatusRow";
import { matchesProviderModelFilter } from "../lib/providerModelHelpers";
import { healthLabel } from "../lib/uiHelpers";

export default function ProvidersTab({
  activeProviderForModels,
  activeProviderModels,
  addManualModel,
  addProviderBusy,
  addProviderForm,
  addProviderStatus,
  applyProviderPreset,
  deleteProvider,
  manualModelDraft,
  providerDrafts,
  providerModelSummaries,
  providerModelViews,
  providerModelsById,
  providerOptions,
  providerPresets,
  providerSecrets,
  providerStatuses,
  providers,
  refreshModels,
  refreshProviderModels,
  saveOrTestProvider,
  saveProvider,
  saveProviderSecret,
  setActiveProviderForModels,
  setManualModelDraft,
  setProviderSecrets,
  testProvider,
  updateAddProviderField,
  updateProviderField,
  updateProviderModelView
}) {
  return (
    <section className="grid">
      <div className="card">
        <h2>Add Provider (OpenAI-Compatible)</h2>
        <div className="grid two">
          <label>
            Preset
            <select value={addProviderForm.preset} onChange={(event) => applyProviderPreset(event.target.value)}>
              {Object.entries(providerPresets).map(([id, preset]) => (
                <option key={id} value={id}>
                  {preset.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Provider ID
            <input
              value={addProviderForm.id}
              onChange={(event) => updateAddProviderField("id", event.target.value)}
              placeholder="provider-id"
            />
          </label>
          <label>
            Base URL
            <input
              value={addProviderForm.base_url}
              onChange={(event) => updateAddProviderField("base_url", event.target.value)}
              placeholder="https://api.example.com"
            />
          </label>
          <label>
            Chat path
            <input
              value={addProviderForm.chat_path}
              onChange={(event) => updateAddProviderField("chat_path", event.target.value)}
              placeholder="/v1/chat/completions"
            />
          </label>
          <label>
            API Key (optional, stored securely)
            <input
              type="password"
              value={addProviderForm.api_key}
              onChange={(event) => updateAddProviderField("api_key", event.target.value)}
              placeholder="sk-..."
            />
          </label>
          <label>
            Optional initial model
            <input
              value={addProviderForm.initial_model}
              onChange={(event) => updateAddProviderField("initial_model", event.target.value)}
              placeholder="gpt-4o-mini"
            />
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={addProviderForm.local}
              onChange={(event) => updateAddProviderField("local", event.target.checked)}
            />
            Local provider
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={addProviderForm.enabled}
              onChange={(event) => updateAddProviderField("enabled", event.target.checked)}
            />
            Enabled
          </label>
        </div>
        <label>
          Default headers JSON (optional)
          <textarea
            value={addProviderForm.default_headers_text}
            onChange={(event) => updateAddProviderField("default_headers_text", event.target.value)}
            placeholder='{"HTTP-Referer":"https://example.com"}'
          />
        </label>
        <label>
          Default query params JSON (optional)
          <textarea
            value={addProviderForm.default_query_params_text}
            onChange={(event) => updateAddProviderField("default_query_params_text", event.target.value)}
            placeholder='{"api-version":"2024-06-01"}'
          />
        </label>
        <div className="row-actions">
          <button disabled={addProviderBusy} onClick={() => saveOrTestProvider({ runTest: false })}>
            {addProviderBusy ? "Saving..." : "Save Provider"}
          </button>
          <button className="button-primary" disabled={addProviderBusy} onClick={() => saveOrTestProvider({ runTest: true })}>
            {addProviderBusy ? "Testing..." : "Save + Test"}
          </button>
        </div>
        <p className="status-line">{addProviderStatus || "Add any OpenAI-compatible endpoint."}</p>
      </div>

      <div className="card">
        <h2>Models</h2>
        <label>
          Provider
          <select value={activeProviderForModels} onChange={(event) => setActiveProviderForModels(event.target.value)}>
            <option value="">(select provider)</option>
            {providerOptions.map((providerId) => (
              <option key={providerId} value={providerId}>
                {providerId}
              </option>
            ))}
          </select>
        </label>
        <div className="row-actions">
          <button disabled={!activeProviderForModels} onClick={() => refreshProviderModels(activeProviderForModels)}>
            Fetch Models
          </button>
          <button onClick={refreshModels}>Refresh All</button>
        </div>
        <div className="model-list">
          {activeProviderModels.length === 0 ? <p className="empty">No models for selected provider.</p> : null}
          {activeProviderModels.map((model) => <ModelStatusRow key={model.id} model={model} />)}
        </div>
        <div className="grid two">
          <label>
            Manual model name
            <input
              value={manualModelDraft.model}
              onChange={(event) => setManualModelDraft((prev) => ({ ...prev, model: event.target.value }))}
              placeholder="model-name"
            />
          </label>
          <label>
            Capabilities (comma-separated)
            <input
              value={manualModelDraft.capabilities}
              onChange={(event) => setManualModelDraft((prev) => ({ ...prev, capabilities: event.target.value }))}
              placeholder="chat,json,tools"
            />
          </label>
        </div>
        <button disabled={!activeProviderForModels} onClick={() => addManualModel(activeProviderForModels)}>
          Add Model Manually
        </button>
      </div>

      {providers.map((provider) => {
        const draftRow = providerDrafts[provider.id] || {};
        const keySourceType = provider.api_key_source?.type || "none";
        const keySourceName = provider.api_key_source?.name || "";
        const providerModels = providerModelsById[provider.id] || [];
        const providerModelSummary = providerModelSummaries[provider.id] || {
          total: 0,
          available: 0,
          routable: 0,
          issues: 0,
          capabilities: []
        };
        const providerModelView = {
          expanded: false,
          query: "",
          filter: "all",
          ...(providerModelViews[provider.id] || {})
        };
        const filteredProviderModels = providerModels.filter((model) => matchesProviderModelFilter(model, providerModelView));

        return (
          <div className="card" key={provider.id}>
            <h2>
              {provider.id} <span className={`badge health-${healthLabel(provider)}`}>{healthLabel(provider)}</span>
            </h2>
            <div className="grid two">
              <label>
                Base URL
                <input
                  value={draftRow.base_url || ""}
                  onChange={(event) => updateProviderField(provider.id, "base_url", event.target.value)}
                />
              </label>
              <label>
                Chat path
                <input
                  value={draftRow.chat_path || "/v1/chat/completions"}
                  onChange={(event) => updateProviderField(provider.id, "chat_path", event.target.value)}
                />
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={!!draftRow.enabled}
                  onChange={(event) => updateProviderField(provider.id, "enabled", event.target.checked)}
                />
                Enabled
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={!!draftRow.local}
                  onChange={(event) => updateProviderField(provider.id, "local", event.target.checked)}
                />
                Local
              </label>
            </div>

            <label>
              API Key (stored securely)
              <input
                type="password"
                value={providerSecrets[provider.id] || ""}
                onChange={(event) =>
                  setProviderSecrets((prev) => ({
                    ...prev,
                    [provider.id]: event.target.value
                  }))
                }
                placeholder="Paste provider key"
              />
            </label>

            <div className="row-actions">
              <button className="button-primary" onClick={() => saveProvider(provider.id)}>
                Save
              </button>
              <button onClick={() => saveProviderSecret(provider.id)}>Set Key</button>
              <button onClick={() => testProvider(provider.id)}>Test</button>
              <button className="button-danger" onClick={() => deleteProvider(provider.id)}>
                Delete
              </button>
            </div>

            <p className="status-line">{providerStatuses[provider.id] || ""}</p>
            {provider.health?.last_error_kind ? (
              <p className="help-text">
                Health: {healthLabel(provider)} · Last error {provider.health.last_error_kind}
                {provider.health.status_code ? ` (${provider.health.status_code})` : ""}
              </p>
            ) : (
              <p className="help-text">Health: {healthLabel(provider)}</p>
            )}
            <p className="help-text">
              Key source: {keySourceType}
              {keySourceName ? ` (${keySourceName})` : ""}
            </p>
            <div className="provider-model-summary">
              <span className="badge">{providerModelSummary.total} models</span>
              <span className="badge health-ok">{providerModelSummary.available} available</span>
              <span className="badge">{providerModelSummary.routable} routable</span>
              {providerModelSummary.issues > 0 ? (
                <span className="badge health-down">{providerModelSummary.issues} issues</span>
              ) : null}
            </div>
            {providerModelSummary.capabilities.length > 0 ? (
              <div className="provider-capability-badges">
                {providerModelSummary.capabilities.map((capability) => (
                  <span key={`${provider.id}-${capability.name}`} className="badge">
                    {capability.name} {capability.count}
                  </span>
                ))}
              </div>
            ) : (
              <p className="help-text">No provider models loaded yet.</p>
            )}
            <div className="row-actions">
              <button
                onClick={() => updateProviderModelView(provider.id, { expanded: !providerModelView.expanded })}
              >
                {providerModelView.expanded ? "Hide Models" : `Show Models (${providerModelSummary.total})`}
              </button>
            </div>
            {providerModelView.expanded ? (
              <div className="provider-model-panel">
                <div className="provider-model-controls">
                  <label>
                    Filter models
                    <input
                      value={providerModelView.query}
                      onChange={(event) => updateProviderModelView(provider.id, { query: event.target.value })}
                      placeholder="Search by name or capability"
                    />
                  </label>
                  <label>
                    Status filter
                    <select
                      value={providerModelView.filter}
                      onChange={(event) => updateProviderModelView(provider.id, { filter: event.target.value })}
                    >
                      <option value="all">All models</option>
                      <option value="available">Available</option>
                      <option value="routable">Routable</option>
                      <option value="issues">Needs attention</option>
                    </select>
                  </label>
                </div>
                <p className="help-text">
                  Showing {filteredProviderModels.length} of {providerModelSummary.total} models
                </p>
                <div className="model-list provider-model-list">
                  {filteredProviderModels.length === 0 ? <p className="empty">No models match the current filter.</p> : null}
                  {filteredProviderModels.map((model) => (
                    <ModelStatusRow
                      key={model.id}
                      capabilityLimit={4}
                      model={model}
                      showCapabilities
                      showLastError
                    />
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </section>
  );
}
