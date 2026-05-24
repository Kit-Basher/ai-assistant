import ModelStatusRow from "./ModelStatusRow";

export default function SetupTab({
  allowRemoteFallback,
  defaultModel,
  defaultModelOptions,
  defaultProvider,
  models,
  providerOptions,
  providerRecommendations,
  refreshModels,
  routingMode,
  routingModes,
  saveDefaults,
  setAllowRemoteFallback,
  setDefaultModel,
  setDefaultProvider,
  setRoutingMode,
  setShowAdvancedModels,
  setShowUnavailableModels,
  setupStatus,
  showAdvancedModels,
  showUnavailableModels
}) {
  return (
    <section className="grid two">
      <div className="card">
        <h2>Chat model</h2>
        <p className="help-text">Choose the provider and chat model used by default. Embedding-only and unavailable models are hidden unless you show them.</p>
        <label>
          Chat routing mode
          <select value={routingMode} onChange={(event) => setRoutingMode(event.target.value)}>
            {routingModes.map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </label>

        <label>
          AI provider
          <select
            value={defaultProvider}
            onChange={(event) => {
              setDefaultProvider(event.target.value);
              setDefaultModel("");
            }}
          >
            <option value="">(none)</option>
            {providerOptions.map((providerId) => (
              <option key={providerId} value={providerId}>
                {providerId}
              </option>
            ))}
          </select>
        </label>

        <label>
          Chat model
          <select value={defaultModel} onChange={(event) => setDefaultModel(event.target.value)}>
            <option value="">(none)</option>
            {defaultModelOptions.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id}
              </option>
            ))}
          </select>
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={allowRemoteFallback}
            onChange={(event) => setAllowRemoteFallback(event.target.checked)}
          />
          Allow remote fallback if local chat models fail
        </label>

        <div className="setup-filter-row">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={showAdvancedModels}
              onChange={(event) => setShowAdvancedModels(event.target.checked)}
            />
            Show advanced models
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={showUnavailableModels}
              onChange={(event) => setShowUnavailableModels(event.target.checked)}
            />
            Show unavailable
          </label>
        </div>

        <div className="row-actions">
          <button className="button-primary" onClick={saveDefaults}>
            Save chat model
          </button>
          <button onClick={refreshModels}>Refresh models</button>
        </div>
        <p className="status-line">{setupStatus || "No pending changes."}</p>
      </div>

      <div className="card">
        <h2>Available models</h2>
        {providerRecommendations.length > 0 ? (
          <div className="recommendations">
            {providerRecommendations.map((note) => (
              <p key={note} className="help-text">
                Recommendation: {note}
              </p>
            ))}
          </div>
        ) : null}
        <div className="model-list">
          {models.length === 0 ? <p className="empty">No chat-ready models match the current filters.</p> : null}
          {models.map((model) => (
            <ModelStatusRow key={model.id} model={model} showLastError showProvider />
          ))}
        </div>
      </div>
    </section>
  );
}
