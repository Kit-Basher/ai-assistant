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
  setupStatus
}) {
  return (
    <section className="grid two">
      <div className="card">
        <h2>Routing Defaults</h2>
        <label>
          Routing mode
          <select value={routingMode} onChange={(event) => setRoutingMode(event.target.value)}>
            {routingModes.map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </label>

        <label>
          Default provider
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
          Default model
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
          Allow remote fallback when local candidates fail
        </label>

        <div className="row-actions">
          <button className="button-primary" onClick={saveDefaults}>
            Save Defaults
          </button>
          <button onClick={refreshModels}>Refresh Models</button>
        </div>
        <p className="status-line">{setupStatus || "No pending changes."}</p>
      </div>

      <div className="card">
        <h2>Model Quick View</h2>
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
          {models.length === 0 ? <p className="empty">No models loaded.</p> : null}
          {models.map((model) => (
            <ModelStatusRow key={model.id} model={model} showLastError showProvider />
          ))}
        </div>
      </div>
    </section>
  );
}
