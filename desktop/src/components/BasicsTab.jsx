import { healthLabel } from "../lib/uiHelpers";

function simpleProviderStatus(provider, statusText) {
  if (!provider) return { label: "Needs setup", tone: "attention", detail: "Add or choose a provider first." };
  const status = healthLabel(provider);
  const normalized = String(statusText || "").toLowerCase();
  if (normalized.includes("working") || normalized.includes("connected")) {
    return { label: "Working", tone: "ok", detail: "Connection test passed." };
  }
  if (normalized.includes("needs api key") || normalized.includes("api key is required")) {
    return { label: "Needs API key", tone: "attention", detail: "Paste an API key, then save and test." };
  }
  if (normalized.includes("could not connect") || normalized.includes("failed")) {
    return { label: "Could not connect", tone: "down", detail: "The test did not connect. Open Provider details for the raw error." };
  }
  if (status === "ok") return { label: "Working", tone: "ok", detail: "This provider looks ready." };
  if (provider.api_key_source?.type === "none" && !provider.local) {
    return { label: "Needs API key", tone: "attention", detail: "Paste an API key, then save and test." };
  }
  return { label: "Could not connect", tone: "down", detail: "Save and test to check this provider." };
}

function capabilitySummaries({ defaultProvider, defaultModel, telegramConfigured, servicesStatus, packsState }) {
  const services = Array.isArray(servicesStatus?.services) ? servicesStatus.services : [];
  const web = services.find((service) => service?.service_id === "searxng") || null;
  const dockerOrPodman = servicesStatus?.docker_available === true || servicesStatus?.podman_available === true;
  const webReady = web?.enabled === true && web?.configured === true && web?.reachable === true;
  const installedPacks = Array.isArray(packsState?.installed) ? packsState.installed : [];
  const reviewPacks = Array.isArray(packsState?.needs_review) ? packsState.needs_review : [];
  return [
    {
      title: "Chat",
      label: defaultProvider && defaultModel ? "Ready" : "Needs setup",
      tone: defaultProvider && defaultModel ? "ok" : "attention",
      detail: defaultProvider && defaultModel ? "The assistant has a chat model selected." : "Choose a chat model to start normal conversations.",
      ask: "Ask: check my model"
    },
    {
      title: "Web search",
      label: webReady ? "Ready" : dockerOrPodman ? "Can set up" : "Needs setup",
      tone: webReady ? "ok" : "attention",
      detail: webReady ? "The assistant can search web result summaries." : "Optional. Ask the assistant to set up web search when you need it.",
      ask: "Ask: set up web search"
    },
    {
      title: "Telegram",
      label: telegramConfigured ? "Ready" : "Needs token",
      tone: telegramConfigured ? "ok" : "attention",
      detail: telegramConfigured ? "Telegram has a saved token." : "Optional. Add a bot token if you want Telegram chat.",
      ask: "Ask: set up Telegram"
    },
    {
      title: "Local models",
      label: defaultProvider ? "Ready" : "Optional",
      tone: defaultProvider ? "ok" : "attention",
      detail: "Optional upgrades can add stronger local models later.",
      ask: "Ask: help me improve local models"
    },
    {
      title: "Skills",
      label: reviewPacks.length ? "Needs review" : installedPacks.length ? "Ready" : "Can add safely",
      tone: reviewPacks.length ? "attention" : "ok",
      detail: "Extra skills use preview, review, approval, setup, and permission steps.",
      ask: "Ask: add a skill for ..."
    }
  ];
}

function runtimeSummary(status) {
  if (status?.ready) {
    return { label: "Running", tone: "ok", detail: status.description || "The local assistant is reachable." };
  }
  return { label: "Needs attention", tone: "attention", detail: status?.description || "The local assistant is still starting or unavailable." };
}

function StatusBadge({ tone, children }) {
  const className = tone === "ok" ? "badge health-ok" : tone === "down" ? "badge health-down" : "badge health-degraded";
  return <span className={className}>{children}</span>;
}

export default function BasicsTab({
  addProviderBusy,
  chatStatus,
  defaultModel,
  defaultModelOptions,
  defaultProvider,
  onRefresh,
  providerOptions,
  providerSecrets,
  providerStatuses,
  providers,
  saveAndTestProviderKey,
  saveDefaults,
  saveTelegramToken,
  selectedProviderId,
  setDefaultModel,
  setDefaultProvider,
  setProviderSecrets,
  setSelectedProviderId,
  setTelegramToken,
  setThemePreference,
  telegramConfigured,
  telegramStatus,
  telegramToken,
  testTelegramToken,
  themePreference,
  servicesStatus,
  packsState
}) {
  const selectedProvider = providers.find((provider) => provider.id === selectedProviderId)
    || providers.find((provider) => provider.id === defaultProvider)
    || providers[0]
    || null;
  const selectedProviderStatus = simpleProviderStatus(selectedProvider, providerStatuses[selectedProvider?.id] || "");
  const selectedProviderKey = selectedProvider?.id ? providerSecrets[selectedProvider.id] || "" : "";
  const selectedProviderHasSavedKey = selectedProvider?.api_key_source?.type && selectedProvider.api_key_source.type !== "none";
  const runtime = runtimeSummary(chatStatus);
  const capabilities = capabilitySummaries({ defaultProvider, defaultModel, telegramConfigured, servicesStatus, packsState });

  return (
    <section className="basics-layout">
      <div className="setup-intro-card">
        <div>
          <p className="product-kicker">Setup</p>
          <h2>Basics</h2>
          <p>Connect a chat model, Telegram, web search, and your preferred theme without opening operator tools.</p>
        </div>
        <button onClick={onRefresh} type="button">Refresh status</button>
      </div>

      <div className="capability-card-grid" aria-label="Capability setup summary">
        {capabilities.map((capability) => (
          <article className="setup-card capability-card" key={capability.title}>
            <div className="setup-card-head">
              <h3>{capability.title}</h3>
              <StatusBadge tone={capability.tone}>{capability.label}</StatusBadge>
            </div>
            <p className="help-text">{capability.detail}</p>
            <p className="assistant-help-text">{capability.ask}</p>
          </article>
        ))}
      </div>

      <div className="setup-card-grid">
        <article className="setup-card">
          <div className="setup-card-head">
            <h3>Chat model</h3>
            <StatusBadge tone={defaultProvider && defaultModel ? "ok" : "attention"}>
              {defaultProvider && defaultModel ? "Selected" : "Choose model"}
            </StatusBadge>
          </div>
          <p className="help-text">Current: {defaultProvider && defaultModel ? `${defaultProvider} / ${defaultModel}` : "not selected"}</p>
          <p className="help-text">Choose what the assistant should use for normal chat.</p>
          <label>
            Provider
            <select
              value={defaultProvider}
              onChange={(event) => {
                setDefaultProvider(event.target.value);
                setDefaultModel("");
                setSelectedProviderId(event.target.value || selectedProvider?.id || "");
              }}
            >
              <option value="">Choose a provider</option>
              {providerOptions.map((providerId) => (
                <option key={providerId} value={providerId}>{providerId}</option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select value={defaultModel} onChange={(event) => setDefaultModel(event.target.value)}>
              <option value="">Choose a model</option>
              {defaultModelOptions.map((model) => (
                <option key={model.id} value={model.id}>{model.id}</option>
              ))}
            </select>
          </label>
          <button className="button-primary" onClick={saveDefaults} type="button">Save chat model</button>
        </article>

        <article className="setup-card">
          <div className="setup-card-head">
            <h3>API key</h3>
            <StatusBadge tone={selectedProviderStatus.tone}>{selectedProviderStatus.label}</StatusBadge>
          </div>
          <p className="help-text">Paste or update the key for one provider. Saved keys are hidden.</p>
          <label>
            Provider
            <select value={selectedProvider?.id || ""} onChange={(event) => setSelectedProviderId(event.target.value)}>
              <option value="">Choose a provider</option>
              {providerOptions.map((providerId) => (
                <option key={providerId} value={providerId}>{providerId}</option>
              ))}
            </select>
          </label>
          <label>
            API key
            <input
              disabled={!selectedProvider}
              type="password"
              value={selectedProviderKey}
              onChange={(event) =>
                setProviderSecrets((prev) => ({
                  ...prev,
                  [selectedProvider.id]: event.target.value
                }))
              }
              placeholder={selectedProviderHasSavedKey ? "Saved key hidden" : "Paste provider key"}
            />
          </label>
          <button
            className="button-primary"
            disabled={addProviderBusy || !selectedProvider}
            onClick={() => saveAndTestProviderKey(selectedProvider.id)}
            type="button"
          >
            {addProviderBusy ? "Testing..." : "Save and test"}
          </button>
          <p className="status-line">{selectedProviderStatus.detail}</p>
        </article>

        <article className="setup-card">
          <div className="setup-card-head">
            <h3>Telegram</h3>
            <StatusBadge tone={telegramConfigured ? "ok" : "attention"}>
              {telegramConfigured ? "Token saved" : "Token missing"}
            </StatusBadge>
          </div>
          <p className="help-text">{telegramStatus || (telegramConfigured ? "Telegram is configured. Test it if you want to check the bot." : "Paste a bot token to connect Telegram.")}</p>
          <label>
            Bot token
            <input
              type="password"
              value={telegramToken}
              onChange={(event) => setTelegramToken(event.target.value)}
              placeholder={telegramConfigured ? "Saved token hidden" : "123456789:AA..."}
            />
          </label>
          <div className="row-actions">
            <button className="button-primary" onClick={saveTelegramToken} type="button">
              Save token
            </button>
            <button onClick={testTelegramToken} type="button">
              Test Telegram
            </button>
          </div>
        </article>

        <article className="setup-card">
          <div className="setup-card-head">
            <h3>Theme</h3>
            <StatusBadge tone="ok">{themePreference === "system" ? "System" : themePreference === "dark" ? "Dark" : "Light"}</StatusBadge>
          </div>
          <p className="help-text">Choose how this Web UI looks on this browser.</p>
          <div className="segmented-control" role="group" aria-label="Theme">
            {["light", "dark", "system"].map((mode) => (
              <button
                className={themePreference === mode ? "active" : ""}
                key={mode}
                onClick={() => setThemePreference(mode)}
                type="button"
              >
                {mode === "system" ? "System" : mode === "dark" ? "Dark" : "Light"}
              </button>
            ))}
          </div>
        </article>

        <article className="setup-card">
          <div className="setup-card-head">
            <h3>Status</h3>
            <StatusBadge tone={runtime.tone}>{runtime.label}</StatusBadge>
          </div>
          <p className="help-text">{runtime.detail}</p>
          <p className="assistant-help-text">
            You can also ask the assistant: “set up Telegram”, “check my model”, or “help me add web search”.
          </p>
        </article>
      </div>
    </section>
  );
}
