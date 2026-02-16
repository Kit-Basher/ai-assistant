import { useEffect, useMemo, useState } from "react";

const ROUTING_MODES = ["auto", "prefer_cheap", "prefer_best", "prefer_local_lowest_cost_capable"];
const PROVIDER_PRESETS = {
  custom: {
    label: "Custom OpenAI-Compatible",
    id: "",
    base_url: "",
    chat_path: "/v1/chat/completions",
    local: false
  },
  openrouter: {
    label: "OpenRouter",
    id: "openrouter",
    base_url: "https://openrouter.ai/api/v1",
    chat_path: "/chat/completions",
    local: false
  },
  ollama: {
    label: "Ollama (Local)",
    id: "ollama",
    base_url: "http://127.0.0.1:11434",
    chat_path: "/v1/chat/completions",
    local: true
  }
};

function healthStatus(entity) {
  return entity?.health?.status || "ok";
}

function healthLabel(entity) {
  const status = healthStatus(entity);
  if (status === "down") return "down";
  if (status === "degraded") return "degraded";
  return "ok";
}

function asErrorText(error) {
  if (!error) return "Unknown error";
  if (typeof error === "string") return error;
  if (error.message) return error.message;
  return JSON.stringify(error);
}

function parseJsonObject(rawText, fieldLabel) {
  const value = String(rawText || "").trim();
  if (!value) return { ok: true, value: {} };
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, error: `${fieldLabel} must be a JSON object` };
    }
    return { ok: true, value: parsed };
  } catch (_error) {
    return { ok: false, error: `${fieldLabel} must be valid JSON` };
  }
}

function formatNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function MessageBubble({ message }) {
  const isAssistant = message.role === "assistant";
  return (
    <div className={`message-row ${isAssistant ? "assistant" : "user"}`}>
      <div className="message-bubble">
        <div className="message-role">{isAssistant ? "Assistant" : "You"}</div>
        <div className="message-content">{message.content || "(empty response)"}</div>
        {isAssistant && message.meta ? (
          <div className="message-meta">
            <span className="badge">{`${message.meta.provider || "none"}/${message.meta.model || "none"}`}</span>
            {message.meta.fallback_used ? <span className="badge fallback">Fallback</span> : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("setup");

  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [routingMode, setRoutingMode] = useState("auto");
  const [defaultProvider, setDefaultProvider] = useState("");
  const [defaultModel, setDefaultModel] = useState("");
  const [allowRemoteFallback, setAllowRemoteFallback] = useState(true);
  const [setupStatus, setSetupStatus] = useState("");

  const [selectedProvider, setSelectedProvider] = useState("");
  const [selectedModel, setSelectedModel] = useState("");

  const [providerDrafts, setProviderDrafts] = useState({});
  const [providerSecrets, setProviderSecrets] = useState({});
  const [providerStatuses, setProviderStatuses] = useState({});
  const [addProviderForm, setAddProviderForm] = useState({
    preset: "custom",
    id: "",
    base_url: "",
    chat_path: "/v1/chat/completions",
    local: false,
    enabled: true,
    api_key: "",
    default_headers_text: "",
    default_query_params_text: "",
    initial_model: ""
  });
  const [addProviderStatus, setAddProviderStatus] = useState("");
  const [addProviderBusy, setAddProviderBusy] = useState(false);
  const [activeProviderForModels, setActiveProviderForModels] = useState("");
  const [manualModelDraft, setManualModelDraft] = useState({
    model: "",
    capabilities: "chat,json,tools"
  });
  const [telegramToken, setTelegramToken] = useState("");
  const [telegramConfigured, setTelegramConfigured] = useState(false);
  const [telegramStatus, setTelegramStatus] = useState("");

  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [chatBusy, setChatBusy] = useState(false);

  const [logs, setLogs] = useState([]);

  const appendLog = (entry) => {
    setLogs((prev) => [
      {
        time: formatNow(),
        ...entry
      },
      ...prev
    ]);
  };

  const request = async (method, path, body) => {
    const init = {
      method,
      headers: { "Content-Type": "application/json" }
    };
    if (body) {
      init.body = JSON.stringify(body);
    }
    const response = await fetch(path, init);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || data.message || `${response.status} ${response.statusText}`);
    }
    return data;
  };

  const refreshRuntimeState = async () => {
    try {
      const [providersPayload, modelsPayload, defaultsPayload, telegramPayload] = await Promise.all([
        request("GET", "/providers"),
        request("GET", "/models"),
        request("GET", "/defaults"),
        request("GET", "/telegram/status").catch(() => null)
      ]);

      const providerRows = providersPayload.providers || [];
      const modelRows = modelsPayload.models || [];

      setProviders(providerRows);
      setModels(modelRows);
      setRoutingMode(defaultsPayload.routing_mode || "auto");
      setDefaultProvider(defaultsPayload.default_provider || "");
      setDefaultModel(defaultsPayload.default_model || "");
      setAllowRemoteFallback(defaultsPayload.allow_remote_fallback !== false);
      if (telegramPayload && telegramPayload.ok) {
        setTelegramConfigured(telegramPayload.configured === true);
      }

      setProviderDrafts((prev) => {
        const next = { ...prev };
        providerRows.forEach((provider) => {
          next[provider.id] = {
            base_url: provider.base_url || "",
            chat_path: provider.chat_path || "/v1/chat/completions",
            local: !!provider.local,
            enabled: provider.enabled !== false
          };
        });
        return next;
      });

      if (!selectedProvider && providerRows.length > 0) {
        setSelectedProvider(defaultsPayload.default_provider || providerRows[0].id);
      }
      if (!selectedModel && modelRows.length > 0) {
        setSelectedModel(defaultsPayload.default_model || modelRows[0].id);
      }
      if (!activeProviderForModels && providerRows.length > 0) {
        setActiveProviderForModels(providerRows[0].id);
      } else if (activeProviderForModels && !providerRows.find((item) => item.id === activeProviderForModels)) {
        setActiveProviderForModels(providerRows[0]?.id || "");
      }

      appendLog({ endpoint: "bootstrap", ok: true, detail: "Loaded /providers, /models, /defaults" });
    } catch (error) {
      appendLog({ endpoint: "bootstrap", ok: false, detail: asErrorText(error) });
    }
  };

  useEffect(() => {
    refreshRuntimeState();
  }, []);

  const providerOptions = useMemo(() => providers.map((item) => item.id), [providers]);
  const modelOptions = useMemo(
    () => models.filter((item) => !selectedProvider || item.provider === selectedProvider),
    [models, selectedProvider]
  );
  const defaultModelOptions = useMemo(
    () => models.filter((item) => !defaultProvider || item.provider === defaultProvider),
    [models, defaultProvider]
  );
  const providerRecommendations = useMemo(() => {
    const notes = [];
    providers.forEach((provider) => {
      const status = healthStatus(provider);
      if (status === "down" && provider.enabled !== false && !provider.local) {
        notes.push(`Disable ${provider.id} until credentials/connectivity recover.`);
        return;
      }
      if (status === "degraded" && !provider.local) {
        notes.push(`${provider.id} degraded: local models are recommended for now.`);
      }
    });
    return notes;
  }, [providers]);
  const activeProviderModels = useMemo(
    () => models.filter((model) => model.provider === activeProviderForModels),
    [models, activeProviderForModels]
  );

  const saveDefaults = async () => {
    setSetupStatus("Saving defaults...");
    try {
      await request("PUT", "/defaults", {
        routing_mode: routingMode,
        default_provider: defaultProvider || null,
        default_model: defaultModel || null,
        allow_remote_fallback: allowRemoteFallback
      });
      setSetupStatus("Defaults saved.");
      appendLog({ endpoint: "/defaults", ok: true, detail: "Updated defaults" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setSetupStatus(`Failed: ${detail}`);
      appendLog({ endpoint: "/defaults", ok: false, detail });
    }
  };

  const refreshModels = async () => {
    try {
      await request("POST", "/models/refresh", {});
      appendLog({ endpoint: "/models/refresh", ok: true, detail: "Refreshed models" });
      await refreshRuntimeState();
    } catch (error) {
      appendLog({ endpoint: "/models/refresh", ok: false, detail: asErrorText(error) });
    }
  };

  const applyProviderPreset = (presetId) => {
    const preset = PROVIDER_PRESETS[presetId] || PROVIDER_PRESETS.custom;
    setAddProviderForm((prev) => ({
      ...prev,
      preset: presetId,
      id: preset.id || prev.id,
      base_url: preset.base_url,
      chat_path: preset.chat_path,
      local: preset.local
    }));
  };

  const updateAddProviderField = (field, value) => {
    setAddProviderForm((prev) => ({ ...prev, [field]: value }));
  };

  const saveOrTestProvider = async ({ runTest = false } = {}) => {
    const providerId = addProviderForm.id.trim().toLowerCase();
    const baseUrl = addProviderForm.base_url.trim();
    const chatPath = addProviderForm.chat_path.trim() || "/v1/chat/completions";
    const initialModel = addProviderForm.initial_model.trim();
    const apiKey = addProviderForm.api_key.trim();

    if (!providerId || !baseUrl) {
      setAddProviderStatus("Provider id and base URL are required.");
      return;
    }

    const headersParsed = parseJsonObject(addProviderForm.default_headers_text, "Headers");
    if (!headersParsed.ok) {
      setAddProviderStatus(headersParsed.error);
      return;
    }
    const queryParsed = parseJsonObject(addProviderForm.default_query_params_text, "Query params");
    if (!queryParsed.ok) {
      setAddProviderStatus(queryParsed.error);
      return;
    }

    setAddProviderBusy(true);
    setAddProviderStatus("Saving provider...");
    try {
      const providerPayload = {
        id: providerId,
        provider_type: "openai_compat",
        base_url: baseUrl,
        chat_path: chatPath,
        local: !!addProviderForm.local,
        enabled: !!addProviderForm.enabled,
        default_headers: headersParsed.value,
        default_query_params: queryParsed.value
      };

      const providerExists = providers.some((item) => item.id === providerId);
      if (!providerExists) {
        await request("POST", "/providers", providerPayload);
      } else {
        await request("PUT", `/providers/${providerId}`, {
          base_url: baseUrl,
          chat_path: chatPath,
          local: !!addProviderForm.local,
          enabled: !!addProviderForm.enabled,
          default_headers: headersParsed.value,
          default_query_params: queryParsed.value
        });
      }

      if (apiKey) {
        await request("POST", `/providers/${providerId}/secret`, { api_key: apiKey });
      }

      if (initialModel) {
        await request("POST", `/providers/${providerId}/models`, {
          model: initialModel,
          capabilities: ["chat", "json", "tools"],
          available: true
        });
      }

      if (runTest) {
        const testPayload = {};
        if (initialModel) {
          testPayload.model = `${providerId}:${initialModel}`;
        }
        const testResult = await request("POST", `/providers/${providerId}/test`, testPayload);
        setAddProviderStatus(`Connected: ${testResult.provider}/${testResult.model}`);
      } else {
        setAddProviderStatus("Provider saved.");
      }

      setAddProviderForm((prev) => ({ ...prev, api_key: "" }));
      setActiveProviderForModels(providerId);
      appendLog({ endpoint: "/providers", ok: true, detail: runTest ? "Saved and tested provider" : "Saved provider" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setAddProviderStatus(`Failed: ${detail}`);
      appendLog({ endpoint: "/providers", ok: false, detail });
    } finally {
      setAddProviderBusy(false);
    }
  };

  const refreshProviderModels = async (providerId) => {
    if (!providerId) return;
    try {
      await request("POST", `/providers/${providerId}/models/refresh`, {});
      setAddProviderStatus(`Fetched models for ${providerId}.`);
      appendLog({ endpoint: `/providers/${providerId}/models/refresh`, ok: true, detail: "Fetched models" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setAddProviderStatus(`Fetch failed: ${detail}`);
      appendLog({ endpoint: `/providers/${providerId}/models/refresh`, ok: false, detail });
    }
  };

  const addManualModel = async (providerId) => {
    if (!providerId) return;
    const modelName = manualModelDraft.model.trim();
    if (!modelName) {
      setAddProviderStatus("Manual model name is required.");
      return;
    }
    const capabilities = manualModelDraft.capabilities
      .split(",")
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean);
    try {
      await request("POST", `/providers/${providerId}/models`, {
        model: modelName,
        capabilities: capabilities.length ? capabilities : ["chat"],
        available: true
      });
      setAddProviderStatus(`Added model ${providerId}:${modelName}.`);
      setManualModelDraft((prev) => ({ ...prev, model: "" }));
      appendLog({ endpoint: `/providers/${providerId}/models`, ok: true, detail: `Added model ${modelName}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setAddProviderStatus(`Manual add failed: ${detail}`);
      appendLog({ endpoint: `/providers/${providerId}/models`, ok: false, detail });
    }
  };

  const updateProviderField = (providerId, field, value) => {
    setProviderDrafts((prev) => ({
      ...prev,
      [providerId]: {
        ...(prev[providerId] || {}),
        [field]: value
      }
    }));
  };

  const saveProvider = async (providerId) => {
    const draftRow = providerDrafts[providerId] || {};
    try {
      await request("PUT", `/providers/${providerId}`, {
        base_url: draftRow.base_url,
        chat_path: draftRow.chat_path,
        local: !!draftRow.local,
        enabled: !!draftRow.enabled
      });
      setProviderStatuses((prev) => ({ ...prev, [providerId]: "Saved." }));
      appendLog({ endpoint: `/providers/${providerId}`, ok: true, detail: "Updated provider" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setProviderStatuses((prev) => ({ ...prev, [providerId]: `Save failed: ${detail}` }));
      appendLog({ endpoint: `/providers/${providerId}`, ok: false, detail });
    }
  };

  const deleteProvider = async (providerId) => {
    if (!window.confirm(`Remove provider ${providerId}?`)) {
      return;
    }
    try {
      await request("DELETE", `/providers/${providerId}`);
      appendLog({ endpoint: `/providers/${providerId}`, ok: true, detail: "Deleted provider" });
      await refreshRuntimeState();
    } catch (error) {
      appendLog({ endpoint: `/providers/${providerId}`, ok: false, detail: asErrorText(error) });
    }
  };

  const saveProviderSecret = async (providerId) => {
    const key = (providerSecrets[providerId] || "").trim();
    if (!key) {
      setProviderStatuses((prev) => ({ ...prev, [providerId]: "API key is required." }));
      return;
    }
    try {
      await request("POST", `/providers/${providerId}/secret`, { api_key: key });
      setProviderSecrets((prev) => ({ ...prev, [providerId]: "" }));
      setProviderStatuses((prev) => ({ ...prev, [providerId]: "Secret saved." }));
      appendLog({ endpoint: `/providers/${providerId}/secret`, ok: true, detail: "Saved provider secret" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setProviderStatuses((prev) => ({ ...prev, [providerId]: `Secret failed: ${detail}` }));
      appendLog({ endpoint: `/providers/${providerId}/secret`, ok: false, detail });
    }
  };

  const testProvider = async (providerId) => {
    const preferredModel = models.find((model) => model.provider === providerId)?.id || null;
    try {
      const result = await request("POST", `/providers/${providerId}/test`, {
        model: preferredModel
      });
      setProviderStatuses((prev) => ({ ...prev, [providerId]: `Connected: ${result.provider}/${result.model}` }));
      appendLog({ endpoint: `/providers/${providerId}/test`, ok: true, detail: `${result.provider}/${result.model}` });
    } catch (error) {
      const detail = asErrorText(error);
      setProviderStatuses((prev) => ({ ...prev, [providerId]: `Test failed: ${detail}` }));
      appendLog({ endpoint: `/providers/${providerId}/test`, ok: false, detail });
    }
  };

  const sendMessage = async () => {
    const content = draft.trim();
    if (!content || chatBusy) return;

    const nextUserMessage = { role: "user", content };
    const nextMessages = [...messages, nextUserMessage];
    setMessages(nextMessages);
    setDraft("");
    setChatBusy(true);

    try {
      const result = await request("POST", "/chat", {
        messages: nextMessages.map((item) => ({ role: item.role, content: item.content })),
        model: selectedModel || undefined,
        provider: selectedProvider || undefined,
        purpose: "chat",
        task_type: "chat"
      });

      const assistantMessage = {
        role: "assistant",
        content: result.assistant?.content || "",
        meta: {
          provider: result.meta?.provider,
          model: result.meta?.model,
          fallback_used: !!result.meta?.fallback_used
        }
      };
      setMessages((prev) => [...prev, assistantMessage]);
      appendLog({
        endpoint: "/chat",
        ok: true,
        detail: `${assistantMessage.meta.provider || "none"}/${assistantMessage.meta.model || "none"}`
      });
    } catch (error) {
      const detail = asErrorText(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${detail}`,
          meta: { provider: "none", model: "none", fallback_used: false }
        }
      ]);
      appendLog({ endpoint: "/chat", ok: false, detail });
    } finally {
      setChatBusy(false);
    }
  };

  const resetConversation = () => {
    setMessages([]);
    appendLog({ endpoint: "chat/reset", ok: true, detail: "Conversation reset" });
  };

  const exportConversation = () => {
    const blob = new Blob([JSON.stringify(messages, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `personal-agent-chat-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    appendLog({ endpoint: "chat/export", ok: true, detail: "Exported conversation" });
  };

  const saveTelegramToken = async () => {
    const token = telegramToken.trim();
    if (!token) {
      setTelegramStatus("Bot token is required.");
      return;
    }
    try {
      await request("POST", "/telegram/secret", { bot_token: token });
      setTelegramToken("");
      setTelegramConfigured(true);
      setTelegramStatus("Telegram token saved.");
      appendLog({ endpoint: "/telegram/secret", ok: true, detail: "Saved Telegram token" });
    } catch (error) {
      const detail = asErrorText(error);
      setTelegramStatus(`Save failed: ${detail}`);
      appendLog({ endpoint: "/telegram/secret", ok: false, detail });
    }
  };

  const testTelegramToken = async () => {
    try {
      const result = await request("POST", "/telegram/test", {});
      const username = result.telegram_user?.username || "unknown";
      setTelegramStatus(`Connected: @${username}`);
      setTelegramConfigured(true);
      appendLog({ endpoint: "/telegram/test", ok: true, detail: `Connected @${username}` });
    } catch (error) {
      const detail = asErrorText(error);
      setTelegramStatus(`Test failed: ${detail}`);
      appendLog({ endpoint: "/telegram/test", ok: false, detail });
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Personal Agent Web UI</h1>
          <p>Manage providers, defaults, and chat via the local API on this host.</p>
        </div>
      </header>

      <nav className="tabs">
        {[
          ["setup", "Defaults"],
          ["providers", "Providers"],
          ["telegram", "Telegram"],
          ["chat", "Chat"],
          ["debug", "Logs/Debug"]
        ].map(([id, label]) => (
          <button key={id} className={activeTab === id ? "active" : ""} onClick={() => setActiveTab(id)}>
            {label}
          </button>
        ))}
      </nav>

      <main className="panel">
        {activeTab === "setup" ? (
          <section className="grid two">
            <div className="card">
              <h2>Routing Defaults</h2>
              <label>
                Routing mode
                <select value={routingMode} onChange={(event) => setRoutingMode(event.target.value)}>
                  {ROUTING_MODES.map((mode) => (
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
                <button onClick={saveDefaults}>Save Defaults</button>
                <button onClick={refreshModels}>Refresh Local Models</button>
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
                  <div key={model.id} className="model-row">
                    <div className="model-head">
                      <span>{model.id}</span>
                      <span className={`badge health-${healthLabel(model)}`}>{healthLabel(model)}</span>
                    </div>
                    <div className="meta-line">
                      {model.provider} · {model.available ? "available" : "unavailable"} · {model.routable ? "routable" : "not routable"}
                    </div>
                    {model.health?.last_error_kind ? (
                      <div className="meta-line">
                        Last error: {model.health.last_error_kind}
                        {model.health.status_code ? ` (${model.health.status_code})` : ""}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          </section>
        ) : null}

        {activeTab === "providers" ? (
          <section className="grid">
            <div className="card">
              <h2>Add Provider (OpenAI-Compatible)</h2>
              <div className="grid two">
                <label>
                  Preset
                  <select value={addProviderForm.preset} onChange={(event) => applyProviderPreset(event.target.value)}>
                    {Object.entries(PROVIDER_PRESETS).map(([id, preset]) => (
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
                <button disabled={addProviderBusy} onClick={() => saveOrTestProvider({ runTest: true })}>
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
                {activeProviderModels.map((model) => (
                  <div key={model.id} className="model-row">
                    <div className="model-head">
                      <span>{model.id}</span>
                      <span className={`badge health-${healthLabel(model)}`}>{healthLabel(model)}</span>
                    </div>
                    <div className="meta-line">
                      {model.available ? "available" : "unavailable"} · {model.routable ? "routable" : "not routable"}
                    </div>
                  </div>
                ))}
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
                    <button onClick={() => saveProvider(provider.id)}>Save</button>
                    <button onClick={() => saveProviderSecret(provider.id)}>Set Key</button>
                    <button onClick={() => testProvider(provider.id)}>Test</button>
                    <button onClick={() => deleteProvider(provider.id)}>Delete</button>
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
                  <p className="help-text">
                    Models: {models.filter((model) => model.provider === provider.id).map((model) => model.id).join(", ") || "none"}
                  </p>
                </div>
              );
            })}
          </section>
        ) : null}

        {activeTab === "telegram" ? (
          <section className="grid">
            <div className="card">
              <h2>Telegram Bot</h2>
              <p className="help-text">
                Status: {telegramConfigured ? "configured" : "not configured"}
              </p>
              <label>
                Bot token (stored securely)
                <input
                  type="password"
                  value={telegramToken}
                  onChange={(event) => setTelegramToken(event.target.value)}
                  placeholder="123456789:AA..."
                />
              </label>
              <div className="row-actions">
                <button onClick={saveTelegramToken}>Save Token</button>
                <button onClick={testTelegramToken}>Test Telegram</button>
              </div>
              <p className="status-line">{telegramStatus || "No Telegram checks run yet."}</p>
            </div>
          </section>
        ) : null}

        {activeTab === "chat" ? (
          <section className="grid chat-layout">
            <div className="card chat-card">
              <h2>Chat</h2>
              <div className="chat-scroll">
                {messages.length === 0 ? <p className="empty">Start a conversation.</p> : null}
                {messages.map((message, index) => (
                  <MessageBubble key={`${message.role}-${index}`} message={message} />
                ))}
              </div>
              <div className="chat-controls">
                <input
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder="Ask your agent"
                />
                <button disabled={chatBusy} onClick={sendMessage}>
                  {chatBusy ? "Sending..." : "Send"}
                </button>
              </div>
              <div className="row-actions">
                <button onClick={resetConversation}>Reset</button>
                <button onClick={exportConversation}>Export</button>
              </div>
            </div>

            <div className="card">
              <h2>Chat Routing</h2>
              <label>
                Provider override
                <select value={selectedProvider} onChange={(event) => setSelectedProvider(event.target.value)}>
                  <option value="">(defaults)</option>
                  {providerOptions.map((providerId) => (
                    <option key={providerId} value={providerId}>
                      {providerId}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Model override
                <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                  <option value="">(defaults)</option>
                  {modelOptions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </section>
        ) : null}

        {activeTab === "debug" ? (
          <section className="card debug-card">
            <h2>Recent Requests</h2>
            <div className="debug-list">
              {logs.length === 0 ? <p className="empty">No requests yet.</p> : null}
              {logs.map((row, index) => (
                <div key={`${row.time}-${index}`} className={`debug-item ${row.ok ? "ok" : "error"}`}>
                  <div className="debug-head">
                    <span>{row.time}</span>
                    <span>{row.endpoint}</span>
                    <span>{row.ok ? "OK" : "ERROR"}</span>
                  </div>
                  <p>{row.detail}</p>
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
