import { useEffect, useMemo, useState } from "react";

const ROUTING_MODES = ["auto", "prefer_cheap", "prefer_best", "prefer_local_lowest_cost_capable"];

function asErrorText(error) {
  if (!error) return "Unknown error";
  if (typeof error === "string") return error;
  if (error.message) return error.message;
  return JSON.stringify(error);
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

  const [newProvider, setNewProvider] = useState({
    id: "",
    base_url: "",
    model: "",
    local: false,
    enabled: true
  });

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
      const [providersPayload, modelsPayload, defaultsPayload] = await Promise.all([
        request("GET", "/providers"),
        request("GET", "/models"),
        request("GET", "/defaults")
      ]);

      const providerRows = providersPayload.providers || [];
      const modelRows = modelsPayload.models || [];

      setProviders(providerRows);
      setModels(modelRows);
      setRoutingMode(defaultsPayload.routing_mode || "auto");
      setDefaultProvider(defaultsPayload.default_provider || "");
      setDefaultModel(defaultsPayload.default_model || "");
      setAllowRemoteFallback(defaultsPayload.allow_remote_fallback !== false);

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
      appendLog({ endpoint: "/models/refresh", ok: true, detail: "Refreshed local models" });
      await refreshRuntimeState();
    } catch (error) {
      appendLog({ endpoint: "/models/refresh", ok: false, detail: asErrorText(error) });
    }
  };

  const addProvider = async () => {
    if (!newProvider.id.trim() || !newProvider.base_url.trim() || !newProvider.model.trim()) {
      setSetupStatus("Provider id, base URL, and initial model are required.");
      return;
    }
    try {
      await request("POST", "/providers", {
        id: newProvider.id.trim().toLowerCase(),
        provider_type: "openai_compat",
        base_url: newProvider.base_url.trim(),
        chat_path: "/v1/chat/completions",
        local: !!newProvider.local,
        enabled: !!newProvider.enabled,
        models: [
          {
            id: `${newProvider.id.trim().toLowerCase()}:${newProvider.model.trim()}`,
            model: newProvider.model.trim(),
            capabilities: ["chat", "json", "tools"],
            pricing: {
              input_per_million_tokens: null,
              output_per_million_tokens: null
            }
          }
        ]
      });
      setNewProvider({ id: "", base_url: "", model: "", local: false, enabled: true });
      appendLog({ endpoint: "/providers", ok: true, detail: "Added provider" });
      await refreshRuntimeState();
    } catch (error) {
      appendLog({ endpoint: "/providers", ok: false, detail: asErrorText(error) });
      setSetupStatus(`Add provider failed: ${asErrorText(error)}`);
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
              <div className="model-list">
                {models.length === 0 ? <p className="empty">No models loaded.</p> : null}
                {models.map((model) => (
                  <div key={model.id} className="model-row">
                    <div>{model.id}</div>
                    <div className="meta-line">
                      {model.provider} · {model.available ? "available" : "unavailable"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        ) : null}

        {activeTab === "providers" ? (
          <section className="grid">
            <div className="card">
              <h2>Add Provider</h2>
              <div className="grid two">
                <label>
                  Provider ID
                  <input
                    value={newProvider.id}
                    onChange={(event) => setNewProvider((prev) => ({ ...prev, id: event.target.value }))}
                    placeholder="acme"
                  />
                </label>
                <label>
                  Base URL
                  <input
                    value={newProvider.base_url}
                    onChange={(event) => setNewProvider((prev) => ({ ...prev, base_url: event.target.value }))}
                    placeholder="https://api.example.com"
                  />
                </label>
                <label>
                  Initial model name
                  <input
                    value={newProvider.model}
                    onChange={(event) => setNewProvider((prev) => ({ ...prev, model: event.target.value }))}
                    placeholder="model-name"
                  />
                </label>
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={newProvider.local}
                    onChange={(event) => setNewProvider((prev) => ({ ...prev, local: event.target.checked }))}
                  />
                  Local provider
                </label>
              </div>
              <button onClick={addProvider}>Add Provider</button>
            </div>

            {providers.map((provider) => {
              const draftRow = providerDrafts[provider.id] || {};
              return (
                <div className="card" key={provider.id}>
                  <h2>{provider.id}</h2>
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
                  <p className="help-text">
                    Models: {models.filter((model) => model.provider === provider.id).map((model) => model.id).join(", ") || "none"}
                  </p>
                </div>
              );
            })}
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
