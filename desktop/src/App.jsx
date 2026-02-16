import { useEffect, useMemo, useState } from "react";

const ROUTING_MODES = ["auto", "prefer_cheap", "prefer_best"];

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
  const [apiBase, setApiBase] = useState("http://127.0.0.1:8765");
  const [activeTab, setActiveTab] = useState("setup");

  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [routingMode, setRoutingMode] = useState("auto");

  const [selectedProvider, setSelectedProvider] = useState("openai");
  const [selectedModel, setSelectedModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [setupStatus, setSetupStatus] = useState("");

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
    const url = `${apiBase}${path}`;
    const init = {
      method,
      headers: { "Content-Type": "application/json" }
    };
    if (body) {
      init.body = JSON.stringify(body);
    }
    const response = await fetch(url, init);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || data.message || `${response.status} ${response.statusText}`);
    }
    return data;
  };

  const loadBootstrap = async () => {
    try {
      const [modelsPayload, configPayload] = await Promise.all([
        request("GET", "/models"),
        request("GET", "/config")
      ]);
      setProviders(modelsPayload.providers || []);
      setModels(modelsPayload.models || []);
      setRoutingMode(configPayload.routing_mode || "auto");

      const availableProvider = (modelsPayload.providers || []).find((item) => item.available)?.name;
      if (availableProvider) {
        setSelectedProvider(availableProvider);
      }

      const availableModel = (modelsPayload.models || []).find((item) => item.available)?.id;
      if (availableModel) {
        setSelectedModel(availableModel);
      }

      appendLog({ endpoint: "bootstrap", ok: true, detail: "Loaded /models and /config" });
    } catch (error) {
      appendLog({ endpoint: "bootstrap", ok: false, detail: asErrorText(error) });
    }
  };

  useEffect(() => {
    loadBootstrap();
  }, []);

  const modelOptions = useMemo(
    () => models.filter((item) => !selectedProvider || item.provider === selectedProvider),
    [models, selectedProvider]
  );

  const testConnection = async () => {
    setSetupStatus("Testing connection...");
    try {
      const result = await request("POST", "/providers/test", {
        provider: selectedProvider,
        model: selectedModel || undefined,
        api_key: apiKey
      });
      setSetupStatus(`Connected: ${result.provider}/${result.model}`);
      appendLog({ endpoint: "/providers/test", ok: true, detail: `${result.provider}/${result.model}` });
      setApiKey("");
      await loadBootstrap();
    } catch (error) {
      const detail = asErrorText(error);
      setSetupStatus(`Connection failed: ${detail}`);
      appendLog({ endpoint: "/providers/test", ok: false, detail });
    }
  };

  const updateRoutingMode = async (nextMode) => {
    setRoutingMode(nextMode);
    try {
      await request("PUT", "/config", { routing_mode: nextMode });
      appendLog({ endpoint: "/config", ok: true, detail: `routing_mode=${nextMode}` });
    } catch (error) {
      appendLog({ endpoint: "/config", ok: false, detail: asErrorText(error) });
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
        purpose: "chat"
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
          <h1>Personal Agent Desktop</h1>
          <p>Local setup + chat + debug over the runtime API</p>
        </div>
        <label className="api-base">
          API Base
          <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} placeholder="http://127.0.0.1:8765" />
        </label>
      </header>

      <nav className="tabs">
        {[
          ["setup", "Setup"],
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
              <h2>Provider Setup</h2>
              <label>
                Provider
                <select value={selectedProvider} onChange={(event) => setSelectedProvider(event.target.value)}>
                  {(providers || []).map((provider) => (
                    <option key={provider.name} value={provider.name}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Model
                <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                  <option value="">(Auto)</option>
                  {modelOptions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id}{model.available ? "" : " (unavailable)"}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                API Key
                <input
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder="Paste provider API key"
                />
              </label>
              <button onClick={testConnection}>Test Connection + Save</button>
              <p className="status-line">{setupStatus || "No test run yet."}</p>
            </div>
            <div className="card">
              <h2>Routing</h2>
              <label>
                Routing mode
                <select value={routingMode} onChange={(event) => updateRoutingMode(event.target.value)}>
                  {ROUTING_MODES.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </label>
              <p className="help-text">
                `auto` balances cost/quality, `prefer_cheap` minimizes spend, `prefer_best` favors strongest model.
              </p>
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
              <h2>Session Model</h2>
              <label>
                Provider
                <select value={selectedProvider} onChange={(event) => setSelectedProvider(event.target.value)}>
                  {(providers || []).map((provider) => (
                    <option key={provider.name} value={provider.name}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Model
                <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                  <option value="">(Auto)</option>
                  {modelOptions.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Routing mode
                <select value={routingMode} onChange={(event) => updateRoutingMode(event.target.value)}>
                  {ROUTING_MODES.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
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
