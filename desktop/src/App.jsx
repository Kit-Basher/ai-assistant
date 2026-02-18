import { useEffect, useMemo, useRef, useState } from "react";

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
const MODELOPS_ACTIONS = [
  "modelops.install_ollama",
  "modelops.pull_ollama_model",
  "modelops.import_gguf_to_ollama",
  "modelops.set_default_model",
  "modelops.enable_disable_provider_or_model",
  "llm.autoconfig.apply",
  "llm.hygiene.apply",
  "llm.registry.prune",
  "llm.registry.rollback",
  "llm.self_heal.apply",
  "llm.autopilot.bootstrap.apply",
  "llm.notifications.test",
  "llm.notifications.send",
  "llm.notifications.prune"
];

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

function formatEpoch(epochSeconds) {
  if (!epochSeconds) return "n/a";
  const asNumber = Number(epochSeconds);
  if (!Number.isFinite(asNumber) || asNumber <= 0) return "n/a";
  return new Date(asNumber * 1000).toLocaleString();
}

function newestNotificationHash(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return "";
  const first = rows[0] || {};
  return String(first.dedupe_hash || "").trim();
}

function normalizeSupportTarget(target) {
  const value = String(target || "").trim();
  if (!value) return "";
  if (value.startsWith("provider:")) return value.slice("provider:".length);
  if (value.startsWith("model:")) return value.slice("model:".length);
  return value;
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
            {message.meta.autopilot?.last_notification?.hash ? (
              <span className="badge">
                {`autopilot ${message.meta.autopilot.last_notification.outcome || "unknown"} · ${message.meta.autopilot.last_notification.delivered_to || "none"}`}
              </span>
            ) : null}
            {Number(message.meta.autopilot?.since_last_user_message || 0) > 0 ? (
              <span className="badge">{`new ops ${Number(message.meta.autopilot.since_last_user_message)}`}</span>
            ) : null}
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
  const [modelScoutStatus, setModelScoutStatus] = useState(null);
  const [modelScoutSuggestions, setModelScoutSuggestions] = useState([]);
  const [modelScoutMessage, setModelScoutMessage] = useState("");
  const [modelScoutRunning, setModelScoutRunning] = useState(false);
  const [llmHealth, setLlmHealth] = useState(null);
  const [llmHealthMessage, setLlmHealthMessage] = useState("");
  const [llmHealthRunning, setLlmHealthRunning] = useState(false);
  const [autoconfigPlan, setAutoconfigPlan] = useState(null);
  const [autoconfigStatus, setAutoconfigStatus] = useState("");
  const [autoconfigBusy, setAutoconfigBusy] = useState(false);
  const [hygienePlan, setHygienePlan] = useState(null);
  const [hygieneStatus, setHygieneStatus] = useState("");
  const [hygieneBusy, setHygieneBusy] = useState(false);
  const [cleanupPlan, setCleanupPlan] = useState(null);
  const [cleanupStatus, setCleanupStatus] = useState("");
  const [cleanupBusy, setCleanupBusy] = useState(false);
  const [capabilitiesReconcilePlan, setCapabilitiesReconcilePlan] = useState(null);
  const [capabilitiesReconcileStatus, setCapabilitiesReconcileStatus] = useState("");
  const [capabilitiesReconcileBusy, setCapabilitiesReconcileBusy] = useState(false);
  const [llmCatalogRows, setLlmCatalogRows] = useState([]);
  const [llmCatalogStatus, setLlmCatalogStatus] = useState(null);
  const [autopilotNotifications, setAutopilotNotifications] = useState([]);
  const [autopilotNotificationsStatus, setAutopilotNotificationsStatus] = useState(null);
  const [autopilotNotificationsPolicy, setAutopilotNotificationsPolicy] = useState(null);
  const [autopilotLastReadHash, setAutopilotLastReadHash] = useState("");
  const [autopilotLastChange, setAutopilotLastChange] = useState(null);
  const [autopilotLastChangeStatus, setAutopilotLastChangeStatus] = useState("");
  const [autopilotLastChangeBusy, setAutopilotLastChangeBusy] = useState(false);
  const [autopilotToast, setAutopilotToast] = useState("");
  const [autopilotNotifyStatus, setAutopilotNotifyStatus] = useState("");
  const [autopilotNotifyBusy, setAutopilotNotifyBusy] = useState(false);
  const [autopilotLedgerEntries, setAutopilotLedgerEntries] = useState([]);
  const [registrySnapshots, setRegistrySnapshots] = useState([]);
  const [safetyStatus, setSafetyStatus] = useState("");
  const [rollbackBusySnapshotId, setRollbackBusySnapshotId] = useState("");
  const [autopilotUndoBusy, setAutopilotUndoBusy] = useState(false);
  const [autopilotBootstrapBusy, setAutopilotBootstrapBusy] = useState(false);
  const [supportBundlePreview, setSupportBundlePreview] = useState(null);
  const [supportDiagnoseTarget, setSupportDiagnoseTarget] = useState("");
  const [supportDiagnoseIntent, setSupportDiagnoseIntent] = useState("fix_routing");
  const [supportDiagnosis, setSupportDiagnosis] = useState(null);
  const [supportRemediationPlan, setSupportRemediationPlan] = useState(null);
  const [supportStatus, setSupportStatus] = useState("");
  const [supportBusy, setSupportBusy] = useState(false);
  const [permissionsConfig, setPermissionsConfig] = useState(null);
  const [permissionsStatus, setPermissionsStatus] = useState("");
  const [auditEntries, setAuditEntries] = useState([]);
  const [modelOpsPlans, setModelOpsPlans] = useState({});
  const [modelOpsStatus, setModelOpsStatus] = useState("");
  const [modelOpsBusy, setModelOpsBusy] = useState({});

  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [chatBusy, setChatBusy] = useState(false);

  const [logs, setLogs] = useState([]);
  const autopilotLastHashRef = useRef("");

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
      const [
        providersPayload,
        modelsPayload,
        defaultsPayload,
        telegramPayload,
        scoutStatusPayload,
        scoutSuggestionsPayload,
        llmHealthPayload,
        llmCatalogPayload,
        llmCatalogStatusPayload,
        llmNotificationsPayload,
        llmNotificationsStatusPayload,
        llmNotificationsPolicyPayload,
        autopilotLedgerPayload,
        registrySnapshotsPayload,
        permissionsPayload,
        auditPayload
      ] = await Promise.all([
        request("GET", "/providers"),
        request("GET", "/models"),
        request("GET", "/defaults"),
        request("GET", "/telegram/status").catch(() => null),
        request("GET", "/model_scout/status").catch(() => null),
        request("GET", "/model_scout/suggestions").catch(() => null),
        request("GET", "/llm/health").catch(() => null),
        request("GET", "/llm/catalog?limit=50").catch(() => null),
        request("GET", "/llm/catalog/status").catch(() => null),
        request("GET", "/llm/notifications?limit=20").catch(() => null),
        request("GET", "/llm/notifications/status").catch(() => null),
        request("GET", "/llm/notifications/policy").catch(() => null),
        request("GET", "/llm/autopilot/ledger?limit=10").catch(() => null),
        request("GET", "/llm/registry/snapshots?limit=20").catch(() => null),
        request("GET", "/permissions").catch(() => null),
        request("GET", "/audit?limit=20").catch(() => null)
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
      if (scoutStatusPayload && scoutStatusPayload.ok) {
        setModelScoutStatus(scoutStatusPayload.status || null);
      }
      if (scoutSuggestionsPayload && scoutSuggestionsPayload.ok) {
        setModelScoutSuggestions(scoutSuggestionsPayload.suggestions || []);
      }
      if (llmHealthPayload && llmHealthPayload.ok) {
        setLlmHealth(llmHealthPayload.health || null);
      }
      if (llmCatalogPayload && llmCatalogPayload.ok) {
        setLlmCatalogRows(Array.isArray(llmCatalogPayload.models) ? llmCatalogPayload.models : []);
      }
      if (llmCatalogStatusPayload && llmCatalogStatusPayload.ok) {
        setLlmCatalogStatus(llmCatalogStatusPayload.status || null);
      }
      if (llmNotificationsPayload && llmNotificationsPayload.ok) {
        const nextRows = Array.isArray(llmNotificationsPayload.notifications) ? llmNotificationsPayload.notifications : [];
        autopilotLastHashRef.current = newestNotificationHash(nextRows);
        setAutopilotNotifications(nextRows);
      }
      if (llmNotificationsStatusPayload && llmNotificationsStatusPayload.ok) {
        const statusRow = llmNotificationsStatusPayload.status || null;
        setAutopilotNotificationsStatus(statusRow);
        if (statusRow && typeof statusRow.last_read_hash === "string") {
          setAutopilotLastReadHash(statusRow.last_read_hash);
        }
      }
      if (llmNotificationsPolicyPayload && llmNotificationsPolicyPayload.ok) {
        setAutopilotNotificationsPolicy(llmNotificationsPolicyPayload.policy || null);
      }
      if (autopilotLedgerPayload && autopilotLedgerPayload.ok) {
        setAutopilotLedgerEntries(Array.isArray(autopilotLedgerPayload.entries) ? autopilotLedgerPayload.entries : []);
      }
      if (registrySnapshotsPayload && registrySnapshotsPayload.ok) {
        setRegistrySnapshots(Array.isArray(registrySnapshotsPayload.snapshots) ? registrySnapshotsPayload.snapshots : []);
      }
      if (permissionsPayload && permissionsPayload.ok) {
        setPermissionsConfig(permissionsPayload.permissions || null);
      }
      if (auditPayload && auditPayload.ok) {
        setAuditEntries(auditPayload.entries || []);
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

      const supportTargetId = normalizeSupportTarget(supportDiagnoseTarget);
      const supportTargetExists = !!(
        (supportTargetId && modelRows.some((item) => item.id === supportTargetId))
        || (supportTargetId && providerRows.some((item) => item.id === supportTargetId))
      );
      if (!supportTargetId || !supportTargetExists) {
        if (modelRows[0]?.id) {
          setSupportDiagnoseTarget(`model:${modelRows[0].id}`);
        } else if (providerRows[0]?.id) {
          setSupportDiagnoseTarget(`provider:${providerRows[0].id}`);
        } else {
          setSupportDiagnoseTarget("");
        }
      }

      appendLog({ endpoint: "bootstrap", ok: true, detail: "Loaded /providers, /models, /defaults" });
    } catch (error) {
      appendLog({ endpoint: "bootstrap", ok: false, detail: asErrorText(error) });
    }
  };

  useEffect(() => {
    refreshRuntimeState();
  }, []);

  useEffect(() => {
    let stopped = false;
    const poll = async () => {
      try {
        const [notificationsPayload, statusPayload] = await Promise.all([
          request("GET", "/llm/notifications?limit=20").catch(() => null),
          request("GET", "/llm/notifications/status").catch(() => null)
        ]);
        if (stopped) return;

        if (notificationsPayload && notificationsPayload.ok) {
          const incoming = Array.isArray(notificationsPayload.notifications) ? notificationsPayload.notifications : [];
          const incomingHash = newestNotificationHash(incoming);
          const previousHash = autopilotLastHashRef.current;

          if (incomingHash && incomingHash !== previousHash) {
            const newest = incoming[0] || {};
            if (String(newest.outcome || "").trim() === "sent") {
              const title = String(newest.message || "").split("\n")[0] || "LLM Autopilot updated configuration";
              setAutopilotToast(title);
            }
          }
          autopilotLastHashRef.current = incomingHash;

          setAutopilotNotifications((current) => {
            const currentHash = newestNotificationHash(current);
            if (currentHash && incomingHash && currentHash === incomingHash) {
              return current;
            }
            return incoming;
          });
        }

        if (statusPayload && statusPayload.ok) {
          const statusRow = statusPayload.status || null;
          setAutopilotNotificationsStatus(statusRow);
          if (statusRow && typeof statusRow.last_read_hash === "string") {
            setAutopilotLastReadHash(statusRow.last_read_hash);
          }
        }
      } catch (_error) {
        // Ignore background poll failures and keep current UI state.
      }
    };

    const timerId = window.setInterval(poll, 5000);
    return () => {
      stopped = true;
      window.clearInterval(timerId);
    };
  }, []);

  useEffect(() => {
    if (!autopilotToast) return undefined;
    const timerId = window.setTimeout(() => {
      setAutopilotToast("");
    }, 3500);
    return () => window.clearTimeout(timerId);
  }, [autopilotToast]);

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
  const canSendAutopilotTest =
    permissionsConfig?.actions?.["llm.notifications.test"] === true
    || autopilotNotificationsPolicy?.allow_test_effective === true;
  const canRollbackRegistry =
    permissionsConfig?.actions?.["llm.registry.rollback"] === true
    || llmHealth?.autopilot?.rollback_policy?.allow_rollback_effective === true;
  const canBootstrapAutopilot =
    permissionsConfig?.actions?.["llm.autopilot.bootstrap.apply"] === true
    || llmHealth?.autopilot?.bootstrap_policy?.allow_apply_effective === true;
  const supportTargetOptions = useMemo(() => {
    const modelOptions = models
      .filter((row) => row && row.id)
      .map((row) => ({
        value: `model:${row.id}`,
        label: `Model · ${row.id}`
      }));
    const providerOptions = providers
      .filter((row) => row && row.id)
      .map((row) => ({
        value: `provider:${row.id}`,
        label: `Provider · ${row.id}`
      }));
    return [...modelOptions, ...providerOptions];
  }, [models, providers]);
  const notifyStatusSummary = useMemo(() => {
    const outcome = String(llmHealth?.notifications?.last_outcome || "").trim();
    const reason = String(llmHealth?.notifications?.last_reason || "").trim();
    const hash = String(llmHealth?.notifications?.last_hash || "").trim();
    if (!outcome && !reason) {
      return "Autopilot notify status is not available yet.";
    }
    return `Autopilot notify status: ${outcome || "unknown"}${reason ? ` (${reason})` : ""}${hash ? ` · hash ${hash.slice(0, 12)}` : ""}`;
  }, [llmHealth]);
  const notificationStoreSummary = useMemo(() => {
    if (!autopilotNotificationsStatus) return "Notification store status unavailable.";
    const stored = Number(autopilotNotificationsStatus.stored_count || 0);
    const pruned = Number(autopilotNotificationsStatus.pruned_count_last_run || 0);
    const pruneAt = autopilotNotificationsStatus.last_prune_at_iso || "never";
    const unread = Number(autopilotNotificationsStatus.unread_count || 0);
    return `Store: ${stored} item(s) · unread ${unread} · last prune removed ${pruned} · at ${pruneAt}`;
  }, [autopilotNotificationsStatus]);

  const autopilotPolicyBadge = useMemo(() => {
    const reason = String(autopilotNotificationsPolicy?.allow_reason || "");
    if (reason === "loopback_auto") {
      return {
        label: "Dev Mode (Loopback Auto-Allow)",
        className: "health-ok"
      };
    }
    if (reason === "permission_required") {
      return {
        label: "Permission Required",
        className: "health-degraded"
      };
    }
    if (reason === "explicit_true") {
      return {
        label: "Explicitly Enabled",
        className: "policy-explicit-true"
      };
    }
    if (reason === "explicit_false") {
      return {
        label: "Explicitly Disabled",
        className: "health-down"
      };
    }
    return null;
  }, [autopilotNotificationsPolicy]);

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
          fallback_used: !!result.meta?.fallback_used,
          autopilot: result.meta?.autopilot || null
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

  const runModelScout = async () => {
    setModelScoutRunning(true);
    setModelScoutMessage("Running scout...");
    try {
      const result = await request("POST", "/model_scout/run", {});
      const total = Array.isArray(result.suggestions) ? result.suggestions.length : 0;
      const fresh = Array.isArray(result.new_suggestions) ? result.new_suggestions.length : 0;
      setModelScoutMessage(`Scout complete: ${total} candidates (${fresh} new).`);
      appendLog({ endpoint: "/model_scout/run", ok: true, detail: `suggestions=${total} new=${fresh}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setModelScoutMessage(`Scout failed: ${detail}`);
      appendLog({ endpoint: "/model_scout/run", ok: false, detail });
    } finally {
      setModelScoutRunning(false);
    }
  };

  const runLlmHealthCheck = async () => {
    setLlmHealthRunning(true);
    setLlmHealthMessage("Running health checks...");
    try {
      const result = await request("POST", "/llm/health/run", {});
      setLlmHealth(result.health || null);
      const total = Number(result.health?.probed?.length || 0);
      setLlmHealthMessage(`Health check complete: probed ${total} candidate(s).`);
      appendLog({ endpoint: "/llm/health/run", ok: true, detail: `probed=${total}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setLlmHealthMessage(`Health check failed: ${detail}`);
      appendLog({ endpoint: "/llm/health/run", ok: false, detail });
    } finally {
      setLlmHealthRunning(false);
    }
  };

  const runLlmCatalogRefresh = async () => {
    setLlmHealthRunning(true);
    setLlmHealthMessage("Refreshing model catalog...");
    try {
      const result = await request("POST", "/llm/catalog/run", { actor: "webui" });
      const added = Number(result.counts?.added || 0);
      const removed = Number(result.counts?.removed || 0);
      const changed = Number(result.counts?.changed || 0);
      setLlmHealthMessage(`Catalog refresh complete: +${added} / -${removed} / ~${changed}.`);
      appendLog({ endpoint: "/llm/catalog/run", ok: true, detail: `added=${added} removed=${removed} changed=${changed}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setLlmHealthMessage(`Catalog refresh failed: ${detail}`);
      appendLog({ endpoint: "/llm/catalog/run", ok: false, detail });
    } finally {
      setLlmHealthRunning(false);
    }
  };

  const planLlmAutoconfig = async () => {
    setAutoconfigBusy(true);
    setAutoconfigStatus("Planning autoconfig...");
    try {
      const result = await request("POST", "/llm/autoconfig/plan", {
        actor: "webui",
        disable_auth_failed_providers: true
      });
      setAutoconfigPlan(result.plan || null);
      const changes = Number(result.plan?.impact?.changes_count || 0);
      setAutoconfigStatus(`Autoconfig plan ready: ${changes} change(s).`);
      appendLog({ endpoint: "/llm/autoconfig/plan", ok: true, detail: `changes=${changes}` });
    } catch (error) {
      const detail = asErrorText(error);
      setAutoconfigStatus(`Plan failed: ${detail}`);
      appendLog({ endpoint: "/llm/autoconfig/plan", ok: false, detail });
    } finally {
      setAutoconfigBusy(false);
    }
  };

  const applyLlmAutoconfig = async () => {
    setAutoconfigBusy(true);
    setAutoconfigStatus("Applying autoconfig...");
    try {
      const result = await request("POST", "/llm/autoconfig/apply", {
        actor: "webui",
        confirm: true,
        disable_auth_failed_providers: true
      });
      const changes = Number(result.plan?.impact?.changes_count || 0);
      setAutoconfigStatus(`Autoconfig applied: ${changes} change(s).`);
      setAutoconfigPlan(result.plan || autoconfigPlan);
      appendLog({ endpoint: "/llm/autoconfig/apply", ok: true, detail: `changes=${changes}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setAutoconfigStatus(`Apply failed: ${detail}`);
      appendLog({ endpoint: "/llm/autoconfig/apply", ok: false, detail });
    } finally {
      setAutoconfigBusy(false);
    }
  };

  const planLlmHygiene = async () => {
    setHygieneBusy(true);
    setHygieneStatus("Planning hygiene...");
    try {
      const result = await request("POST", "/llm/hygiene/plan", {
        actor: "webui"
      });
      setHygienePlan(result.plan || null);
      const changes = Number(result.plan?.impact?.changes_count || 0);
      setHygieneStatus(`Hygiene plan ready: ${changes} change(s).`);
      appendLog({ endpoint: "/llm/hygiene/plan", ok: true, detail: `changes=${changes}` });
    } catch (error) {
      const detail = asErrorText(error);
      setHygieneStatus(`Plan failed: ${detail}`);
      appendLog({ endpoint: "/llm/hygiene/plan", ok: false, detail });
    } finally {
      setHygieneBusy(false);
    }
  };

  const applyLlmHygiene = async () => {
    setHygieneBusy(true);
    setHygieneStatus("Applying hygiene...");
    try {
      const result = await request("POST", "/llm/hygiene/apply", {
        actor: "webui",
        confirm: true
      });
      const changes = Number(result.plan?.impact?.changes_count || 0);
      setHygieneStatus(`Hygiene applied: ${changes} change(s).`);
      setHygienePlan(result.plan || hygienePlan);
      appendLog({ endpoint: "/llm/hygiene/apply", ok: true, detail: `changes=${changes}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setHygieneStatus(`Apply failed: ${detail}`);
      appendLog({ endpoint: "/llm/hygiene/apply", ok: false, detail });
    } finally {
      setHygieneBusy(false);
    }
  };

  const planLlmCleanup = async () => {
    setCleanupBusy(true);
    setCleanupStatus("Planning cleanup...");
    try {
      const result = await request("POST", "/llm/cleanup/plan", {
        actor: "webui"
      });
      setCleanupPlan(result.plan || null);
      const changes = Number(result.plan?.impact?.changes_count || 0);
      const candidates = Number(result.plan?.impact?.prune_candidates_count || 0);
      setCleanupStatus(`Cleanup plan ready: ${changes} change(s), ${candidates} prune candidate(s).`);
      appendLog({ endpoint: "/llm/cleanup/plan", ok: true, detail: `changes=${changes} prune=${candidates}` });
    } catch (error) {
      const detail = asErrorText(error);
      setCleanupStatus(`Plan failed: ${detail}`);
      appendLog({ endpoint: "/llm/cleanup/plan", ok: false, detail });
    } finally {
      setCleanupBusy(false);
    }
  };

  const applyLlmCleanup = async () => {
    setCleanupBusy(true);
    setCleanupStatus("Applying cleanup...");
    try {
      const result = await request("POST", "/llm/cleanup/apply", {
        actor: "webui",
        confirm: true
      });
      const changes = Number(result.plan?.impact?.changes_count || 0);
      setCleanupStatus(`Cleanup applied: ${changes} change(s).`);
      setCleanupPlan(result.plan || cleanupPlan);
      appendLog({ endpoint: "/llm/cleanup/apply", ok: true, detail: `changes=${changes}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setCleanupStatus(`Apply failed: ${detail}`);
      appendLog({ endpoint: "/llm/cleanup/apply", ok: false, detail });
    } finally {
      setCleanupBusy(false);
    }
  };

  const planLlmCapabilitiesReconcile = async () => {
    setCapabilitiesReconcileBusy(true);
    setCapabilitiesReconcileStatus("Planning capability reconcile...");
    try {
      const result = await request("POST", "/llm/capabilities/reconcile/plan", {
        actor: "webui"
      });
      setCapabilitiesReconcilePlan(result.plan || null);
      const changes = Number(result.plan?.impact?.changes_count || 0);
      const mismatches = Number(result.plan?.impact?.models_with_mismatch || 0);
      setCapabilitiesReconcileStatus(`Capability reconcile plan ready: ${changes} change(s), ${mismatches} mismatched model(s).`);
      appendLog({ endpoint: "/llm/capabilities/reconcile/plan", ok: true, detail: `changes=${changes}` });
    } catch (error) {
      const detail = asErrorText(error);
      setCapabilitiesReconcileStatus(`Plan failed: ${detail}`);
      appendLog({ endpoint: "/llm/capabilities/reconcile/plan", ok: false, detail });
    } finally {
      setCapabilitiesReconcileBusy(false);
    }
  };

  const applyLlmCapabilitiesReconcile = async () => {
    setCapabilitiesReconcileBusy(true);
    setCapabilitiesReconcileStatus("Applying capability reconcile...");
    try {
      const result = await request("POST", "/llm/capabilities/reconcile/apply", {
        actor: "webui",
        confirm: true
      });
      const changes = Number(result.plan?.impact?.changes_count || 0);
      setCapabilitiesReconcileStatus(`Capability reconcile applied: ${changes} change(s).`);
      setCapabilitiesReconcilePlan(result.plan || capabilitiesReconcilePlan);
      appendLog({ endpoint: "/llm/capabilities/reconcile/apply", ok: true, detail: `changes=${changes}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setCapabilitiesReconcileStatus(`Apply failed: ${detail}`);
      appendLog({ endpoint: "/llm/capabilities/reconcile/apply", ok: false, detail });
    } finally {
      setCapabilitiesReconcileBusy(false);
    }
  };

  const rollbackRegistryToSnapshot = async (snapshotId) => {
    const target = String(snapshotId || "").trim();
    if (!target) return;
    setRollbackBusySnapshotId(target);
    setSafetyStatus(`Rolling back to ${target}...`);
    try {
      const result = await request("POST", "/llm/registry/rollback", {
        actor: "webui",
        snapshot_id: target,
        confirm: true
      });
      const hash = String(result.resulting_registry_hash || "").trim();
      setSafetyStatus(`Rollback complete: ${target}${hash ? ` · ${hash.slice(0, 12)}` : ""}`);
      appendLog({ endpoint: "/llm/registry/rollback", ok: true, detail: `snapshot=${target}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setSafetyStatus(`Rollback failed: ${detail}`);
      appendLog({ endpoint: "/llm/registry/rollback", ok: false, detail });
    } finally {
      setRollbackBusySnapshotId("");
    }
  };

  const sendAutopilotTestNotification = async () => {
    setAutopilotNotifyBusy(true);
    setAutopilotNotifyStatus("Sending test notification...");
    try {
      const result = await request("POST", "/llm/notifications/test", {
        actor: "webui",
        confirm: true
      });
      const outcome = result.result?.outcome || "unknown";
      const delivered = result.result?.delivered_to || "none";
      setAutopilotNotifyStatus(`Test notification outcome: ${outcome} (delivered_to=${delivered}).`);
      appendLog({ endpoint: "/llm/notifications/test", ok: true, detail: `outcome=${outcome}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setAutopilotNotifyStatus(`Test notification failed: ${detail}`);
      appendLog({ endpoint: "/llm/notifications/test", ok: false, detail });
    } finally {
      setAutopilotNotifyBusy(false);
    }
  };

  const markAutopilotRead = async (hash, { silent = false } = {}) => {
    const normalizedHash = String(hash || "").trim();
    if (!normalizedHash) return;
    try {
      const result = await request("POST", "/llm/notifications/mark_read", { hash: normalizedHash });
      if (result.ok && result.status) {
        setAutopilotNotificationsStatus(result.status);
        setAutopilotLastReadHash(String(result.status.last_read_hash || ""));
      }
      if (!silent) {
        setAutopilotLastChangeStatus(`Marked read: ${normalizedHash.slice(0, 12)}`);
      }
    } catch (error) {
      if (!silent) {
        setAutopilotLastChangeStatus(`Mark read failed: ${asErrorText(error)}`);
      }
    }
  };

  const explainLastAutopilotChange = async () => {
    setAutopilotLastChangeBusy(true);
    setAutopilotLastChangeStatus("Loading last autopilot change...");
    try {
      const result = await request("GET", "/llm/autopilot/explain_last");
      if (!result.ok || !result.found || !result.last_apply) {
        setAutopilotLastChange(null);
        setAutopilotLastChangeStatus("No autopilot apply action found.");
        return;
      }
      setAutopilotLastChange(result.last_apply);
      setAutopilotLastChangeStatus("Loaded latest autopilot apply rationale.");
    } catch (error) {
      setAutopilotLastChange(null);
      setAutopilotLastChangeStatus(`Explain failed: ${asErrorText(error)}`);
    } finally {
      setAutopilotLastChangeBusy(false);
    }
  };

  const undoLastAutopilotChange = async () => {
    setAutopilotUndoBusy(true);
    setSafetyStatus("Rolling back most recent autopilot apply...");
    try {
      const result = await request("POST", "/llm/autopilot/undo", {
        actor: "webui",
        confirm: true
      });
      const snapshot = String(result.rolled_back_to_snapshot_id || "").trim();
      const hash = String(result.resulting_registry_hash || "").trim();
      setSafetyStatus(`Undo complete${snapshot ? `: ${snapshot}` : ""}${hash ? ` · ${hash.slice(0, 12)}` : ""}`);
      appendLog({ endpoint: "/llm/autopilot/undo", ok: true, detail: snapshot || "ok" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setSafetyStatus(`Undo failed: ${detail}`);
      appendLog({ endpoint: "/llm/autopilot/undo", ok: false, detail });
    } finally {
      setAutopilotUndoBusy(false);
    }
  };

  const bootstrapAutopilotDefaults = async () => {
    setAutopilotBootstrapBusy(true);
    setSafetyStatus("Running bootstrap defaults...");
    try {
      const result = await request("POST", "/llm/autopilot/bootstrap", {
        actor: "webui",
        confirm: true
      });
      const changes = Number(result.plan?.impact?.changes_count || 0);
      if (result.applied) {
        setSafetyStatus(`Bootstrap applied: ${changes} change(s).`);
      } else {
        const reason = String((result.plan?.reasons || [])[0] || "already_configured");
        setSafetyStatus(`Bootstrap no-op: ${reason}.`);
      }
      appendLog({ endpoint: "/llm/autopilot/bootstrap", ok: true, detail: `changes=${changes}` });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setSafetyStatus(`Bootstrap failed: ${detail}`);
      appendLog({ endpoint: "/llm/autopilot/bootstrap", ok: false, detail });
    } finally {
      setAutopilotBootstrapBusy(false);
    }
  };

  const exportSupportBundle = async () => {
    setSupportBusy(true);
    setSupportStatus("Exporting support bundle...");
    try {
      const result = await request("GET", "/llm/support/bundle");
      const bundle = result.bundle || {};
      setSupportBundlePreview(bundle);
      const rendered = JSON.stringify(bundle, null, 2);
      let copied = false;
      try {
        if (navigator?.clipboard?.writeText) {
          await navigator.clipboard.writeText(rendered);
          copied = true;
        }
      } catch (_error) {
        copied = false;
      }
      if (!copied) {
        const blob = new Blob([rendered], { type: "application/json;charset=utf-8" });
        const url = window.URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "personal-agent-support-bundle.json";
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        window.URL.revokeObjectURL(url);
      }
      setSupportStatus(copied ? "Support bundle copied to clipboard." : "Support bundle downloaded.");
      appendLog({ endpoint: "/llm/support/bundle", ok: true, detail: copied ? "copied" : "downloaded" });
    } catch (error) {
      const detail = asErrorText(error);
      setSupportStatus(`Support bundle export failed: ${detail}`);
      appendLog({ endpoint: "/llm/support/bundle", ok: false, detail });
    } finally {
      setSupportBusy(false);
    }
  };

  const runSupportDiagnosis = async () => {
    const targetId = normalizeSupportTarget(supportDiagnoseTarget);
    if (!targetId) {
      setSupportStatus("Select a provider or model to diagnose.");
      return;
    }
    setSupportBusy(true);
    setSupportStatus(`Diagnosing ${targetId}...`);
    try {
      const result = await request("GET", `/llm/support/diagnose?id=${encodeURIComponent(targetId)}`);
      setSupportDiagnosis(result.diagnosis || null);
      setSupportStatus(`Diagnosis ready for ${result.kind || "target"} ${result.id || targetId}.`);
      appendLog({ endpoint: "/llm/support/diagnose", ok: true, detail: targetId });
    } catch (error) {
      const detail = asErrorText(error);
      setSupportDiagnosis(null);
      setSupportStatus(`Diagnosis failed: ${detail}`);
      appendLog({ endpoint: "/llm/support/diagnose", ok: false, detail });
    } finally {
      setSupportBusy(false);
    }
  };

  const planSupportRemediation = async () => {
    const targetId = normalizeSupportTarget(supportDiagnoseTarget);
    setSupportBusy(true);
    setSupportStatus("Building remediation plan...");
    try {
      const payload = {
        intent: supportDiagnoseIntent
      };
      if (targetId) {
        payload.target = targetId;
      }
      const result = await request("POST", "/llm/support/remediate/plan", payload);
      setSupportRemediationPlan(result.plan || null);
      const stepCount = Number((result.plan?.steps || []).length || 0);
      setSupportStatus(`Remediation plan ready (${stepCount} step(s), plan-only).`);
      appendLog({ endpoint: "/llm/support/remediate/plan", ok: true, detail: `steps=${stepCount}` });
    } catch (error) {
      const detail = asErrorText(error);
      setSupportRemediationPlan(null);
      setSupportStatus(`Remediation plan failed: ${detail}`);
      appendLog({ endpoint: "/llm/support/remediate/plan", ok: false, detail });
    } finally {
      setSupportBusy(false);
    }
  };

  const dismissScoutSuggestion = async (suggestionId) => {
    try {
      await request("POST", `/model_scout/suggestions/${encodeURIComponent(suggestionId)}/dismiss`, {});
      setModelScoutMessage(`Dismissed ${suggestionId}.`);
      appendLog({ endpoint: "/model_scout/suggestions/*/dismiss", ok: true, detail: suggestionId });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setModelScoutMessage(`Dismiss failed: ${detail}`);
      appendLog({ endpoint: "/model_scout/suggestions/*/dismiss", ok: false, detail });
    }
  };

  const markScoutSuggestionInstalled = async (suggestionId) => {
    try {
      await request("POST", `/model_scout/suggestions/${encodeURIComponent(suggestionId)}/mark_installed`, {});
      setModelScoutMessage(`Marked installed: ${suggestionId}.`);
      appendLog({ endpoint: "/model_scout/suggestions/*/mark_installed", ok: true, detail: suggestionId });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setModelScoutMessage(`Mark installed failed: ${detail}`);
      appendLog({ endpoint: "/model_scout/suggestions/*/mark_installed", ok: false, detail });
    }
  };

  const updatePermissionAction = (actionName, value) => {
    setPermissionsConfig((prev) => {
      const base = prev || {};
      const actions = base.actions || {};
      return {
        ...base,
        actions: {
          ...actions,
          [actionName]: !!value
        }
      };
    });
  };

  const updatePermissionConstraint = (field, value) => {
    setPermissionsConfig((prev) => {
      const base = prev || {};
      const constraints = base.constraints || {};
      return {
        ...base,
        constraints: {
          ...constraints,
          [field]: value
        }
      };
    });
  };

  const savePermissions = async () => {
    if (!permissionsConfig) {
      setPermissionsStatus("Permissions are not loaded yet.");
      return;
    }
    const maxGb = Number((permissionsConfig.constraints || {}).max_download_gb || 0);
    const payload = {
      mode: permissionsConfig.mode || "manual_confirm",
      actions: permissionsConfig.actions || {},
      constraints: {
        ...(permissionsConfig.constraints || {}),
        max_download_bytes: Math.max(0, Math.floor(maxGb * 1024 * 1024 * 1024))
      }
    };
    try {
      const result = await request("PUT", "/permissions", payload);
      setPermissionsConfig(result.permissions || permissionsConfig);
      setPermissionsStatus("Permissions saved.");
      appendLog({ endpoint: "/permissions", ok: true, detail: "Updated ModelOps permissions" });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setPermissionsStatus(`Save failed: ${detail}`);
      appendLog({ endpoint: "/permissions", ok: false, detail });
    }
  };

  const toModelOpsRequest = (suggestion) => {
    if (!suggestion) return null;
    if (suggestion.kind === "local") {
      const repoId = suggestion.repo_id || "";
      if (!repoId) return null;
      return {
        action: "modelops.pull_ollama_model",
        params: {
          model: `hf.co/${repoId}`,
          estimated_download_gb: 4
        }
      };
    }

    const modelId = suggestion.model_id || "";
    if (!modelId) return null;
    const provider = suggestion.provider_id || modelId.split(":")[0] || "";
    return {
      action: "modelops.set_default_model",
      params: {
        default_provider: provider,
        default_model: modelId
      }
    };
  };

  const planModelOpForSuggestion = async (suggestion) => {
    const modelOpsRequest = toModelOpsRequest(suggestion);
    if (!modelOpsRequest) {
      setModelOpsStatus("Unable to build ModelOps request for this suggestion.");
      return;
    }
    setModelOpsBusy((prev) => ({ ...prev, [suggestion.id]: true }));
    try {
      const result = await request("POST", "/modelops/plan", {
        action: modelOpsRequest.action,
        params: modelOpsRequest.params,
        dry_run: true
      });
      setModelOpsPlans((prev) => ({
        ...prev,
        [suggestion.id]: result
      }));
      setModelOpsStatus(
        result.decision?.allow
          ? `Plan ready for ${suggestion.id}. Confirm to execute.`
          : `Plan denied for ${suggestion.id}: ${result.decision?.reason || "policy_denied"}`
      );
      appendLog({
        endpoint: "/modelops/plan",
        ok: true,
        detail: `${modelOpsRequest.action} allow=${result.decision?.allow === true}`
      });
    } catch (error) {
      const detail = asErrorText(error);
      setModelOpsStatus(`Plan failed: ${detail}`);
      appendLog({ endpoint: "/modelops/plan", ok: false, detail });
    } finally {
      setModelOpsBusy((prev) => ({ ...prev, [suggestion.id]: false }));
    }
  };

  const executeModelOpForSuggestion = async (suggestion) => {
    const planned = modelOpsPlans[suggestion.id];
    const modelOpsRequest = toModelOpsRequest(suggestion);
    if (!planned || !modelOpsRequest) {
      setModelOpsStatus("Run plan first.");
      return;
    }
    setModelOpsBusy((prev) => ({ ...prev, [suggestion.id]: true }));
    try {
      const result = await request("POST", "/modelops/execute", {
        action: modelOpsRequest.action,
        params: modelOpsRequest.params,
        dry_run: false,
        confirm: true
      });
      const success = result.result?.ok === true;
      setModelOpsStatus(success ? `Executed ${suggestion.id}.` : `Execution failed for ${suggestion.id}.`);
      appendLog({
        endpoint: "/modelops/execute",
        ok: success,
        detail: `${modelOpsRequest.action} ${success ? "success" : "failed"}`
      });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setModelOpsStatus(`Execution failed: ${detail}`);
      appendLog({ endpoint: "/modelops/execute", ok: false, detail });
    } finally {
      setModelOpsBusy((prev) => ({ ...prev, [suggestion.id]: false }));
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
          ["permissions", "Permissions"],
          ["model_scout", "Model Scout"],
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
                <button disabled={llmHealthRunning} onClick={runLlmHealthCheck}>
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
                  <div key={`${entry.ts || "llm-action"}-${index}`} className="model-row">
                    <div className="model-head">
                      <span>{entry.action}</span>
                      <span className={`badge ${entry.outcome === "success" ? "health-ok" : "health-degraded"}`}>
                        {entry.outcome || "unknown"}
                      </span>
                    </div>
                    <div className="meta-line">
                      {entry.ts || "n/a"} · {entry.reason || "n/a"} · {entry.duration_ms || 0}ms
                    </div>
                  </div>
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
                  <div key={`${entry.ts || "autopilot-note"}-${index}`} className="model-row">
                    <div className="model-head">
                      <span>{String(entry.message || "").split("\n")[0] || "LLM Autopilot updated configuration"}</span>
                      <span className={`badge ${entry.outcome === "sent" ? "health-ok" : "health-degraded"}`}>
                        {entry.outcome || "unknown"}
                      </span>
                    </div>
                    <div className="meta-line">
                      {entry.ts_iso || "n/a"} · {entry.reason || "n/a"} · delivered_to {entry.delivered_to || "none"} ·{" "}
                      {entry.deferred ? "deferred" : "immediate"}
                    </div>
                    <div className="meta-line">
                      {String(entry.message || "")
                        .split("\n")
                        .slice(1, 3)
                        .join(" ")
                        .trim() || "(no body preview)"}
                    </div>
                    <div className="meta-line">hash: {entry.dedupe_hash || "n/a"}</div>
                  </div>
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
                <div className="model-row">
                  <div className="model-head">
                    <span>{autopilotLastChange.action || "llm.autopilot.apply"}</span>
                    <span className="badge">
                      {autopilotLastChange.snapshot_id_before || "no-snapshot"}
                    </span>
                  </div>
                  <div className="meta-line">
                    {formatEpoch(autopilotLastChange.ts)} · {autopilotLastChange.reason || "n/a"} · hash{" "}
                    {String(autopilotLastChange.registry_hash_after || "").slice(0, 12) || "n/a"}
                  </div>
                  {(autopilotLastChange.rationale_lines || []).map((line, index) => (
                    <div key={`${line}-${index}`} className="meta-line">
                      {line}
                    </div>
                  ))}
                  <div className="meta-line">changed: {(autopilotLastChange.changed_ids || []).join(", ") || "none"}</div>
                </div>
              ) : null}

              <div className="row-actions">
                <button disabled={autoconfigBusy} onClick={planLlmAutoconfig}>
                  {autoconfigBusy ? "Working..." : "Plan Autoconfig"}
                </button>
                <button disabled={autoconfigBusy || !autoconfigPlan} onClick={applyLlmAutoconfig}>
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
                <button disabled={hygieneBusy || !hygienePlan} onClick={applyLlmHygiene}>
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
                  <div key={row.id} className="model-row">
                    <div className="model-head">
                      <span>{row.id}</span>
                      <span className="badge">{(row.capabilities || []).join(",") || "chat"}</span>
                    </div>
                    <div className="meta-line">
                      ctx {row.max_context_tokens || "?"} · in {row.input_cost_per_million_tokens ?? "n/a"} · out {row.output_cost_per_million_tokens ?? "n/a"} · {row.source}
                    </div>
                  </div>
                ))}
              </div>

              <h3>Cleanup</h3>
              <div className="row-actions">
                <button disabled={cleanupBusy} onClick={planLlmCleanup}>
                  {cleanupBusy ? "Working..." : "Plan Cleanup"}
                </button>
                <button disabled={cleanupBusy || !cleanupPlan} onClick={applyLlmCleanup}>
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
                <button disabled={!canRollbackRegistry || autopilotUndoBusy} onClick={undoLastAutopilotChange}>
                  {autopilotUndoBusy ? "Undoing..." : "Undo Last Autopilot Change"}
                </button>
                <button disabled={!canBootstrapAutopilot || autopilotBootstrapBusy} onClick={bootstrapAutopilotDefaults}>
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
                    <div key={entry.id} className="model-row">
                      <div className="model-head">
                        <span>{entry.action || "llm.apply"}</span>
                        <span className={`badge ${entry.outcome === "success" ? "health-ok" : "health-degraded"}`}>
                          {entry.outcome || "unknown"}
                        </span>
                      </div>
                      <div className="meta-line">
                        {entry.ts_iso || "n/a"} · {entry.reason || "n/a"} · changed {(entry.changed_ids || []).join(", ") || "none"}
                      </div>
                      <div className="meta-line">
                        snapshot {snapshotId || "n/a"} · hash {entry.resulting_registry_hash || "n/a"}
                      </div>
                      <div className="row-actions">
                        <button
                          disabled={!snapshotId || !canRollbackRegistry || rollbackBusySnapshotId === snapshotId}
                          onClick={() => rollbackRegistryToSnapshot(snapshotId)}
                        >
                          {rollbackBusySnapshotId === snapshotId ? "Rolling Back..." : "Rollback"}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="model-list">
                {registrySnapshots.length === 0 ? <p className="empty">No snapshots available.</p> : null}
                {registrySnapshots.slice(0, 10).map((row) => (
                  <div key={row.snapshot_id} className="model-row">
                    <div className="model-head">
                      <span>{row.snapshot_id}</span>
                      <span className="badge">{row.size_bytes || 0} bytes</span>
                    </div>
                    <div className="meta-line">hash {row.registry_hash || "n/a"}</div>
                    <div className="row-actions">
                      <button
                        disabled={!canRollbackRegistry || rollbackBusySnapshotId === row.snapshot_id}
                        onClick={() => rollbackRegistryToSnapshot(row.snapshot_id)}
                      >
                        {rollbackBusySnapshotId === row.snapshot_id ? "Rolling Back..." : "Rollback"}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <h2>Support</h2>
              <p className="help-text">
                Export a deterministic local support bundle, diagnose a provider/model, and generate a plan-only remediation sequence.
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
                  {supportBusy ? "Working..." : "Plan Remediation"}
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
                  <div className="model-row">
                    <div className="model-head">
                      <span>Status</span>
                      <span className={`badge ${String(supportDiagnosis.status || "") === "ok" ? "health-ok" : "health-degraded"}`}>
                        {supportDiagnosis.status || "unknown"}
                      </span>
                    </div>
                    <div className="meta-line">
                      error {supportDiagnosis.last_error_kind || "none"} · code {supportDiagnosis.status_code || "n/a"} · streak{" "}
                      {Number(supportDiagnosis.failure_streak || 0)}
                    </div>
                    <div className="meta-line">
                      root causes: {(supportDiagnosis.root_causes || []).join(", ") || "none"}
                    </div>
                    {(supportDiagnosis.recommended_actions || []).map((line, index) => (
                      <div key={`${line}-${index}`} className="meta-line">
                        {line}
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
              <div className="model-list">
                {!supportRemediationPlan ? <p className="empty">Run remediation plan to view next steps.</p> : null}
                {supportRemediationPlan ? (
                  <div className="model-row">
                    <div className="model-head">
                      <span>Remediation Plan ({supportRemediationPlan.intent || "fix_routing"})</span>
                      <span className="badge">{supportRemediationPlan.plan_only ? "plan-only" : "apply"}</span>
                    </div>
                    <div className="meta-line">reasons: {(supportRemediationPlan.reasons || []).join(" | ") || "n/a"}</div>
                    {(supportRemediationPlan.steps || []).map((step) => (
                      <div key={step.id || step.action} className="meta-line">
                        {(step.id || "step").replaceAll("_", " ")}: {step.action} ({step.reason})
                      </div>
                    ))}
                  </div>
                ) : null}
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

        {activeTab === "permissions" ? (
          <section className="grid two">
            <div className="card">
              <h2>ModelOps Permissions</h2>
              <label>
                Mode
                <select
                  value={permissionsConfig?.mode || "manual_confirm"}
                  onChange={(event) => setPermissionsConfig((prev) => ({ ...(prev || {}), mode: event.target.value }))}
                >
                  <option value="manual_confirm">manual_confirm</option>
                  <option value="auto">auto</option>
                </select>
              </label>

              <div className="grid">
                {MODELOPS_ACTIONS.map((actionName) => (
                  <label key={actionName} className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={permissionsConfig?.actions?.[actionName] === true}
                      onChange={(event) => updatePermissionAction(actionName, event.target.checked)}
                    />
                    {actionName}
                  </label>
                ))}
              </div>

              <label>
                Max download GB
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
                allow_install_ollama
              </label>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={permissionsConfig?.constraints?.allow_remote_models !== false}
                  onChange={(event) => updatePermissionConstraint("allow_remote_models", event.target.checked)}
                />
                allow_remote_models
              </label>

              <label>
                Allowed providers (comma separated)
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
                Allowed model patterns (comma separated)
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
                <button onClick={savePermissions}>Save Permissions</button>
              </div>
              <p className="status-line">{permissionsStatus || "Default is deny for all ModelOps actions."}</p>
            </div>

            <div className="card">
              <h2>Recent Audit Events</h2>
              <div className="model-list">
                {auditEntries.length === 0 ? <p className="empty">No audit events yet.</p> : null}
                {auditEntries.map((entry, index) => (
                  <div className="model-row" key={`${entry.ts || "entry"}-${index}`}>
                    <div className="model-head">
                      <span>{entry.action}</span>
                      <span className={`badge ${entry.decision === "allow" ? "health-ok" : "health-down"}`}>{entry.decision}</span>
                    </div>
                    <div className="meta-line">
                      {entry.ts} · outcome {entry.outcome} · reason {entry.reason}
                    </div>
                    <div className="meta-line">dry_run {entry.dry_run ? "yes" : "no"} · duration {entry.duration_ms}ms</div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        ) : null}

        {activeTab === "model_scout" ? (
          <section className="grid">
            <div className="card">
              <h2>Model Scout</h2>
              <p className="help-text">
                Status: {modelScoutStatus?.last_run?.ok === false ? "degraded" : "ready"} · Backend: {modelScoutStatus?.backend || "unknown"}
              </p>
              <p className="help-text">
                Suggestions: {modelScoutStatus?.total || 0} total · {(modelScoutStatus?.counts || {}).new || 0} new
              </p>
              <p className="help-text">
                Sources: HF {modelScoutStatus?.sources?.huggingface?.available ? "ok" : "down"} · Ollama{" "}
                {modelScoutStatus?.sources?.ollama?.available ? "ok" : "down"} · OpenRouter{" "}
                {modelScoutStatus?.sources?.openrouter?.available ? "ok" : "down"}
              </p>
              <div className="row-actions">
                <button disabled={modelScoutRunning} onClick={runModelScout}>
                  {modelScoutRunning ? "Running..." : "Run Scout"}
                </button>
              </div>
              <p className="status-line">{modelScoutMessage || "Scout recommends only. It never changes defaults automatically."}</p>
            </div>

            <div className="card">
              <h2>Suggestions</h2>
              <div className="model-list">
                {modelScoutSuggestions.length === 0 ? <p className="empty">No suggestions yet.</p> : null}
                {modelScoutSuggestions.map((item) => (
                  <div key={item.id} className="model-row">
                    <div className="model-head">
                      <span>{item.kind === "local" ? item.repo_id : item.model_id}</span>
                      <span className="badge">{item.kind}</span>
                    </div>
                    <div className="meta-line">score {Number(item.score || 0).toFixed(1)} · status {item.status}</div>
                    <div className="meta-line">{item.rationale}</div>
                    {item.install_cmd ? <div className="meta-line">Try: {item.install_cmd}</div> : null}
                    <div className="row-actions">
                      <button disabled={modelOpsBusy[item.id] === true} onClick={() => planModelOpForSuggestion(item)}>
                        {modelOpsBusy[item.id] === true ? "Planning..." : "Try This Model"}
                      </button>
                      <button
                        disabled={!modelOpsPlans[item.id]?.decision?.allow || modelOpsBusy[item.id] === true}
                        onClick={() => executeModelOpForSuggestion(item)}
                      >
                        Confirm Execute
                      </button>
                      <button onClick={() => dismissScoutSuggestion(item.id)}>Dismiss</button>
                      <button onClick={() => markScoutSuggestionInstalled(item.id)}>Mark Installed</button>
                    </div>
                    {modelOpsPlans[item.id] ? (
                      <div className="meta-line">
                        plan: {modelOpsPlans[item.id].decision?.allow ? "allow" : "deny"} · reason{" "}
                        {modelOpsPlans[item.id].decision?.reason || "n/a"} · steps{" "}
                        {(modelOpsPlans[item.id].plan?.steps || []).length}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
              <p className="status-line">{modelOpsStatus || "Run plan first, then confirm execution."}</p>
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
