import { useEffect, useMemo, useRef, useState } from "react";
import AdminPanel from "./components/AdminPanel";
import ChatExperience from "./components/ChatExperience";
import DebugTab from "./components/DebugTab";
import ModelScoutTab from "./components/ModelScoutTab";
import PacksTab from "./components/PacksTab";
import OperationsTab from "./components/OperationsTab";
import PermissionsTab from "./components/PermissionsTab";
import ProvidersTab from "./components/ProvidersTab";
import SetupTab from "./components/SetupTab";
import StateTab from "./components/StateTab";
import TelegramTab from "./components/TelegramTab";
import {
  buildAssistantMessage,
  buildComposerPlaceholder,
  buildStarterPrompts,
  buildStatusSummary
} from "./lib/chatUiHelpers";
import { matchesProviderModelFilter, summarizeProviderCapabilities } from "./lib/providerModelHelpers";
import {
  asErrorText,
  formatNow,
  healthStatus,
  newestNotificationHash,
  normalizeSupportTarget,
  parseJsonObject
} from "./lib/uiHelpers";

const ROUTING_MODES = ["auto", "prefer_cheap", "prefer_best", "prefer_local_lowest_cost_capable"];
const CHAT_SESSION_STORAGE_KEY = "personal-agent-chat-session-id";
const CHAT_THREAD_STORAGE_KEY = "personal-agent-chat-thread-id";
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
  {
    id: "modelops.install_ollama",
    label: "Install Ollama",
    description: "Allow the app to install the local Ollama runtime when it is missing."
  },
  {
    id: "modelops.pull_ollama_model",
    label: "Download Ollama Models",
    description: "Allow pulling local Ollama models onto this machine."
  },
  {
    id: "modelops.import_gguf_to_ollama",
    label: "Import GGUF Models",
    description: "Allow importing GGUF files into Ollama for local use."
  },
  {
    id: "modelops.set_default_model",
    label: "Change Default Model",
    description: "Allow updating the default provider/model used for routing."
  },
  {
    id: "modelops.enable_disable_provider_or_model",
    label: "Enable or Disable Providers",
    description: "Allow turning providers or models on and off."
  },
  {
    id: "llm.autoconfig.apply",
    label: "Apply Autoconfig",
    description: "Allow automated configuration fixes based on detected runtime state."
  },
  {
    id: "llm.hygiene.apply",
    label: "Apply Registry Hygiene",
    description: "Allow cleanup of stale registry entries and metadata drift."
  },
  {
    id: "llm.registry.prune",
    label: "Prune Registry Snapshots",
    description: "Allow removing old registry data and cleanup candidates."
  },
  {
    id: "llm.registry.rollback",
    label: "Rollback Registry",
    description: "Allow restoring an earlier registry snapshot."
  },
  {
    id: "llm.self_heal.apply",
    label: "Apply Self-Heal Actions",
    description: "Allow automatic recovery actions when health checks detect issues."
  },
  {
    id: "llm.autopilot.bootstrap.apply",
    label: "Bootstrap Defaults",
    description: "Allow autopilot to seed or repair baseline routing defaults."
  },
  {
    id: "llm.notifications.test",
    label: "Send Test Notifications",
    description: "Allow sending test notifications through the configured channel."
  },
  {
    id: "llm.notifications.send",
    label: "Send Live Notifications",
    description: "Allow real autopilot notifications to be delivered."
  },
  {
    id: "llm.notifications.prune",
    label: "Prune Notification History",
    description: "Allow cleanup of stored notification records."
  }
];

const createChatId = (prefix) => `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

const loadStoredChatId = (storageKey, prefix) => {
  if (typeof window === "undefined") {
    return createChatId(prefix);
  }
  try {
    const existing = window.sessionStorage.getItem(storageKey);
    if (existing) {
      return existing;
    }
    const created = createChatId(prefix);
    window.sessionStorage.setItem(storageKey, created);
    return created;
  } catch (_error) {
    return createChatId(prefix);
  }
};

const saveStoredChatId = (storageKey, value) => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.sessionStorage.setItem(storageKey, value);
  } catch (_error) {
    // Ignore storage write failures and keep using the in-memory id.
  }
};

const MODEL_SCOUT_PURPOSES = ["chat", "code", "organize", "story"];
const MODEL_SCOUT_PURPOSE_LABELS = {
  chat: "chat",
  code: "code",
  organize: "organize",
  story: "story"
};

const buildCanonicalScoutStatus = ({ checkPayload, lifecyclePayload }) => {
  const envelope = checkPayload && checkPayload.ok ? checkPayload.envelope || {} : {};
  const lifecycle = lifecyclePayload && lifecyclePayload.ok !== false ? lifecyclePayload || {} : {};
  const recommendationsByPurpose = envelope.recommendations_by_purpose || {};
  const recommendationCount = Object.values(recommendationsByPurpose).reduce((total, rows) => {
    return total + (Array.isArray(rows) ? rows.length : 0);
  }, 0);
  return {
    mode: String((envelope.policy || {}).mode || "unknown"),
    currentModel: String(((envelope.current_model || {}).model) || lifecycle.active_model || "").trim(),
    currentProvider: String(((envelope.current_model || {}).provider) || lifecycle.active_provider || "").trim(),
    availableCount: Number(envelope.available_count || 0),
    newModelsCount: Number(envelope.new_models_count || 0),
    recommendationCount,
    providerCounts: envelope.provider_counts || {},
    warnings: Array.isArray(envelope.warnings) ? envelope.warnings : [],
    lifecycleCounts: lifecycle.counts || {},
    source: "canonical:/llm/models/check+/llm/models/lifecycle"
  };
};

const buildCanonicalScoutSuggestions = ({ checkPayload, providerRows }) => {
  const envelope = checkPayload && checkPayload.ok ? checkPayload.envelope || {} : {};
  const recommendationsByPurpose = envelope.recommendations_by_purpose || {};
  const localByProvider = new Map(
    (Array.isArray(providerRows) ? providerRows : [])
      .filter((row) => row && row.id)
      .map((row) => [String(row.id).trim().toLowerCase(), row.local === true])
  );
  const suggestions = [];
  MODEL_SCOUT_PURPOSES.forEach((purpose) => {
    const rows = Array.isArray(recommendationsByPurpose[purpose]) ? recommendationsByPurpose[purpose] : [];
    rows.forEach((row, index) => {
      const canonicalModelId = String(row.canonical_model_id || "").trim();
      const providerId = String(row.provider || canonicalModelId.split(":")[0] || "").trim().toLowerCase();
      const modelId = String(row.model_id || canonicalModelId).trim();
      const local = localByProvider.has(providerId) ? localByProvider.get(providerId) === true : providerId === "ollama";
      const whyBetter = Array.isArray(row.why_better_than_current) ? row.why_better_than_current : [];
      const tradeoffs = Array.isArray(row.tradeoffs) ? row.tradeoffs : [];
      suggestions.push({
        id: `${purpose}:${canonicalModelId}:${index}`,
        purpose,
        purposeLabel: MODEL_SCOUT_PURPOSE_LABELS[purpose] || purpose,
        canonical_model_id: canonicalModelId,
        provider_id: providerId,
        model_id: modelId,
        local,
        score: Number(row.score || 0),
        tier: row.tier || null,
        reason: String(row.reason || "policy_selected"),
        whyBetter,
        tradeoffs
      });
    });
  });
  return suggestions;
};

export default function App() {
  const [adminOpen, setAdminOpen] = useState(false);
  const [adminTab, setAdminTab] = useState("setup");
  const [adminLoaded, setAdminLoaded] = useState(false);
  const [adminBusy, setAdminBusy] = useState(false);
  const [readyState, setReadyState] = useState(null);
  const [uiState, setUiState] = useState(null);

  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [routingMode, setRoutingMode] = useState("auto");
  const [defaultProvider, setDefaultProvider] = useState("");
  const [defaultModel, setDefaultModel] = useState("");
  const [allowRemoteFallback, setAllowRemoteFallback] = useState(true);
  const [setupStatus, setSetupStatus] = useState("");

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
  const [providerModelViews, setProviderModelViews] = useState({});
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
  const [packsState, setPacksState] = useState(null);
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
  const [supportRemediationResult, setSupportRemediationResult] = useState(null);
  const [supportStatus, setSupportStatus] = useState("");
  const [supportBusy, setSupportBusy] = useState(false);
  const [permissionsConfig, setPermissionsConfig] = useState(null);
  const [permissionsStatus, setPermissionsStatus] = useState("");
  const [auditEntries, setAuditEntries] = useState([]);

  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [chatSessionId] = useState(() => loadStoredChatId(CHAT_SESSION_STORAGE_KEY, "chat-session"));
  const [chatThreadId, setChatThreadId] = useState(() => loadStoredChatId(CHAT_THREAD_STORAGE_KEY, "chat-thread"));

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

  const request = async (method, path, body, options = {}) => {
    const allowError = options.allowError === true;
    const init = {
      method,
      headers: { "Content-Type": "application/json" }
    };
    if (body) {
      init.body = JSON.stringify(body);
    }
    const response = await fetch(path, init);
    const data = await response.json().catch(() => ({}));
    if (!response.ok && !allowError) {
      throw new Error(data.error || data.message || `${response.status} ${response.statusText}`);
    }
    data.__http_ok = response.ok;
    data.__status = response.status;
    return data;
  };

  const refreshReadyState = async () => {
    try {
      const readyPayload = await request("GET", "/ready");
      setReadyState(readyPayload);
    } catch (error) {
      const detail = asErrorText(error);
      setReadyState({
        ready: false,
        phase: "degraded",
        runtime_mode: "FAILED",
        onboarding: { summary: "I could not reach the agent right now." },
        recovery: { summary: "The service may still be starting or unavailable." }
      });
      appendLog({ endpoint: "/ready", ok: false, detail });
    }
  };

  const refreshAdminState = async () => {
    setAdminBusy(true);
    try {
      const [
        providersPayload,
        modelsPayload,
        defaultsPayload,
        statePayload,
        packsStatePayload,
        telegramPayload,
        modelCheckPayload,
        modelLifecyclePayload,
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
        request("GET", "/state").catch(() => null),
        request("GET", "/packs/state").catch(() => null),
        request("GET", "/telegram/status").catch(() => null),
        request("POST", "/llm/models/check", { purposes: MODEL_SCOUT_PURPOSES }).catch(() => null),
        request("GET", "/llm/models/lifecycle").catch(() => null),
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
      if (statePayload) {
        setUiState(statePayload);
      }
      if (packsStatePayload && packsStatePayload.ok) {
        setPacksState(packsStatePayload);
      }
      if (telegramPayload && telegramPayload.ok) {
        setTelegramConfigured(telegramPayload.configured === true);
      }
      const nextScoutStatus = buildCanonicalScoutStatus({
        checkPayload: modelCheckPayload,
        lifecyclePayload: modelLifecyclePayload
      });
      setModelScoutStatus(nextScoutStatus);
      setModelScoutSuggestions(
        buildCanonicalScoutSuggestions({
          checkPayload: modelCheckPayload,
          providerRows
        })
      );
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

      setAdminLoaded(true);
      appendLog({ endpoint: "admin/bootstrap", ok: true, detail: "Loaded admin state" });
    } catch (error) {
      appendLog({ endpoint: "admin/bootstrap", ok: false, detail: asErrorText(error) });
    } finally {
      setAdminBusy(false);
    }
  };

  const refreshRuntimeState = async ({ includeAdmin = adminOpen } = {}) => {
    await refreshReadyState();
    if (includeAdmin) {
      await refreshAdminState();
    }
  };

  useEffect(() => {
    refreshRuntimeState({ includeAdmin: false });
  }, []);

  useEffect(() => {
    if (!adminOpen || adminLoaded) return;
    void refreshAdminState();
  }, [adminLoaded, adminOpen]);

  useEffect(() => {
    if (!adminOpen) return undefined;
    let stopped = false;
    const poll = async () => {
      try {
        const [statePayload, packsStatePayload, notificationsPayload, statusPayload] = await Promise.all([
          request("GET", "/state").catch(() => null),
          request("GET", "/packs/state").catch(() => null),
          request("GET", "/llm/notifications?limit=20").catch(() => null),
          request("GET", "/llm/notifications/status").catch(() => null)
        ]);
        if (stopped) return;

        if (statePayload) {
          setUiState(statePayload);
        }

        if (packsStatePayload && packsStatePayload.ok) {
          setPacksState(packsStatePayload);
        }

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
  }, [adminOpen]);

  useEffect(() => {
    if (!autopilotToast) return undefined;
    const timerId = window.setTimeout(() => {
      setAutopilotToast("");
    }, 3500);
    return () => window.clearTimeout(timerId);
  }, [autopilotToast]);

  const providerOptions = useMemo(() => providers.map((item) => item.id), [providers]);
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
  const providerModelsById = useMemo(() => {
    const next = {};
    models.forEach((model) => {
      const providerId = String(model?.provider || "").trim();
      if (!providerId) return;
      if (!next[providerId]) next[providerId] = [];
      next[providerId].push(model);
    });
    return next;
  }, [models]);
  const providerModelSummaries = useMemo(() => {
    const next = {};
    providers.forEach((provider) => {
      const providerId = String(provider?.id || "").trim();
      const rows = providerModelsById[providerId] || [];
      next[providerId] = {
        total: rows.length,
        available: rows.filter((row) => row?.available === true).length,
        routable: rows.filter((row) => row?.routable === true).length,
        issues: rows.filter((row) => healthStatus(row) !== "ok" || row?.available === false || row?.routable === false).length,
        capabilities: summarizeProviderCapabilities(rows)
      };
    });
    return next;
  }, [providerModelsById, providers]);
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
  const chatStatus = useMemo(() => buildStatusSummary(readyState), [readyState]);
  const starterPrompts = useMemo(() => buildStarterPrompts(readyState), [readyState]);
  const composerPlaceholder = useMemo(() => buildComposerPlaceholder(readyState), [readyState]);

  const updateProviderModelView = (providerId, patch) => {
    setProviderModelViews((prev) => ({
      ...prev,
      [providerId]: {
        expanded: false,
        query: "",
        filter: "all",
        ...(prev[providerId] || {}),
        ...patch
      }
    }));
  };

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

  const sendMessage = async (overrideText) => {
    const content = String(typeof overrideText === "string" ? overrideText : draft).trim();
    if (!content || chatBusy) return;

    const nextUserMessage = { role: "user", content };
    const nextMessages = [...messages, nextUserMessage];
    setMessages(nextMessages);
    setDraft("");
    setChatBusy(true);

    try {
      const result = await request("POST", "/chat", {
        messages: nextMessages.map((item) => ({ role: item.role, content: item.content })),
        session_id: chatSessionId,
        thread_id: chatThreadId,
        source_surface: "webui",
        purpose: "chat",
        task_type: "chat"
      }, { allowError: true });

      const assistantMessage = buildAssistantMessage(result);
      setMessages((prev) => [...prev, assistantMessage]);
      appendLog({
        endpoint: "/chat",
        ok: result.ok === true,
        detail:
          result.ok === true
            ? "Conversation updated"
            : result.error_kind || result.error || "needs_attention"
      });
    } catch (error) {
      const detail = asErrorText(error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `I ran into a problem: ${detail}`,
          tone: "error",
          ui: { confirmation: null, clarification: null }
        }
      ]);
      appendLog({ endpoint: "/chat", ok: false, detail });
    } finally {
      setChatBusy(false);
      void refreshReadyState();
    }
  };

  const resetConversation = () => {
    const nextThreadId = createChatId("chat-thread");
    setChatThreadId(nextThreadId);
    saveStoredChatId(CHAT_THREAD_STORAGE_KEY, nextThreadId);
    setMessages([]);
    appendLog({ endpoint: "chat/reset", ok: true, detail: "Conversation reset" });
  };

  const exportConversation = () => {
    const exportPayload = {
      exported_at: new Date().toISOString(),
      messages: messages.map((message) => ({
        role: message.role,
        content: message.content
      }))
    };
    const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: "application/json" });
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
      await refreshRuntimeState();
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
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setTelegramStatus(`Test failed: ${detail}`);
      appendLog({ endpoint: "/telegram/test", ok: false, detail });
    }
  };

  const runModelScout = async () => {
    setModelScoutRunning(true);
    setModelScoutMessage("Refreshing canonical recommendations...");
    try {
      const [checkPayload, lifecyclePayload] = await Promise.all([
        request("POST", "/llm/models/check", { purposes: MODEL_SCOUT_PURPOSES }),
        request("GET", "/llm/models/lifecycle")
      ]);
      const nextSuggestions = buildCanonicalScoutSuggestions({
        checkPayload,
        providerRows: providers
      });
      setModelScoutStatus(
        buildCanonicalScoutStatus({
          checkPayload,
          lifecyclePayload
        })
      );
      setModelScoutSuggestions(nextSuggestions);
      setModelScoutMessage(`Recommendations refreshed: ${nextSuggestions.length} candidate(s).`);
      appendLog({ endpoint: "/llm/models/check", ok: true, detail: `recommendations=${nextSuggestions.length}` });
      await refreshReadyState();
    } catch (error) {
      const detail = asErrorText(error);
      setModelScoutMessage(`Recommendation refresh failed: ${detail}`);
      appendLog({ endpoint: "/llm/models/check", ok: false, detail });
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
      setSupportRemediationResult(null);
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

  const executeSupportRemediation = async () => {
    const targetId = normalizeSupportTarget(supportDiagnoseTarget);
    setSupportBusy(true);
    setSupportStatus("Executing safe remediation steps...");
    try {
      const payload = {
        intent: supportDiagnoseIntent,
        confirm: true
      };
      if (targetId) {
        payload.target = targetId;
      }
      const result = await request("POST", "/llm/support/remediate/execute", payload);
      setSupportRemediationResult(result);
      setSupportRemediationPlan(result.plan || supportRemediationPlan);
      const executedCount = Number((result.executed_steps || []).length || 0);
      const blockedCount = Number((result.blocked_steps || []).length || 0);
      setSupportStatus(
        result.message || `Executed ${executedCount} step(s), blocked ${blockedCount}.`
      );
      appendLog({
        endpoint: "/llm/support/remediate/execute",
        ok: !!result.ok,
        detail: `executed=${executedCount},blocked=${blockedCount}`
      });
      await refreshRuntimeState();
    } catch (error) {
      const detail = asErrorText(error);
      setSupportStatus(`Remediation execute failed: ${detail}`);
      appendLog({ endpoint: "/llm/support/remediate/execute", ok: false, detail });
    } finally {
      setSupportBusy(false);
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

  const adminSections = [
    {
      id: "setup",
      label: "Setup",
      content: (
        <SetupTab
          allowRemoteFallback={allowRemoteFallback}
          defaultModel={defaultModel}
          defaultModelOptions={defaultModelOptions}
          defaultProvider={defaultProvider}
          models={models}
          providerOptions={providerOptions}
          providerRecommendations={providerRecommendations}
          refreshModels={refreshModels}
          routingMode={routingMode}
          routingModes={ROUTING_MODES}
          saveDefaults={saveDefaults}
          setAllowRemoteFallback={setAllowRemoteFallback}
          setDefaultModel={setDefaultModel}
          setDefaultProvider={setDefaultProvider}
          setRoutingMode={setRoutingMode}
          setupStatus={setupStatus}
        />
      )
    },
    {
      id: "state",
      label: "State",
      content: <StateTab stateSnapshot={uiState} />
    },
    {
      id: "packs",
      label: "Packs",
      content: <PacksTab packsSnapshot={packsState} />
    },
    {
      id: "operations",
      label: "Operations",
      content: (
        <OperationsTab
          autopilotBootstrapBusy={autopilotBootstrapBusy}
          autopilotLastChange={autopilotLastChange}
          autopilotLastChangeBusy={autopilotLastChangeBusy}
          autopilotLastChangeStatus={autopilotLastChangeStatus}
          autopilotLastReadHash={autopilotLastReadHash}
          autopilotLedgerEntries={autopilotLedgerEntries}
          autopilotNotifications={autopilotNotifications}
          autopilotNotificationsPolicy={autopilotNotificationsPolicy}
          autopilotNotificationsStatus={autopilotNotificationsStatus}
          autopilotNotifyBusy={autopilotNotifyBusy}
          autopilotNotifyStatus={autopilotNotifyStatus}
          autopilotToast={autopilotToast}
          autopilotUndoBusy={autopilotUndoBusy}
          autoconfigBusy={autoconfigBusy}
          autoconfigPlan={autoconfigPlan}
          autoconfigStatus={autoconfigStatus}
          bootstrapAutopilotDefaults={bootstrapAutopilotDefaults}
          canBootstrapAutopilot={canBootstrapAutopilot}
          canRollbackRegistry={canRollbackRegistry}
          canSendAutopilotTest={canSendAutopilotTest}
          capabilitiesReconcileBusy={capabilitiesReconcileBusy}
          capabilitiesReconcilePlan={capabilitiesReconcilePlan}
          capabilitiesReconcileStatus={capabilitiesReconcileStatus}
          cleanupBusy={cleanupBusy}
          cleanupPlan={cleanupPlan}
          cleanupStatus={cleanupStatus}
          executeSupportRemediation={executeSupportRemediation}
          explainLastAutopilotChange={explainLastAutopilotChange}
          exportSupportBundle={exportSupportBundle}
          hygieneBusy={hygieneBusy}
          hygienePlan={hygienePlan}
          hygieneStatus={hygieneStatus}
          llmCatalogRows={llmCatalogRows}
          llmCatalogStatus={llmCatalogStatus}
          llmHealth={llmHealth}
          llmHealthMessage={llmHealthMessage}
          llmHealthRunning={llmHealthRunning}
          markAutopilotRead={markAutopilotRead}
          notificationStoreSummary={notificationStoreSummary}
          notifyStatusSummary={notifyStatusSummary}
          planLlmAutoconfig={planLlmAutoconfig}
          planLlmCapabilitiesReconcile={planLlmCapabilitiesReconcile}
          planLlmCleanup={planLlmCleanup}
          planLlmHygiene={planLlmHygiene}
          planSupportRemediation={planSupportRemediation}
          registrySnapshots={registrySnapshots}
          rollbackBusySnapshotId={rollbackBusySnapshotId}
          rollbackRegistryToSnapshot={rollbackRegistryToSnapshot}
          runLlmCatalogRefresh={runLlmCatalogRefresh}
          runLlmHealthCheck={runLlmHealthCheck}
          runSupportDiagnosis={runSupportDiagnosis}
          safetyStatus={safetyStatus}
          sendAutopilotTestNotification={sendAutopilotTestNotification}
          setSupportDiagnoseIntent={setSupportDiagnoseIntent}
          setSupportDiagnoseTarget={setSupportDiagnoseTarget}
          supportBundlePreview={supportBundlePreview}
          supportBusy={supportBusy}
          supportDiagnosis={supportDiagnosis}
          supportDiagnoseIntent={supportDiagnoseIntent}
          supportDiagnoseTarget={supportDiagnoseTarget}
          supportRemediationPlan={supportRemediationPlan}
          supportRemediationResult={supportRemediationResult}
          supportStatus={supportStatus}
          supportTargetOptions={supportTargetOptions}
          undoLastAutopilotChange={undoLastAutopilotChange}
          applyLlmAutoconfig={applyLlmAutoconfig}
          applyLlmCapabilitiesReconcile={applyLlmCapabilitiesReconcile}
          applyLlmCleanup={applyLlmCleanup}
          applyLlmHygiene={applyLlmHygiene}
        />
      )
    },
    {
      id: "providers",
      label: "Providers",
      content: (
        <ProvidersTab
          activeProviderForModels={activeProviderForModels}
          activeProviderModels={activeProviderModels}
          addManualModel={addManualModel}
          addProviderBusy={addProviderBusy}
          addProviderForm={addProviderForm}
          addProviderStatus={addProviderStatus}
          applyProviderPreset={applyProviderPreset}
          deleteProvider={deleteProvider}
          manualModelDraft={manualModelDraft}
          providerDrafts={providerDrafts}
          providerModelSummaries={providerModelSummaries}
          providerModelViews={providerModelViews}
          providerModelsById={providerModelsById}
          providerOptions={providerOptions}
          providerPresets={PROVIDER_PRESETS}
          providerSecrets={providerSecrets}
          providerStatuses={providerStatuses}
          providers={providers}
          refreshModels={refreshModels}
          refreshProviderModels={refreshProviderModels}
          saveOrTestProvider={saveOrTestProvider}
          saveProvider={saveProvider}
          saveProviderSecret={saveProviderSecret}
          setActiveProviderForModels={setActiveProviderForModels}
          setManualModelDraft={setManualModelDraft}
          setProviderSecrets={setProviderSecrets}
          testProvider={testProvider}
          updateAddProviderField={updateAddProviderField}
          updateProviderField={updateProviderField}
          updateProviderModelView={updateProviderModelView}
        />
      )
    },
    {
      id: "telegram",
      label: "Telegram",
      content: (
        <TelegramTab
          saveTelegramToken={saveTelegramToken}
          setTelegramToken={setTelegramToken}
          telegramConfigured={telegramConfigured}
          telegramStatus={telegramStatus}
          telegramToken={telegramToken}
          testTelegramToken={testTelegramToken}
        />
      )
    },
    {
      id: "permissions",
      label: "Permissions",
      content: (
        <PermissionsTab
          actions={MODELOPS_ACTIONS}
          auditEntries={auditEntries}
          permissionsConfig={permissionsConfig}
          permissionsStatus={permissionsStatus}
          savePermissions={savePermissions}
          setPermissionsConfig={setPermissionsConfig}
          updatePermissionAction={updatePermissionAction}
          updatePermissionConstraint={updatePermissionConstraint}
        />
      )
    },
    {
      id: "model_scout",
      label: "Model Scout",
      content: (
        <ModelScoutTab
          modelScoutMessage={modelScoutMessage}
          modelScoutRunning={modelScoutRunning}
          modelScoutStatus={modelScoutStatus}
          modelScoutSuggestions={modelScoutSuggestions}
          runModelScout={runModelScout}
        />
      )
    },
    {
      id: "debug",
      label: "Logs",
      content: <DebugTab logs={logs} />
    }
  ];

  return (
    <>
      <ChatExperience
        chatBusy={chatBusy}
        composerPlaceholder={composerPlaceholder}
        draft={draft}
        messages={messages}
        onDraftChange={setDraft}
        onExportConversation={exportConversation}
        onOpenAdmin={() => setAdminOpen(true)}
        onResetConversation={resetConversation}
        onSendMessage={sendMessage}
        onStarterPrompt={sendMessage}
        starterPrompts={starterPrompts}
        status={chatStatus}
      />
      <AdminPanel
        activeSection={adminTab}
        loading={adminBusy && !adminLoaded}
        onClose={() => setAdminOpen(false)}
        onRefresh={() => {
          void refreshRuntimeState({ includeAdmin: true });
        }}
        onSelectSection={setAdminTab}
        open={adminOpen}
        sections={adminSections}
        status={chatStatus}
      />
    </>
  );
}
