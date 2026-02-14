PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    pitch TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    priority INTEGER DEFAULT 3,
    tags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    title TEXT NOT NULL,
    details TEXT,
    effort_mins INTEGER,
    impact_1to5 INTEGER,
    status TEXT NOT NULL DEFAULT 'todo',
    due_date TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    content TEXT NOT NULL,
    tags TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS open_loops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due_date TEXT,
    priority INTEGER NOT NULL DEFAULT 3,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    when_ts TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    sent_at TEXT,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS user_prefs (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    action_id TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS disk_baselines (
    user_id TEXT PRIMARY KEY,
    snapshot_json TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS disk_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    snapshot_local_date TEXT NOT NULL,
    hostname TEXT NOT NULL,
    mountpoint TEXT NOT NULL,
    filesystem TEXT,
    total_bytes INTEGER NOT NULL,
    used_bytes INTEGER NOT NULL,
    free_bytes INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_disk_snapshots_date_mount
    ON disk_snapshots(snapshot_local_date, mountpoint);

CREATE INDEX IF NOT EXISTS idx_disk_snapshots_mount_taken
    ON disk_snapshots(mountpoint, taken_at);

CREATE TABLE IF NOT EXISTS dir_size_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    scope TEXT NOT NULL,
    path TEXT NOT NULL,
    bytes INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dir_size_samples_scope_taken
    ON dir_size_samples(scope, taken_at);

CREATE TABLE IF NOT EXISTS storage_scan_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    scope TEXT NOT NULL,
    dirs_scanned INTEGER NOT NULL,
    errors_skipped INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_storage_scan_stats_taken_scope
    ON storage_scan_stats(taken_at, scope);

CREATE INDEX IF NOT EXISTS idx_storage_scan_stats_scope_taken
    ON storage_scan_stats(scope, taken_at);

CREATE TABLE IF NOT EXISTS resource_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    snapshot_local_date TEXT NOT NULL,
    hostname TEXT NOT NULL,
    load_1m REAL NOT NULL,
    load_5m REAL NOT NULL,
    load_15m REAL NOT NULL,
    mem_total INTEGER NOT NULL,
    mem_used INTEGER NOT NULL,
    mem_free INTEGER NOT NULL,
    swap_total INTEGER NOT NULL,
    swap_used INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_resource_snapshots_date_host
    ON resource_snapshots(snapshot_local_date, hostname);

CREATE INDEX IF NOT EXISTS idx_resource_snapshots_taken
    ON resource_snapshots(taken_at);

CREATE TABLE IF NOT EXISTS resource_process_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    category TEXT NOT NULL,
    pid INTEGER NOT NULL,
    name TEXT NOT NULL,
    cpu_ticks INTEGER NOT NULL,
    rss_bytes INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resource_process_samples_taken_category
    ON resource_process_samples(taken_at, category);

CREATE TABLE IF NOT EXISTS resource_scan_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    scope TEXT NOT NULL,
    procs_scanned INTEGER NOT NULL,
    errors_skipped INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_resource_scan_stats_taken_scope
    ON resource_scan_stats(taken_at, scope);

CREATE INDEX IF NOT EXISTS idx_resource_scan_stats_scope_taken
    ON resource_scan_stats(scope, taken_at);

CREATE TABLE IF NOT EXISTS network_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    snapshot_local_date TEXT NOT NULL,
    hostname TEXT NOT NULL,
    default_iface TEXT NOT NULL,
    default_gateway TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_network_snapshots_date_host
    ON network_snapshots(snapshot_local_date, hostname);

CREATE INDEX IF NOT EXISTS idx_network_snapshots_taken
    ON network_snapshots(taken_at);

CREATE TABLE IF NOT EXISTS network_interfaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    name TEXT NOT NULL,
    state TEXT NOT NULL,
    rx_bytes INTEGER NOT NULL,
    tx_bytes INTEGER NOT NULL,
    rx_errors INTEGER NOT NULL,
    tx_errors INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_network_interfaces_taken
    ON network_interfaces(taken_at);

CREATE TABLE IF NOT EXISTS network_nameservers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    taken_at TEXT NOT NULL,
    nameserver TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_network_nameservers_taken
    ON network_nameservers(taken_at);

CREATE TABLE IF NOT EXISTS pending_clarifications (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    intent_type TEXT NOT NULL,
    partial_args_json TEXT NOT NULL,
    question TEXT NOT NULL,
    options_json TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    snapshot_id INTEGER,
    source TEXT NOT NULL,
    anomaly_key TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    metric_unit TEXT,
    context_json TEXT NOT NULL,
    UNIQUE(user_id, observed_at, anomaly_key)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_user_time
    ON anomaly_events(user_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_anomaly_events_user_key_time
    ON anomaly_events(user_id, anomaly_key, observed_at);

-- Unified, JSON-serialized system snapshots (v0.2 brief/delta).
CREATE TABLE IF NOT EXISTS system_facts_snapshots (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    boot_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    facts_json TEXT NOT NULL,
    content_hash_sha256 TEXT NOT NULL,
    partial INTEGER NOT NULL DEFAULT 0,
    errors_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_system_facts_user_taken
    ON system_facts_snapshots(user_id, taken_at);

-- Registry for last "rendered report" payloads (used for followups).
CREATE TABLE IF NOT EXISTS last_report_registry (
    user_id TEXT NOT NULL,
    report_key TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    audit_ref TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, report_key)
);

CREATE TABLE IF NOT EXISTS report_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    report_key TEXT NOT NULL,
    taken_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    audit_ref TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_report_history_user_key_taken
    ON report_history(user_id, report_key, taken_at);

CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_notes_project_id ON notes(project_id);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status);
CREATE INDEX IF NOT EXISTS idx_pending_clarifications_user_id ON pending_clarifications(user_id);
CREATE INDEX IF NOT EXISTS idx_pending_clarifications_chat_id ON pending_clarifications(chat_id);
CREATE INDEX IF NOT EXISTS idx_pending_clarifications_expires_at ON pending_clarifications(expires_at);
