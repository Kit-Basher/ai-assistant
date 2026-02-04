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

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    when_ts TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL
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

CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_notes_project_id ON notes(project_id);
CREATE INDEX IF NOT EXISTS idx_reminders_status ON reminders(status);
CREATE INDEX IF NOT EXISTS idx_pending_clarifications_user_id ON pending_clarifications(user_id);
CREATE INDEX IF NOT EXISTS idx_pending_clarifications_chat_id ON pending_clarifications(chat_id);
CREATE INDEX IF NOT EXISTS idx_pending_clarifications_expires_at ON pending_clarifications(expires_at);
