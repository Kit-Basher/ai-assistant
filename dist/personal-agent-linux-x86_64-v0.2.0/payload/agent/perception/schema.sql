CREATE TABLE IF NOT EXISTS metrics_snapshot (
    id INTEGER PRIMARY KEY,
    ts INTEGER,
    cpu_usage REAL,
    cpu_freq REAL,
    mem_used INTEGER,
    mem_available INTEGER,
    root_disk_used_pct REAL,
    gpu_usage REAL,
    gpu_mem_used INTEGER,
    gpu_temp REAL
);

CREATE INDEX IF NOT EXISTS idx_metrics_snapshot_ts
    ON metrics_snapshot(ts);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts INTEGER,
    kind TEXT,
    severity TEXT,
    summary TEXT,
    evidence_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_ts
    ON events(ts);
