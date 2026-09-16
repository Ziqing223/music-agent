CREATE TABLE capability_probe_recovery_attempts (
    attempt_id TEXT PRIMARY KEY,
    probe_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (
        state IN ('started', 'command_succeeded', 'command_failed')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (probe_id) REFERENCES capability_probes (probe_id)
);
