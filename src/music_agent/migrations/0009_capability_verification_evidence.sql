CREATE TABLE capability_verification_evidence (
    probe_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL CHECK (operation = 'set_favorited'),
    target_canonical_id TEXT NOT NULL,
    target_persistent_id TEXT NOT NULL CHECK (target_persistent_id <> ''),
    baseline_favorited INTEGER NOT NULL CHECK (baseline_favorited IN (0, 1)),
    baseline_disliked INTEGER NOT NULL CHECK (baseline_disliked IN (0, 1)),
    verification_contract_version INTEGER NOT NULL CHECK (verification_contract_version >= 1),
    verified_at TEXT NOT NULL CHECK (verified_at <> ''),
    FOREIGN KEY (probe_id) REFERENCES capability_probes (probe_id)
);
