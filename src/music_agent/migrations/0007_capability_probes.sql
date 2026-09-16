CREATE TABLE capability_probes (
    probe_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL CHECK (operation = 'set_favorited'),
    target_canonical_id TEXT NOT NULL,
    target_persistent_id TEXT NOT NULL CHECK (target_persistent_id <> ''),
    baseline_favorited INTEGER NOT NULL CHECK (baseline_favorited IN (0, 1)),
    baseline_disliked INTEGER NOT NULL CHECK (baseline_disliked IN (0, 1)),
    step_state TEXT NOT NULL CHECK (
        step_state IN ('baseline_captured', 'forward_started', 'forward_observed', 'restore_started', 'restore_observed')
    ),
    recovery_status TEXT NOT NULL CHECK (
        recovery_status IN ('baseline_confirmed', 'restored', 'needs_manual_check')
    ),
    verification_verdict TEXT NOT NULL CHECK (
        verification_verdict IN ('pending', 'verified', 'failed', 'inconclusive')
    ),
    forward_command_outcome TEXT CHECK (
        forward_command_outcome IN ('success', 'failed', 'unknown')
    ),
    forward_favorited_state TEXT CHECK (
        forward_favorited_state IN ('missing', 'value')
    ),
    forward_favorited_value INTEGER CHECK (forward_favorited_value IN (0, 1)),
    forward_disliked_state TEXT CHECK (
        forward_disliked_state IN ('missing', 'value')
    ),
    forward_disliked_value INTEGER CHECK (forward_disliked_value IN (0, 1)),
    restore_command_outcome TEXT CHECK (
        restore_command_outcome IN ('success', 'failed', 'unknown')
    ),
    restore_favorited_state TEXT CHECK (
        restore_favorited_state IN ('missing', 'value')
    ),
    restore_favorited_value INTEGER CHECK (restore_favorited_value IN (0, 1)),
    restore_disliked_state TEXT CHECK (
        restore_disliked_state IN ('missing', 'value')
    ),
    restore_disliked_value INTEGER CHECK (restore_disliked_value IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (forward_favorited_state IS NULL AND forward_favorited_value IS NULL)
        OR (forward_favorited_state = 'missing' AND forward_favorited_value IS NULL)
        OR (forward_favorited_state = 'value' AND forward_favorited_value IS NOT NULL)
    ),
    CHECK (
        (forward_disliked_state IS NULL AND forward_disliked_value IS NULL)
        OR (forward_disliked_state = 'missing' AND forward_disliked_value IS NULL)
        OR (forward_disliked_state = 'value' AND forward_disliked_value IS NOT NULL)
    ),
    CHECK (
        (restore_favorited_state IS NULL AND restore_favorited_value IS NULL)
        OR (restore_favorited_state = 'missing' AND restore_favorited_value IS NULL)
        OR (restore_favorited_state = 'value' AND restore_favorited_value IS NOT NULL)
    ),
    CHECK (
        (restore_disliked_state IS NULL AND restore_disliked_value IS NULL)
        OR (restore_disliked_state = 'missing' AND restore_disliked_value IS NULL)
        OR (restore_disliked_state = 'value' AND restore_disliked_value IS NOT NULL)
    )
);
