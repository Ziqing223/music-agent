CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE canonical_entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('track', 'artist', 'album', 'playlist', 'playlist_membership')),
    UNIQUE (id, entity_type)
);

CREATE TABLE artists (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL DEFAULT 'artist' CHECK (entity_type = 'artist'),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    name TEXT NOT NULL,
    FOREIGN KEY (id, entity_type) REFERENCES canonical_entities (id, entity_type)
);

CREATE TABLE albums (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL DEFAULT 'album' CHECK (entity_type = 'album'),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    name TEXT NOT NULL,
    release_date TEXT,
    FOREIGN KEY (id, entity_type) REFERENCES canonical_entities (id, entity_type)
);

CREATE TABLE playlists (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL DEFAULT 'playlist' CHECK (entity_type = 'playlist'),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    name TEXT NOT NULL,
    FOREIGN KEY (id, entity_type) REFERENCES canonical_entities (id, entity_type)
);

CREATE TABLE tracks (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL DEFAULT 'track' CHECK (entity_type = 'track'),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    name TEXT NOT NULL,
    album_id TEXT,
    duration_ms INTEGER,
    track_number INTEGER,
    disc_number INTEGER,
    release_date TEXT,
    composer TEXT,
    favorited INTEGER CHECK (favorited IN (0, 1) OR favorited IS NULL),
    disliked INTEGER CHECK (disliked IN (0, 1) OR disliked IS NULL),
    rating INTEGER CHECK (rating BETWEEN 0 AND 100 OR rating IS NULL),
    play_count INTEGER CHECK (play_count >= 0 OR play_count IS NULL),
    skip_count INTEGER CHECK (skip_count >= 0 OR skip_count IS NULL),
    added_to_library_at TEXT,
    last_played_at TEXT,
    FOREIGN KEY (id, entity_type) REFERENCES canonical_entities (id, entity_type),
    FOREIGN KEY (album_id) REFERENCES albums (id)
);

CREATE TABLE playlist_memberships (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL DEFAULT 'playlist_membership' CHECK (entity_type = 'playlist_membership'),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    playlist_id TEXT NOT NULL,
    track_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    added_at TEXT,
    FOREIGN KEY (id, entity_type) REFERENCES canonical_entities (id, entity_type),
    FOREIGN KEY (playlist_id) REFERENCES playlists (id),
    FOREIGN KEY (track_id) REFERENCES tracks (id)
);

CREATE TABLE track_artists (
    track_id TEXT NOT NULL,
    artist_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (track_id, position),
    UNIQUE (track_id, artist_id),
    FOREIGN KEY (track_id) REFERENCES tracks (id),
    FOREIGN KEY (artist_id) REFERENCES artists (id)
);

CREATE TABLE album_artists (
    album_id TEXT NOT NULL,
    artist_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (album_id, position),
    UNIQUE (album_id, artist_id),
    FOREIGN KEY (album_id) REFERENCES albums (id),
    FOREIGN KEY (artist_id) REFERENCES artists (id)
);

CREATE TABLE track_genres (
    track_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    value TEXT NOT NULL,
    PRIMARY KEY (track_id, position),
    FOREIGN KEY (track_id) REFERENCES tracks (id)
);

CREATE TABLE track_tags (
    track_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    value TEXT NOT NULL,
    PRIMARY KEY (track_id, position),
    FOREIGN KEY (track_id) REFERENCES tracks (id)
);

CREATE TABLE external_identity_bindings (
    source_system TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    external_id TEXT NOT NULL CHECK (external_id <> ''),
    canonical_id TEXT NOT NULL,
    PRIMARY KEY (source_system, entity_type, external_id),
    FOREIGN KEY (canonical_id, entity_type) REFERENCES canonical_entities (id, entity_type)
);

CREATE INDEX ix_external_identity_canonical ON external_identity_bindings (canonical_id);
CREATE UNIQUE INDEX ux_apple_music_scalar_identity
    ON external_identity_bindings (canonical_id, entity_type)
    WHERE source_system = 'apple_music';
