"""Read-only Apple Music adapter for already-resolved canonical Tracks."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from music_agent.identity import EntityType
from music_agent.identity import generate_canonical_id
from music_agent.source_observation import ObservedValue, SourceObservation


class AppleMusicReadError(RuntimeError):
    code = "source_lookup_failed"


class AppleMusicMappingError(ValueError):
    code = "validation_error"


class SourceReadStatus(StrEnum):
    FOUND = "found"
    CONFIRMED_NOT_FOUND = "confirmed_not_found"
    LOOKUP_FAILED = "lookup_failed"


class MusicCommandRunner(Protocol):
    def run(self, persistent_id: str) -> str: ...


@dataclass(frozen=True, slots=True)
class RawTrackRecord:
    persistent_id: str
    fields: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SourceReadResult:
    status: SourceReadStatus
    record: RawTrackRecord | None = None
    error: str | None = None


READ_TRACK_SCRIPT = r'''
on replaceText(findText, replacementText, sourceText)
    set previousDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to findText
    set textItems to every text item of sourceText
    set AppleScript's text item delimiters to replacementText
    set replacedText to textItems as text
    set AppleScript's text item delimiters to previousDelimiters
    return replacedText
end replaceText

on jsonString(sourceValue)
    set valueText to sourceValue as text
    set valueText to my replaceText("\\", "\\\\", valueText)
    set valueText to my replaceText(quote, "\\\"", valueText)
    set valueText to my replaceText(return, "\\n", valueText)
    set valueText to my replaceText(linefeed, "\\n", valueText)
    return quote & valueText & quote
end jsonString

on jsonNullableString(sourceValue)
    if sourceValue is missing value then return "null"
    if (sourceValue as text) is "" then return "null"
    return my jsonString(sourceValue)
end jsonNullableString

on jsonInteger(sourceValue)
    if sourceValue is missing value then return "null"
    return sourceValue as integer as text
end jsonInteger

on jsonBoolean(sourceValue)
    if sourceValue is missing value then return "null"
    if sourceValue then return "true"
    return "false"
end jsonBoolean

on run argv
    if (count of argv) is not 1 then error "one persistent ID is required"
    set targetID to item 1 of argv
    tell application "Music"
        set matchingTracks to every track of library playlist 1 whose persistent ID is targetID
        if (count of matchingTracks) is 0 then return "{\"status\":\"confirmed_not_found\"}"
        set sourceTrack to item 1 of matchingTracks
        set trackName to name of sourceTrack
        set artistName to artist of sourceTrack
        set albumName to album of sourceTrack
        set durationMilliseconds to round ((duration of sourceTrack) * 1000)
        set playedCount to played count of sourceTrack
        set favoriteState to favorited of sourceTrack
        set dislikeState to disliked of sourceTrack
        set ratingValue to rating of sourceTrack
    end tell
    return "{\"status\":\"found\",\"fields\":{" & ¬
        "\"name\":" & my jsonString(trackName) & "," & ¬
        "\"artist\":" & my jsonNullableString(artistName) & "," & ¬
        "\"album\":" & my jsonNullableString(albumName) & "," & ¬
        "\"duration_ms\":" & my jsonInteger(durationMilliseconds) & "," & ¬
        "\"played_count\":" & my jsonInteger(playedCount) & "," & ¬
        "\"favorited\":" & my jsonBoolean(favoriteState) & "," & ¬
        "\"disliked\":" & my jsonBoolean(dislikeState) & "," & ¬
        "\"rating\":" & my jsonInteger(ratingValue) & "}}"
end run
'''


class OsascriptMusicRunner:
    """Invoke the bundled read-only AppleScript without shell interpolation."""

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, persistent_id: str) -> str:
        _require_persistent_id(persistent_id)
        try:
            completed = subprocess.run(
                ["osascript", "-e", READ_TRACK_SCRIPT, "--", persistent_id],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AppleMusicReadError(str(error)) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"osascript exited {completed.returncode}"
            raise AppleMusicReadError(detail)
        return completed.stdout.strip()


TRACK_OBSERVATION_FIELDS = (
    "name", "artist_ids", "album_id", "duration_ms", "genres", "track_number",
    "disc_number", "release_date", "composer", "library_state.favorited",
    "library_state.disliked", "library_state.rating", "library_state.play_count",
    "library_state.skip_count", "library_state.added_to_library_at",
    "library_state.last_played_at",
)


class AppleMusicSourceAdapter:
    """Parse structured runner output and map supported Track facts to observations."""

    def __init__(self, runner: MusicCommandRunner) -> None:
        self.runner = runner

    def read_track(self, persistent_id: str) -> SourceReadResult:
        _require_persistent_id(persistent_id)
        try:
            output = self.runner.run(persistent_id)
            payload = json.loads(output)
            return _parse_source_result(payload, persistent_id)
        except (AppleMusicReadError, json.JSONDecodeError, TypeError, ValueError) as error:
            return SourceReadResult(SourceReadStatus.LOOKUP_FAILED, error=str(error))

    def build_observation(
        self, canonical_id: str, read_result: SourceReadResult
    ) -> SourceObservation:
        if read_result.status is not SourceReadStatus.FOUND or read_result.record is None:
            raise AppleMusicMappingError("a FOUND Track record is required")
        fields = {path: ObservedValue.missing() for path in TRACK_OBSERVATION_FIELDS}
        raw = read_result.record.fields
        _map_field(raw, "name", fields, "name", _nonempty_string)
        _map_field(raw, "duration_ms", fields, "duration_ms", _nonnegative_integer)
        _map_field(raw, "favorited", fields, "library_state.favorited", _boolean)
        _map_field(raw, "disliked", fields, "library_state.disliked", _boolean)
        _map_field(raw, "rating", fields, "library_state.rating", _rating)
        _map_field(raw, "played_count", fields, "library_state.play_count", _nonnegative_integer)
        _map_field(
            raw, "date_added", fields, "library_state.added_to_library_at", _timezone_datetime
        )
        _map_field(
            raw, "played_date", fields, "library_state.last_played_at", _timezone_datetime
        )
        return SourceObservation(EntityType.TRACK, canonical_id, "apple_music", fields)


def materialize_library_track_relations(
    model: dict[str, Any], canonical_id: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Create only the source-local relation entities needed by one Library track.

    Music.app exposes artist/album names but no stable relation identity in this
    read contract.  Therefore names are never matched against Catalog entities
    and never used to merge Tracks.  An existing relation on this exact track is
    reused only when its displayed name still matches; otherwise a fresh unbound
    Artist/Album is added and returned for the track observation to reference.
    """
    track = next((item for item in model["tracks"] if item["id"] == canonical_id), None)
    if track is None:
        raise AppleMusicMappingError("canonical Track is required")
    relation_fields: dict[str, Any] = {}

    artist_name = raw.get("artist")
    artist_ids: list[str] = []
    if isinstance(artist_name, str) and artist_name.strip():
        artist_name = artist_name.strip()
        current = [
            artist
            for artist in model["artists"]
            if artist["id"] in track.get("artist_ids", ())
            and artist.get("name") == artist_name
        ]
        if len(current) == 1:
            artist_id = current[0]["id"]
        else:
            artist_id = generate_canonical_id(EntityType.ARTIST)
            model["artists"].append(
                {
                    "id": artist_id,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": artist_name,
                }
            )
        artist_ids = [artist_id]
        relation_fields["artist_ids"] = artist_ids

    album_name = raw.get("album")
    if isinstance(album_name, str) and album_name.strip():
        album_name = album_name.strip()
        current_album = next(
            (
                album
                for album in model["albums"]
                if album["id"] == track.get("album_id")
                and album.get("name") == album_name
            ),
            None,
        )
        if current_album is not None:
            album_id = current_album["id"]
        else:
            album_id = generate_canonical_id(EntityType.ALBUM)
            model["albums"].append(
                {
                    "id": album_id,
                    "external_ids": {"apple_music_persistent_id": None},
                    "name": album_name,
                    "artist_ids": artist_ids,
                    "release_date": None,
                }
            )
        relation_fields["album_id"] = album_id
    return relation_fields


def add_library_relation_values(
    observation: SourceObservation, relation_fields: Mapping[str, Any]
) -> SourceObservation:
    """Return the same Track observation plus source-confirmed canonical relations."""
    fields = dict(observation.fields)
    if "artist_ids" in relation_fields:
        fields["artist_ids"] = ObservedValue.value(list(relation_fields["artist_ids"]))
    if "album_id" in relation_fields:
        fields["album_id"] = ObservedValue.value(relation_fields["album_id"])
    return SourceObservation(
        observation.entity_type,
        observation.canonical_id,
        observation.source_system,
        fields,
        source_presence=observation.source_presence,
    )


def _parse_source_result(payload: object, persistent_id: str) -> SourceReadResult:
    if not isinstance(payload, dict):
        raise AppleMusicReadError("source output must be a JSON object")
    status = payload.get("status")
    if status == SourceReadStatus.CONFIRMED_NOT_FOUND.value:
        return SourceReadResult(SourceReadStatus.CONFIRMED_NOT_FOUND)
    if status != SourceReadStatus.FOUND.value:
        raise AppleMusicReadError("source output has an unknown status")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise AppleMusicReadError("FOUND source output requires a fields object")
    record = RawTrackRecord(persistent_id, MappingProxyType(dict(fields)))
    return SourceReadResult(SourceReadStatus.FOUND, record=record)


def _map_field(
    raw: Mapping[str, Any],
    source_property: str,
    fields: dict[str, ObservedValue],
    canonical_path: str,
    converter: Any,
) -> None:
    if source_property not in raw or raw[source_property] is None:
        return
    fields[canonical_path] = ObservedValue.value(converter(raw[source_property]))


def _require_persistent_id(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise AppleMusicMappingError("persistent_id must be a non-empty string")
    return value


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise AppleMusicMappingError("source string must be non-empty")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise AppleMusicMappingError("source boolean has the wrong type")
    return value


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AppleMusicMappingError("source count must be a non-negative integer")
    return value


def _rating(value: object) -> int:
    value = _nonnegative_integer(value)
    if value > 100:
        raise AppleMusicMappingError("source rating must be within 0..100")
    return value


def _timezone_datetime(value: object) -> str:
    value = _nonempty_string(value)
    if not value.isascii():
        raise AppleMusicMappingError("source datetime must use ASCII canonical representation")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AppleMusicMappingError("source datetime must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AppleMusicMappingError("source datetime must include a timezone")
    return value
