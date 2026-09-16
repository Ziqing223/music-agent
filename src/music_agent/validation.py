"""Structural and synthetic graph validation for the canonical music model."""

from __future__ import annotations

import json
from datetime import date, datetime
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


class StructuralValidationError(ValueError):
    def __init__(self, issue: ValidationError) -> None:
        self.issue = issue
        path = ".".join(str(part) for part in issue.absolute_path) or "<root>"
        super().__init__(f"{path}: {issue.message}")


class GraphValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


FORMAT_CHECKER = FormatChecker()


@FORMAT_CHECKER.checks("canonical-datetime", raises=(TypeError, ValueError))
def _is_timezone_datetime(value: object) -> bool:
    if not isinstance(value, str):
        return True
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


@FORMAT_CHECKER.checks("canonical-release-date", raises=(TypeError, ValueError))
def _is_release_date(value: object) -> bool:
    if value is None or not isinstance(value, str):
        return True
    if not value.isascii():
        return False
    if len(value) == 4:
        return value.isdigit() and 1 <= int(value) <= 9999
    if len(value) == 7:
        datetime.strptime(value, "%Y-%m")
        return True
    if len(value) == 10:
        date.fromisoformat(value)
        return True
    return False


def load_schema() -> dict[str, Any]:
    schema_path = resources.files("music_agent").joinpath("canonical_music_model.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


SCHEMA = load_schema()
Draft202012Validator.check_schema(SCHEMA)
STRUCTURAL_VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FORMAT_CHECKER)


def validate_structure(fixture: dict[str, Any]) -> None:
    errors = sorted(STRUCTURAL_VALIDATOR.iter_errors(fixture), key=lambda error: list(error.absolute_path))
    if errors:
        raise StructuralValidationError(errors[0])


def validate_graph(fixture: dict[str, Any]) -> None:
    collections = ("tracks", "artists", "albums", "playlists", "playlist_memberships")
    all_ids: set[str] = set()
    for collection in collections:
        for entity in fixture[collection]:
            entity_id = entity["id"]
            if entity_id in all_ids:
                raise GraphValidationError("duplicate_canonical_id", f"Duplicate canonical ID: {entity_id}")
            all_ids.add(entity_id)

    artist_ids = {entity["id"] for entity in fixture["artists"]}
    album_ids = {entity["id"] for entity in fixture["albums"]}
    track_ids = {entity["id"] for entity in fixture["tracks"]}
    playlist_ids = {entity["id"] for entity in fixture["playlists"]}

    for album in fixture["albums"]:
        _require_references(album["artist_ids"], artist_ids, "dangling_artist_id")
    for track in fixture["tracks"]:
        _require_references(track["artist_ids"], artist_ids, "dangling_artist_id")
        if track["album_id"] is not None and track["album_id"] not in album_ids:
            raise GraphValidationError("dangling_album_id", f"Unknown album ID: {track['album_id']}")
    for membership in fixture["playlist_memberships"]:
        if membership["playlist_id"] not in playlist_ids:
            raise GraphValidationError("dangling_playlist_id", f"Unknown playlist ID: {membership['playlist_id']}")
        if membership["track_id"] not in track_ids:
            raise GraphValidationError("dangling_track_id", f"Unknown track ID: {membership['track_id']}")


def _require_references(references: list[str], known_ids: set[str], code: str) -> None:
    for reference in references:
        if reference not in known_ids:
            raise GraphValidationError(code, f"Unknown canonical reference: {reference}")


def validate_fixture(fixture: dict[str, Any]) -> None:
    validate_structure(fixture)
    validate_graph(fixture)
