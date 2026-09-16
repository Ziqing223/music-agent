"""P10.10: Project GENUINE durable Music Agent events into the bounded Vault surfaces.

The P10.9 writer renders record objects; this module builds them from the DURABLE store --
a real ``rcm_`` recommendation run or a real ``fbk_`` feedback observation -- so the Vault
projection can never fabricate events. SQLite stays canonical; the Vault receives only the
bounded, replay-safe records the reconciliation approved.

Mappings (deterministic, documented):

- Recommendation: one table row per run item. 歌曲 = the canonical track name; 艺人 = the
  linked artist names (or 未知艺人); 匹配理由 = ``匹配度 NN%`` from the run's bounded score
  total; 潜在风险 = 待反馈. The run id becomes the replay marker, so re-projecting is a
  no-op and never duplicates records.
- Feedback: the P08 kind maps to the Vault's fixed fields without conflating semantics --
  skipped → 场景：跳过（跳过≠不喜欢）; completed → 喜欢：完整听完; liked/disliked → 喜欢/不喜欢.
  Unmapped kinds leave every field empty (the record still lands with its date + track
  identity). The feedback id becomes the replay marker.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

from music_agent.feedback_history_repository import FeedbackHistoryRepository
from music_agent.obsidian_projection import FeedbackRecord, RecommendationItem, RecommendationRecord
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.repository import CanonicalRepository


class ObsidianEventProjectionError(ValueError):
    code = "obsidian_event_projection_error"


_KIND_FIELD_MAP: Mapping[str, tuple[str, str]] = {
    "liked": ("喜欢", "是"),
    "disliked": ("不喜欢", "是"),
    "completed": ("喜欢", "完整听完"),
    "skipped": ("场景", "跳过（跳过≠不喜欢）"),
    "replayed": ("场景", "重复播放"),
    "played": ("场景", "播放"),
    "favorited": ("是否加入收藏", "是"),
    "corrected": ("喜欢", "修正反馈"),
    "direction_good": ("喜欢", "方向正确"),
    "attribution_correction": ("对画像的影响", "归属修正"),
}


def build_recommendation_record(
    database_path: str | Path, run_id: str, *, theme: str = "每日推荐"
) -> RecommendationRecord:
    """Build one projection record from a real persisted recommendation run."""
    if not isinstance(theme, str) or not theme.strip():
        raise ObsidianEventProjectionError("theme must be a non-empty string")
    with RecommendationHistoryRepository(database_path) as history:
        run = history.get_result(run_id)
    if run is None:
        raise ObsidianEventProjectionError(f"recommendation run does not exist: {run_id}")
    with CanonicalRepository(database_path) as repository:
        model = repository.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        items: list[RecommendationItem] = []
        for item in run.items:
            target_id = item.candidate.target.target_id
            track = track_by_id.get(target_id)
            if track is None:
                raise ObsidianEventProjectionError(
                    f"run {run_id} references an unknown track {target_id}"
                )
            artist_names = [
                artist_by_id[artist_id]["name"]
                for artist_id in track["artist_ids"]
                if artist_id in artist_by_id
            ]
            items.append(
                RecommendationItem(
                    song=track["name"],
                    artist="、".join(artist_names) if artist_names else "未知艺人",
                    match_reason=f"匹配度 {round(item.score.total * 100)}%",
                    risk="待反馈",
                )
            )
    if not items:
        raise ObsidianEventProjectionError(f"run {run_id} produced no items to project")
    return RecommendationRecord(
        run_id=run.run_id,
        date=_date_part(run.produced_at),
        theme=theme,
        items=tuple(items),
    )


def build_feedback_record(
    database_path: str | Path, feedback_id: str
) -> FeedbackRecord:
    """Build one projection record from a real persisted feedback observation."""
    with FeedbackHistoryRepository(database_path) as history:
        observation = history.get_observation(feedback_id)
    if observation is None:
        raise ObsidianEventProjectionError(
            f"feedback observation does not exist: {feedback_id}"
        )
    song, artist = _resolve_feedback_subject(database_path, observation)
    fields: dict[str, str] = {}
    mapping = _KIND_FIELD_MAP.get(observation.kind.value)
    if mapping:
        field, value = mapping
        fields[field] = value
    return FeedbackRecord(
        feedback_id=observation.feedback_id,
        date=_date_part(observation.observed_at),
        song=song,
        artist=artist,
        fields=fields,
    )


def _resolve_feedback_subject(database_path: Path, observation) -> tuple[str, str]:
    """Resolve the feedback's track/artist display identity from the durable store."""
    with CanonicalRepository(database_path) as repository:
        model = repository.load_model()
        track_by_id = {track["id"]: track for track in model["tracks"]}
        artist_by_id = {artist["id"]: artist for artist in model["artists"]}
        track_id = None
        if observation.target is not None:
            track_id = observation.target.target_id
        elif observation.recommendation is not None:
            # The candidate lives in the referenced run's items.
            with RecommendationHistoryRepository(database_path) as history:
                run = history.get_result(observation.recommendation.run_id)
            if run is not None:
                for item in run.items:
                    if item.candidate.candidate_id == observation.recommendation.candidate_id:
                        track_id = item.candidate.target.target_id
                        break
        if track_id is None or track_id not in track_by_id:
            return "未知歌曲", "未知艺人"
        track = track_by_id[track_id]
        artist_names = [
            artist_by_id[artist_id]["name"]
            for artist_id in track["artist_ids"]
            if artist_id in artist_by_id
        ]
        return track["name"], "、".join(artist_names) if artist_names else "未知艺人"


def _date_part(timestamp) -> str:
    """The YYYY-MM-DD part of a durable timestamp (datetime or ISO string)."""
    if isinstance(timestamp, datetime):
        return timestamp.date().isoformat()
    try:
        return datetime.fromisoformat(timestamp).date().isoformat()
    except (ValueError, TypeError) as error:
        raise ObsidianEventProjectionError(
            f"cannot derive a date from timestamp: {timestamp!r}"
        ) from error
