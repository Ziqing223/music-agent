"""P10.9: Bounded Obsidian projection writer (human-readable knowledge projection only).

The SQLite/domain repositories remain the canonical machine state; this module writes a
deterministic, bounded projection into the EXISTING Vault structure -- never a second
knowledge structure. The contract was fixed by the read-only Vault reconciliation:

- ``首页.md`` ``## 当前状态`` auto-sync status bullet: exact single-line bounded update.
- ``03_Recommendations/推荐历史.md``: append real recommendation record sections before
  the ``## 记录格式`` anchor (the file's own designed use), with an ``rcm_`` replay marker.
- ``04_Feedback/推荐反馈.md``: append real feedback record sections before the
  ``## 推荐记录方式`` anchor, with an ``fbk_`` replay marker.

Hard rules (fail closed, no exceptions):

- Paths resolve deterministically under ``<vault_root>/03_Music/Music_Agent/``; a missing vault
  root, partition, target file, anchor, or exact-match text is an error -- never a guess.
- Every write is computed fully in memory from the pre-write content, written atomically
  (temp file + ``os.replace``), then read back and byte-verified against the expected
  content. On any failure nothing is written.
- Only the approved surfaces exist as API; the USER-OWNED surfaces (01_Library, 02_Profile,
  05_System, rule files) have no write path whatsoever.
- Replay protection: an already-present ``<!-- rcm_run: ... -->`` / ``<!-- fbk_obs: ... -->``
  marker turns the append into a no-op (``replayed=True``), so repeated projections never
  duplicate records.
- The optional "当前尚无...记录。" intro sentence is updated only when it matches exactly
  and no prior record exists; otherwise it is left untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

MUSIC_AGENT_PARTITION = "03_Music/Music_Agent"
_HOMEPAGE_RELATIVE = "首页.md"
_RECOMMENDATIONS_RELATIVE = "03_Recommendations/推荐历史.md"
_FEEDBACK_RELATIVE = "04_Feedback/推荐反馈.md"

_STATUS_SECTION_HEADING = "## 当前状态"
_STATUS_BULLET_PREFIX = "- 自动同步："
_RECOMMENDATIONS_ANCHOR = "## 记录格式"
_FEEDBACK_ANCHOR = "## 推荐记录方式"
_RECOMMENDATIONS_EMPTY_SENTENCE = "当前尚无推荐记录。"
_RECOMMENDATIONS_EMPTY_REPLACEMENT = "已有真实推荐记录，见下方。"
_FEEDBACK_EMPTY_SENTENCE = "当前尚无反馈记录。"
_FEEDBACK_EMPTY_REPLACEMENT = "已有真实反馈记录，见下方。"

_FEEDBACK_FIELD_ORDER = ("评分", "喜欢", "不喜欢", "场景", "是否加入收藏", "对画像的影响")


class ObsidianProjectionError(ValueError):
    code = "obsidian_projection_error"


class ObsidianProjectionValidationError(ObsidianProjectionError):
    code = "validation_error"


class ObsidianProjectionStructureError(ObsidianProjectionError):
    """The Vault does not have the expected structure; nothing was written."""

    code = "obsidian_structure_error"


class ObsidianProjectionVerificationError(ObsidianProjectionError):
    """The post-write readback did not match the expected content."""

    code = "obsidian_verification_error"


@dataclass(frozen=True, slots=True)
class ObsidianProjectionConfig:
    """Deterministic target resolution: one explicit vault root, one fixed partition."""

    vault_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.vault_root, Path) or self.vault_root == Path(""):
            raise ObsidianProjectionValidationError("vault_root must be a non-empty Path")


@dataclass(frozen=True, slots=True)
class RecommendationItem:
    """One recommendation table row (pipes and newlines are refused, not escaped).

    ``feedback`` (P10 addition) is the optional 后续反馈 cell text; when omitted the
    writer renders the default [[03_Music/Music_Agent/04_Feedback/推荐反馈|待反馈]] link.
    """

    song: str
    artist: str
    match_reason: str
    risk: str
    feedback: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("song", self.song),
            ("artist", self.artist),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ObsidianProjectionValidationError(f"{label} must be a non-empty string")
        for label, value in (
            ("song", self.song),
            ("artist", self.artist),
            ("match_reason", self.match_reason),
            ("risk", self.risk),
            ("feedback", self.feedback),
        ):
            if value is None:
                continue
            if not isinstance(value, str):
                raise ObsidianProjectionValidationError(f"{label} must be a string")
            if "\n" in value or "|" in value or "<!--" in value or "-->" in value:
                raise ObsidianProjectionValidationError(
                    f"{label} must not contain newlines, pipes, or comment markers"
                )


@dataclass(frozen=True, slots=True)
class RecommendationRecord:
    """One dated recommendation section (``rcm_`` run identity is the replay key)."""

    run_id: str
    date: str
    theme: str
    items: tuple[RecommendationItem, ...]

    def __post_init__(self) -> None:
        _require_single_line("run_id", self.run_id, prefix="rcm_")
        _require_date(self.date)
        _require_single_line("theme", self.theme)
        if not isinstance(self.items, tuple) or not self.items:
            raise ObsidianProjectionValidationError("items must be a non-empty tuple")
        if not all(isinstance(item, RecommendationItem) for item in self.items):
            raise ObsidianProjectionValidationError("items must be RecommendationItem values")


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """One dated feedback section (``fbk_`` identity is the replay key)."""

    feedback_id: str
    date: str
    song: str
    artist: str
    fields: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _require_single_line("feedback_id", self.feedback_id, prefix="fbk_")
        _require_date(self.date)
        _require_single_line("song", self.song)
        _require_single_line("artist", self.artist)
        if not isinstance(self.fields, Mapping):
            raise ObsidianProjectionValidationError("fields must be a mapping")
        copied: dict[str, str] = {}
        for key, value in self.fields.items():
            if key not in _FEEDBACK_FIELD_ORDER:
                raise ObsidianProjectionValidationError(
                    f"unsupported feedback field {key!r}; expected one of {_FEEDBACK_FIELD_ORDER}"
                )
            if not isinstance(value, str) or "\n" in value:
                raise ObsidianProjectionValidationError(
                    f"feedback field {key!r} must be a single-line string"
                )
            copied[key] = value
        object.__setattr__(self, "fields", MappingProxyType(copied))


class ObsidianSurface(StrEnum):
    HOMEPAGE_STATUS = "homepage_status"
    RECOMMENDATION_HISTORY = "recommendation_history"
    FEEDBACK_HISTORY = "feedback_history"


class ObsidianWriteAction(StrEnum):
    UPDATED = "updated"
    APPENDED = "appended"
    NOOP_REPLAY = "noop_replay"
    NOOP_UNCHANGED = "noop_unchanged"


@dataclass(frozen=True, slots=True)
class ObsidianWriteResult:
    """One verified projection write (or no-op)."""

    path: Path
    surface: ObsidianSurface
    action: ObsidianWriteAction
    verified: bool


class ObsidianProjectionWriter:
    """Writes the three approved projection surfaces; everything else is out of reach."""

    def __init__(self, config: ObsidianProjectionConfig) -> None:
        if not isinstance(config, ObsidianProjectionConfig):
            raise ObsidianProjectionValidationError("config must be an ObsidianProjectionConfig")
        self.config = config

    # --- surface targets (deterministic resolution) -------------------------------

    def _target(self, relative: str) -> Path:
        root = self.config.vault_root
        if not root.is_dir():
            raise ObsidianProjectionStructureError(f"vault root does not exist: {root}")
        partition = root / MUSIC_AGENT_PARTITION
        if not partition.is_dir():
            raise ObsidianProjectionStructureError(
                f"Music Agent partition does not exist: {partition}"
            )
        return partition / relative

    # --- surface 1: homepage auto-sync status bullet ------------------------------

    def update_homepage_autosync_status(self, status_text: str) -> ObsidianWriteResult:
        """Replace exactly the one ``- 自动同步：…`` bullet inside ``## 当前状态``."""
        if not isinstance(status_text, str) or not status_text.strip():
            raise ObsidianProjectionValidationError("status_text must be a non-empty string")
        if "\n" in status_text or "-->" in status_text or "<!--" in status_text:
            raise ObsidianProjectionValidationError("status_text must be a single, safe line")
        path = self._target(_HOMEPAGE_RELATIVE)
        if not path.is_file():
            raise ObsidianProjectionStructureError(f"homepage does not exist: {path}")
        content = path.read_text(encoding="utf-8")
        section = _bounded_section(content, _STATUS_SECTION_HEADING)
        bullets = [line for line in section.splitlines() if line.startswith(_STATUS_BULLET_PREFIX)]
        if not bullets:
            raise ObsidianProjectionStructureError(
                f"no {_STATUS_BULLET_PREFIX!r} bullet found in {_STATUS_SECTION_HEADING}"
            )
        if len(bullets) != 1:
            raise ObsidianProjectionStructureError(
                f"expected exactly one auto-sync bullet, found {len(bullets)}"
            )
        old_line = bullets[0]
        new_line = f"{_STATUS_BULLET_PREFIX}{status_text}"
        if old_line == new_line:
            return ObsidianWriteResult(
                path, ObsidianSurface.HOMEPAGE_STATUS,
                ObsidianWriteAction.NOOP_UNCHANGED, verified=True,
            )
        expected = content.replace(old_line, new_line, 1)
        self._atomic_write(path, expected)
        return ObsidianWriteResult(
            path, ObsidianSurface.HOMEPAGE_STATUS,
            ObsidianWriteAction.UPDATED, verified=True,
        )

    def update_homepage_library_count(self, count: int) -> ObsidianWriteResult:
        """P10.14: bounded homepage surface -- exactly one ``- 同步曲库：N 首`` bullet.

        Long-term knowledge projection of the synced library size (the canonical count,
        never derived from names). Inserts the bullet right after the auto-sync status
        bullet the first time; later runs update only that one bullet. Idempotent.
        """
        if not isinstance(count, int) or count < 0:
            raise ObsidianProjectionValidationError("count must be a non-negative integer")
        path = self._target(_HOMEPAGE_RELATIVE)
        if not path.is_file():
            raise ObsidianProjectionStructureError(f"homepage does not exist: {path}")
        content = path.read_text(encoding="utf-8")
        section = _bounded_section(content, _STATUS_SECTION_HEADING)
        prefix = "- 同步曲库："
        existing = [line for line in section.splitlines() if line.startswith(prefix)]
        if len(existing) > 1:
            raise ObsidianProjectionStructureError(
                f"expected at most one library-count bullet, found {len(existing)}"
            )
        new_line = f"{prefix}{count} 首"
        if existing:
            old_line = existing[0]
            if old_line == new_line:
                return ObsidianWriteResult(
                    path, ObsidianSurface.HOMEPAGE_STATUS,
                    ObsidianWriteAction.NOOP_UNCHANGED, verified=True,
                )
            expected = content.replace(old_line, new_line, 1)
            self._atomic_write(path, expected)
            return ObsidianWriteResult(
                path, ObsidianSurface.HOMEPAGE_STATUS,
                ObsidianWriteAction.UPDATED, verified=True,
            )
        # First run: insert after the auto-sync status bullet (the P10.9 surface).
        status_bullets = [
            line for line in section.splitlines() if line.startswith(_STATUS_BULLET_PREFIX)
        ]
        if len(status_bullets) != 1:
            raise ObsidianProjectionStructureError(
                "the auto-sync status bullet must exist exactly once before adding the library count"
            )
        anchor = status_bullets[0]
        expected = content.replace(anchor, anchor + "\n" + new_line, 1)
        self._atomic_write(path, expected)
        return ObsidianWriteResult(
            path, ObsidianSurface.HOMEPAGE_STATUS,
            ObsidianWriteAction.UPDATED, verified=True,
        )

    # --- surface 2: recommendation history append ---------------------------------

    def append_recommendation_record(self, record: RecommendationRecord) -> ObsidianWriteResult:
        """Append one dated recommendation section before the ``## 记录格式`` anchor."""
        if not isinstance(record, RecommendationRecord):
            raise ObsidianProjectionValidationError("record must be a RecommendationRecord")
        path = self._target(_RECOMMENDATIONS_RELATIVE)
        if not path.is_file():
            raise ObsidianProjectionStructureError(f"recommendation history does not exist: {path}")
        content = path.read_text(encoding="utf-8")
        marker = _recommendation_marker(record.run_id)
        if marker in content:
            return ObsidianWriteResult(
                path, ObsidianSurface.RECOMMENDATION_HISTORY,
                ObsidianWriteAction.NOOP_REPLAY, verified=True,
            )
        block = _render_recommendation_block(record, marker)
        expected = _insert_before_anchor(content, _RECOMMENDATIONS_ANCHOR, block)
        if not _any_recommendation_marker(content):
            expected = expected.replace(
                _RECOMMENDATIONS_EMPTY_SENTENCE, _RECOMMENDATIONS_EMPTY_REPLACEMENT, 1
            )
        self._atomic_write(path, expected)
        return ObsidianWriteResult(
            path, ObsidianSurface.RECOMMENDATION_HISTORY,
            ObsidianWriteAction.APPENDED, verified=True,
        )

    # --- surface 3: feedback history append ---------------------------------------

    def append_feedback_record(self, record: FeedbackRecord) -> ObsidianWriteResult:
        """Append one dated feedback section before the ``## 推荐记录方式`` anchor."""
        if not isinstance(record, FeedbackRecord):
            raise ObsidianProjectionValidationError("record must be a FeedbackRecord")
        path = self._target(_FEEDBACK_RELATIVE)
        if not path.is_file():
            raise ObsidianProjectionStructureError(f"feedback history does not exist: {path}")
        content = path.read_text(encoding="utf-8")
        marker = _feedback_marker(record.feedback_id)
        if marker in content:
            return ObsidianWriteResult(
                path, ObsidianSurface.FEEDBACK_HISTORY,
                ObsidianWriteAction.NOOP_REPLAY, verified=True,
            )
        block = _render_feedback_block(record, marker)
        expected = _insert_before_anchor(content, _FEEDBACK_ANCHOR, block)
        if not _any_feedback_marker(content):
            expected = expected.replace(
                _FEEDBACK_EMPTY_SENTENCE, _FEEDBACK_EMPTY_REPLACEMENT, 1
            )
        self._atomic_write(path, expected)
        return ObsidianWriteResult(
            path, ObsidianSurface.FEEDBACK_HISTORY,
            ObsidianWriteAction.APPENDED, verified=True,
        )

    # --- atomic write + readback verification -------------------------------------

    def _atomic_write(self, path: Path, expected: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(expected, encoding="utf-8")
            os.replace(temporary, path)
        except OSError as error:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise ObsidianProjectionError(f"could not write {path}: {error}") from error
        if path.read_text(encoding="utf-8") != expected:
            raise ObsidianProjectionVerificationError(f"readback verification failed for {path}")


# --- pure renderers / helpers -------------------------------------------------------


def _require_single_line(label: str, value: str, *, prefix: str | None = None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ObsidianProjectionValidationError(f"{label} must be a non-empty string")
    if "\n" in value or "<!--" in value or "-->" in value:
        raise ObsidianProjectionValidationError(f"{label} must be a single, safe line")
    if prefix is not None and not value.startswith(prefix):
        raise ObsidianProjectionValidationError(f"{label} must start with {prefix!r}")


def _require_date(value: str) -> None:
    _require_single_line("date", value)
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise ObsidianProjectionValidationError("date must be YYYY-MM-DD") from error


def _bounded_section(content: str, heading: str) -> str:
    """The text of one ``## `` section (heading line until the next same-or-higher heading)."""
    lines = content.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.strip() == heading), None)
    if start is None:
        raise ObsidianProjectionStructureError(f"section {heading!r} not found")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return "".join(lines[start + 1 : end])


def _insert_before_anchor(content: str, anchor: str, block: str) -> str:
    """Insert ``block`` immediately before the anchor heading; the anchor must exist once."""
    count = content.count(anchor + "\n")
    if count == 0:
        raise ObsidianProjectionStructureError(f"anchor {anchor!r} not found")
    if count > 1:
        raise ObsidianProjectionStructureError(
            f"anchor {anchor!r} appears {count} times; expected exactly once"
        )
    index = content.index(anchor + "\n")
    return content[:index] + block + content[index:]


def _recommendation_marker(run_id: str) -> str:
    return f"<!-- rcm_run: {run_id} -->"


def _feedback_marker(feedback_id: str) -> str:
    return f"<!-- fbk_obs: {feedback_id} -->"


def _any_recommendation_marker(content: str) -> bool:
    return "<!-- rcm_run: " in content


def _any_feedback_marker(content: str) -> bool:
    return "<!-- fbk_obs: " in content


def _render_recommendation_block(record: RecommendationRecord, marker: str) -> str:
    lines = [
        f"### {record.date}｜{record.theme}",
        marker,
        "",
        "| 歌曲 | 艺人 | 匹配理由 | 潜在风险 | 后续反馈 |",
        "|---|---|---|---|---|",
    ]
    for item in record.items:
        feedback_cell = item.feedback or "[[03_Music/Music_Agent/04_Feedback/推荐反馈|待反馈]]"
        lines.append(
            f"| {item.song} | {item.artist} | {item.match_reason} | {item.risk} | "
            f"{feedback_cell} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_feedback_block(record: FeedbackRecord, marker: str) -> str:
    lines = [
        f"### {record.date}｜{record.song} — {record.artist}",
        marker,
        "",
    ]
    for key in _FEEDBACK_FIELD_ORDER:
        value = record.fields.get(key, "")
        lines.append(f"- {key}：{value}")
    lines.append("")
    return "\n".join(lines) + "\n"
