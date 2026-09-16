"""P20 Quality Fix 05: deterministic direction-shift semantics for re-recommendation.

Until now 「再来一批，换个方向」 reached the provider loop as a plain
re-recommendation and the model answered with ``generate_recommendation``/
``generate_inferred_recommendation`` carrying only ``exclude_target_ids`` --
"a different batch" instead of "a different DIRECTION". This module defines the
closed, pure, provider-free pieces of the real direction-shift behavior:

- the closed phrase set that requests a shift (deterministic recognition, same
  whole-line discipline as the routing table -- a near-miss never shifts);
- the explicit-direction mapping (日系 → J-Pop 等, plus the verified canonical
  genre-key passthrough) so 「换成日系」/「来点摇滚」 travel the same
  deterministic path instead of another model guess;
- the deterministic replacement-direction picker: from the previous batch's
  DURABLE genre basis (``get_recommendation_run`` items' ``evidence.basis``,
  never the model's song-name guessing) and the user's real positive direction
  evidence (the sealed P10 genre-affinity reducer, same machinery the inferred
  recommendation channel already trusts), the picker selects one different
  real direction or fails honest with ``None`` -- it never fabricates a
  direction the user has no positive evidence for.

Everything here is pure and stateless: nothing is read, routed or executed.
The executor (``direction_coach.run_direction_shift``) drives the actual tool
calls on the caller's surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

# Same normalization discipline as intent_router: whole-line exact after
# strip + optional trailing punctuation + casefold. Near-misses refuse.
_TRAILING_PUNCTUATION = "。．？！?！!·…"


def _normalize(text: str) -> str:
    return text.strip().strip(_TRAILING_PUNCTUATION).strip().casefold()


# The closed direction-shift phrase set. The four re-recommend verbs with the
# two direction tails (comma-separated and run-together spellings both land
# here after normalization), plus the bare shift lines. Deliberately absent:
# any verb + other tail (再来一批好听的 -- that stays the ordinary full-batch
# re-recommend), so a shift is only ever recognized for the exact set.
_SHIFT_VERBS: tuple[str, ...] = ("再来一批", "再推荐一批", "换一批", "换一组")
_SHIFT_TAILS: tuple[str, ...] = ("换个方向", "换方向")

DIRECTION_SHIFT_PHRASES: frozenset[str] = frozenset(
    [*[f"{verb}{tail}" for verb in _SHIFT_VERBS for tail in _SHIFT_TAILS],
     *[f"{verb}，{tail}" for verb in _SHIFT_VERBS for tail in _SHIFT_TAILS],
     "换个方向", "换方向", "换一种风格", "来点不一样的"],
)


def is_direction_shift_request(text: object) -> bool:
    """True when ``text`` is a closed whole-line direction-shift form (Fix05).

    Pure classification mirroring ``intent_router``: exact whole-line match
    after normalization; anything else (extra words, near-misses) is False.
    """
    if not isinstance(text, str):
        return False
    return _normalize(text) in DIRECTION_SHIFT_PHRASES


# Explicit-direction words the system supports deterministically. The mapping
# is closed and grounded: 日系→J-Pop is the established prompt rule, and
# J-Pop/Rock are verified canonical genre keys of the real store; the
# passthrough set only contains canonical genre keys that exist in the store
# (case is preserved by the frozen canonicalizer, so only exact spellings
# pass through). An unknown word is refused (None) and stays the provider
# loop's own direction-word rule (平缓 → search terms etc.).
DIRECTION_WORD_GENRES: dict[str, str] = {
    "日系": "J-Pop",
    "日语": "J-Pop",
    "日文": "J-Pop",
    "摇滚": "Rock",
    "摇滚乐": "Rock",
}

_DIRECTION_GENRE_KEY_PASSTHROUGH: frozenset[str] = frozenset(
    (
        "J-Pop", "Rock", "Mandopop", "K-Pop", "Pop", "Alternative",
    )
)

# classify() casefolds the WHOLE line before mapping, so the passthrough must
# match casefolded spellings back to the verified canonical keys; the alias
# table stays exact (its words carry no case).
_DIRECTION_GENRE_KEY_PASSTHROUGH_BY_FOLD: dict[str, str] = {
    key.casefold(): key for key in _DIRECTION_GENRE_KEY_PASSTHROUGH
}


def map_direction_word(word: object) -> str | None:
    """Map one explicit direction word to a canonical genre key, or None.

    Only the closed alias table and the verified genre-key passthrough
    (case-insensitively -- the classifier casefolds the line first) map;
    everything else refuses (None) so the caller remits to the provider loop
    and its existing direction-word rule -- never an invented genre.
    """
    if not isinstance(word, str):
        return None
    stripped = word.strip()
    if not stripped:
        return None
    if stripped in DIRECTION_WORD_GENRES:
        return DIRECTION_WORD_GENRES[stripped]
    key = _DIRECTION_GENRE_KEY_PASSTHROUGH_BY_FOLD.get(stripped.casefold())
    if key is not None:
        return key
    return None


@dataclass(frozen=True, slots=True)
class DirectionRequest:
    """One classified direction request (Fix05)."""

    kind: str  # "shift" (direction switch) or "explicit" (user-named direction)
    genre: str | None  # the mapped canonical genre key for "explicit" requests


_EXPLICIT_DIRECTION_RE: re.Pattern[str] = re.compile(r"^(?:换成|来点)\s*(\S.+)$")


def classify_direction_request(text: object) -> DirectionRequest | None:
    """Classify ``text`` as a direction request: shift / explicit / None (Fix05).

    The closed shift set wins first (来点不一样的 is a shift, not a direction
    word); then the open 换成/来点<word> form maps through
    :func:`map_direction_word` -- a mapped word becomes an explicit-direction
    request, an unmapped word (平缓, 轻松的, ...) stays the provider loop's
    surface and yields None. Pure; nothing is executed here.
    """
    if not isinstance(text, str):
        return None
    normalized = _normalize(text)
    if normalized in DIRECTION_SHIFT_PHRASES:
        return DirectionRequest(kind="shift", genre=None)
    match = _EXPLICIT_DIRECTION_RE.match(normalized)
    if match is None:
        return None
    mapped = map_direction_word(match.group(1))
    if mapped is None:
        return None
    return DirectionRequest(kind="explicit", genre=mapped)


@dataclass(frozen=True, slots=True)
class ShiftDecision:
    """One deterministic replacement-direction decision."""

    selected: str | None  # the canonical genre key to generate in; None = fail-honest
    excluded_genres: frozenset[str]  # the previous batch's main direction(s)


def pick_shifted_direction(
    batch_genre_counts: Mapping[str, int],
    positive_directions: tuple[str, ...],
) -> ShiftDecision:
    """Pick one real replacement direction, or fail honest with ``None``.

    ``batch_genre_counts`` maps the previous batch's DURABLE genre basis labels
    to their item counts (from ``get_recommendation_run`` items' evidence);
    ``positive_directions`` is the user's real positive direction evidence in
    preference order (the sealed genre-affinity reducer, strongest first --
    the caller supplies the order, this function only preserves it).

    Policy (simple, deterministic, testable -- no new scoring):

    * the previous batch's main direction(s) = the genre(s) with the maximum
      basis count; those become ``excluded_genres``;
    * candidates are the positive directions ABSENT from the batch first, then
      the positive directions present below the main count -- both in the
      caller's strength order, so the strongest real alternative wins and the
      choice never jumps between otherwise-equal candidates;
    * a mixed batch (no unique main) naturally picks a low-share or absent
      direction the same way;
    * when no positive direction survives (a single fake-proof direction that
      IS the batch's main direction, or no positive evidence at all), the
      decision fails honest: ``selected`` is None and the caller must ask the
      user instead of generating the same direction again.
    """
    counts = {
        genre: count
        for genre, count in batch_genre_counts.items()
        if isinstance(genre, str) and isinstance(count, int) and count > 0
    }
    max_count = max(counts.values(), default=0)
    excluded = frozenset(
        genre for genre, count in counts.items() if count == max_count
    ) if max_count > 0 else frozenset()
    absent_winner: str | None = None
    present_alternative: str | None = None
    for genre in positive_directions:
        if not isinstance(genre, str) or not genre or genre in excluded:
            continue
        if genre in counts:
            # Present in the batch below the main share: a legitimate
            # alternative, but any ABSENT positive direction outranks it.
            if present_alternative is None:
                present_alternative = genre
            continue
        absent_winner = genre  # first absent positive direction wins outright
        break
    selected = absent_winner if absent_winner is not None else present_alternative
    return ShiftDecision(selected=selected, excluded_genres=excluded)


def shifted_direction_note(selected: str) -> str:
    """One honest presentation line for a shift that happened (Fix05)."""
    return f"已换到「{selected}」方向（依据你偏好记录里真实存在的 {selected} 证据）。"


def explicit_direction_note(selected: str) -> str:
    """One honest presentation line for an explicit user-named direction (Fix05)."""
    return f"已按「{selected}」方向推荐。"


# The fail-honest reply for a shift without any other real direction (§九).
# The caller must NOT generate anything when this is the answer: no
# fabrication, no silently regenerating the previous direction.
NO_ALTERNATIVE_DIRECTION_REPLY = (
    "目前从你的偏好记录里还没有足够明确的另一个方向。"
    "你想换成日系、摇滚，还是给我一个方向？"
)