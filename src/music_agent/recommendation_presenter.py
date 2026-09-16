"""P20 Quality Fix 10: deterministic user-facing rendering of a successful
recommendation batch.

Once ``generate_recommendation`` / ``generate_inferred_recommendation`` has
succeeded, the user-visible list -- and each item's one-line reason -- is
rendered HERE, from the authoritative generation payload (the P20-Fix09
evidence-carrying item projection), never from provider free text. The module
is a near-pure function layer: no LLM calls, no catalog reads, no re-ranking,
no encyclopedia knowledge -- the order is the persisted selected order and the
only facts consumed are each item's name/artist/playback fact/fresh identity
plus its shared evidence block.

Evidence-to-copy mapping (mandate §八, A--F):

- direct provenance with a reliable label: specific copying per basis kind
  (genre / the track itself / another track / an artist); the generic
  「这首来自你已有的明确偏好。」 remains the copy for a direct row with no
  specific shape, and inferred provenance is never written as direct;
- inferred genre basis: 「这首按 {label} 方向推断出来。」;
- inferred track-self basis (the Fix09 definition: basis 是曲目自身, label
  is this item's own name): 「这首按对这首曲目本身的偏好推断递选。」;
- fresh + evidence: 「这是本次新发现，{reason}。」;
- fresh + no direct matching evidence: 「这是本次新发现，目前没有更直接的
  偏好匹配证据。」;
- a non-fresh item is never called 本次新发现.

The module never emits internal terms (novel / fresh=true / mechanism /
provenance / basis / score fields / candidate / run ids), never expands into
music criticism, and never references the current-playing track.

P20 Quality Fix 11 builds on this same layer: a closed explanation request
(为什么推荐这些？ etc.) is answered deterministically from the authoritative
recommendation RUN (``get_recommendation_run`` payload) --
``render_recommendation_explanation_for_user`` below reuses the SAME
evidence-to-copy mapping (each item's explanation reason is fact-identical to
its first-presentation line, mandate §八) and adds a program-computed
direction summary over the batch's real genre evidence (mandate §六/§七:
three directions are three, track-self and no-evidence items never join a
genre count). Zero provider rounds -- no SSL exposure, no planning narration,
no re-querying the persisted preference evidence.

Strict fail-safe: any payload that does not match the post-Fix09 item contract
(missing ``items``, a non-string ``name``, or a missing ``evidence`` block)
renders ``None`` -- the caller replies with the fixed fail-honest sentence
instead of a guessed reason (still closed by the P20-Fix08 final-response
boundary). This keeps legacy replayed result shapes and every other
free-text scene on their existing honest paths.
"""

from __future__ import annotations

from collections.abc import Mapping


_DIRECT_TRACK_SELF = "来自你对这首曲目本身的已有偏好"
_INFERRED_TRACK_SELF = "按对这首曲目本身的偏好推断递选"
_DIRECT_GENERIC = "来自你已有的明确偏好"

_NO_DIRECT_EVIDENCE_FRESH = "这是本次新发现，目前没有更直接的偏好匹配证据。"
_NO_DIRECT_EVIDENCE_PLAIN = "这首目前没有更直接的偏好匹配证据。"

_PREVIEW_ONLY_CUE = "这批曲目都只能试听 30 秒。需要试听哪一首，直接告诉我。"
_GENERIC_CUE = "需要试听哪一首，直接告诉我。"
_SIMILARITY_MECHANISM = "曲目元数据相似"

# The multi-direction summary speaks the count as a Chinese numeral (mandate
# §六: 「…Mandopop 三个方向」) -- mapped deterministically up to ten, plain
# digits beyond (a batch that wide is not a real product shape).
_DIRECTION_COUNT_WORDS = {
    2: "两",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
}


def _direction_count_word(count: int) -> str:
    return _DIRECTION_COUNT_WORDS.get(count, str(count))


def render_recommendation_for_user(payload: Mapping | None) -> str | None:
    """The complete deterministic batch presentation, or None (fail-safe).

    Layout: the item-count header, the ordered items each with its one reason
    line, and the closing playback cue -- blank-line separated.
    """
    items = _validated_items(payload)
    if items is None:
        return None
    lines = render_recommendation_items(payload)
    if lines is None:
        return None
    cue = render_recommendation_cue(payload)
    if cue is None:
        return None
    return "\n\n".join((f"为你推荐这 {len(items)} 首：", lines, cue))


def render_recommendation_items(payload: Mapping | None) -> str | None:
    """The ordered items block (position. name — artist + reason), or None.

    Used by the full renderer above and by the Fix05 direction-shift fast paths
    (whose own note line stands in for the header). Order is exactly the
    payload's selected order; an item-less/legacy payload renders None.
    """
    items = _validated_items(payload)
    if items is None:
        return None
    blocks: list[str] = []
    for position, item in enumerate(items, start=1):
        line = f"{position}. {item['name']}"
        artist = item.get("artist_name")
        if isinstance(artist, str) and artist.strip():
            line += f" — {artist}"
        blocks.append(line + "\n" + _reason_for_item(item))
    return "\n\n".join(blocks)


def render_recommendation_cue(payload: Mapping | None) -> str | None:
    """The deterministic closing cue, or None when the payload is unusable.

    An all-preview_only batch carries the 30-second preview fact; everything
    else (library-playable, mixed, or no playback fact) gets the plain cue.
    """
    items = _validated_items(payload)
    if items is None:
        return None
    routes: set[str] = set()
    for item in items:
        playback = item.get("playback")
        route = playback.get("route") if isinstance(playback, Mapping) else None
        if isinstance(route, str):
            routes.add(route)
    return _PREVIEW_ONLY_CUE if routes == {"preview_only"} else _GENERIC_CUE


def render_recommendation_explanation_for_user(
    payload: Mapping | None,
) -> str | None:
    """P20 Fix 11: the deterministic batch explanation, or None (fail-safe).

    A closed explanation request (为什么推荐这些？ …) is answered from the
    authoritative recommendation RUN (the ``get_recommendation_run`` payload),
    never from provider free text. Layout: the program-computed direction
    summary over the batch's real genre evidence, then the ordered items each
    carrying the SAME evidence-derived reason copy as the first presentation
    (mandate §八 reuse -- one vocabulary, fact-identical per item). No
    playback cue, no current-playing reference, no score, no encyclopedia:
    the only facts consumed are each item's name/artist and its shared
    evidence block, plus the durable run reader's fresh-identity note for a
    zero-basis this-request discovery. A payload outside the post-Fix09 item
    contract renders None: the caller replies with its fixed fail-honest
    sentence instead (a guessed reason never reaches the user).
    """
    items = _validated_items(payload)
    if items is None:
        return None
    blocks: list[str] = []
    for position, item in enumerate(items, start=1):
        line = f"{position}. {item['name']}"
        artist = item.get("artist_name")
        if isinstance(artist, str) and artist.strip():
            line += f" — {artist}"
        blocks.append(line + "\n" + _explanation_reason_for_item(item))
    body = "\n\n".join(blocks)
    summary = _explanation_direction_summary(items)
    if summary is None:
        return body
    return "\n\n".join((summary, body))


def _validated_items(payload: Mapping | None) -> list[Mapping] | None:
    """The payload's items under the post-Fix09 display contract, or None.

    Every item must be a mapping with a non-empty string ``name`` and an
    ``evidence`` mapping -- the two facts the deterministic reason can never
    invent. Anything else (including legacy replayed shapes with no evidence
    block) fails closed so the caller keeps its ordinary text path.
    """
    if not isinstance(payload, Mapping):
        return None
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return None
    validated: list[Mapping] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(item.get("evidence"), Mapping):
            return None
        validated.append(item)
    return validated


def _reason_for_item(item: Mapping) -> str:
    """One item's deterministic reason line from its evidence block alone."""
    similarity_reason = _similarity_reason(item)
    if similarity_reason is not None:
        return similarity_reason
    fresh = item.get("fresh_this_request") is True
    clauses = [
        clause
        for clause in (_clause_for_basis_row(entry, item["name"])
                       for entry in _basis_rows(item))
        if clause
    ]
    if not clauses:
        return _NO_DIRECT_EVIDENCE_FRESH if fresh else _NO_DIRECT_EVIDENCE_PLAIN
    reason = "；".join(clauses)
    return f"这是本次新发现，{reason}。" if fresh else f"这首{reason}。"


def _similarity_reason(item: Mapping) -> str | None:
    """Explain only code-owned shared categorical metadata for similarity items."""

    evidence = item.get("evidence")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("mechanism") != _SIMILARITY_MECHANISM
    ):
        return None
    clauses: list[str] = []
    for entry in _basis_rows(item):
        label = entry.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        kind = entry.get("kind")
        if kind == "genre":
            clauses.append(f"流派「{label}」")
        elif kind == "artist":
            clauses.append(f"艺人「{label}」")
        elif kind == "composer":
            clauses.append(f"作曲者「{label}」")
        elif kind == "tag":
            clauses.append(f"标签「{label}」")
    if not clauses:
        return "这首没有足够的可说明元数据相似证据。"
    seed = evidence.get("seed")
    seed_name = seed.get("name") if isinstance(seed, Mapping) else None
    if isinstance(seed_name, str) and seed_name.strip():
        return f"这首与《{seed_name.strip()}》共享" + "、".join(clauses) + "。"
    return "这首与相似度 seed 共享" + "、".join(clauses) + "。"


def _basis_rows(item: Mapping) -> list[Mapping]:
    evidence = item.get("evidence")
    basis = evidence.get("basis") if isinstance(evidence, Mapping) else None
    if not isinstance(basis, list):
        return []
    return [entry for entry in basis if isinstance(entry, Mapping)]


def _clause_for_basis_row(entry: Mapping, item_name: str) -> str | None:
    """One basis row mapped to its reason clause (no trailing punctuation).

    Only the three real basis kinds the shared Fix09 evidence builder emits --
    genre / track / artist -- have specific copy. A direct row of an unknown
    kind keeps the generic direct sentence (mandate §八 A), an inferred row of
    an unknown kind is dropped (its specifics cannot be honestly claimed).
    Track-self is the row whose label IS this item's own name (the builder's
    own definition of basis 曲目自身).
    """
    kind = entry.get("kind")
    if not isinstance(kind, str):
        return None
    label = entry.get("label")
    if not isinstance(label, str) or not label.strip():
        return None
    direct = entry.get("provenance") == "直接"
    is_self = kind == "track" and label == item_name
    if kind == "genre":
        if direct:
            return f"来自你对 {label} 方向的已有偏好"
        return f"按 {label} 方向推断出来"
    if kind == "track":
        if direct:
            return _DIRECT_TRACK_SELF if is_self else f"来自你对《{label}》的已有偏好"
        return _INFERRED_TRACK_SELF if is_self else f"根据你对《{label}》的偏好推断选入"
    if kind == "artist":
        if direct:
            return f"来自你对「{label}」的已有偏好"
        return f"按「{label}」的艺人偏好推断出来"
    if direct:
        return _DIRECT_GENERIC
    return None


def _explanation_reason_for_item(item: Mapping) -> str:
    """One item's explanation reason -- the Fix10 first-presentation copy.

    A batch-history item carries no per-item ``fresh_this_request`` flag, so
    freshness is only restored from the evidence note the durable run reader
    records for zero-basis this-request discoveries (「本次目录搜索的新发现
    …」): that item reads the same honest fresh sentence its first view showed.
    Items with recorded evidence reuse the exact presentation clause copy --
    the fact layer of the explanation is identical to the first view by
    construction, never re-derived.
    """
    if _basis_rows(item):
        return _reason_for_item(item)
    evidence = item.get("evidence")
    note = evidence.get("note") if isinstance(evidence, Mapping) else None
    if isinstance(note, str) and note.strip():
        return _NO_DIRECT_EVIDENCE_FRESH
    return _NO_DIRECT_EVIDENCE_PLAIN


def _explanation_direction_summary(items: list[Mapping]) -> str | None:
    """The program-computed direction summary, or None (no genre evidence).

    Only genre-kind basis rows are counted (mandate §六/§七): the summary is
    derived from the batch's real direction evidence -- three directions are
    three, never "two". Track-self and no-evidence items never join a genre
    count; they keep their own per-item reasons. A batch with no genre rows at
    all gets no direction claim at all: the items alone are the explanation.
    One direction shared by every item reads the all-batch sentence; one
    direction alongside items with no genre row, and two-or-more directions,
    read the honest 主要来自 form.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    all_genre = True
    for item in items:
        evidence = item.get("evidence")
        if (
            isinstance(evidence, Mapping)
            and evidence.get("mechanism") == _SIMILARITY_MECHANISM
        ):
            all_genre = False
            continue
        labels: list[str] = []
        for entry in _basis_rows(item):
            if entry.get("kind") != "genre":
                continue
            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            label = label.strip()
            labels.append(label)
            if label not in seen:
                seen.add(label)
                ordered.append(label)
        if not labels:
            all_genre = False
    if not ordered:
        return None
    if len(ordered) == 1 and all_genre:
        return f"这一批 {len(items)} 首都来自 {ordered[0]} 方向。"
    if len(ordered) == 1:
        return f"这一批主要来自 {ordered[0]} 方向。"
    joined = "、".join(ordered[:-1]) + " 和 " + ordered[-1]
    count_word = _direction_count_word(len(ordered))
    return f"这一批主要来自 {joined} {count_word}个方向："
