"""chat-session 控制指令快路由：本地精确意图映射（第三阶段 B 快速通道）。

Pure, provider-free mapping from one user line to a P09 tool name. Only exact,
unambiguous control commands are routed straight to a direct tool call; every
other input returns ``None`` and falls back to the full ``ProviderAgentLoop``.

P14-C06.3a/b: the context-sensitive forms below gained ActiveMusicContext-aware
routing. Their decision needs the live register + ownership judgment + runner
truth, which the caller supplies through optional keyword arguments -- the
function itself stays pure and stateless. ``channel`` is the service's last
audio action (``"none"``/``"library"``/``"preview"``, from get_active_context);
``context`` is the ownership judgment (``"agent_selected"``/``"own_queue"``/
``"unknown"``); ``preview_sounding`` is the real runner read (``is_preview_active``,
the 3b endpoint field). P15-S1 adds ``preview_session_state`` -- the live
continuous-preview session state from ``get_playback_context`` (None = none).
The defaults (``None`` = unknown, ``False`` = not sounding) are exactly the
pre-C06.3a behavior: the context-sensitive forms refuse to route rather than
guess (宁可不路由，不误触发).

Context-sensitive rules -- each mirrors the provider prompt, which stays
authoritative for everything else:

- ``stop`` / ``停止`` routes to ``stop_preview`` when ``preview_sounding``
  is True -- the real runner truth (read-only ``is_preview_active``), not the
  channel action log -- or when a continuous preview session is RUNNING (which
  also covers the inter-clip gaps where nothing is sounding yet). ``channel ==
  "preview"`` no longer routes on its own: a preview may have ended naturally
  (or the channel may say ``none`` while audio is still sounding), and only the
  runner knows. ``stop_preview`` is idempotent, so the truth read and the
  execution can race harmlessly.
- ``换一首`` routes to ``next_track`` only when ``channel == "none"`` AND
  ``context == "own_queue"`` -- the single combination the provider prompt
  allows for next_track (cutting through the user's own queue). Every other
  combination (preview/library channel, agent_selected/unknown ownership) needs
  the model to pick a concrete next track, so it falls back to the provider loop.

P15-S1 session-aware additions (the same caller-supplied-truth shape): the
third context token ``preview_session_state`` holds the live continuing-preview
session state ("running"/"completed"/"cancelled"; None = no session -- the
default stays the exact pre-P15 behavior, session awareness is conditional):
- While a session is RUNNING, ``暂停``/``pause`` routes to ``stop_preview``
  (暂停 ≙ 停止试听 -- afplay has no mid-clip pause, decided honestly). Off any
  session, both keep their pre-P15 route to the plain ``pause``.
- The continue family (``继续``/``继续播放``/``continue``/``play``/``resume``)
  keeps routing to ``play`` under every session state: the *caller* intercepts a
  play route while a session is RUNNING and reports progress instead of
  executing (resuming Music.app over sounding preview audio is forbidden by the
  single-audio-source rule). The router never blocks them itself.

P15-S1 C02 (interaction-boundary fixes from the 真机 round):
- While a session is RUNNING, ``下一首``/``next``/``下一首试听`` all route to
  ``advance_preview`` -- one in-place skip of the live session (never a new
  session, never a cancel, never Music.app).
- Off a session, ``下一首``/``next`` keep their pre-P15 ``next_track`` route and
  ``下一首试听`` refuses (None) so the provider loop decides: the router never
  invents a session. The three forms join the context-sensitive set above
  because their outcome now depends on the live session state.

Deliberately closed and exact -- no fuzzy matching, no prefixes, no keyword
detection. A false positive here would execute a playback command the user
never issued, so the table is intentionally narrower than the provider prompt
rules (which remain authoritative for everything else).

P15-S4-M3-B adds the V1 playback-status phrase set: exactly five whole-line
forms route to the ``playback_status`` pseudo-command. ``playback_status`` is a
CLI-side marker, NOT an ``AgentToolName`` -- the chat-session caller
intercepts it before any tool dispatch and renders the answer by itself (one
``get_playback_context`` read, deterministic formatting, zero provider
rounds); it must never be forwarded to ``loop.client.call``. The status forms
do NOT join ``_CONTEXT_SENSITIVE_FORMS``: their answer needs no pre-fetched
context tokens, the CLI reads the authoritative observation directly.

Matching also now tolerates trailing punctuation (。．？！?！!·…, stripped from
both ends after whitespace) so the natural ``现在在播放什么？`` hits its entry.
Anything beyond optional surrounding whitespace/punctuation is still refused.
Keep every other tool-name string in sync with ``AgentToolName`` values in
``agent_tools.py``.

P16-S3 adds the V1 formal-playback phrase set (播放一首正式歌曲 等): exactly
five whole-line forms route to the ``formal_play`` pseudo-command. Like
``playback_status`` it is a CLI-side marker, NOT an ``AgentToolName`` -- the
chat-session caller intercepts it before any tool dispatch and runs the full
deterministic chain (locate the active batch -> pick the first library-routed
item -> play_track -> get_now_playing verification -> fixed answer) through the
same P09 client, zero provider rounds. The formal forms do NOT join
``_CONTEXT_SENSITIVE_FORMS`` either: the runner reads its own live truth.
Deliberately excluded from the set: the delegation family (随便播放一首 etc.)
stays with the provider loop -- it permits a preview fallback the deterministic
runner refuses, so stealing it would change behavior instead of just latency.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

# Optional trailing punctuation tolerated on the exact match (both ends, after
# whitespace strip). Enables 现在在播放什么？/暂停。 without any fuzzy matching.
_TRAILING_PUNCTUATION = "。．？！?！!·…"


def _normalize(text: str) -> str:
    return text.strip().strip(_TRAILING_PUNCTUATION).strip().casefold()


_DELEGATED_PLAYBACK_FORMS: frozenset[str] = frozenset(
    {
        "你来决定",
        "你选",
        "随便",
        "随便播放一首",
        "放首歌",
    }
)


def is_delegated_playback_intent(text: str) -> bool:
    """True only for the existing explicit play-or-preview delegation forms.

    The set preserves the prompt's existing explicit delegation vocabulary but
    does not accept longer or merely similar sentences. It authorizes one
    selection plus its playback/preview action; it does not authorize a broader
    task or a second recommendation generation.
    """
    return isinstance(text, str) and _normalize(text) in _DELEGATED_PLAYBACK_FORMS


@dataclass(frozen=True, slots=True)
class PreferenceStatementTurnSemantics:
    """Explicit artist/catalog-level preference statement for this turn.

    This is deliberately narrower than track feedback. Forms such as
    ``我喜欢这首歌`` stay in the feedback family, while ``我喜欢 X 的歌``
    means the user is expressing a preference about X's body of music and is
    NOT asking for a recommendation batch. P20 stabilizes that distinction
    without inventing an artist-level persistence capability that the current
    Agent surface does not expose.
    """

    polarity: str
    target: str


_PREFERENCE_STATEMENT_RE = re.compile(
    r"^我\s*(?P<verdict>喜欢|不喜欢|讨厌)\s*"
    r"(?P<target>[^，,。！？!?；;]+?)\s*的(?:音乐|歌曲|歌)$",
    re.IGNORECASE,
)


def resolve_preference_statement_turn_semantics(
    text: str,
) -> PreferenceStatementTurnSemantics | None:
    """Parse a collection-level preference statement, never a recommendation."""
    if not isinstance(text, str):
        return None
    value = text.strip().strip(_TRAILING_PUNCTUATION).strip()
    if not value:
        return None
    # A preference seed plus an explicit recommendation clause is a
    # recommendation turn, not a standalone preference statement. Keep this
    # parser from stealing forms such as
    # ``我喜欢 IU，推荐几首适合晚上听的歌`` before recommendation routing.
    if resolve_recommendation_turn_semantics(text) is not None:
        return None
    match = _PREFERENCE_STATEMENT_RE.fullmatch(value)
    if match is None:
        return None
    target = match.group("target").strip()
    if not target:
        return None
    verdict = match.group("verdict")
    polarity = "positive" if verdict == "喜欢" else "negative"
    return PreferenceStatementTurnSemantics(polarity=polarity, target=target)


@dataclass(frozen=True, slots=True)
class RecommendationTurnSemantics:
    """Explicit recommendation meaning carried by this user turn.

    ``mode`` is one of ``generic``, ``artist_constraint``,
    ``similarity_seed`` or ``preference_seed``. The parser intentionally
    captures only explicit, whole-line grammar; entity lookup decides whether
    a named similarity/preference target is an artist or track later, against
    canonical facts. ``seed_source=current_track`` is an abstract reference,
    never a fabricated canonical identity.
    """

    mode: str
    target: str | None
    target_kind: str | None
    scene: str | None
    requested_count: int | None = None
    seed_source: str | None = None


_GENERIC_RECOMMENDATION_FORMS: frozenset[str] = frozenset(
    {
        "推荐音乐",
        "推荐几首歌",
        "给我推荐",
        "给我推荐几首歌",
        "帮我推荐",
        "帮我推荐几首歌",
        "帮忙推荐几首歌",
    }
)
_CURRENT_TRACK_SIMILARITY_FORMS: frozenset[str] = frozenset(
    {
        "找类似这首的",
        # Explicit current-track similarity continuations. Keep this closed:
        # the follow-up verb alone (再推荐几首 / 再来几首) does not establish a
        # current-track seed and must not be upgraded to similarity.
        "再找一些类似这首的",
        "再找几首类似这首的",
        "再来一些类似这首的",
        "再推荐几首类似这首的",
    }
)


_EVENING_MARKERS = ("晚上", "夜晚", "夜间", "睡前", "入眠")
_PREFERENCE_RECOMMENDATION_RE = re.compile(
    r"^我喜欢\s*(?P<target>.+?)[，,]\s*"
    r"(?:(?:请)?(?:给我|帮我))?推荐(?:几首|一些|点)?"
    r"(?:适合(?:晚上|夜晚|夜间|睡前|入眠)听的)?(?:音乐|歌曲|歌)$",
    re.IGNORECASE,
)
_SIMILAR_RECOMMENDATION_PATTERNS = (
    re.compile(
        r"^(?:(?:请)?(?:给我|帮我))?推荐(?:几首|一些|点)?类似\s*"
        r"(?P<target>.+?)\s*的(?:音乐|歌曲|歌)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^找(?:几首|一些|点)?(?:和|与)\s*(?P<target>.+?)\s*"
        r"(?:风格)?(?:相近|类似)的(?:音乐|歌曲|歌)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:请)?(?:给我|帮我))?推荐(?:几首|一些|点)?像\s*"
        r"(?P<target>.+?)\s*的(?:音乐|歌曲|歌)$",
        re.IGNORECASE,
    ),
)
_ARTIST_RECOMMENDATION_RE = re.compile(
    r"^(?:(?:请)?(?:给我|帮我))?"
    r"(?:推荐|找|来点)(?:几首|一些|点)?"
    r"(?:适合(?:晚上|夜晚|夜间|睡前|入眠)听的)?\s*"
    r"(?P<target>.+?)\s*的(?:音乐|歌曲|歌)$",
    re.IGNORECASE,
)


def resolve_recommendation_turn_semantics(
    text: str,
) -> RecommendationTurnSemantics | None:
    """Parse explicit target semantics without guessing an entity identity."""
    if not isinstance(text, str):
        return None
    value = text.strip().strip(_TRAILING_PUNCTUATION).strip()
    if not value:
        return None
    normalized = _normalize(text)
    if normalized in _GENERIC_RECOMMENDATION_FORMS:
        return RecommendationTurnSemantics(
            mode="generic",
            target=None,
            target_kind=None,
            scene=None,
            requested_count=5,
            seed_source=None,
        )
    if normalized in _CURRENT_TRACK_SIMILARITY_FORMS:
        return RecommendationTurnSemantics(
            mode="similarity_seed",
            target=None,
            target_kind="track",
            scene=None,
            requested_count=5,
            seed_source="current_track",
        )
    # Fresh-discovery grammar (找一些新的歌 / 推荐没听过的歌) owns these
    # adjective forms.  They do not name an artist entity called "新".
    if is_fresh_discovery_intent(text):
        return None
    scene = "evening" if any(marker in value for marker in _EVENING_MARKERS) else None
    match = _PREFERENCE_RECOMMENDATION_RE.fullmatch(value)
    if match is not None:
        target = match.group("target").strip()
        if target:
            return RecommendationTurnSemantics(
                "preference_seed", target, None, scene
            )
    for pattern in _SIMILAR_RECOMMENDATION_PATTERNS:
        match = pattern.fullmatch(value)
        if match is not None:
            target = match.group("target").strip()
            if target:
                return RecommendationTurnSemantics(
                    "similarity_seed", target, None, scene
                )
    match = _ARTIST_RECOMMENDATION_RE.fullmatch(value)
    if match is None:
        return None
    target = match.group("target").strip()
    # Generic scene requests have no explicit entity ("推荐适合晚上听的歌").
    # Do not manufacture an artist named after the scene phrase.
    if not target or target in {
        "新",
        "新的",
        "好听",
        "好听的",
        "适合晚上听",
        "适合夜晚听",
        "适合夜间听",
        "适合睡前听",
        "适合入眠听",
    }:
        return None
    return RecommendationTurnSemantics("artist_constraint", target, "artist", scene)

# Only exact whole-line matches (keyed by _normalize: strip + trailing
# punctuation + casefold). Extending this table loosens routing -- every entry
# must be an unambiguous control command.
_ROUTED_COMMANDS: dict[str, str] = {
    "pause": "pause",
    "暂停": "pause",
    "stop preview": "stop_preview",
    "停止试听": "stop_preview",
    # P19-T14-F-R2: the explicit stop-object form joins the same plain table
    # entry -- 停止试听它 (and its 他/她 feedstock via normalization) means
    # stop the preview, referent or not; stop_preview is idempotent.
    "停止试听它": "stop_preview",
    # P19-T14-A: 暂停试听 names the preview explicitly, so it stays a plain
    # table entry (context-free stop_preview), same as 停止试听.
    "暂停试听": "stop_preview",
    "play": "play",
    "resume": "play",
    "continue": "play",
    "继续播放": "play",
    "继续": "play",
    "next": "next_track",
    "下一首": "next_track",
    "下一首试听": "advance_preview",
    "prev": "previous_track",
    "previous": "previous_track",
    "上一首": "previous_track",
    # P15-S4-M3-B: V1 playback-status phrase set, closed to exactly these five.
    # playback_status is a CLI-side pseudo-command (never an AgentToolName):
    # the caller intercepts it before tool dispatch and answers deterministically.
    "现在在播放什么": "playback_status",
    "现在播放的是什么": "playback_status",
    "当前在播放什么": "playback_status",
    "当前播放什么": "playback_status",
    "现在是什么歌": "playback_status",
    # P16-S3: V1 formal-playback phrase set, closed to exactly these five.
    # formal_play is a second CLI-side pseudo-command: the caller intercepts it
    # and runs the deterministic locate->select->play->verify chain itself.
    "播放一首正式歌曲": "formal_play",
    "放一首正式歌曲": "formal_play",
    "播放一首正式的歌": "formal_play",
    "来一首正式歌曲": "formal_play",
    "正式播放一首": "formal_play",
}

# Exact forms whose routing depends on the live Playback Context. Only these
# trigger a context fetch on the caller's fast path; everything else routes (or
# falls back) context-free. P15-S1 extends the set: the session state can flip
# 暂停 -> stop_preview and gates the continue-family interception, so 暂停/pause
# and the play-family forms join the set (they now pay one local read).
# P15-S1 C02 adds 下一首/next/下一首试听: only while a session literally runs do
# they flip to advance_preview, so they pay the same one local read.
_CONTEXT_SENSITIVE_FORMS: frozenset[str] = frozenset(
    (
        "stop",
        "停止",
        "换一首",
        "暂停",
        "pause",
        "继续",
        "继续播放",
        "continue",
        "play",
        "resume",
        "下一首",
        "next",
        "下一首试听",
        # P19-T14-A: 不听了 flips to stop_preview only while a preview really
        # sounds -- off one it falls back to the loop untouched. The flip
        # decision itself is made on the authority's runner truth; the loop's
        # OWN reads remain their own surface (the split-brain the stop family
        # used to trust is fixed only for the intercepted forms -- see the
        # T14 closeout report residuals for 停/别放了/关掉).
        "不听了",
    )
)


def needs_active_context(text: str) -> bool:
    """True when routing ``text`` could change with the live Playback Context.

    Callers use this to fetch the context read only for the forms where it can
    change the outcome -- the plain fast path for every other command stays
    free of any service read. Uses the same normalization as ``route_intent``
    so a punctuated form (暂停。) keeps its context-dependent routing.
    """
    return isinstance(text, str) and _normalize(text) in _CONTEXT_SENSITIVE_FORMS


def route_intent(
    text: str,
    *,
    channel: str | None = None,
    context: str | None = None,
    preview_sounding: bool = False,
    preview_session_state: str | None = None,
) -> str | None:
    """Map one user line to a direct tool name, or ``None`` to use the agent loop.

    Surrounding whitespace and optional trailing punctuation (。．？！等) are
    tolerated; anything else -- extra words, near-misses like 请问怎么暂停,
    interior punctuation -- is refused (None) so the provider loop handles it.

    ``channel`` and ``context`` feed the 换一首 rule; ``preview_sounding`` is
    the real runner truth (get_playback_context ``preview_sounding``) and feeds
    the stop rule; ``preview_session_state`` (the same endpoint's ``session``
    state, or None) feeds the P15-S1 rules above. Tokens are matched strictly,
    with no normalization: ``preview_sounding`` counts only when ``is True`` and
    the session only when literally "running" -- the defaults never invent a
    route and the plain table routes identically with or without them.

    ``playback_status`` (the V1 status-query result) and ``formal_play``
    (P16-S3, the V1 formal-playback result) are CLI-side pseudo-commands: the
    caller must intercept them instead of forwarding either as a tool call.
    """
    if not isinstance(text, str):
        return None
    exact = _normalize(text)
    running = preview_session_state == "running"
    tool = _ROUTED_COMMANDS.get(exact)
    if running and tool == "pause":
        return "stop_preview"  # 暂停 ≙ 停止试听 while the session runs
    if running and exact in ("下一首", "next"):
        # 下一首 ≙ 试听下一首 while the session runs: the skip belongs to the
        # live session (never Music.app, never a fresh session).
        return "advance_preview"
    if tool is not None:
        if tool == "advance_preview":
            # 下一首试听 only has a meaning on a live session. Off one it must
            # NOT invent a session (or touch Music.app): defer to the loop.
            return "advance_preview" if running else None
        return tool  # play family unchanged; the caller intercepts play while running
    # P19-T14-A: 不听了 joins the stop family -- it flips to stop_preview only
    # on real runner truth (preview_sounding is True / session running), and
    # stays None otherwise so the provider loop answers as it always has.
    if exact in ("stop", "停止", "不听了"):
        if preview_sounding is True or running:
            return "stop_preview"
        return None
    if exact == "换一首":
        if channel == "none" and context == "own_queue":
            return "next_track"
        return None
    return None


# P19-T14-B legacy closed recommendation-intent form set. Production reply-door
# logic now consumes ``expects_recommendation_batch`` so provider routing and
# presentation share one truth. ``is_recommendation_intent`` remains temporarily
# as a compatibility classifier while repo-wide callers are checked before deletion.
# Historically it replaced a recommendation-shaped request that finished WITHOUT
# a generated batch with
# a single honest fallback sentence (never the model's own prose, which
# otherwise degrades into a hand-enumerated pseudo-list). Unlike the routing
# table this is not an execution map -- it never executes anything, it only
# classifies. The set is exact whole-line forms (same ``_normalize``), kept
# deliberately narrow for the same reason the routing table is closed: a false
# positive here silently discards a legitimate reply, so near-miss phrasings
# are NOT added speculatively. They stay covered by the prompt rule and the
# numbered-song-list shape detector in ``provider_agent``.
#
# Excluded by construction: 创建歌单 (playlist creation, not recommendation --
# its produce is a write intent, not a recommendation run) and any request that
# merely mentions recommendation mechanics (「推荐功能怎么用」etc.).
_RECOMMENDATION_INTENT_FORMS: frozenset[str] = frozenset(
    (
        # The web shell's own shortcut chips (ui/index.html).
        "推荐音乐",
        "找类似这首的",
        "换个心情",
        # The similar-to-current family (the T14-B live failure wording).
        "类似这首的",
        "类似的歌",
        "类似的歌曲",
        "找类似的",
        "找类似的歌",
        "找相似的歌",
        "推荐类似的",
        "推荐和这首类似的",
        "类似刚才这首",
        "相似刚才这首",
        # The recommend-a-batch family (mirrors the provider prompt's triggers).
        "推荐几首歌",
        "给我推荐",
        "给我推荐几首歌",
        "帮我推荐",
        "帮忙推荐几首歌",
        "来一批",
        "再来一批",
        "再来点新的",
        "换一批",
        "换一组",
        "推荐新的",
        "推荐点新的",
        "推荐一些新歌",
    )
)


def is_recommendation_intent(text: str) -> bool:
    """True when ``text`` is a closed recommendation-request form (T14-B guard).

    Pure and stateless like ``route_intent``: exact whole-line match after the
    same normalization (strip + optional trailing punctuation + casefold). The
    caller decides what to do with the classification; nothing is routed or
    executed here. Only the forms in ``_RECOMMENDATION_INTENT_FORMS`` count --
    anything else (extra words, near-miss spellings) returns False so the
    provider loop's own answer stays untouched.
    """
    if not isinstance(text, str):
        return False
    return _normalize(text) in _RECOMMENDATION_INTENT_FORMS


# P19-T14-E: explicit play-intent recognition for the Play-vs-Preview guard.
# Like the recommendation set this is a CLASSIFIER, not a routing table: it
# never executes anything, it only feeds the harness door that forbids a
# play-intent turn from ending with preview audio. The forms are deliberately
# wider than ``route_intent``'s closed table because play requests are open
# (any song name follows 播放), but every admission rule below is written so a
# false positive can only cost a redundant post-run check -- never a wrong
# execution. 试听-touching texts are excluded (mixed intent stays the preview
# family's own surface), and question/feature phrasings are excluded so
# 播放器怎么用 never reads as a play command.
_PLAY_INTENT_EXACT_FORMS: frozenset[str] = frozenset(
    (
        "播放",
        "播放这首",
        "播放那首",
        "播放这一首",
        "播放那一首",
        "播放上一首",
        "放这首",
        "放这一首",
    )
)

# 播放第N首 family (N in arabic or common Chinese numerals), e.g. 播放第2首.
_PLAY_INTENT_NTH_RE = re.compile(
    r"^播放第(?P<ordinal>[0-9一二三四五六七八九十两]+)首$"
)

# Prefixes that mention the play infra without being play commands.
_PLAY_INTENT_FEATURE_PREFIXES: tuple[str, ...] = (
    "播放器",
    "播放列表",
    "播放队列",
    "播放历史",
    "播放功能",
    "播放状态",
)

# Question markers that turn a 播放... line into a question, not a command.
# P20-Fix02 adds 区别: the feature-comparison question (播放和推荐有什么区别？)
# must keep the full surface -- a false negative here only costs tokens, never
# a wrong execution.
_PLAY_INTENT_QUESTION_MARKERS: tuple[str, ...] = ("怎么", "如何", "为什么", "是什么", "吗", "呢", "区别")


def is_explicit_play_intent(text: str) -> bool:
    """True when ``text`` is an explicit formal-play request (T14-E guard).

    Recognizes the product contract's play family: bare 播放, the batch forms
    (播放这首/那首/上一首), 播放第N首, and named-track requests (播放 <歌名/艺人>).
    Conservative exclusions keep feature/question phrasings (播放器/播放列表/
    怎么播放/…吗) and any 试听-touching text out -- a false positive would only
    pay one redundant post-run check in the caller's door, but the exclusions
    keep that door silent for everything that is not clearly a play command.
    """
    if not isinstance(text, str):
        return False
    normalized = _normalize(text)
    if normalized.startswith("开始播放"):
        normalized = normalized[2:]
    if normalized in _PLAY_INTENT_EXACT_FORMS:
        return True
    if _PLAY_INTENT_NTH_RE.match(normalized):
        return True
    if not normalized.startswith("播放"):
        return False
    if "试听" in normalized:
        return False
    if normalized.startswith(_PLAY_INTENT_FEATURE_PREFIXES):
        return False
    if any(marker in normalized for marker in _PLAY_INTENT_QUESTION_MARKERS):
        return False
    return bool(normalized[2:].strip())


def _named_play_target_text(text: str) -> str | None:
    """Return only the user-language target of a named formal-play command.

    This projection deliberately excludes every closed playback command
    (bare/resume, current/recommendation referents, ordinals and the routed
    formal-play pseudo command).  The returned text is *not* identity: later
    code must resolve it against structured search/discovery facts before it
    can authorize any action or offer.
    """
    if not isinstance(text, str) or not is_explicit_play_intent(text):
        return None
    # Existing deterministic commands retain their own ownership.
    if route_intent(text) is not None or needs_active_context(text):
        return None

    raw = text.strip().strip(_TRAILING_PUNCTUATION).strip()
    normalized = _normalize(text)
    if normalized.startswith("开始播放"):
        raw = raw[len("开始播放") :].strip()
        normalized = normalized[len("开始播放") :].strip()
        # Keep the same grammar as is_explicit_play_intent after stripping
        # the optional 开始 prefix.
        if normalized.startswith("播放"):
            raw = raw[len("播放") :].strip()
            normalized = normalized[len("播放") :].strip()
    elif normalized.startswith("播放"):
        raw = raw[len("播放") :].strip()
        normalized = normalized[len("播放") :].strip()
    else:
        return None

    if not raw or not normalized:
        return None
    # Closed referents/ordinal forms are not named targets.
    if normalized in {"这首", "那首", "这一首", "那一首", "上一首"}:
        return None
    if re.fullmatch(r"第[0-9一二三四五六七八九十两]+首", normalized):
        return None
    return raw


def is_read_only_library_intent(text: str) -> bool:
    """True for explicit search / Library-membership questions.

    This is a safety classifier, not an answer router.  Its only execution
    consequence is narrowing the provider to read/search tools so a question
    such as ``Spring Thief 在我的资料库里吗`` cannot drift into ``play_track``.
    Explicit playback wording is checked first and always stays on the formal
    playback surface.
    """
    if not isinstance(text, str):
        return False
    if is_explicit_play_intent(text):
        return False
    normalized = _normalize(text)
    if not normalized:
        return False
    if normalized.startswith(("搜索", "查找")) and len(normalized) > 2:
        return True
    if normalized.startswith("有没有") and len(normalized) > 3:
        return True
    if "在我的资料库" in normalized and normalized.endswith(("吗", "么")):
        return True
    return normalized.startswith("我的资料库有哪些")


# P19-T14-F: track-reference pronoun normalization. Through the live-track
# context, 试听它 resolves today, but the equally common spoken variants
# 试听他 / 试听她 do not (the provider asks "which song" instead of reusing the
# already-resolved track). At the intent/reference parsing boundary, a
# track-reference verb (试听 / 播放, optionally prefixed 停止) whose DIRECT,
# sentence-final object is the pronoun 他/她 is rewritten exactly once to 它 so
# every variant reuses the proven 它 path. This is deliberately NOT a global
# pronoun rewrite: the pronoun only normalizes as the final object of a
# track-reference verb -- 试听他的歌 / 她说 / 他是谁 and every other reading
# pass through untouched. Pure text transform; nothing is routed or executed
# here, and the output never collides with _ROUTED_COMMANDS (no 它-object form
# is routed), so the fast paths' behavior is bit-identical for both spellings.
_TRACK_REFERENCE_VERBS: tuple[str, ...] = ("停止试听", "试听", "播放")

# Group 1 is anchored on the verb alternation built from the tuple above
# (all pure literals -- no metacharacters); group 3 tolerates the same
# trailing token set the routing table tolerates (module
# _TRAILING_PUNCTUATION + whitespace). Anything else after the pronoun
# (的歌/吗/说/一下...) refuses the rewrite.
_TRACK_REF_PRONOUN_RE = re.compile(
    r"^(\s*\S*(?:"
    + "|".join(_TRACK_REFERENCE_VERBS)
    + r"))([他她])([。．？！?！!·…\s]*)$"
)


def normalize_track_reference_pronouns(text: str) -> str:
    """Rewrite 他/她 to 它 inside sentence-final track-reference patterns (T14-F).

    ``试听他`` -> ``试听它``, ``播放她`` -> ``播放它``, ``停止试听他`` ->
    ``停止试听它`` (whitespace/punctuation preserved); everything else --
    including non-strings and any line whose trailing pronoun is not the
    track-reference object -- is returned unchanged.
    """
    if not isinstance(text, str) or not text or ("他" not in text and "她" not in text):
        return text
    match = _TRACK_REF_PRONOUN_RE.match(text)
    if match is None:
        return text
    return match.group(1) + "它" + match.group(3)


# P19-T14-F-R2/R4: deterministic pronoun binding. T14-F's spelling rewrite
# was NOT enough -- the downstream 它 referent resolution was still
# model-side, and the live failure wandered into recommendation generation
# (「暂时没有找到合适的推荐…」). These classifiers/pure extractors let the
# caller bind the pronoun to the ONE authoritative referent BEFORE any
# provider round. R4 corrected the referent source: the session-local
# ``referent_canonical_id`` (last explicit conversational track target,
# survives preview stops) wins; the action-log ``channel`` is only the
# legacy fallback when no referent exists yet -- it is cleared on
# stop_preview by constitution and can never outlive the referent.
_PRONOUN_TRACK_REFERENCE_FORMS: dict[str, str] = {
    "试听它": "preview",
    "播放它": "play",
}


def pronoun_track_reference(text: str) -> str | None:
    """Classify ``text`` as a pronoun track reference: "preview" / "play" / None.

    Closed whole-line forms only (试听它 / 播放它, optional trailing
    punctuation) -- after the T14-F pronoun normalization, every 它/他/她
    feedstock spelling lands here. Purely classification: nothing is read,
    routed or executed; the caller decides what the referent is and what to do.
    """
    if not isinstance(text, str):
        return None
    return _PRONOUN_TRACK_REFERENCE_FORMS.get(
        _normalize(normalize_track_reference_pronouns(text))
    )


def pronoun_track_reference_target(context_payload: object) -> str | None:
    """Extract the single authoritative referent from a context observation.

    P19-T14-F-R4: ``referent_canonical_id`` (get_active_context /
    get_playback_context, P19-T14-F-R4) is the conversational track target --
    the last successfully resolved explicit track interaction (试听 X / 播放 X
    / an explicit batch-item interaction). It survives preview stops by design,
    which ``channel`` (an action log cleared on stop_preview) does not.

    A valid referent wins outright. When it is absent (fresh session, no
    explicit interaction yet), the observation's ``channel`` acts as the
    legacy fallback -- a library/preview channel with a canonical id yields
    exactly that id -- and anything else (state none/unknown, missing ids,
    malformed payload) yields None: ambiguity and absence are never resolved
    by guessing. Pure; no I/O.
    """
    if not isinstance(context_payload, Mapping):
        return None
    referent = context_payload.get("referent_canonical_id")
    if isinstance(referent, str) and referent:
        return referent
    channel = context_payload.get("channel")
    if not isinstance(channel, Mapping):
        return None
    state = channel.get("state")
    canonical_id = channel.get("canonical_id")
    if state in ("library", "preview") and isinstance(canonical_id, str) and canonical_id:
        return canonical_id
    return None


def continuous_preview_session_running(context_payload: object) -> bool:
    """True when the observation carries a literally RUNNING continuous
    preview session (its own multi-track surface; the pronoun referent is
    contested there and the caller defers to the provider's session rules)."""
    if not isinstance(context_payload, Mapping):
        return False
    session = context_payload.get("session")
    return isinstance(session, Mapping) and session.get("state") == "running"


# Fixed replies for the deterministic pronoun-binding fast path. The play
# failure reuses the T14-E contract sentence (provider_agent constant) -- the
# pronoun path must say exactly what the play path says.
PRONOUN_PREVIEW_START_REPLY = "已开始试听（30 秒）。"
PRONOUN_PREVIEW_UNAVAILABLE_REPLY = "这首没有可用的 30 秒试听。"
PRONOUN_PLAY_START_REPLY = "已开始正式播放。"
PRONOUN_PREVIEW_ASK = "想试听哪首歌？"
PRONOUN_PLAY_ASK = "想播放哪首歌？"


# S1 (token/round-cost optimization): the closed plain-chat form set. Exactly
# these whole-line social forms run with ZERO tool schemas (the provider loop
# consults this classifier once per user message); every other line -- every
# music question, command, near-miss, or unlisted chitchat -- keeps the full
# tool set. A false positive here STRIPS tools from a request that may need
# them (the model could not read real music state), while a false negative
# only costs tokens, so the set is deliberately narrow and exact (same
# _normalize discipline as the routing table: surrounding whitespace + the
# optional trailing punctuation set tolerated, nothing else).
#
# Deliberately excluded:
# - The consent/acknowledgment family (好的/行/可以/嗯/收到/没问题/…): those
#   answer pending offers -- a bare 好的 after the model's 「要试听吗?」 must
#   keep every tool so the preview can actually start.
# - Any form carrying a music word (谢谢你的推荐, 他唱得真好听, 我喜欢这首):
#   truncation or composition never classifies -- only the exact closed set.
# - Capability questions (你能做什么/你会什么): the model answers those
#   through get_agent_capabilities, so tools stay.
# - 好的/可以-style acknowledgments and every routed command (暂停/继续/…):
#   those ride the routing table / the loop with tools, untouched.
_PLAIN_CHAT_FORMS: frozenset[str] = frozenset(
    (
        # Greetings / wellbeing.
        "你好", "您好", "你好呀", "你好啊", "你好吗", "最近好吗",
        "嗨", "哈喽", "哈啰", "哈罗",
        "hi", "hello", "hey", "morning",
        "早上好", "早", "早安", "中午好", "下午好", "晚上好", "晚安",
        "在吗", "在不在", "在么", "有人吗", "有人么",
        # Thanks.
        "谢谢", "谢谢你", "谢谢啦", "谢谢了", "多谢", "感谢", "太感谢了", "非常感谢",
        "辛苦了", "麻烦你了",
        "thanks", "thank you", "thx",
        # Farewell.
        "再见", "拜拜", "回见", "回头见", "下次见", "回聊", "晚安啦",
        "bye", "bye bye", "see you",
        # Identity / compliments / zero-music-tool small talk.
        "你是谁", "你叫什么名字",
        "你真聪明", "你真厉害", "你真棒",
        "今天天气怎么样", "今天天气如何", "天气怎么样",
        "你吃了吗", "吃了吗",
        "讲个笑话", "讲个冷笑话",
    )
)


def is_plain_chat(text: str) -> bool:
    """True when ``text`` is a closed plain-chat form (S1 zero-tool gate).

    Pure and stateless exactly like ``route_intent``: exact whole-line match
    after the same normalization (strip + optional trailing punctuation +
    casefold). Only the forms in ``_PLAIN_CHAT_FORMS`` count -- everything
    else (extra words, near-miss chitchat, mixed sentences, greetings glued
    to a request like 你好帮我推荐几首歌) returns False so the provider loop
    keeps its full tool set. Nothing is read, routed or executed here; the
    caller (ProviderAgentLoop) decides whether to drop tools for this turn.
    """
    if not isinstance(text, str):
        return False
    return _normalize(text) in _PLAIN_CHAT_FORMS


# S3 (token-cost optimization): per-task tool-surface classifiers. Like every
# function in this module they are pure text judgments -- the caller
# (ProviderAgentLoop) maps a classification to a narrowed tool set and, when
# nothing matches, keeps the full set fail-safe. A false positive here only
# narrows the surface (never widens it), so every family is deliberately
# closed: exact whole-line forms or a small anchored pattern, and any
# near-miss / mixed phrasing falls through to the full set. P20 consolidates
# recommendation presentation through ``expects_recommendation_batch`` so the
# provider and web reply door consume the same recommendation/fresh expectation.

# The 推荐点<方向> open form (推荐点日系的 / 推荐点轻松的 / …). 推荐点 something
# never names a track, so the provider can narrow it to the recommendation
# chain. P20's unified batch expectation also lets the web reply door fail
# honest when such a request produces no batch.
_RECOMMEND_DIRECTION_RE = re.compile(r"^推荐点.+$")

# P20-Fix02 recommendation-surface extensions, whole-line exact. They narrow
# the S3 tool list and now also participate in the unified batch expectation;
# every near-miss (推荐几首好听的 / 再来一批好听的 / …) still falls through to
# the full set.
_RECOMMENDATION_SURFACE_FORMS: frozenset[str] = frozenset(
    (
        "最近给我推荐几首歌",
        "帮我推荐几首歌",
        "来几首推荐",
        "再推荐一批",
    )
)

# The re-recommend family with an optional direction tail (再来一批，换个方向 /
# 换一批换个方向 …). Bounded to exactly the four closed re-recommend verbs plus
# the two direction tails -- anything else after the verb (换一批是什么 /
# 再来一批好听的) stays on the full set, and a bare 换个方向 never matches
# (it has no batch verb and stays the model's own line).
_RECOMMENDATION_RECOMMEND_RE = re.compile(
    r"^(?:再来一批|再推荐一批|换一批|换一组)(?:[，,]?(?:换个方向|换方向))?$"
)

# Natural recommendation requests join the recommendation tool/prompt surface.
# Both forms are whole-line anchored and require a music noun; questions,
# search/library reads, playback commands and chained actions remain outside.
_NATURAL_RECOMMENDATION_REQUEST_RE = re.compile(
    r"^(?:(?:给我|帮我|请给我|请帮我)?推荐.+(?:音乐|歌曲|歌)"
    r"|找(?:几首|一些|点)?和.+类似的(?:音乐|歌曲|歌))$"
)


def is_recommendation_request(text: str) -> bool:
    """True when ``text`` names a recommendation task (S3 surface classifier).

    The closed ``_RECOMMENDATION_INTENT_FORMS`` set, P20-Fix02 extensions,
    the open 推荐点<方向> form, and the bounded re-recommend pattern (normalized
    the same way). Pure and stateless. The web reply door consumes this fact
    through ``expects_recommendation_batch`` rather than maintaining a second
    production definition of recommendation intent.
    """
    if not isinstance(text, str):
        return False
    normalized = _normalize(text)
    return (
        resolve_recommendation_turn_semantics(text) is not None
        or normalized in _RECOMMENDATION_INTENT_FORMS
        or normalized in _RECOMMENDATION_SURFACE_FORMS
        or bool(_RECOMMEND_DIRECTION_RE.fullmatch(normalized))
        or bool(_RECOMMENDATION_RECOMMEND_RE.fullmatch(normalized))
        or bool(_NATURAL_RECOMMENDATION_REQUEST_RE.fullmatch(normalized))
    )


def recommendation_presentation_label(text: str) -> str | None:
    """Stable label for an explicitly resolved current-turn recommendation."""
    return _recommendation_label_for_semantics(
        resolve_recommendation_turn_semantics(text)
    )


def _recommendation_label_for_semantics(
    semantics: RecommendationTurnSemantics | None,
) -> str | None:
    """Project the existing typed recommendation meaning into its UI label."""
    if semantics is None:
        return None
    if semantics.mode == "similarity_seed":
        if semantics.seed_source == "current_track":
            return "类似当前曲目"
        return f"类似 {semantics.target}"
    if semantics.scene == "evening":
        return "今晚推荐"
    return "新推荐"


# The closed fresh-discovery set. These lines ask for catalog search plus an
# inferred/fresh recommendation -- the same task tools as the recommendation
# chain (discover + query_catalog_discovery_state + the two generation tools
# + context reads), never the play/preview/feedback families. P20-Fix02 adds
# the natural phrasings the live audit found falling to the FULL surface.
# Overlap with ``_RECOMMENDATION_INTENT_FORMS`` (推荐一些新歌) is deliberate:
# the S3/S4 selectors check fresh FIRST so those lines travel the catalog path;
# P20's unified batch expectation still marks them as recommendation-producing
# for presentation. Pinned near-misses (找点新歌/找新歌/
# 再来点新的/找些新歌听听) stay excluded -- an admitted fresh form must name
# 没听过, pair 新歌 with 推荐/给我找, or ask 库外 explicitly.
_FRESH_DISCOVERY_FORMS: frozenset[str] = frozenset(
    (
        "找些新的",
        "找点新的",
        "推荐没听过的",
        "找没听过的",
        "找库外歌曲",
        # P20-Fix02 fresh/catalog phrasings.
        "推荐一些新歌",
        "推荐一些没听过的新歌",
        "推荐一些我没听过的新歌",
        # Explicit library-absence wording: this is a discovery constraint,
        # never an artist entity named “我没有”. Keep the admission exact
        # until the recommendation contract grows first-class constraints.
        "推荐几首我没有的歌",
        "找点我没听过的歌",
        "给我找些新歌",
        "推荐点库外的歌",
        "找一些新的",
        "来点没听过的",
    )
)


def is_fresh_discovery_intent(text: str) -> bool:
    """True when ``text`` is a closed fresh/catalog-discovery form (S3)."""
    if not isinstance(text, str):
        return False
    return _normalize(text) in _FRESH_DISCOVERY_FORMS


def expects_recommendation_batch(text: str) -> bool:
    """Single production truth for whether this turn asks for a new batch.

    Provider tool/prompt routing and the web reply door must agree on this fact.
    Historically the web shell used the narrower ``is_recommendation_intent``
    closed set while the provider used ``is_recommendation_request`` plus fresh
    discovery forms, allowing the two layers to disagree about the same turn.
    Keep the old classifier as a compatibility helper for now, but production
    presentation should consume this unified expectation instead.
    """
    if not isinstance(text, str):
        return False
    return is_recommendation_request(text) or is_fresh_discovery_intent(text)


# P20-Fix02: the closed recommendation-EXPLANATION set. Why-questions about
# the current batch or the recommendation reasoning itself. Read-only by
# construction: these lines explain, they never ask for ANOTHER batch -- so
# the S3 explanation surface excludes both generation tools and
# discover_catalog_tracks, and a new recommendation run is structurally
# unreachable on this class. Kept closed and exact: questions about the
# recommendation ALGORITHM or feature (为什么推荐算法这么慢？/推荐系统是怎么
# 工作的？/你会推荐吗？) and refusals (不要推荐了/我不想听推荐) are NOT
# explanation requests and keep the full set. Feeds only the tool-surface /
# system-prompt selection, never the T14-B reply door.
_RECOMMENDATION_EXPLANATION_FORMS: frozenset[str] = frozenset(
    (
        "为什么推荐这些",
        "为什么给我推荐这些",
        "为什么这些适合我",
        "这几首为什么适合我",
        "这批为什么适合我",
        # P20-Fix06: the UAT-live verb-first 这一批/这批 spellings the
        # closed set missed -- 为什么这一批适合我？ fell through to the
        # full general surface (31 tools, DEFAULT prompt) and answered with
        # ungrounded score/encyclopedia/current-playing reasoning. Same
        # read-only explanation class, same normalization discipline.
        "为什么这一批适合我",
        "为什么这批适合我",
        "这批推荐为什么适合我",
        "推荐理由是什么",
        "为什么推荐这几首",
    )
)


def is_recommendation_explanation_intent(text: str) -> bool:
    """True when ``text`` is a closed recommendation-explanation form (S3).

    Pure and stateless like every classifier here: exact whole-line match
    after the same normalization (strip + optional trailing punctuation +
    casefold). Nothing is read, routed or executed; the caller maps the
    classification to the read-only explanation surface.
    """
    if not isinstance(text, str):
        return False
    return _normalize(text) in _RECOMMENDATION_EXPLANATION_FORMS


# Feedback task forms. Three shapes, all whole-line exact:
#  - direction praise: 这个方向不错 / 这个方向好 (a LIKE with a genre/direction
#    attribution is the only honest recording -- the model decides);
#  - track verdict on a batch referent: (我)?(喜欢|不喜欢|讨厌) + 第N首/这首/那首/
#    上一首;
#  - bare 这首 verdicts on the current track: 这首不错 / 这首好听 / 这首一般 / …
# Deliberately refused: bare 喜欢/不喜欢 (no object -- could be a statement
# about future taste, not a verdict), named-song verdicts (喜欢夜曲 -- the
# batch-locating chain cannot resolve them without search tools), and any
# multi-clause sentence. All of those keep the full set.
_FEEDBACK_DIRECTION_FORMS: frozenset[str] = frozenset(
    (
        "这个方向不错",
        "这个方向好",
        "这方向不错",
        "这方向好",
    )
)

_FEEDBACK_VERDICT_RE = re.compile(
    r"^我?(喜欢|不喜欢|讨厌)(第[0-9一二三四五六七八九十两]+首|这首(?:歌|歌曲)?|那首(?:歌|歌曲)?|上一首(?:歌|歌曲)?)$"
)

_FEEDBACK_THIS_TRACK_RE = re.compile(
    r"^这首(不错|好听|不好听|很喜欢|喜欢|不喜欢|一般|还行)$"
)


@dataclass(frozen=True, slots=True)
class CurrentTrackFeedbackTurnSemantics:
    """One explicit verdict whose referent is the current player track.

    This semantic is intentionally narrower than the general feedback family:
    batch ordinals, direction feedback and collection-level artist statements
    are not current-player verdicts and keep their existing paths.
    """

    kind: str


_CURRENT_TRACK_FEEDBACK_RE = re.compile(
    r"^我?(?P<verdict>喜欢|不喜欢)"
    r"(?:这首(?:歌|歌曲)?|当前正在播放的这首(?:歌|歌曲)?)$"
)


def resolve_current_track_feedback_turn_semantics(
    text: str,
) -> CurrentTrackFeedbackTurnSemantics | None:
    """Parse the closed current-player feedback forms used by the runtime."""
    if not isinstance(text, str):
        return None
    match = _CURRENT_TRACK_FEEDBACK_RE.fullmatch(_normalize(text))
    if match is None:
        return None
    return CurrentTrackFeedbackTurnSemantics(
        kind="liked" if match.group("verdict") == "喜欢" else "disliked"
    )


def is_feedback_intent(text: str) -> bool:
    """True when ``text`` is a closed feedback-verdict form (S3).

    Feeds only the tool-surface selection: a matching line may still end in a
    plain acknowledgment (the model owns the record-vs-answer decision); the
    point is that no tool outside the feedback/learning chain can be needed.
    """
    if not isinstance(text, str):
        return False
    normalized = _normalize(text)
    if normalized in _FEEDBACK_DIRECTION_FORMS:
        return True
    if resolve_current_track_feedback_turn_semantics(text) is not None:
        return True
    return bool(
        _FEEDBACK_VERDICT_RE.fullmatch(normalized)
        or _FEEDBACK_THIS_TRACK_RE.fullmatch(normalized)
    )


# Preview task forms (S3). Members:
#  - the 试听 named-track open form (试听夜曲 -- any non-empty remainder), with
#    question/feature phrasings excluded so 试听怎么弄/试听是什么意思 never
#    narrows;
#  - 试听第N首 and the closed batch-referent preview set (试听这首/那首/上一首/
#    一下/这个);
#  - the batch-preview command surface (都放一遍/都试听一遍/… -- preview_batch's
#    own phrasings);
#  - the stop residuals the routing table hands back to the loop when nothing
#    truly sounds (停/停止/别放了/关掉 -- the prompt's own stop clause: read
#    get_active_context.preview_sounding, stop_preview if true, else say so
#    honestly and pause only if the user means the music). 停止试听 and 暂停试听
#    join the exact set too: the routing table always intercepts them
#    caller-side, but loop-level texts (tests, non-CLI callers) classify the
#    same way.
# The pronoun forms (试听它/他/她) never reach the loop (the outer referent
# fast path executes them or answers the honest question), so they need no
# entry -- though the 试听 prefix would classify them harmlessly anyway.
_PREVIEW_NTH_RE = re.compile(
    r"^试听第(?P<ordinal>[0-9一二三四五六七八九十两]+)首$"
)

_PREVIEW_EXACT_FORMS: frozenset[str] = frozenset(
    (
        "试听这首",
        "试听那首",
        "试听这一首",
        "试听那一首",
        "试听上一首",
        "试听一下",
        "试听这个",
        "都试听一遍",
        "都放一遍",
        "全试听一遍",
        "全放一遍",
        "把这一批都试听一遍",
        "这一批都试听一遍",
        "这批都试听一遍",
        "把这一批都放一遍",
        "停止试听",
        "暂停试听",
    )
)

_PREVIEW_STOP_RESIDUAL_FORMS: frozenset[str] = frozenset(
    (
        "停",
        "停止",
        "别放了",
        "关掉",
    )
)

_PREVIEW_NAMED_TRACK_RE = re.compile(r"^试听.+$")

_PREVIEW_QUESTION_MARKERS: tuple[str, ...] = (
    "怎么",
    "如何",
    "为什么",
    "是什么",
    "是不是",
    "吗",
    "呢",
    "哪",
    "什么",
    "不",
)


def is_preview_intent(text: str) -> bool:
    """True when ``text`` is a preview/stop task form (S3 surface classifier)."""
    if not isinstance(text, str):
        return False
    normalized = _normalize(text)
    if normalized in _PREVIEW_EXACT_FORMS or normalized in _PREVIEW_STOP_RESIDUAL_FORMS:
        return True
    if _PREVIEW_NTH_RE.fullmatch(normalized):
        return True
    if not _PREVIEW_NAMED_TRACK_RE.fullmatch(normalized):
        return False
    # Mixed-intent guard: a line that names 播放 alongside 试听 (试听后再播放,
    # 试听还是播放…) is not a pure preview task -- the full set keeps both
    # halves capable.
    if "播放" in normalized:
        return False
    return not any(marker in normalized for marker in _PREVIEW_QUESTION_MARKERS)


class TurnPrimarySemantic(str, Enum):
    """The single top-level semantic owner for one user turn."""

    RECOMMENDATION = "recommendation"
    PREFERENCE_STATEMENT = "preference_statement"
    FEEDBACK = "feedback"
    PLAYBACK_ACTION = "playback_action"
    LIBRARY_QUERY = "library_query"
    RECOMMENDATION_EXPLANATION = "recommendation_explanation"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class TurnExpectedResult(str, Enum):
    """The result shape requested by the turn, never its execution outcome."""

    RECOMMENDATION_BATCH = "recommendation_batch"
    ACKNOWLEDGEMENT = "acknowledgement"
    FEEDBACK_RESULT = "feedback_result"
    ACTION_RESULT = "action_result"
    ANSWER = "answer"
    CLARIFICATION = "clarification"


class TurnSemanticSource(str, Enum):
    """How the final semantic plan for this turn was obtained."""

    DETERMINISTIC = "deterministic"
    LLM_INTERPRETED = "llm_interpreted"
    UNRESOLVED = "unresolved"


class TurnTaskSurface(str, Enum):
    """Compatibility projection of the existing provider task-surface decision."""

    PLAIN_CHAT = "plain_chat"
    FEEDBACK = "feedback"
    PREVIEW = "preview"
    PLAYBACK = "playback"
    LIBRARY_QUERY = "library_query"
    RECOMMENDATION_EXPLANATION = "recommendation_explanation"
    FRESH_DISCOVERY = "fresh_discovery"
    RECOMMENDATION = "recommendation"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class PlaybackActionTurnSemantics:
    """Abstract playback meaning without target, route, or execution facts.

    ``source`` and ``selection_mode`` describe how a later workflow may locate
    a target. They never authorize or select a canonical entity. In particular,
    ``delegated`` mirrors the original closed delegation classifier. Slice 3 also
    authorizes the already-recognized ``choose_another`` mode only when it names the
    active recommendation source; this adds no raw-text classifier.
    """

    kind: str
    source: str | None = None
    selection_mode: str | None = None
    explicit_index: int | None = None
    routed_command: str | None = None
    delegated: bool = False
    explicit_play: bool = False
    # User-language target for a named formal-play request (e.g.
    # ``播放地球最后一夜``). This is semantic input only: it is never a
    # canonical identity or execution authority. Canonical resolution remains
    # code-owned and must come from structured tool facts.
    target_text: str | None = None


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """Side-effect-free, unified semantic projection for one user turn.

    Existing resolvers remain authoritative. This object references their
    outputs and records their current provider-surface projection; it owns no
    canonical identity, recommendation run, playback route, or execution state.
    """

    user_text: str
    primary: TurnPrimarySemantic
    expected_result: TurnExpectedResult
    task_surface: TurnTaskSurface
    recommendation: RecommendationTurnSemantics | None = None
    preference_statement: PreferenceStatementTurnSemantics | None = None
    current_track_feedback: CurrentTrackFeedbackTurnSemantics | None = None
    playback_action: PlaybackActionTurnSemantics | None = None
    fresh_discovery: bool = False
    semantic_source: TurnSemanticSource = TurnSemanticSource.DETERMINISTIC
    requires_clarification: bool = False
    clarification_reason: str | None = None

    @property
    def expects_recommendation_batch(self) -> bool:
        return self.expected_result is TurnExpectedResult.RECOMMENDATION_BATCH

    @property
    def recommendation_label(self) -> str | None:
        return _recommendation_label_for_semantics(self.recommendation)

    @property
    def delegated_action_authorized(self) -> bool:
        action = self.playback_action
        return bool(
            action
            and (
                action.delegated
                or (
                    action.source == "active_recommendation"
                    and action.selection_mode == "choose_another"
                )
            )
        )

    @property
    def explicit_play_intent(self) -> bool:
        return bool(self.playback_action and self.playback_action.explicit_play)


_CHINESE_DIGITS: Mapping[str, int] = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def _ordinal_value(token: str) -> int | None:
    """Parse the ordinal token already admitted by the existing Nth regex."""
    if token.isdecimal():
        value = int(token)
        return value if value > 0 else None
    if token == "十":
        return 10
    if token.count("十") == 1:
        tens_text, ones_text = token.split("十")
        tens = 1 if not tens_text else _CHINESE_DIGITS.get(tens_text)
        ones = 0 if not ones_text else _CHINESE_DIGITS.get(ones_text)
        if tens is not None and ones is not None:
            return tens * 10 + ones
        return None
    return _CHINESE_DIGITS.get(token)


def _playback_action_semantics(text: str) -> PlaybackActionTurnSemantics | None:
    """Project existing playback classifiers into non-authoritative semantics."""
    normalized = _normalize(text)
    delegated = is_delegated_playback_intent(text)
    explicit_play = is_explicit_play_intent(text)
    preview = is_preview_intent(text)

    if delegated:
        return PlaybackActionTurnSemantics(
            kind="play_or_preview",
            source="active_recommendation",
            selection_mode="agent_choose_one",
            delegated=True,
        )

    play_nth = _PLAY_INTENT_NTH_RE.fullmatch(normalized)
    if play_nth is not None:
        return PlaybackActionTurnSemantics(
            kind="play",
            source="active_recommendation",
            selection_mode="explicit_index",
            explicit_index=_ordinal_value(play_nth.group("ordinal")),
            explicit_play=True,
        )

    preview_nth = _PREVIEW_NTH_RE.fullmatch(normalized)
    if preview_nth is not None:
        return PlaybackActionTurnSemantics(
            kind="preview",
            source="active_recommendation",
            selection_mode="explicit_index",
            explicit_index=_ordinal_value(preview_nth.group("ordinal")),
        )

    # Current provider behavior already interprets this exact follow-up in its
    # broad fallback. Slice 1 records that abstract meaning but deliberately
    # leaves the task surface FULL and does not grant delegated authority.
    if normalized == "再换一首":
        return PlaybackActionTurnSemantics(
            kind="play_or_preview",
            source="active_recommendation",
            selection_mode="choose_another",
        )

    routed_command = None if needs_active_context(text) else route_intent(text)
    if normalized == "换一首":
        return PlaybackActionTurnSemantics(
            kind="contextual",
            selection_mode="choose_another",
        )
    if preview:
        return PlaybackActionTurnSemantics(kind="preview")
    if explicit_play:
        return PlaybackActionTurnSemantics(
            kind="play",
            explicit_play=True,
            target_text=_named_play_target_text(text),
        )
    if routed_command is not None or needs_active_context(text):
        return PlaybackActionTurnSemantics(
            kind="command", routed_command=routed_command
        )
    return None


def _task_surface_for_turn(
    *,
    plain_chat: bool,
    feedback: bool,
    preview: bool,
    explicit_play: bool,
    library_query: bool,
    recommendation_explanation: bool,
    fresh_discovery: bool,
    recommendation_request: bool,
) -> TurnTaskSurface:
    """Preserve the existing ProviderAgent task-surface priority exactly."""
    if plain_chat:
        return TurnTaskSurface.PLAIN_CHAT
    if feedback:
        return TurnTaskSurface.FEEDBACK
    if preview:
        return TurnTaskSurface.PREVIEW
    if explicit_play:
        return TurnTaskSurface.PLAYBACK
    if library_query:
        return TurnTaskSurface.LIBRARY_QUERY
    if recommendation_explanation:
        return TurnTaskSurface.RECOMMENDATION_EXPLANATION
    if fresh_discovery:
        return TurnTaskSurface.FRESH_DISCOVERY
    if recommendation_request:
        return TurnTaskSurface.RECOMMENDATION
    return TurnTaskSurface.FULL


def resolve_turn_plan(text: str) -> TurnPlan:
    """Build the one top-level semantic projection from existing pure resolvers."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    recommendation = resolve_recommendation_turn_semantics(text)
    preference = resolve_preference_statement_turn_semantics(text)
    current_feedback = resolve_current_track_feedback_turn_semantics(text)
    feedback = is_feedback_intent(text)
    delegated = is_delegated_playback_intent(text)
    preview = is_preview_intent(text)
    explicit_play = is_explicit_play_intent(text)
    library_query = is_read_only_library_intent(text)
    explanation = is_recommendation_explanation_intent(text)
    fresh = is_fresh_discovery_intent(text)
    recommendation_request = is_recommendation_request(text)
    plain_chat = is_plain_chat(text)
    action = _playback_action_semantics(text)

    surface = _task_surface_for_turn(
        plain_chat=plain_chat,
        feedback=feedback,
        preview=preview,
        explicit_play=explicit_play,
        library_query=library_query,
        recommendation_explanation=explanation,
        fresh_discovery=fresh,
        recommendation_request=recommendation_request,
    )

    if preference is not None:
        primary = TurnPrimarySemantic.PREFERENCE_STATEMENT
        expected = TurnExpectedResult.ACKNOWLEDGEMENT
    elif feedback:
        primary = TurnPrimarySemantic.FEEDBACK
        expected = TurnExpectedResult.FEEDBACK_RESULT
    elif delegated or action is not None:
        primary = TurnPrimarySemantic.PLAYBACK_ACTION
        expected = TurnExpectedResult.ACTION_RESULT
    elif library_query:
        primary = TurnPrimarySemantic.LIBRARY_QUERY
        expected = TurnExpectedResult.ANSWER
    elif explanation:
        primary = TurnPrimarySemantic.RECOMMENDATION_EXPLANATION
        expected = TurnExpectedResult.ANSWER
    elif recommendation_request or fresh:
        primary = TurnPrimarySemantic.RECOMMENDATION
        expected = TurnExpectedResult.RECOMMENDATION_BATCH
    elif plain_chat:
        primary = TurnPrimarySemantic.GENERAL_CHAT
        expected = TurnExpectedResult.ANSWER
    else:
        primary = TurnPrimarySemantic.UNKNOWN
        expected = TurnExpectedResult.ANSWER

    return TurnPlan(
        user_text=text,
        primary=primary,
        expected_result=expected,
        task_surface=surface,
        recommendation=recommendation,
        preference_statement=preference,
        current_track_feedback=current_feedback,
        playback_action=action,
        fresh_discovery=fresh,
        semantic_source=(
            TurnSemanticSource.UNRESOLVED
            if primary is TurnPrimarySemantic.UNKNOWN
            else TurnSemanticSource.DETERMINISTIC
        ),
    )
