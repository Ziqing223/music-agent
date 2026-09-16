"""User-facing output sanitizer for the P14-R4.1 output contract.

The model's context necessarily contains internal tokens (tool payloads are
appended verbatim and the system prompt teaches field semantics by naming
them), so prompt discipline alone cannot guarantee the user-facing surface
stays free of ``rcm_…`` ids, ``active_batch``/``fresh_item_count``-style
field names, playback-route labels, internal id field names, ``context``
values, or fixed process-narration sentences.  This module is the
deterministic output gate: it scrubs model text at the user-facing doors
(cli.py ``_print_chat_result`` and web_shell.py's /api/chat reply) with a
closed vocabulary and fixed patterns.

P20 Fix 04 extends the P14-R4.1 vocabulary: route composites
(``playback.route 为 preview_only`` collapses to the full natural phrase
instead of the P14 shape 「playback.route 为 只能试听」), boolean/count
fields (``preview_sounding``, ``fresh_item_count``, ``fresh_this_request``),
bare internal id field names (``canonical_id``/``target_id``/``candidate_id``/
``run_id``/``referent_canonical_id``), a longer process-narration lead-in
family (用户明确说/按规则/信息充足/让我确认/…), and a fail-safe
fallback: a text scrubbed down to nothing returns one stable honest sentence
instead of an empty answer.

Design: P14-R4.1-DESIGN.md §3.2.  Rules are deliberately closed — no open
language rewriting — so behaviour is model-independent and unit-testable.
Deletions are confined to matched internal spans and fixed process lead-ins;
song/artist names and natural sentences pass through byte-identical.  The
verbose provider/tool trace (stderr, hidden by default) is deliberately
left unsanitized.
"""

import re

_ID_PREFIX_NAMES = {
    "rcm": "推荐编号",
    "cnd": "候选编号",
    "fbk": "反馈编号",
    "trk": "曲目编号",
    "art": "艺人编号",
    "alb": "专辑编号",
    "pl": "歌单编号",
    "pm": "歌单成员编号",
    "int": "意图编号",
    "agt": "服务请求编号",
}

# Closed field-name/value/route-label literals, mapped to readable Chinese.
# Order matters: longer tokens first so `preview_only` never fades into a
# bare `preview` and `active_batch` wins over `active`.  Full-width and
# half-width ASCII forms of every token are both matched (design rule 2).
_FIELD_VALUE_NAMES = {
    "playback.route": "播放情况",
    "playback_route": "播放情况",
    "referent_canonical_id": "指代曲目标识",
    "fresh_item_count": "本次新发现的曲目数",
    "fresh_this_request": "是否本次新发现",
    # P20 Fix03 evidence-projection field names (live-observed leakage).
    "evidence": "推荐依据",
    "basis": "依据",
    "canonical_id": "曲目标识",
    "target_id": "目标曲目标识",
    "candidate_id": "候选曲目标识",
    "run_id": "推荐批次标识",
    "preview_sounding": "试听状态",
    "active_batch": "当前批次",
    "runs_total": "推荐批总数",
    "preview_only": "只能试听",
    "unavailable": "不可播放",
    "library": "可正式播放",
    "agent_selected": "Agent 选的曲目",
    "own_queue": "你自己的队列",
}

# Fixed process-narration lead-ins banned by the R1 prompt (provider_agent.py
# output contract), extended in P14-R4.3 with the live-observed lead-in set
# and again in P20 Fix 04 (第三人称复述用户意图、规则自述、信息充足/让我确认
# self-instruction、工具结果复述).  The whole sentence — up to its
# terminating punctuation or newline — is stripped.  Deliberately closed:
# near-misses in natural speech are kept.
_SELF_NARRATION_PREFIXES = (
    "我如实告知用户",
    "我应该向用户说明",
    "我应该向用户",
    "我应该如实",
    "我需要如实向用户",
    "我需要向用户说明",
    "我需要向用户",
    "我需要如实说明",
    "用户明确说",
    "用户明确表示",
    "用户明确要求",
    "用户要求我",
    "用户说了",
    "用户让我",
    "用户点名",
    "按规则",
    "根据规则",
    "根据播放意图规则",
    "按播放意图规则",
    "按照播放意图规则",
    "播放意图规则",
    "我不能自动降级",
    "按照规则",
    "用户要求",
    "应如实告知用户",
    "如实告知用户",
    "我需要如实向你",
    "我需要向你说明",
    "我来为用户",
    "我要为用户",
    "让我为用户",
    "让我向用户",
    "让我向你",
    "我已经收集到",
    "搜索确认",
    "我理解了当前批次",
    "信息充足",
    "我已经掌握",
    "让我逐条说明",
    "让我为你说明",
    "让我说明",
    # P20-Fix06 (UAT-live): 让我结合…来回答你 / 让我结合…说明理由 -- the
    # whole sentence is process narration, not an answer.
    "让我结合",
    "让我依据",
    "让我梳理",
    "让我核实",
    "让我整理",
    "让我介绍",
    "让我列出",
    "让我分析",
    "让我确认",
    "让我来确认",
    "让我来核实",
    "让我来梳理",
    "让我来介绍",
    "让我来整理",
    "让我来说明",
    "让我来分析",
    "让我检查",
    "让我看看",
    "让我先查",
    "我查一下",
    "我来查",
    "我来整理",
    "我来逐条",
    "按照顺序给出",
    "我从工具结果得知",
    "根据工具结果",
    "根据工具返回",
    "工具返回",
    "工具结果显示",
    # Live-observed English thinking preambles (P20 Fix 04 live run):
    # the model wrote its whole self-instruction answer in English.
    "I have all the information I need",
    "Let me explain",
)

# P20 Fix 04 fail-safe: a reply scrubbed down to nothing must never surface
# as an empty answer — this single stable honest sentence stands in instead.
_EMPTY_TEXT_FALLBACK = "抱歉，这次没能整理出可展示的回答，请换个说法再试一次。"

_WORD_EDGE = r"(?<![A-Za-z0-9_０-９Ａ-Ｚａ-ｚ＿])"
_WORD_EDGE_END = r"(?![A-Za-z0-9_０-９Ａ-Ｚａ-ｚ＿])"

# Full-width ASCII (U+FF01..U+FF5E, incl. U+FF3F full-width underscore) →
# half-width, table-built for the value group normalisation used by the
# boolean composites.
_FULLWIDTH_TO_ASCII = {
    0xFF01 + offset: chr(0x21 + offset) for offset in range(0x5F)
}


def _wide_ascii_pattern(token: str) -> str:
    """Pattern fragment for ``token`` accepting half- or full-width ASCII.

    Each ASCII character alternatively matches its full-width equivalent
    (U+FF00 block letters/digits, U+FF3F full-width underscore), so model
    rewrites in full-width form are caught on equal footing.
    """
    fragments = []
    for char in token:
        if ("a" <= char <= "z") or ("A" <= char <= "Z") or ("0" <= char <= "9"):
            fragments.append(f"[{char}{chr(ord(char) + 0xFEE0)}]")
        elif char == "_":
            fragments.append("[_＿]")
        elif char == ".":
            fragments.append("[.．]")
        else:
            fragments.append(re.escape(char))
    return "".join(fragments)


def _normalize_width(value: str) -> str:
    """Map full-width ASCII (letters/digits/underscore) back to half-width."""
    return value.translate(_FULLWIDTH_TO_ASCII)


_FIELD_VALUE_RULES = [
    (
        re.compile(_WORD_EDGE + _wide_ascii_pattern(token) + _WORD_EDGE_END),
        replacement,
    )
    for token, replacement in _FIELD_VALUE_NAMES.items()
]

# A value attached to a field name: wrapped separators (为/是/is with
# optional spacing and optional 的/情况/状态/标注/标 lead-ins — the live runs
# wrote 「标注为」「的情况是」 shapes), a bare colon/'=' (half or full
# width), or plain whitespace.  ``context为unknown`` (contracted) is caught
# on equal footing with the spaced forms.
_VALUE_SEPARATOR = (
    r"(?:[\s:：=＝]*(?:的)?(?:情况|状态)?[\s:：=＝]*(?:标注|标)?"
    r"(?:为|是|is)[\s:：=＝]*|[\s:：=＝]+)"
)

# Markdown bold around a value (``**value**``) must not defeat the
# field+value composites: both the leading and trailing ``**`` are consumed
# with the match so the replacement never leaves unbalanced bold behind.
_AST_BEFORE = r"\**[ \t]*"
_AST_AFTER = r"[ \t]*\**"

# Route values collapse to their full natural capability phrase when attached
# to a route field — the P20 Fix 04 improvement over the P14 shape
# 「route 为 只能试听」.  The bare token map above still covers a standalone
# value, so an attached value is consumed whole and can never dangle.
_ROUTE_REPLACEMENTS = {
    "preview_only": "只能试听 30 秒",
    "library": "可以正式播放",
    "unavailable": "无法播放",
}

_ROUTE_COMPOSITES = [
    re.compile(
        _WORD_EDGE
        + _wide_ascii_pattern(field)
        + _VALUE_SEPARATOR
        + _AST_BEFORE
        + "(?P<value>"
        + _wide_ascii_pattern(value)
        + ")"
        + _AST_AFTER
        + _WORD_EDGE_END,
        re.IGNORECASE,
    )
    for field in ("playback.route", "playback_route", "route")
    for value in _ROUTE_REPLACEMENTS
]


def _route_composite_replacement(match: re.Match) -> str:
    return _ROUTE_REPLACEMENTS[_normalize_width(match.group("value")).lower()]


# preview_sounding boolean: the field plus its true/false value collapses to
# the honest audible-state sentences (the same wording the fixed fast-path
# doors already answer with), so 「试听状态 为 false」 can never reach the
# user.
_PREVIEW_SOUNDING_COMPOSITE = re.compile(
    _WORD_EDGE
    + _wide_ascii_pattern("preview_sounding")
    + _VALUE_SEPARATOR
    + _AST_BEFORE
    + r"(?P<flag>"
    + _wide_ascii_pattern("true")
    + r"|"
    + _wide_ascii_pattern("false")
    + r")"
    + _AST_AFTER
    + _WORD_EDGE_END,
    re.IGNORECASE,
)

_PREVIEW_SOUNDING_SUB = {
    "true": "有试听正在播放",
    "false": "当前没有正在播放的试听",
}


def _preview_sounding_replacement(match: re.Match) -> str:
    return _PREVIEW_SOUNDING_SUB[_normalize_width(match.group("flag")).lower()]


# fresh_item_count N: the batch-level count collapses to the natural sentence
# 这 N 首都是本次新发现 (N kept verbatim, half- or full-width digits).
_FRESH_ITEM_COUNT_COMPOSITE = re.compile(
    _WORD_EDGE
    + _wide_ascii_pattern("fresh_item_count")
    + _VALUE_SEPARATOR
    + _AST_BEFORE
    + r"(?P<count>[0-9０-９]+)"
    + r"(?:[ \t]*首)?"
    + _AST_AFTER
    + _WORD_EDGE_END,
    re.IGNORECASE,
)


def _fresh_item_count_replacement(match: re.Match) -> str:
    return f"这 {match.group('count')} 首都是本次新发现"


# fresh_this_request boolean (per-item flag): collapses to the item-level
# phrase; false keeps the honest negative rather than over-claiming novelty.
_FRESH_THIS_REQUEST_COMPOSITE = re.compile(
    _WORD_EDGE
    + _wide_ascii_pattern("fresh_this_request")
    + _VALUE_SEPARATOR
    + _AST_BEFORE
    + r"(?P<flag>"
    + _wide_ascii_pattern("true")
    + r"|"
    + _wide_ascii_pattern("false")
    + r")"
    + _AST_AFTER
    + _WORD_EDGE_END,
    re.IGNORECASE,
)

_FRESH_THIS_REQUEST_SUB = {
    "true": "本次新发现",
    "false": "不是本次新发现",
}


def _fresh_this_request_replacement(match: re.Match) -> str:
    return _FRESH_THIS_REQUEST_SUB[_normalize_width(match.group("flag")).lower()]


# `unknown` is generic English, so it is only scrubbed when tied to the
# `context` field (risk §4.1): the whole "context 为 unknown"-shaped span is
# collapsed to the mapped value instead of leaving a dangling "context 为".
_CONTEXT_UNKNOWN_COMPOSITE = re.compile(
    _wide_ascii_pattern("context")
    + _VALUE_SEPARATOR
    + _wide_ascii_pattern("unknown"),
    re.IGNORECASE,
)

# Mapped-form composites: when the English field+value composite could not
# match (e.g. 「标注为」 or mixed formatting before the extension covered
# it), rule 2's bare map leaves a Chinese field name attached to its value —
# those residues collapse to the same natural phrases so no Chinese-dressed
# field name ever reaches the user.  The replacement helpers are shared with
# the English composites.
_CHINESE_ROUTE_SUB = {
    "只能试听": "只能试听 30 秒",
    "可正式播放": "可以正式播放",
    "不可播放": "无法播放",
}

_CHINESE_ROUTE_COMPOSITES = [
    re.compile(
        "播放情况"
        + _VALUE_SEPARATOR
        + _AST_BEFORE
        + "(?P<value>"
        + value
        + r")"
        + _AST_AFTER
    )
    for value in _CHINESE_ROUTE_SUB
]


def _chinese_route_replacement(match: re.Match) -> str:
    return _CHINESE_ROUTE_SUB[match.group("value")]


_CHINESE_SOUNDING_COMPOSITE = re.compile(
    "试听状态"
    + _VALUE_SEPARATOR
    + _AST_BEFORE
    + r"(?P<flag>"
    + _wide_ascii_pattern("true")
    + r"|"
    + _wide_ascii_pattern("false")
    + r")"
    + _AST_AFTER,
    re.IGNORECASE,
)

_CHINESE_FRESH_COUNT_COMPOSITE = re.compile(
    "本次新发现的曲目数"
    + _VALUE_SEPARATOR
    + _AST_BEFORE
    + r"(?P<count>[0-9０-９]+)"
    + r"(?:[ \t]*首)?"
    + _AST_AFTER
)

_CHINESE_FRESH_FLAG_COMPOSITE = re.compile(
    "是否本次新发现"
    + _VALUE_SEPARATOR
    + _AST_BEFORE
    + r"(?P<flag>"
    + _wide_ascii_pattern("true")
    + r"|"
    + _wide_ascii_pattern("false")
    + r")"
    + _AST_AFTER,
    re.IGNORECASE,
)

# The enclosing phrase is stripped: the match walks back to the nearest
# boundary — sentence-ender, comma/semicolon/colon, or the start of the text
# (a lead-in like 信息充足 often sits mid-sentence: 「现在信息充足，我来为
# 你解释…」 is the live Fix03 shape; real content in the same 逗号 run-on
# survives because the strip starts at the nearest 标点, not the distant
# sentence start), consumes to the next sentence end, and keeps only the
# boundary so the neighbouring text stays intact.
_SELF_NARRATION = re.compile(
    r"(?P<lead>[。！？!?\n，；：,;:]|^)[^。！？!?\n，；：,;:]*?(?:"
    + "|".join(re.escape(prefix) for prefix in _SELF_NARRATION_PREFIXES)
    + r")[^。！？!?\n]*[。！？!?]?"
)

_ID_PATTERN = re.compile(
    r"(rcm|cnd|fbk|trk|art|alb|pl|pm|int|agt)_[0-9a-fA-F-]{32,}"
)


def sanitize_user_text(text: str) -> str:
    """Scrub internal tokens out of model text headed for the user.

    Returns a new string; non-string input returns unchanged.
    The three closed rules (design §3.2): internal ID patterns, field/value
    literals and composites (both half- and full-width ASCII), and fixed
    process-narration sentence lead-ins.  A text scrubbed down to nothing
    returns the stable ``_EMPTY_TEXT_FALLBACK`` sentence instead of an empty
    answer (P20 Fix 04 fail-safe).
    """
    if not isinstance(text, str):
        return text

    sanitized = _scrub_internal_ids(text)
    sanitized = _scrub_field_value_literals(sanitized)
    sanitized = _scrub_self_narration(sanitized)
    sanitized = sanitized.strip()
    if not sanitized:
        return _EMPTY_TEXT_FALLBACK
    return sanitized


def _scrub_internal_ids(text: str) -> str:
    """Rule 1: prefix+UUID ids → readable Chinese class words."""
    return _ID_PATTERN.sub(
        lambda match: _ID_PREFIX_NAMES[match.group(1)], text
    )


def _scrub_field_value_literals(text: str) -> str:
    """Rule 2: composites first (route/boolean/count fields with their
    values), then the bare field/value token map, then ``context unknown``.
    Composites run first so the attached value is consumed whole."""
    for composite in _ROUTE_COMPOSITES:
        text = composite.sub(_route_composite_replacement, text)
    text = _PREVIEW_SOUNDING_COMPOSITE.sub(_preview_sounding_replacement, text)
    text = _FRESH_ITEM_COUNT_COMPOSITE.sub(_fresh_item_count_replacement, text)
    text = _FRESH_THIS_REQUEST_COMPOSITE.sub(_fresh_this_request_replacement, text)
    for pattern, replacement in _FIELD_VALUE_RULES:
        text = pattern.sub(replacement, text)
    for composite in _CHINESE_ROUTE_COMPOSITES:
        text = composite.sub(_chinese_route_replacement, text)
    text = _CHINESE_SOUNDING_COMPOSITE.sub(_preview_sounding_replacement, text)
    text = _CHINESE_FRESH_COUNT_COMPOSITE.sub(_fresh_item_count_replacement, text)
    text = _CHINESE_FRESH_FLAG_COMPOSITE.sub(_fresh_this_request_replacement, text)
    text = _CONTEXT_UNKNOWN_COMPOSITE.sub("无法判断", text)
    return text


def _scrub_self_narration(text: str) -> str:
    """Rule 3: strip fixed-prefix self-narration sentences, keep the rest."""
    # subn-iterate to convergence: a stripped sentence consumes its terminal
    # stop, and the next sentence then has no leading boundary for its
    # ``^``-anchored match in the same pass — one pass per stripped sentence
    # resolves run-on narration.  Each productive pass strictly shrinks the
    # text, and a pass with zero replacements breaks, so this terminates.
    scrubbed = text
    while True:
        def _replace(match: re.Match) -> str:
            lead = match.group("lead")
            # A comma/semicolon/colon lead-in to a wholly removed process
            # sentence dangling before a line break (「…推断），让我向你
            # 说明…。」) is dropped with it; a mid-line comma is kept as the
            # conjunction between the surviving clauses.
            if lead in "，；：" and (
                match.end() == len(scrubbed) or scrubbed[match.end()] == "\n"
            ):
                return ""
            return lead

        scrubbed, count = _SELF_NARRATION.subn(_replace, scrubbed)
        if count == 0:
            break
    # Collapse a run of spaces left behind mid-line, and blank lines fully
    # consumed by a removed sentence; paragraph breaks elsewhere are kept.
    # Two adjacent sentences both stripped leave ``。。`` behind — collapse
    # the doubled stop.
    scrubbed = re.sub(r"[ \t]{2,}", " ", scrubbed)
    scrubbed = re.sub(r"\n[ \t]*\n", "\n", scrubbed)
    scrubbed = re.sub(r"([。！？])\1+", r"\1", scrubbed)
    # A stripped opening sentence can still leave a dangling 「。」 held by
    # the following sentence's lead, so a reply can never open with a stop.
    scrubbed = re.sub(r"^[。]+", "", scrubbed)
    return scrubbed