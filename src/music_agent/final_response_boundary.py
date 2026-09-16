"""P20 Fix 08: the final user response boundary (Layer 2 over Fix 04).

The UAT after Fix 04 still surfaced the model's whole internal process to the
user: abbreviated internal ids (``rcm_1b4e0124``, ``trk_a7827505`` -- the
Layer-1 id pattern only replaces the full 32-hex UUID form), bare internal
field names Layer 1 does not map (``source_path``, ``basis_targets``, ...),
and whole process-narration blocks in phrasings the closed Layer-1 prefix
vocabulary could never enumerate (让我核对 / 规则说 / 我不该说 / 向用户如实
呈现 / ...).  Chasing more synonyms is a losing game (Fix 04 already proved
the model rephrases indefinitely), so this module adds the structural layer
the mandate demands: a fail-closed VALIDATOR on the scrubbed text, shared by
every user-facing door (cli chat, cli chat-session, web /api/chat).

Two layers (sec.4):

* Layer 1 ``sanitize_user_text`` -- unchanged -- keeps doing the safe small
  transforms it is good at (route phrases, counts/booleans, full-form ids,
  known narration lead-ins).  Its public contract (including the pinned
  short-id passthrough behaviour) is deliberately untouched.
* Layer 2 this module -- detects what Layer 1 cannot safely rewrite:

  A. internal ids of ANY length (the abbreviated residues Layer 1 ignores),
  B. bare internal field names that reached the user text (residues of the
     unmapped set; the mapped ones are already Chinese by the time Layer 2
     runs),
  C. hard process-leak markers: self-instruction (让我核对/让我组织/我需要先/
     我只能严格按照/...), self-restraint (我不该说/我不能基于…自编), rule
     recitement (规则说/按照规则/系统提示/内部字段), third-person 用户
     narration (向用户/用户当前/用户要求/…), and tool-result narration
     (工具结果/工具返回/…).

Fail-closed semantics (sec.5/6): contamination is never patched word-by-word
(the broken leftovers of internal reasoning must not reach the user).  A
contaminated text is reduced to ONE coherent clean region -- the answer tail
after a contaminated head (the UAT shape: process block, then the real final
answer), or the answer head before a contaminated tail -- and only when that
region is a certain boundary (non-blank, ≥12 chars).  Anything else collapses
to a stable per-task fallback sentence (sec.5): the explanation fallback, the
recommendation fallback, or the default scrubbed-empty fallback reused from
Layer 1.  NEVER the raw provider text (sec.6): sanitizer exceptions fall back
too, and there is no code path that echoes the unsanitized text.

Deterministic and closed: no NLP, no model judgment, pure regex over the
text.  The closed marker vocabulary is deliberately hard-nosed -- normal
first-person assistant sentences (我可以帮你试听这首歌 / 我不能正式播放这
首歌，但可以试听 30 秒 / 根据你的偏好记录…) contain none of the markers
and pass byte-identical (sec.9).
"""

from __future__ import annotations

import re

from music_agent.output_sanitizer import (
    _WORD_EDGE,
    _WORD_EDGE_END,
    _wide_ascii_pattern,
    sanitize_user_text,
)

__all__ = ["FINAL_RESPONSE_FALLBACKS", "present_final_text", "presentation_fallback_kind"]

# Per-task stable fallbacks (sec.5).  Canned and trusted: none of them needs
# re-validation, and each reads as a natural assistant reply rather than a
# generic error.
_EXPLANATION_FALLBACK = (
    "这批推荐主要依据你现有的偏好证据生成。"
    "当前这次解释没有整理好，我可以重新为你说明每一首的推荐依据。"
)
_RECOMMENDATION_FALLBACK = (
    "这次推荐已经生成，但回答内容没有整理好。你可以让我重新列出这一批推荐。"
)
_DEFAULT_FALLBACK = "抱歉，这次没能整理出可展示的回答，请换个说法再试一次。"

FINAL_RESPONSE_FALLBACKS = {
    "default": _DEFAULT_FALLBACK,
    "recommendation": _RECOMMENDATION_FALLBACK,
    "explanation": _EXPLANATION_FALLBACK,
}

# --- Layer 2 detectors -------------------------------------------------------

# Any-length internal id: the shared prefix family plus one or more hexish
# characters after the underscore, half- or full-width.  Layer 1 has already
# replaced every full 32-hex UUID with its Chinese class word, so whatever
# matches here is a residue the model wrote itself (usually abbreviated) --
# an internal id must never reach any user surface (sec.7). The prefix
# itself is wide-aware too: 全角 ｒｃｍ＿１ｂ４ｅ０１２４ is just as internal.
_ID_PREFIXFAMILY = (
    "(?:"
    + "|".join(
        _wide_ascii_pattern(prefix)
        for prefix in ("rcm", "cnd", "fbk", "trk", "art", "alb", "pl", "pm", "int", "agt")
    )
    + ")"
)
_INTERNAL_ID_RE = re.compile(
    _WORD_EDGE
    + _ID_PREFIXFAMILY
    + r"[_＿][0-9a-fA-F０-９Ａ-Ｆａ-ｆ][0-9a-fA-F０-９Ａ-Ｚａ-ｚ-]{0,}"
    + _WORD_EDGE_END
)

# Bare internal field names (closed set; sec.4-B).  The mapped ones are
# Chinese replacements by the time Layer 2 runs, so this exists to catch the
# unmapped residues the model copies from tool payloads.
_FIELD_NAMES = (
    "basis_targets",
    "source_path",
    "source_system",
    "source_event_id",
    "canonical_id",
    "target_id",
    "candidate_id",
    "run_id",
    "referent_canonical_id",
    "fresh_this_request",
    "fresh_item_count",
    "preview_sounding",
    "active_batch",
    "runs_total",
    "provenance",
    "provenance.kind",
    "route",
    "bindings",
    "artist_ids",
    "artist_name",
    "play_count",
    "persistent_id",
    "apple_music_persistent_id",
    "apple_music_catalog_id",
    "itunes_store_id",
    "persistent id",
    "canonical",
    "apple_music_library",
)

_FIELD_NAME_RE = re.compile(
    _WORD_EDGE
    + "(?:"
    + "|".join(_wide_ascii_pattern(field) for field in _FIELD_NAMES)
    + ")"
    + _WORD_EDGE_END,
    re.IGNORECASE,
)

# Hard process-leak markers (sec.4-C/8).  Deliberately closed and hard-nosed:
# every entry is a phrase a final answer to a real person essentially never
# contains.  The single-word traps of sec.9 are absent on purpose -- 我, 可
# 以, 用户, 规则, 依据 alone are not markers; 某些正常句 (我不能正式播放这
# 首歌 / 我可以帮你试听这首歌 / 根据你的偏好记录…) contain none of these.
_PROCESS_LEAK_MARKERS = (
    # self-instruction / planning
    "让我核对", "让我组织", "让我回顾", "让我判断", "让我先",
    "我来组织", "我来布置", "我现在来", "现在我来",
    "我需要先", "我需要按", "我需要严格", "我需要只",
    "我需要组织", "我需要整理", "我需要检查", "我需要核对",
    "我应该", "让我重新考虑", "用户编号",
    # self-restraint / rule recitement
    "我不该说", "我不能基于", "我不能自编", "我不能编", "我不会编", "我不应编",
    "规则说", "按照规则", "按规则", "规则规定", "规则要求", "规则里",
    "内部规则", "系统提示", "提示词", "内部字段",
    "推荐依据字段", "依据字段",
    # third-person 用户 narration
    "向用户", "用户当前", "用户明确", "用户要求", "用户说", "用户正在",
    "用户刚才", "用户已经", "告诉用户", "呈现给用户", "对用户", "为用户",
    "用户可见", "用户侧", "用户需要", "用户喜欢", "用户偏好",
    # tool-result narration
    "工具结果", "工具返回", "从工具", "工具里", "工具给我",
    "工具信息", "工具输出", "工具显示",
    # strictness recitement / honest-report preambles (live UAT shapes)
    "我只能严格", "我严格按照", "严格按照", "须严格按照",
    "需要如实", "必须如实", "需要说清", "必须说明",
    "应当如实", "应该如实", "如实呈现给",
    # provider scratch/reasoning in English
    "I should", "Let me reconsider", "Let me think", "The user implies",
    "The user asked", "I need to", "We need to",
)

_PROCESS_LEAK_RE = re.compile("|".join(_PROCESS_LEAK_MARKERS), re.IGNORECASE)

# A rejected region must still be a real answer, not one dangling clause
# ("并给出后续选项。" is exactly the tail that must NOT be salvaged).
_MIN_ACCEPTABLE_REGION_CHARS = 12

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")

# Provider terminal content has an explicit delivery envelope. Reasoning belongs
# outside it; malformed/multiple envelopes never become a partially patched reply.
FINAL_ANSWER_CONTRACT = (
    "\n最终交付契约：最后一个无工具调用的回复必须只包含一个 "
    "<final_answer>面向用户的完整回答</final_answer>。"
    "不要在其中放计划、思考过程或对用户意图的第三人称叙述。"
)
_FINAL_ENVELOPE_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
# Detect discourse roles, not track names or ordinary English answer vocabulary.
# Legacy unframed content remains supported only when it is not self-directed
# planning or third-person request analysis. Never salvage an arbitrary tail.
_SCRATCH_DISCOURSE_RE = re.compile(
    r"(?:^|[\n.!?]\s*)\s*(?:"
    r"Let me (?!know\b)|(?:I|We) (?:should|need to)\b|"
    r"(?:The )?user (?:wants|asked|needs|implies)\b|"
    r"This appears to be (?:a|an) [^.!?\n]*request\b|Since the message\b)", re.IGNORECASE
)


def extract_provider_final(text: str, *, fallback_kind: str = "default") -> str:
    """Extract one explicit final block, or accept clean legacy terminal prose.

    Mixed unframed scratch is rejected whole. This never removes words from an
    answer and never reads the provider's reasoning_content transport field.
    """
    if not isinstance(text, str):
        return _fallback_for(fallback_kind)
    if "<final_answer" in text or "</final_answer" in text:
        matches = list(_FINAL_ENVELOPE_RE.finditer(text))
        if (len(matches) != 1 or text.count("<final_answer>") != 1
                or text.count("</final_answer>") != 1):
            return _fallback_for(fallback_kind)
        candidate = matches[0].group(1).strip()
        if not candidate or _SCRATCH_DISCOURSE_RE.search(candidate) or _contaminated(candidate):
            return _fallback_for(fallback_kind)
        return candidate
    if _SCRATCH_DISCOURSE_RE.search(text):
        return _fallback_for(fallback_kind)
    return text


def _contaminated(unit: str) -> bool:
    """True when one sentence unit carries any Layer-2 leakage class."""
    return bool(
        _INTERNAL_ID_RE.search(unit)
        or _FIELD_NAME_RE.search(unit)
        or _PROCESS_LEAK_RE.search(unit)
    )


def _fallback_for(kind: str) -> str:
    return FINAL_RESPONSE_FALLBACKS.get(kind, _DEFAULT_FALLBACK)


def presentation_fallback_kind(
    user_text: str | None = None, tool_executions: object = ()
) -> str:
    """The per-task fallback class for one turn, shared by every door.

    Explanation-shaped turns get the explanation fallback; recommendation
    turns (classified or actually generated) get the recommendation one;
    everything else gets the default.  Deterministic from the same intent
    classifiers and generation record the doors already consult.
    """
    from music_agent.intent_router import (
        is_fresh_discovery_intent,
        is_recommendation_explanation_intent,
        is_recommendation_intent,
    )
    from music_agent.provider_agent import generation_succeeded

    if user_text is not None and is_recommendation_explanation_intent(user_text):
        return "explanation"
    if user_text is not None and (
        is_recommendation_intent(user_text)
        or is_fresh_discovery_intent(user_text)
    ):
        return "recommendation"
    try:
        generated = generation_succeeded(tool_executions)
    except Exception:
        generated = False
    return "recommendation" if generated else "default"


def present_final_text(
    text: str, *, fallback_kind: str = "default"
) -> str:
    """The ONE last step between provider text and a user surface.

    Layer 1 scrubs (safe replacements), Layer 2 validates: a clean text passes
    byte-identical, a contaminated text is reduced to its single coherent
    clean region (answer head or tail, only on a certain boundary), anything
    else becomes the stable per-task fallback.  A sanitizer exception falls
    back too -- the raw provider text is never returned (sec.6).
    """
    if not isinstance(text, str):
        return _fallback_for(fallback_kind)
    text = extract_provider_final(text, fallback_kind=fallback_kind)
    try:
        # A product name is prose, not the bare internal playback route enum.
        product_names: list[str] = []
        def protect_product(match: re.Match) -> str:
            product_names.append(match.group())
            return f"\ue000{len(product_names) - 1}\ue001"
        if "\ue000" in text or "\ue001" in text:
            return _fallback_for(fallback_kind)
        protected = re.sub(r"\bApple Music library\b", protect_product, text, flags=re.IGNORECASE)
        scrubbed = sanitize_user_text(protected)
        for index, name in enumerate(product_names):
            scrubbed = scrubbed.replace(f"\ue000{index}\ue001", name)
    except Exception:
        return _fallback_for(fallback_kind)
    scrubbed = scrubbed.strip()
    if not scrubbed:
        return _fallback_for(fallback_kind)

    units = [unit for unit in _SENTENCE_SPLIT_RE.split(scrubbed) if unit.strip()]
    flags = [_contaminated(unit) for unit in units]
    if not any(flags):
        return scrubbed

    last = max(index for index, flag in enumerate(flags) if flag)
    if flags[0]:
        # Contaminated head: the answer may follow the process block.
        candidate = "".join(units[last + 1 :])
    else:
        # Clean head: the answer leads and the process tails -- keep the head.
        first = next(index for index, flag in enumerate(flags) if flag)
        candidate = "".join(units[:first])
    candidate = candidate.strip()
    if candidate and len(candidate) >= _MIN_ACCEPTABLE_REGION_CHARS:
        return candidate
    return _fallback_for(fallback_kind)
