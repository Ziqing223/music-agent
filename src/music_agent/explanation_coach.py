"""P20 Quality Fix 11: deterministic recommendation-explanation fast path.

The explanation requests (为什么推荐这些？ / 为什么这些适合我？ / 为什么这一批
适合我？ / …) were answered by provider free text -- the P20-Fix03/Fix06 link
kept the phrasing grounded, but the provider round itself exposed two UAT
failures: a raw transport traceback printed to the user
(provider_unavailable_error → urlopen → SSL), and planning narration (让我补充
查看这些被喜欢曲目的偏好画像…… / 让我基于已获取的信息来回答) leaking ahead of
the real answer. This module replaces that path for the closed request set
with deterministic rendering: the authoritative recommendation run
(``get_recommendation_run``) holds every fact an explanation needs -- each
item's evidence mechanism, basis rows and the fresh-discovery note -- so the
whole turn executes through ONE context read + ONE run read + the shared
presenter (``recommendation_presenter``). Zero provider rounds: no SSL
exposure, no planning narration, and the Fix09-eliminated re-query list
(list_feedback_observations / list_learning_applications /
query_track_preference×N / get_now_playing / get_canonical_entity) never
returns.

Reply contract (shared by cli chat, cli chat-session and web /api/chat):

* a closed explanation line with a readable active batch renders the
  deterministic explanation (program-computed direction summary + per-item
  reason copy fact-identical to the first presentation);
* no active batch → the fixed sentence 「当前没有可以解释的推荐批次。你可以先让
  我推荐几首歌。」 -- never a generation call;
* an unreadable context, an unreadable run, or a run payload outside the
  post-Fix09 item contract → the fixed sentence 「我现在没能读取到这一批的推荐
  依据。可以重新生成一批后再看推荐理由。」 -- never
  provider_unavailable_error / urlopen / SSL / execution_error or a raw
  exception;
* any other line → ``None``, and the caller keeps its ordinary provider path
  untouched (the recommendation-generation, catalog and everything else stay
  exactly as before).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from music_agent.direction_coach import _is_ok_tool_result
from music_agent.intent_router import is_recommendation_explanation_intent
from music_agent.recommendation_presenter import (
    render_recommendation_explanation_for_user,
)

__all__ = [
    "NO_ACTIVE_BATCH_REPLY",
    "RUN_READ_FAILURE_REPLY",
    "run_recommendation_explanation",
]

# Mandate §十一: no batch to explain -- the honest refusal, never a generation
# call.
NO_ACTIVE_BATCH_REPLY = "当前没有可以解释的推荐批次。你可以先让我推荐几首歌。"

# Mandate §十: the batch exists but its evidence cannot be read (unreadable
# context, failed run read, or a run payload outside the post-Fix09 item
# contract) -- a stable natural sentence, never a raw provider/transport error.
RUN_READ_FAILURE_REPLY = "我现在没能读取到这一批的推荐依据。可以重新生成一批后再看推荐理由。"


def run_recommendation_explanation(
    client: Any, line: str
) -> dict[str, str] | None:
    """Execute a closed explanation line deterministically, or remit (None).

    The active batch resolves through the household active-context rule --
    ``get_active_context().active_batch`` (the register pointer of this
    service, else the derived newest persisted run; the exact rule the
    direction coach and the T14 decoder doors already read, deliberately not
    redesigned here). One ``get_recommendation_run`` read then yields the
    evidence-carrying detail items the shared presenter renders. Fail-honest
    by construction: a cleanly-read context with no batch answers the fixed
    no-batch sentence, and any unreadable/failed read answers the fixed
    read-failure sentence. ``None`` means the line is not this fast path's
    business (not a closed explanation form): the caller keeps its ordinary
    provider path untouched.
    """
    if not isinstance(line, str) or not is_recommendation_explanation_intent(line):
        return None
    try:
        context = client.call("get_active_context", {})
        if not _is_ok_tool_result(context):
            return {"kind": "reply", "text": RUN_READ_FAILURE_REPLY}
        payload = context.payload
        batch = payload.get("active_batch") if isinstance(payload, Mapping) else None
        run_id: str | None = None
        if isinstance(batch, Mapping):
            candidate = batch.get("run_id")
            if isinstance(candidate, str) and candidate:
                run_id = candidate
    except Exception:
        # Unreadable context (transport/registry failure): the batch's basis
        # cannot be read -- the fixed sentence, never the raw exception.
        return {"kind": "reply", "text": RUN_READ_FAILURE_REPLY}
    if run_id is None:
        return {"kind": "reply", "text": NO_ACTIVE_BATCH_REPLY}
    try:
        run = client.call("get_recommendation_run", {"run_id": run_id})
    except Exception:
        return {"kind": "reply", "text": RUN_READ_FAILURE_REPLY}
    if not _is_ok_tool_result(run):
        return {"kind": "reply", "text": RUN_READ_FAILURE_REPLY}
    rendered = render_recommendation_explanation_for_user(run.payload)
    if rendered is None:
        # The payload is outside the post-Fix09 item contract (e.g. a legacy
        # replayed run with no evidence blocks): nothing can be explained
        # deterministically -- fail honest, never a guessed reason.
        return {"kind": "reply", "text": RUN_READ_FAILURE_REPLY}
    return {"kind": "rendered", "text": rendered}