"""Session-local continuation state for assistant-offered actions.

P22-S2.1 deliberately keeps this state tiny and non-durable.  It remembers
only a single code-authorized action that the assistant has offered and lets
the next user turn either accept it, decline it, or replace it with a new
request.  It never parses prior assistant prose and never owns playback truth.
"""

from __future__ import annotations

from dataclasses import dataclass


PREVIEW_TRACK = "preview_track"


@dataclass(frozen=True, slots=True)
class OfferedAction:
    """One action the current conversation session may accept on the next turn."""

    kind: str
    target_canonical_id: str
    source: str
    verified_title: str | None = None
    verified_artist: str | None = None

    def __post_init__(self) -> None:
        if self.kind != PREVIEW_TRACK:
            raise ValueError("S2.1 only supports preview_track offered actions")
        if not isinstance(self.target_canonical_id, str) or not self.target_canonical_id:
            raise ValueError("offered action requires a canonical target")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("offered action requires a source")
        for field_name in ("verified_title", "verified_artist"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    """Result of arbitrating one user turn against the pending offered action."""

    outcome: str
    action: OfferedAction | None = None
    replacement_text: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"none", "accepted", "declined", "override"}:
            raise ValueError("invalid continuation outcome")
        if self.outcome == "accepted" and self.action is None:
            raise ValueError("accepted continuation requires an action")
        if self.outcome != "accepted" and self.action is not None:
            raise ValueError("only accepted continuation carries an action")
        if self.replacement_text is not None:
            if self.outcome != "override":
                raise ValueError("only override continuation may carry replacement text")
            if not isinstance(self.replacement_text, str) or not self.replacement_text:
                raise ValueError("replacement_text must be a non-empty string or None")


_GENERIC_ACCEPT = frozenset({"好的", "好", "可以", "行"})
_PREVIEW_ACCEPT = frozenset({"开始试听", "试听吧"})
_DECLINE = frozenset({"不用", "不用了", "算了", "不要"})
_TERMINAL_PUNCTUATION = "。！？!?"
_OVERRIDE_SEPARATORS = frozenset({"，", ",", "；", ";", "：", ":"})


def _closed_form(text: str) -> str:
    """Normalize only a complete short form; internal punctuation is preserved.

    Keeping commas and other internal punctuation is what makes
    ``不用，播放第二首`` an override instead of a pure decline.
    """

    return text.strip().casefold().rstrip(_TERMINAL_PUNCTUATION).strip()


def _replacement_text_after_decline(text: str) -> str | None:
    """Return a clearly delimited substantive request after a decline prefix.

    This is continuation arbitration, not intent parsing: the decline applies only
    to the pending OfferedAction, while the text after a comma/semicolon/colon is
    handed unchanged to the existing TurnPlan pipeline.  No assistant prose or
    target identity is inspected here.
    """

    stripped = text.strip()
    folded = stripped.casefold()
    for decline in sorted(_DECLINE, key=len, reverse=True):
        if not folded.startswith(decline):
            continue
        remainder = stripped[len(decline):].lstrip()
        if not remainder or remainder[0] not in _OVERRIDE_SEPARATORS:
            continue
        replacement = remainder[1:].strip()
        return replacement or None
    return None


class OfferedActionRegister:
    """Single-slot, session-local owner for assistant-offered actions."""

    def __init__(self) -> None:
        self._current: OfferedAction | None = None

    @property
    def current(self) -> OfferedAction | None:
        return self._current

    def arm(self, action: OfferedAction) -> None:
        if not isinstance(action, OfferedAction):
            raise TypeError("action must be an OfferedAction")
        # A new offer supersedes the previous single-slot value by definition.
        self._current = action

    def consume(self) -> OfferedAction | None:
        action = self._current
        self._current = None
        return action

    def expire(self) -> None:
        self._current = None

    def resolve(self, user_text: str) -> ContinuationDecision:
        """Resolve before normal TurnPlan/routing and apply lifecycle atomically.

        Accepted and declined offers are consumed immediately, before any
        external action executes.  Therefore an execution failure can never
        make a stale acceptance replayable.  Any other non-empty turn expires
        the offer and proceeds through the ordinary semantic pipeline.
        """

        if not isinstance(user_text, str):
            raise TypeError("user_text must be a string")
        action = self._current
        if action is None:
            return ContinuationDecision("none")

        form = _closed_form(user_text)
        if form in _DECLINE:
            self.consume()
            return ContinuationDecision("declined")

        accepted = form in _GENERIC_ACCEPT
        if action.kind == PREVIEW_TRACK:
            accepted = accepted or form in _PREVIEW_ACCEPT
        if accepted:
            consumed = self.consume()
            assert consumed is not None
            return ContinuationDecision("accepted", consumed)

        # Every other non-empty/substantive turn wins over the old offer.  The
        # host then runs its existing TurnPlan/routing/provider path unchanged.
        self.expire()
        return ContinuationDecision(
            "override", replacement_text=_replacement_text_after_decline(user_text)
        )
