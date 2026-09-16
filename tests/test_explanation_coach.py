"""P20 Fix 11: the deterministic recommendation-explanation fast path.

Three layers share one file:

* ``ExplanationCoachFakeClientTest`` -- the deterministic executor over
  scripted tool results: closed-form classification, active-batch resolution,
  zero provider rounds (the exact call sequence is asserted), and the two
  fixed fail-honest sentences (no batch / unreadable or legacy run);
* ``ExplanationCliSurfaceTest`` -- the CLI wiring (_run_explanation): the
  closed explanation lines print the deterministic text through the Fix08
  boundary and never reach the provider loop, everything else remits;
* ``ExplanationWebSurfaceTest`` -- the web fast path: the same reply through
  the /api/chat machinery (deterministic rendered text or the fixed
  sentences), batch stays null, and the provider loop is never invoked.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import music_agent.cli  # noqa: F401  (the surface tests import the helper)
from music_agent.explanation_coach import (
    NO_ACTIVE_BATCH_REPLY,
    RUN_READ_FAILURE_REPLY,
    run_recommendation_explanation,
)
from tests.test_direction_coach import error_result, ok_result

ISO = "2026-08-16T00:00:00+00:00"
CLIENT_ID = "agt_eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"

# The six closed forms of the mandate, plus the household neighbours the
# classifier also covers.
MANDATE_FORMS = (
    "为什么推荐这些？",
    "为什么这些适合我？",
    "为什么这一批适合我？",
    "为什么这批适合我？",
    "这批为什么适合我？",
    "这批推荐为什么适合我？",
)
NEIGHBOUR_FORMS = (
    "为什么给我推荐这些",
    "这几首为什么适合我",
    "推荐理由是什么",
    "为什么推荐这几首",
)


def anime_batch_run() -> dict:
    """The default active run: Anime + track-self + zero-basis-with-note."""
    return {
        "run_id": "rcm_active",
        "item_count": 3,
        "items": [
            {
                "candidate_id": "cnd_1",
                "target_id": "trk_d1111111-1111-4d11-8d11-000000000001",
                "name": "Anime Song",
                "artist_name": "Anime Artist",
                "evidence": {
                    "mechanism": "推断偏好",
                    "basis": [{"kind": "genre", "label": "Anime", "provenance": "推断"}],
                },
            },
            {
                "candidate_id": "cnd_2",
                "target_id": "trk_d1111111-1111-4d11-8d11-000000000002",
                "name": "夜曲",
                "artist_name": None,
                "evidence": {
                    "mechanism": "推断偏好",
                    "basis": [{"kind": "track", "label": "夜曲", "provenance": "推断"}],
                },
            },
            {
                "candidate_id": "cnd_3",
                "target_id": "trk_d1111111-1111-4d11-8d11-000000000003",
                "name": "Unknown",
                "artist_name": None,
                "evidence": {
                    "mechanism": "exploration",
                    "basis": [],
                    "note": "本次目录搜索的新发现（暂无偏好匹配证据）",
                },
            },
        ],
    }


class ExplanationScriptedClient:
    """Closed script of the two reads the fast path performs -- anything else
    (a generation call, a preference re-query, a provider round) fails the
    test by construction."""

    def __init__(
        self,
        *,
        active_run_id: str | None = "rcm_active",
        context_outcome: str = "ok",
        context_raises: bool = False,
        run_payload: dict | None = None,
        run_outcome: str = "ok",
        run_raises: bool = False,
    ) -> None:
        self.active_run_id = active_run_id
        self.context_outcome = context_outcome
        self.context_raises = context_raises
        self.run_payload = run_payload if run_payload is not None else anime_batch_run()
        self.run_outcome = run_outcome
        self.run_raises = run_raises
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool: str, payload: dict):
        self.calls.append((tool, dict(payload)))
        if tool == "get_active_context":
            if self.context_raises:
                raise RuntimeError("agent offline")
            if self.context_outcome != "ok":
                return error_result(tool, self.context_outcome)
            batch = None
            if self.active_run_id is not None:
                batch = {
                    "run_id": self.active_run_id,
                    "source": "register",
                    "produced_at": ISO,
                    "item_count": 3,
                }
            return ok_result(tool, {"active_batch": batch})
        if tool == "get_recommendation_run":
            if self.run_raises:
                raise RuntimeError("run read failed")
            if self.run_outcome != "ok":
                return error_result(tool, self.run_outcome)
            return ok_result(tool, self.run_payload)
        raise AssertionError(f"unexpected tool {tool}")

    def close_event_listener(self) -> None:
        pass


class ExplanationCoachFakeClientTest(unittest.TestCase):
    def _coach_call(self, client: ExplanationScriptedClient, line: str) -> dict | None:
        return run_recommendation_explanation(client, line)

    def test_rendered_batch_is_byte_deterministic(self) -> None:
        client = ExplanationScriptedClient()
        outcome = self._coach_call(client, "为什么推荐这些？")
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["kind"], "rendered")
        self.assertEqual(
            outcome["text"],
            "这一批主要来自 Anime 方向。\n\n"
            "1. Anime Song — Anime Artist\n这首按 Anime 方向推断出来。\n\n"
            "2. 夜曲\n这首按对这首曲目本身的偏好推断递选。\n\n"
            "3. Unknown\n这是本次新发现，目前没有更直接的偏好匹配证据。",
        )
        # Zero provider rounds by construction: the ONLY reads are the context
        # read and the run read -- no generation, no preference re-query.
        self.assertEqual(
            client.calls,
            [
                ("get_active_context", {}),
                ("get_recommendation_run", {"run_id": "rcm_active"}),
            ],
        )

    def test_all_closed_mandate_forms_claim_the_fast_path(self) -> None:
        for form in MANDATE_FORMS:
            with self.subTest(form=form):
                client = ExplanationScriptedClient()
                outcome = self._coach_call(client, form)
                self.assertIsNotNone(outcome)
                self.assertIn("这一批主要来自 Anime 方向。", outcome["text"])
                self.assertEqual(len(client.calls), 2)

    def test_neighbour_explanation_forms_claim_it_too(self) -> None:
        for form in NEIGHBOUR_FORMS:
            with self.subTest(form=form):
                client = ExplanationScriptedClient()
                outcome = self._coach_call(client, form)
                self.assertIsNotNone(outcome)

    def test_non_explanation_lines_remit_untouched(self) -> None:
        for line in ("推荐几首歌", "再来一批", "再来一批，换个方向", "你好", 42, None):
            with self.subTest(line=line):
                client = ExplanationScriptedClient()
                self.assertIsNone(self._coach_call(client, line))
                self.assertEqual(client.calls, [])

    def test_no_active_batch_fails_honest_without_generation(self) -> None:
        # Cleanly read context, no batch: the fixed sentence -- and nothing
        # else (a generation call would fail the script's AssertionError).
        client = ExplanationScriptedClient(active_run_id=None)
        outcome = self._coach_call(client, "为什么推荐这些？")
        self.assertEqual(outcome, {"kind": "reply", "text": NO_ACTIVE_BATCH_REPLY})
        self.assertEqual(client.calls, [("get_active_context", {})])

    def test_context_error_outcome_fails_honest(self) -> None:
        client = ExplanationScriptedClient(context_outcome="agent_runtime_offline")
        outcome = self._coach_call(client, "为什么推荐这些？")
        self.assertEqual(outcome, {"kind": "reply", "text": RUN_READ_FAILURE_REPLY})
        self.assertEqual(len(client.calls), 1)

    def test_context_transport_failure_fails_honest(self) -> None:
        client = ExplanationScriptedClient(context_raises=True)
        outcome = self._coach_call(client, "为什么推荐这些？")
        self.assertEqual(outcome, {"kind": "reply", "text": RUN_READ_FAILURE_REPLY})

    def test_run_read_error_fails_honest(self) -> None:
        client = ExplanationScriptedClient(run_outcome="recommendation_run_not_found")
        outcome = self._coach_call(client, "为什么推荐这些？")
        self.assertEqual(outcome, {"kind": "reply", "text": RUN_READ_FAILURE_REPLY})
        self.assertEqual(len(client.calls), 2)

    def test_run_read_transport_failure_fails_honest(self) -> None:
        client = ExplanationScriptedClient(run_raises=True)
        outcome = self._coach_call(client, "为什么推荐这些？")
        self.assertEqual(outcome, {"kind": "reply", "text": RUN_READ_FAILURE_REPLY})

    def test_legacy_run_without_evidence_fails_honest(self) -> None:
        # A pre-Fix09 replayed run (name-only items): the shared presenter
        # fails closed and the fixed sentence answers -- no guessed reason.
        client = ExplanationScriptedClient(
            run_payload={
                "run_id": "rcm_old",
                "item_count": 2,
                "items": [
                    {"name": "Old Song", "artist_name": "Old Artist"},
                    {"name": "Another Old Song"},
                ],
            }
        )
        outcome = self._coach_call(client, "为什么这一批适合我？")
        self.assertEqual(outcome, {"kind": "reply", "text": RUN_READ_FAILURE_REPLY})


class ExplanationFakeLoop:
    """The provider loop must never be reached by a closed explanation line."""

    def __init__(self, client: ExplanationScriptedClient) -> None:
        self.client = client
        self.messages: list[str] = []

    def run(self, text: str):
        self.messages.append(text)
        raise AssertionError(f"provider loop invoked for {text!r}")


class ExplanationCliSurfaceTest(unittest.TestCase):
    def test_cli_fast_path_prints_deterministic_explanation(self) -> None:
        from music_agent.cli import _run_explanation

        client = ExplanationScriptedClient()
        loop = ExplanationFakeLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_explanation(loop, "为什么推荐这些？")
        self.assertTrue(handled)
        output = stdout.getvalue()
        self.assertIn("这一批主要来自 Anime 方向。", output)
        self.assertIn("1. Anime Song — Anime Artist\n这首按 Anime 方向推断出来。", output)
        self.assertIn("这是本次新发现，目前没有更直接的偏好匹配证据。", output)
        self.assertEqual(loop.messages, [])  # provider loop never reached

    def test_cli_fast_path_prints_fixed_no_batch_sentence(self) -> None:
        from music_agent.cli import _run_explanation

        client = ExplanationScriptedClient(active_run_id=None)
        loop = ExplanationFakeLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_explanation(loop, "为什么这批适合我？")
        self.assertTrue(handled)
        self.assertEqual(stdout.getvalue(), NO_ACTIVE_BATCH_REPLY + "\n")
        self.assertEqual(loop.messages, [])
        self.assertEqual(len(client.calls), 1)  # one context read, nothing else

    def test_cli_fast_path_prints_fixed_read_failure_sentence(self) -> None:
        from music_agent.cli import _run_explanation

        client = ExplanationScriptedClient(run_raises=True)
        loop = ExplanationFakeLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_explanation(loop, "为什么推荐这些？")
        self.assertTrue(handled)
        output = stdout.getvalue()
        self.assertEqual(output, RUN_READ_FAILURE_REPLY + "\n")
        self.assertNotIn("Traceback", output)
        self.assertNotIn("SSL", output)
        self.assertNotIn("provider", output)

    def test_cli_fast_path_remits_non_explanation_lines(self) -> None:
        from music_agent.cli import _run_explanation

        client = ExplanationScriptedClient()
        loop = ExplanationFakeLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_explanation(loop, "推荐几首歌")
        self.assertFalse(handled)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(client.calls, [])


class _FakeCards:
    def __init__(self) -> None:
        self.run_id: str | None = None
        self.cards: list[dict] = []

    def latest_run_id(self) -> str | None:
        return self.run_id

    def latest_cards(self) -> list[dict]:
        return self.cards


class ExplanationWebSurfaceTest(unittest.TestCase):
    def _app(self, client: ExplanationScriptedClient):
        from music_agent.web_shell import ShellConfig, WebShellApp

        with tempfile.TemporaryDirectory() as temp:
            config = ShellConfig(
                database_path=Path(temp) / "store.db",
                provider_factory=lambda: None,
                agent_client=(CLIENT_ID, "full"),
                mode="standalone",
            )
            app = WebShellApp(config)
        app._loop = ExplanationFakeLoop(client)
        app._cards = _FakeCards()
        return app

    def test_web_fast_path_renders_deterministically(self) -> None:
        client = ExplanationScriptedClient()
        app = self._app(client)
        reply = app._explanation_fast_path("为什么推荐这些？")
        self.assertIsNotNone(reply)
        self.assertIn("这一批主要来自 Anime 方向。", reply["reply"])
        self.assertIn("1. Anime Song — Anime Artist\n这首按 Anime 方向推断出来。", reply["reply"])
        self.assertTrue(reply["reply_html"])
        self.assertFalse(reply["rounds_capped"])
        self.assertIsNone(reply["batch"])  # an explanation never generates a batch
        for banned in ("score", "满分", "rcm_", "cnd_", "trk_", "mechanism",
                       "provenance", "basis", "让我补充", "让我核对"):
            self.assertNotIn(banned, reply["reply"])
        self.assertEqual(app._loop.messages, [])  # provider loop never reached

    def test_web_fast_path_fixed_sentences(self) -> None:
        no_batch = self._app(ExplanationScriptedClient(active_run_id=None))
        reply = no_batch._explanation_fast_path("为什么推荐这些？")
        self.assertEqual(reply["reply"], NO_ACTIVE_BATCH_REPLY)
        self.assertIsNone(reply["batch"])
        unreadable = self._app(ExplanationScriptedClient(run_raises=True))
        reply = unreadable._explanation_fast_path("这批为什么适合我？")
        self.assertEqual(reply["reply"], RUN_READ_FAILURE_REPLY)

    def test_web_fast_path_remits_non_explanation_lines(self) -> None:
        app = self._app(ExplanationScriptedClient())
        self.assertIsNone(app._explanation_fast_path("再来一批，换个方向"))
        self.assertIsNone(app._explanation_fast_path("推荐几首歌"))
        self.assertIsNone(app._explanation_fast_path("你好"))


if __name__ == "__main__":
    unittest.main()