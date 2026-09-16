"""P20 Quality Fix 05: direction-shift coach, surfaces and service truth.

Three layers share one file:

* ``DirectionCoachFakeClientTest`` -- the deterministic executor's policy over
  scripted tool results (selection, exclusion, fail-honest remit/fallback);
* ``DirectionCoachServiceIntegrationTest`` -- the whole chain against a REAL
  SharedAgentService on a fixture DB: real preference evidence, real
  ``generate_recommendation`` / ``generate_inferred_recommendation`` runs, the
  coach on top (mandate §十一 1-9);
* ``DirectionShiftSurfacesTest`` -- the CLI and web fast-path wiring: the
  closed shift lines never reach the provider loop, everything else still does
  (§十一(9): plain 再来一批 continues to the loop untouched).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentToolOutcome,
    AgentToolResult,
    generate_request_id,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.direction_coach import run_direction_shift
from music_agent.direction_shift import NO_ALTERNATIVE_DIRECTION_REPLY
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.source_observation import ObservedValue
from music_agent.validation import validate_fixture

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ISO = "2026-08-16T00:00:00+00:00"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)

M1 = "trk_d1111111-1111-4d11-8d11-000000000001"
M2 = "trk_d1111111-1111-4d11-8d11-000000000002"
M3 = "trk_d1111111-1111-4d11-8d11-000000000003"
J1 = "trk_d2222222-2222-4d22-8d22-000000000001"
J2 = "trk_d2222222-2222-4d22-8d22-000000000002"
R1 = "trk_d3333333-3333-4d33-8d33-000000000001"
R2 = "trk_d3333333-3333-4d33-8d33-000000000002"

ARTIST_ALPHA = "art_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ARTIST_BETA = "art_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
ARTIST_GAMMA = "art_cccccccc-cccc-4ccc-8ccc-cccccccccccc"


def ok_result(tool: str, payload: dict) -> AgentToolResult:
    return AgentToolResult(
        request_id=generate_request_id(),
        tool=tool,
        outcome=AgentToolOutcome.OK,
        payload=payload,
        error_code=None,
        error_message=None,
        completed_at=NOW,
    )


def error_result(tool: str, code: str) -> AgentToolResult:
    return AgentToolResult(
        request_id=generate_request_id(),
        tool=tool,
        outcome=AgentToolOutcome.EXECUTION_ERROR,
        payload=None,
        error_code=code,
        error_message=code,
        completed_at=NOW,
    )


def batch_items(target_ids: list[str], genre: str) -> list[dict]:
    return [
        {
            "candidate_id": f"cnd_{index}",
            "target_id": target_id,
            "name": f"Song {index}",
            "artist_name": "Artist",
            "evidence": {
                "mechanism": "直接偏好",
                "basis": [{"kind": "genre", "label": genre, "provenance": "直接"}],
            },
        }
        for index, target_id in enumerate(target_ids)
    ]


class ScriptedClient:
    """Closed script of the three tools the coach reads/writes + a truth reader."""

    def __init__(
        self,
        *,
        true_genres: tuple[str, ...],
        true_scope: tuple[str, ...],
        active_run_id: str | None = "rcm_prev",
        prev_targets: tuple[str, ...] = (M1, M2),
        prev_count: int | None = None,
        generated: list[dict] | None = None,
        gen_error: bool = False,
        empty_basis: bool = False,
    ) -> None:
        self.truth = (true_genres, true_scope)
        self.active_run_id = active_run_id
        self.prev_targets = prev_targets
        self.prev_count = prev_count
        self.generated = generated
        self.gen_error = gen_error
        self.empty_basis = empty_basis
        self.calls: list[tuple[str, dict]] = []

    @property
    def service(self):  # duck-typed: anything with direction_shift_inputs
        return self

    def direction_shift_inputs(self, source_system: str | None = None):
        return self.truth

    def call(self, tool: str, payload: dict) -> AgentToolResult:
        self.calls.append((tool, dict(payload)))
        if tool == "get_active_context":
            batch = None
            if self.active_run_id is not None:
                batch = {
                    "run_id": self.active_run_id,
                    "source": "register",
                    "produced_at": "2026-08-28T00:00:00+00:00",
                    "item_count": len(self.prev_targets),
                }
            return ok_result(tool, {"active_batch": batch})
        if tool == "get_recommendation_run":
            count = (
                self.prev_count
                if self.prev_count is not None
                else len(self.prev_targets)
            )
            items = batch_items(list(self.prev_targets), "Mandopop")
            if self.empty_basis:
                for item in items:
                    item["evidence"]["basis"] = []
            return ok_result(
                tool,
                {
                    "run_id": payload["run_id"],
                    "item_count": count,
                    "items": items,
                },
            )
        if tool == "generate_inferred_recommendation":
            if self.gen_error:
                return error_result(tool, "empty_recommendation")
            items = self.generated
            if items is None:
                items = [
                    {
                        "target_id": J1,
                        "name": "J Song 1",
                        "artist_name": "J Artist",
                        "fresh_this_request": False,
                    },
                    {
                        "target_id": J2,
                        "name": "J Song 2",
                        "artist_name": "J Artist",
                        "fresh_this_request": False,
                    },
                ]
            return ok_result(
                tool,
                {
                    "run_id": "rcm_new",
                    "item_count": len(items),
                    "items": items,
                },
            )
        raise AssertionError(f"unexpected tool {tool}")

    def close_event_listener(self) -> None:
        pass


class DirectionCoachFakeClientTest(unittest.TestCase):
    def test_shift_generates_new_direction_with_scope_and_exclusion(self) -> None:
        client = ScriptedClient(
            true_genres=("J-Pop", "Rock"), true_scope=(M1, M2, J1, R1)
        )
        outcome = run_direction_shift(client, "再来一批，换个方向。")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "J-Pop")
        generation = client.calls[-1]
        self.assertEqual(generation[0], "generate_inferred_recommendation")
        self.assertEqual(set(generation[1]["target_ids"]), {M1, M2, J1, R1})
        self.assertEqual(generation[1]["genres"], ["J-Pop"])
        self.assertEqual(set(generation[1]["exclude_target_ids"]), {M1, M2})
        self.assertEqual(generation[1]["limit"], 2)
        self.assertIn("J-Pop", outcome["note"])
        self.assertEqual(outcome["run_id"], "rcm_new")
        self.assertEqual(len(outcome["items"]), 2)

    def test_shift_excludes_previous_batch_direction(self) -> None:
        # The previous batch's main direction is a candidate direction TOO:
        # the picker must skip it and land on the next real positive.
        client = ScriptedClient(
            true_genres=("Mandopop", "Rock"), true_scope=(M1, M2, R1)
        )
        outcome = run_direction_shift(client, "换一种风格")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "Rock")

    def test_no_alternative_fails_honest_without_generation(self) -> None:
        client = ScriptedClient(true_genres=("Mandopop",), true_scope=(M1, M2))
        outcome = run_direction_shift(client, "换个方向")
        self.assertEqual(outcome, {"kind": "reply", "text": NO_ALTERNATIVE_DIRECTION_REPLY})
        self.assertFalse(
            any(tool == "generate_inferred_recommendation" for tool, _ in client.calls)
        )

    def test_unreadable_truth_remits_to_provider(self) -> None:
        client = ScriptedClient(true_genres=("J-Pop",), true_scope=(J1,))
        client.truth = None  # a broken reader
        self.assertIsNone(run_direction_shift(client, "换个方向"))

    def test_explicit_direction_wins_and_never_auto_reselects(self) -> None:
        client = ScriptedClient(
            true_genres=("Mandopop", "J-Pop"), true_scope=(M1, J1)
        )
        outcome = run_direction_shift(client, "换成日系")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "J-Pop")
        self.assertEqual(client.calls[-1][1]["genres"], ["J-Pop"])

    def test_explicit_direction_without_positive_scope_remits(self) -> None:
        # The explicit word keeps its old provider-loop surface when the real
        # positive pool cannot carry a generation (never a fabricated batch).
        client = ScriptedClient(true_genres=("Mandopop",), true_scope=())
        self.assertIsNone(run_direction_shift(client, "换成摇滚"))

    def test_generation_failure_answers_honestly(self) -> None:
        client = ScriptedClient(
            true_genres=("J-Pop",), true_scope=(J1,), gen_error=True
        )
        outcome = run_direction_shift(client, "来点不一样的")
        self.assertEqual(outcome["kind"], "reply")
        self.assertIn("暂时没有找到合适的推荐", outcome["text"])

    def test_explicit_generation_failure_remits_to_provider(self) -> None:
        # §八: an explicit named direction keeps its capable old surface on
        # failure; the failed attempt persisted nothing (empty runs never do).
        client = ScriptedClient(
            true_genres=("J-Pop",), true_scope=(J1,), gen_error=True
        )
        self.assertIsNone(run_direction_shift(client, "换成摇滚"))

    def test_no_active_batch_remits(self) -> None:
        # One get_active_context read happens before the remit: the coach
        # must confirm there is no batch to shift FROM; nothing else runs.
        client = ScriptedClient(
            true_genres=("J-Pop",), true_scope=(J1,), active_run_id=None
        )
        self.assertIsNone(run_direction_shift(client, "换个方向"))
        self.assertEqual(
            [tool for tool, _ in client.calls], ["get_active_context"]
        )

    def test_plain_re_recommend_and_non_shifts_never_claimed(self) -> None:
        client = ScriptedClient(true_genres=("J-Pop",), true_scope=(J1,))
        for line in ("再来一批", "再推荐一批", "换一批", "再来一批好听的", "推荐几首歌", "换个方向吧"):
            self.assertIsNone(run_direction_shift(client, line), line)
        self.assertEqual(len(client.calls), 0)

    def test_empty_basis_batch_falls_back_to_strongest_positive(self) -> None:
        # A batch of zero-evidence fresh-discovery items (targets present, no
        # basis entries) has no main direction; the strongest positive
        # direction becomes the shift target.
        client = ScriptedClient(
            true_genres=("Rock", "J-Pop"),
            true_scope=(R1, J1),
            prev_targets=(M1, M2),
            empty_basis=True,
        )
        outcome = run_direction_shift(client, "换个方向")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "Rock")


# --- real-service integration -------------------------------------------------


class DirectionCoachServiceIntegrationTest(unittest.TestCase):
    """The mandate §十一 chain over a REAL service: durable seeds -> real batch
    -> coach -> real generated batch -> durable reads of what happened."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        self.seed_extended_fixture()
        self.service = SharedAgentService(
            self.database_path,
            clients=AgentClientRegistry({CLIENT_ID: AgentClientPolicy.FULL}),
        )
        self.addCleanup(self.service.close)
        self.client = self._recording_client()

    def _recording_client(self) -> "RecordingClient":
        return RecordingClient(self.service)

    def seed_extended_fixture(self) -> None:
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        extra = (
            (M1, "Mandopop Track 1", "Mandopop", ARTIST_ALPHA),
            (M2, "Mandopop Track 2", "Mandopop", ARTIST_ALPHA),
            (M3, "Mandopop Track 3", "Mandopop", ARTIST_ALPHA),
            (J1, "J Track 1", "J-Pop", ARTIST_GAMMA),
            (J2, "J Track 2", "J-Pop", ARTIST_GAMMA),
            (R1, "Rock Track 1", "Rock", ARTIST_BETA),
            (R2, "Rock Track 2", "Rock", ARTIST_BETA),
        )
        for track_id, name, genre, artist_id in extra:
            fixture["tracks"].append(
                {
                    "id": track_id,
                    "external_ids": {
                        "apple_music_persistent_id": None,
                        "itunes_store_id": None,
                    },
                    "name": name,
                    "artist_ids": [artist_id],
                    "album_id": None,
                    "duration_ms": None,
                    "genres": [genre],
                    "track_number": None,
                    "disc_number": None,
                    "release_date": None,
                    "composer": None,
                    "library_state": {
                        "favorited": None,
                        "disliked": None,
                        "rating": None,
                        "play_count": None,
                        "skip_count": None,
                        "added_to_library_at": None,
                        "last_played_at": None,
                    },
                    "agent_metadata": {"tags": []},
                }
            )
        validate_fixture(fixture)
        from music_agent.repository import CanonicalRepository

        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)

    def seed_positive(self, track_id: str) -> None:
        with PreferencePersistenceRepository(self.database_path) as repository:
            repository.record_observation(
                SignalIdentity(
                    PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id),
                    "apple_music",
                    "favorited",
                ),
                ObservedValue.value(True),
                observed_at=ISO,
                provenance="fixture_seed",
            )

    def read_run(self, run_id: str) -> dict:
        result = self.client.call("get_recommendation_run", {"run_id": run_id})
        self.assertIs(result.outcome, AgentToolOutcome.OK)
        self.client.calls.clear()
        return result.payload

    def generate(self, target_ids: tuple[str, ...], limit: int) -> dict:
        result = self.client.call(
            "generate_recommendation", {"target_ids": list(target_ids), "limit": limit}
        )
        self.assertIs(
            result.outcome, AgentToolOutcome.OK, getattr(result, "error_code", None)
        )
        self.client.calls.clear()
        return result.payload

    def item_targets(self, payload: dict) -> list[str]:
        return [item["target_id"] for item in payload["items"]]

    def test_direction_truth_reads_durable_evidence(self) -> None:
        for track_id in (M1, M2, J1, R1):
            self.seed_positive(track_id)
        genres, scope = self.service.direction_shift_inputs()
        self.assertEqual(genres[0], "Mandopop")
        self.assertEqual(set(genres), {"Mandopop", "J-Pop", "Rock"})
        self.assertEqual(set(scope), {M1, M2, J1, R1})

    def test_shift_from_mandopop_batch_generates_real_direction(self) -> None:
        # §十一(1): Mandopop main + real J-Pop/Rock positives -> the shift
        # filters a REAL different direction in, keeps the previous batch out,
        # and the new run records its durable direction basis.
        from music_agent.direction_coach import (
            _batch_genre_counts,
            _genre_resolver,
            _resolve_track_genres,
            _track_basis_item_ids,
        )

        for track_id in (M1, M2, J1, R1):
            self.seed_positive(track_id)
        batch1 = self.generate((M1, M2), 2)
        run1 = self.read_run(batch1["run_id"])
        self.assertEqual(set(self.item_targets(run1)), {M1, M2})
        # Direct candidates carry TRACK-kind durable basis (display names
        # only); the batch direction derives from the items' canonical genres.
        for item in run1["items"]:
            self.assertTrue(
                any(entry["kind"] == "track" for entry in item["evidence"]["basis"]),
                item,
            )
        track_genres = _resolve_track_genres(
            _genre_resolver(self.client), _track_basis_item_ids(run1)
        )
        self.assertEqual(_batch_genre_counts(run1, track_genres)[0], {"Mandopop": 2})

        outcome = run_direction_shift(self.client, "再来一批换个方向")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "J-Pop")
        generation = [c for c in self.client.calls if c[0] == "generate_inferred_recommendation"]
        self.assertEqual(len(generation), 1)
        payload = generation[0][1]
        self.assertEqual(payload["genres"], ["J-Pop"])
        self.assertEqual(
            set(payload["exclude_target_ids"]), {M1, M2}  # the previous batch
        )
        self.assertEqual(set(payload["target_ids"]), {M1, M2, J1, R1})

        run2 = self.read_run(outcome["run_id"])
        self.assertTrue(run2["items"])
        # §十一(7): the previous batch never reappears.
        self.assertFalse(set(self.item_targets(run2)) & set(self.item_targets(run1)))
        # §十一(8): the new run records its real direction basis -- every item
        # resolves to the selected direction through the durable evidence.
        track_genres = _resolve_track_genres(
            _genre_resolver(self.client), _track_basis_item_ids(run2)
        )
        self.assertEqual(set(_batch_genre_counts(run2, track_genres)[0]), {"J-Pop"})

    def test_jpop_main_batch_shifts_away_from_jpop(self) -> None:
        # §十一(2): J-Pop main -> the new direction is not J-Pop.
        for track_id in (J1, R1):
            self.seed_positive(track_id)
        self.generate((J1,), 1)
        outcome = run_direction_shift(self.client, "换个方向")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "Rock")

    def test_mixed_batch_picks_absent_direction_deterministically(self) -> None:
        # §十一(3): mixed evidence -> one pinned deterministic outcome.
        for track_id in (M1, J1, R1):
            self.seed_positive(track_id)
        self.generate((M1, J1), 2)
        outcome = run_direction_shift(self.client, "换个方向")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "Rock")  # pinned, never model-chosen

    def test_no_other_direction_answers_with_question_no_generation(self) -> None:
        # §十一(6): a single fake-proof direction -> NO generation call, the
        # natural question instead.
        for track_id in (M1, M2):
            self.seed_positive(track_id)
        self.generate((M1, M2), 2)
        outcome = run_direction_shift(self.client, "换个方向")
        self.assertEqual(outcome, {"kind": "reply", "text": NO_ALTERNATIVE_DIRECTION_REPLY})
        self.assertFalse(
            any(c[0] == "generate_inferred_recommendation" for c in self.client.calls)
        )

    def test_explicit_japanese_direction(self) -> None:
        # §十一(4): 换成日系 -> J-Pop, never another direction.
        for track_id in (M1, J1):
            self.seed_positive(track_id)
        self.generate((M1,), 1)
        outcome = run_direction_shift(self.client, "换成日系")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "J-Pop")
        generation = [c for c in self.client.calls if c[0] == "generate_inferred_recommendation"]
        self.assertEqual(generation[0][1]["genres"], ["J-Pop"])

    def test_explicit_rock_direction(self) -> None:
        # §十一(5): 来点摇滚 -> Rock.
        for track_id in (M1, R1):
            self.seed_positive(track_id)
        self.generate((M1,), 1)
        outcome = run_direction_shift(self.client, "来点摇滚")
        self.assertEqual(outcome["kind"], "generated")
        self.assertEqual(outcome["genre"], "Rock")
        generation = [c for c in self.client.calls if c[0] == "generate_inferred_recommendation"]
        self.assertEqual(generation[0][1]["genres"], ["Rock"])

    def test_plain_re_recommend_remits_untouched(self) -> None:
        # §十一(9): the plain re-recommend surface is not the coach's business.
        for track_id in (M1, J1):
            self.seed_positive(track_id)
        self.generate((M1,), 1)
        self.assertIsNone(run_direction_shift(self.client, "再来一批"))
        self.assertEqual(len(self.client.calls), 0)


class RecordingClient(AgentClient):
    def __init__(self, service: SharedAgentService) -> None:
        super().__init__(AgentClientIdentity(CLIENT_ID, "codex"), service)
        self.calls: list[tuple[str, dict]] = []

    def call(self, tool: str, payload: dict, **kwargs):
        self.calls.append((tool, dict(payload)))
        return super().call(tool, payload, **kwargs)


# --- surfaces: CLI fast path + web fast path ---------------------------------


class FakeShiftLoop:
    def __init__(self, client: ScriptedClient) -> None:
        self.client = client
        self.messages: list[str] = []

    def run(self, text: str):
        self.messages.append(text)
        # _print_chat_result reads a result object (final_text in particular);
        # mirror the test_chat_session fake_result idiom.
        return SimpleNamespace(
            final_text=f"reply:{text}",
            rounds=1,
            tool_executions=(),
            context_trimmed=False,
            rounds_capped=False,
            total_elapsed_ms=12.3,
        )


class _OldService:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class DirectionShiftCliSurfaceTest(unittest.TestCase):
    def test_cli_fast_path_presents_shifted_batch(self) -> None:
        from music_agent.cli import _run_direction_shift

        client = ScriptedClient(true_genres=("J-Pop", "Rock"), true_scope=(M1, J1))
        loop = FakeShiftLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_direction_shift(loop, "换个方向")
        self.assertTrue(handled)
        output = stdout.getvalue()
        self.assertIn("已换到「J-Pop」方向", output)
        self.assertIn("1. J Song 1 — J Artist", output)
        self.assertIn("2. J Song 2 — J Artist", output)
        self.assertEqual(loop.messages, [])

    def test_cli_fast_path_presents_honest_reply(self) -> None:
        from music_agent.cli import _run_direction_shift

        client = ScriptedClient(true_genres=("Mandopop",), true_scope=(M1,))
        loop = FakeShiftLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_direction_shift(loop, "换个方向")
        self.assertTrue(handled)
        self.assertEqual(stdout.getvalue(), NO_ALTERNATIVE_DIRECTION_REPLY + "\n")
        self.assertEqual(loop.messages, [])

    def test_cli_fast_path_remits_non_shift_lines(self) -> None:
        from music_agent.cli import _run_direction_shift

        client = ScriptedClient(true_genres=("J-Pop",), true_scope=(J1,))
        loop = FakeShiftLoop(client)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            handled = _run_direction_shift(loop, "再来一批")
        self.assertFalse(handled)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(len(client.calls), 0)

    def _run_chat(self, message: str) -> tuple[int, str, FakeShiftLoop]:
        loop = FakeShiftLoop(
            ScriptedClient(true_genres=("J-Pop", "Rock"), true_scope=(M1, J1))
        )
        service = _OldService()

        def fake_build(args, schemas=None):
            return service, loop

        from music_agent.cli import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                [
                    "chat", "--db", "/unused/store.db", "--message", message,
                    "--agent-client", f"{CLIENT_ID}:full",
                ]
            )
        return exit_code, stdout.getvalue(), loop

    def test_chat_command_claims_direction_shift_before_provider(self) -> None:
        exit_code, stdout, loop = self._run_chat("再来一批，换个方向")
        self.assertEqual(exit_code, 0)
        self.assertIn("J-Pop", stdout)
        self.assertEqual(loop.messages, [])  # zero provider rounds

    def test_chat_plain_re_recommend_still_reaches_provider(self) -> None:
        # §十一(9) at the surface: plain 再来一批 keeps its provider loop.
        exit_code, stdout, loop = self._run_chat("再来一批")
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["再来一批"])

    def test_chat_one_shot_tears_down_session_after_shift(self) -> None:
        # The fast path runs inside the try/finally: service closes on return.
        loop = FakeShiftLoop(
            ScriptedClient(true_genres=("J-Pop", "Rock"), true_scope=(M1, J1))
        )
        service = _OldService()

        def fake_build(args, schemas=None):
            return service, loop

        from music_agent.cli import main

        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = main(
                [
                    "chat", "--db", "/unused/store.db", "--message", "换个方向",
                    "--agent-client", f"{CLIENT_ID}:full",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertTrue(service.closed)
        self.assertEqual(loop.messages, [])


class _FakeCards:
    def __init__(self, run_id: str | None, cards: list[dict]) -> None:
        self.run_id = run_id
        self.cards = cards

    def latest_run_id(self) -> str | None:
        return self.run_id

    def latest_cards(self) -> list[dict]:
        return self.cards


class DirectionShiftWebSurfaceTest(unittest.TestCase):
    def _app(self, client: ScriptedClient, cards: _FakeCards):
        from music_agent.web_shell import ShellConfig, WebShellApp

        with tempfile.TemporaryDirectory() as temp:
            config = ShellConfig(
                database_path=Path(temp) / "store.db",
                provider_factory=lambda: None,
                agent_client=(CLIENT_ID, "full"),
                mode="standalone",
            )
            app = WebShellApp(config)
        app._loop = FakeShiftLoop(client)
        app._cards = cards
        return app

    def test_web_fast_path_generated_batch(self) -> None:
        card = {"name": "card"}
        client = ScriptedClient(true_genres=("J-Pop",), true_scope=(M1, J1))
        app = self._app(client, _FakeCards("rcm_new", [card]))
        reply = app._direction_shift_fast_path("换个方向")
        self.assertIsNotNone(reply)
        self.assertIn("已换到「J-Pop」方向", reply["reply"])
        self.assertIn("1. J Song 1", reply["reply"])
        self.assertTrue(reply["reply_html"])
        self.assertFalse(reply["rounds_capped"])
        self.assertEqual(reply["batch"]["run_id"], "rcm_new")
        self.assertEqual(reply["batch"]["cards"], [card])

    def test_web_fast_path_batch_identity_must_match_cards_source(self) -> None:
        # The batch travels only when its run id is the newest persisted run;
        # a mismatch (e.g. a concurrent generation) keeps the reply honest.
        client = ScriptedClient(true_genres=("J-Pop",), true_scope=(M1, J1))
        app = self._app(client, _FakeCards("rcm_other", [{"name": "card"}]))
        reply = app._direction_shift_fast_path("换个方向")
        self.assertIn("J-Pop", reply["reply"])
        self.assertIsNone(reply["batch"])

    def test_web_fast_path_honest_reply_and_remit(self) -> None:
        client = ScriptedClient(true_genres=("Mandopop",), true_scope=(M1,))
        app = self._app(client, _FakeCards(None, []))
        reply = app._direction_shift_fast_path("换个方向")
        self.assertEqual(reply["reply"], NO_ALTERNATIVE_DIRECTION_REPLY)
        self.assertIsNone(reply["batch"])
        # A non-shift line is not the fast path's business.
        self.assertIsNone(app._direction_shift_fast_path("再来一批"))
        self.assertIsNone(app._direction_shift_fast_path("推荐几首歌"))


if __name__ == "__main__":
    unittest.main()