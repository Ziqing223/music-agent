"""P12-C06: interactive chat-session CLI shell tests.

Fakes only: the session-build helper is patched, so no real provider, service,
or network call happens here. ``_build_chat_session`` is the reuse boundary --
asserting it is called once while several messages run proves the session does
not rebuild configuration per input.
"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

from music_agent.agent_contract import (
    AgentToolOutcome,
    AgentToolResult,
    generate_request_id,
)
from music_agent.agent_socket import (
    AGENT_CLIENT_NOT_REGISTERED_CODE,
    AGENT_CLIENT_NOT_REGISTERED_HINT,
)

CLIENT_ID = "agt_44444444-4444-4444-8444-444444444444"


def fake_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        final_text=f"reply:{text}",
        rounds=1,
        tool_executions=(),
        context_trimmed=False,
        rounds_capped=False,
        total_elapsed_ms=12.3,
    )


class FakeClient:
    """Records routed tool calls and replies with an OK result + minimal payload.

    ``session_state`` ("running" or None) drives the P15-S1 get_playback_context
    observation and the stop_preview session-cancellation flag; ``suspended``
    feeds the observed AudioSuspension entry (the P15-S1 C02 restore target).
    The CLI reads them for session-aware routing, the play interception and the
    continue-family restore gate. P15-S4-M3-B adds the status-answer knobs:
    ``player`` / ``preview_sounding`` shape the get_playback_context observation,
    ``offline`` makes get_playback_context refuse with agent_runtime_offline
    (the run socket), ``not_registered`` makes it refuse with the P20-Fix01
    translated ``agent_client_not_registered`` envelope (a reachable run whose
    permission gate refused the session client), and ``now_playing`` feeds the
    local get_now_playing degrade.
    P16-S3 adds the formal-play knobs: ``active_batch`` feeds the
    get_active_context observation, ``run_items`` feeds get_recommendation_run,
    and ``play_persistent_id`` is the play_track resolution payload's
    persistent_id (the verification comparison target).
    P19-T14-F-R2 adds the pronoun-binding knobs: ``channel_state`` /
    ``channel_canonical_id`` feed the get_playback_context channel register
    (the action-log fallback; default "none"/None), and
    preview_catalog_track answers OK with ``started`` true (the deterministic
    binding path's success predicate). P19-T14-F-R4 adds
    ``referent_canonical_id`` -- the session-local conversational target
    that survives preview stops and wins over the channel.
    All defaults reproduce the pre-M3/pre-S3 behavior exactly.
    """

    def __init__(
        self,
        session_state: str | None = None,
        suspended: Mapping[str, Any] | None = None,
        *,
        player: Mapping[str, Any] | None = None,
        preview_sounding: bool | None = None,
        offline: bool = False,
        not_registered: bool = False,
        now_playing: Mapping[str, Any] | None = None,
        active_batch: Mapping[str, Any] | None = None,
        run_items: list | None = None,
        play_persistent_id: str | None = "P-LIB-1",
        channel_state: str = "none",
        channel_canonical_id: str | None = None,
        referent_canonical_id: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.session_state = session_state
        self.suspended = suspended
        self.player = player
        self.preview_sounding = preview_sounding
        self.offline = offline
        self.not_registered = not_registered
        self.now_playing = now_playing
        self.active_batch = active_batch
        self.run_items = run_items
        self.play_persistent_id = play_persistent_id
        self.channel_state = channel_state
        self.channel_canonical_id = channel_canonical_id
        self.referent_canonical_id = referent_canonical_id
        self.last_played_canonical_id: str | None = None
        self.control_state: str | None = None

    def call(self, tool: str, payload: Mapping[str, Any]) -> AgentToolResult:
        self.calls.append((tool, dict(payload)))
        if tool == "get_playback_context":
            if self.offline:
                return AgentToolResult(
                    request_id=generate_request_id(),
                    tool=tool,
                    outcome=AgentToolOutcome.EXECUTION_ERROR,
                    payload=None,
                    error_code="agent_runtime_offline",
                    error_message="agent runtime offline",
                    completed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                )
            if self.not_registered:
                return AgentToolResult(
                    request_id=generate_request_id(),
                    tool=tool,
                    outcome=AgentToolOutcome.EXECUTION_ERROR,
                    payload=None,
                    error_code=AGENT_CLIENT_NOT_REGISTERED_CODE,
                    error_message=AGENT_CLIENT_NOT_REGISTERED_HINT,
                    completed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                )
            payload = {
                "channel": {
                    "state": self.channel_state,
                    "canonical_id": self.channel_canonical_id,
                },
                "referent_canonical_id": self.referent_canonical_id,
                "preview_sounding": (
                    self.preview_sounding
                    if self.preview_sounding is not None
                    else False
                ),
                "player": dict(self.player) if self.player is not None else None,
                "session": (
                    {
                        "state": self.session_state,
                        "total": 5,
                        "position": 2,
                        "current_canonical_id": "trk_x",
                        "current_name": "Synthetic Solo",
                        "skipped": [],
                    }
                    if self.session_state is not None
                    else None
                ),
                "suspended": (
                    dict(self.suspended) if self.suspended is not None else None
                ),
            }
        elif tool == "get_now_playing":
            now_playing = dict(self.now_playing) if self.now_playing is not None else None
            if now_playing is None and self.control_state is not None:
                now_playing = {"state": self.control_state}
            observed_pid = (
                now_playing.get("persistent_id")
                if isinstance(now_playing, Mapping)
                else None
            )
            player_canonical_id = (
                self.last_played_canonical_id
                if self.last_played_canonical_id is not None
                and observed_pid == self.play_persistent_id
                else None
            )
            payload = {
                "now_playing": now_playing,
                "agent_channel": {
                    "state": "library",
                    "canonical_id": self.last_played_canonical_id,
                },
                "player_canonical_id": player_canonical_id,
                "canonical_resolution": (
                    "binding" if player_canonical_id is not None else None
                ),
            }
        elif tool == "get_active_context":
            payload = {
                "preview_sounding": (
                    self.preview_sounding
                    if self.preview_sounding is not None
                    else False
                ),
                "active_batch": (
                    dict(self.active_batch) if self.active_batch is not None else None
                ),
            }
        elif tool == "get_recommendation_run":
            payload = {
                "run_id": payload.get("run_id"),
                "items": (
                    [dict(item) for item in self.run_items]
                    if self.run_items is not None
                    else []
                ),
            }
        elif tool == "play_track":
            self.last_played_canonical_id = payload.get("canonical_id")
            payload = {
                "command": tool,
                "ok": True,
                "persistent_id": self.play_persistent_id,
                "resolution": "binding",
                "elapsed_ms": 5,
            }
        elif tool == "preview_catalog_track":
            payload = {
                "command": tool,
                "canonical_id": payload.get("canonical_id"),
                "started": True,
            }
        elif tool == "stop_preview":
            payload = {
                "command": tool,
                "ok": True,
                "stopped": True,
                "preview_session_cancelled": self.session_state == "running",
            }
        elif tool in {"play", "pause"}:
            self.control_state = "playing" if tool == "play" else "paused"
            payload = {"command": tool, "ok": True}
        else:
            payload = {"command": tool, "ok": True}
        return AgentToolResult(
            request_id=generate_request_id(),
            tool=tool,
            outcome=AgentToolOutcome.OK,
            payload=payload,
            error_code=None,
            error_message=None,
            completed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )

    def close_event_listener(self) -> None:
        """P15-S2-IPC S4: the teardown surface the chat commands call in their
        ``finally`` (the real RoutedAgentClient tears its event sink down;
        the fake has no sink)."""


class FakeLoop:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.client = FakeClient()

    def run(self, text: str):
        self.messages.append(text)
        return fake_result(text)


class FakeService:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ChatSessionCliTest(unittest.TestCase):
    def _run(self, command, stdin_lines):
        loop = FakeLoop()
        service = FakeService()
        stdin = iter(stdin_lines)
        prompts: list[str] = []

        def fake_input(prompt: str = "") -> str:
            prompts.append(prompt)
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        build_calls = []

        def fake_build(args, schemas=None):
            build_calls.append(args)
            return service, loop

        from music_agent.cli import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(command)
        return exit_code, stdout.getvalue(), stderr.getvalue(), loop, service, build_calls, prompts

    def test_chat_session_parser_does_not_require_message(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args([
            "chat-session", "--db", "store.db",
            "--agent-client", f"{CLIENT_ID}:full",
            "--provider", "codex", "--model", "gpt-5", "--max-rounds", "4",
        ])
        self.assertEqual(args.provider, "codex")
        self.assertEqual(args.model, "gpt-5")
        self.assertEqual(args.max_rounds, 4)

    def test_chat_still_runs_single_message(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, stderr, loop, service, build_calls, prompts = self._run(
                ["chat", "--db", db, "--message", "你好",
                 "--agent-client", f"{CLIENT_ID}:full"],
                [],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["你好"])
        self.assertEqual(len(build_calls), 1)
        self.assertIn("reply:你好", stdout)
        self.assertNotIn("[provider=", stderr)  # trace is hidden by default
        self.assertTrue(service.closed)

    def test_session_runs_sequential_inputs_and_skips_blank_lines(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, stderr, loop, service, build_calls, prompts = self._run(
                ["chat-session", "--db", db, "--agent-client", f"{CLIENT_ID}:full"],
                ["推荐几首歌", "", "   ", "播放第一首", "/exit"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["推荐几首歌", "播放第一首"])
        # Built exactly once for the whole session, not once per input.
        self.assertEqual(len(build_calls), 1)
        self.assertTrue(prompts and all(p == "你> " for p in prompts))
        self.assertGreaterEqual(len(prompts), 2)  # re-prompts after every line
        self.assertIn("reply:推荐几首歌", stdout)
        self.assertIn("reply:播放第一首", stdout)
        self.assertEqual(stdout.count("reply:"), 2)
        self.assertTrue(service.closed)

    def test_shared_builder_wires_playback_resolver_for_both_commands(self) -> None:
        """Both commands build through ``_build_chat_session`` (the existing
        tests assert each command calls it exactly once), so one real build
        proves chat and chat-session receive the identical playback stack."""
        from music_agent.agent_service import SharedAgentService
        from music_agent.cli import _build_chat_session
        from music_agent.playback_control import (
            MusicLibraryResolver,
            MusicPlaybackAdapter,
        )

        captured: dict = {}

        class SpyService(SharedAgentService):
            def __init__(self, db, **kwargs):
                captured.update(kwargs)

            def close(self):
                pass

        args = SimpleNamespace(
            db="store.db",
            agent_client=[f"{CLIENT_ID}:full"],
            provider="deepseek",
            model=None,
            api_key_env=None,
            base_url=None,
            timeout=120.0,
            max_rounds=8,
        )
        with patch("music_agent.agent_service.SharedAgentService", SpyService):
            built_service, _ = _build_chat_session(args)
        self.assertIsInstance(captured["playback_adapter"], MusicPlaybackAdapter)
        self.assertIsInstance(captured["playback_resolver"], MusicLibraryResolver)
        self.assertIsNotNone(captured["catalog_search_source"])
        # P15-S1: every chat mode presents preview-session events locally.
        self.assertTrue(callable(built_service.preview_event_handler))

    def test_session_quit_exits_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, _, _, loop, service, _, _ = self._run(
                ["chat-session", "--db", db, "--agent-client", f"{CLIENT_ID}:full"],
                ["一条", "/quit", "不该执行的"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["一条"])
        self.assertTrue(service.closed)

    def test_routed_pause_bypasses_the_provider_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, _, loop, service, _, _ = self._run(
                ["chat-session", "--db", db, "--agent-client", f"{CLIENT_ID}:full"],
                ["暂停", "/quit"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        # P15-S1: 暂停 is a session-aware form -- one local read decides (no
        # session here, so the plain pause route wins), still zero model rounds.
        self.assertEqual(
            loop.client.calls,
            [
                ("get_playback_context", {}),
                ("pause", {}),
                ("get_now_playing", {}),
            ],
        )
        self.assertIn("已暂停播放", stdout)
        self.assertTrue(service.closed)

    def test_routed_stop_preview_bypasses_the_provider_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, _, loop, service, _, _ = self._run(
                ["chat-session", "--db", db, "--agent-client", f"{CLIENT_ID}:full"],
                ["停止试听", "/exit"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(loop.client.calls, [("stop_preview", {})])
        self.assertIn("已停止试听", stdout)

    def _run_session_scenario(self, session_state, lines, suspended=None):
        """Session with a FakeClient that reports the given preview session state
        and (optionally) the observed AudioSuspension entry."""
        from music_agent.cli import main

        loop = FakeLoop()
        loop.client = FakeClient(session_state=session_state, suspended=suspended)
        service = FakeService()
        stdin = iter(lines)

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        def fake_build(args, schemas=None):
            return service, loop

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        return exit_code, stdout.getvalue(), stderr.getvalue(), loop, service

    def test_pause_during_running_session_stops_the_preview_run(self) -> None:
        """P15-S1 §8/A3: 暂停 ≙ stop_preview while the session runs -- no Music.app
        pause, no provider round."""
        exit_code, stdout, _, loop, service = self._run_session_scenario(
            "running", ["暂停", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls, [("get_playback_context", {}), ("stop_preview", {})]
        )
        self.assertNotIn("已暂停播放", stdout)
        self.assertTrue(service.closed)

    def test_continue_during_running_session_reports_progress_without_play(self) -> None:
        """P15-S1 §8/A5: the continue family never resumes Music.app over sounding
        preview audio -- progress is reported locally instead."""
        for line in ("继续", "继续播放", "continue", "play", "resume"):
            with self.subTest(line=line):
                exit_code, stdout, _, loop, service = self._run_session_scenario(
                    "running", [line, "/quit"]
                )
                self.assertEqual(exit_code, 0)
                self.assertEqual(loop.messages, [])
                # One read for the interception; the play route is NOT executed.
                self.assertEqual(
                    loop.client.calls, [("get_playback_context", {})]
                )
                self.assertIn("连播进行中（第 2/共 5 首）", stdout)
                self.assertTrue(service.closed)

    def test_continue_without_a_suspension_declines_to_invent_a_restore(self) -> None:
        """P15-S1 C02: after any terminal (completed / failed / stopped) the
        register is cleared; with no AudioSuspension there is no honest restore
        target, so 继续 must NOT execute play -- the refusal is explicit and the
        chat session stays alive."""
        for line in ("继续", "继续播放"):
            with self.subTest(line=line):
                exit_code, stdout, _, loop, service = self._run_session_scenario(
                    None, [line, "/quit"]
                )
                self.assertEqual(exit_code, 0)
                self.assertEqual(loop.messages, [])
                self.assertEqual(loop.client.calls, [("get_playback_context", {})])
                self.assertIn("没有可恢复的播放", stdout)
                self.assertNotIn("已继续播放", stdout)
                self.assertTrue(service.closed)

    def test_continue_with_a_suspension_still_restores_play(self) -> None:
        """P15-S1 C02: the suspension memo IS the restore target -- 继续播放 with
        one resumes formal playback through the plain play route (restore-by-intent
        unchanged by the C02 gate)."""
        suspended = {
            "player_state": "playing",
            "persistent_id": "REAL-PID-1",
            "name": "起风了 (旧版)",
            "pause_ok": True,
        }
        exit_code, stdout, _, loop, service = self._run_session_scenario(
            None, ["继续播放", "/quit"], suspended=suspended
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [
                ("get_playback_context", {}),
                ("play", {}),
                ("get_now_playing", {}),
            ],
        )
        self.assertIn("已继续播放", stdout)
        self.assertNotIn("没有可恢复的播放", stdout)
        self.assertTrue(service.closed)

    def test_next_during_a_running_session_advances_the_preview(self) -> None:
        """P15-S1 C02: 下一首/下一首试听 while the session runs route to
        advance_preview -- no provider round, no next_track, no new session."""
        for line in ("下一首", "下一首试听"):
            with self.subTest(line=line):
                exit_code, stdout, _, loop, service = self._run_session_scenario(
                    "running", [line, "/quit"]
                )
                self.assertEqual(exit_code, 0)
                self.assertEqual(loop.messages, [])
                self.assertEqual(
                    loop.client.calls,
                    [("get_playback_context", {}), ("advance_preview", {})],
                )
                self.assertNotIn("已切换到下一首", stdout)
                self.assertTrue(service.closed)

    def test_next_off_a_session_keeps_the_pre_p15_routes(self) -> None:
        """P15-S1 C02: off a session 下一首 stays the plain fast route (now with
        one local context read); 下一首试听 has nothing to advance and falls to
        the provider loop instead of inventing a session."""
        exit_code, stdout, _, loop, service = self._run_session_scenario(
            None, ["下一首", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls, [("get_playback_context", {}), ("next_track", {})]
        )
        self.assertIn("已切换到下一首", stdout)

        exit_code, stdout, _, loop, service = self._run_session_scenario(
            None, ["下一首试听", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        # No advance target: the ONE local session read still happens (the form
        # is session-sensitive), then the provider loop decides -- no tool was
        # minted and no session was invented.
        self.assertEqual(loop.messages, ["下一首试听"])
        self.assertEqual(loop.client.calls, [("get_playback_context", {})])
        self.assertIn("reply:下一首试听", stdout)
        self.assertTrue(service.closed)

    def test_one_shot_chat_hides_preview_batch_from_the_tool_list(self) -> None:
        """P15-S1 §6 one-shot interception: ``chat --message`` builds with the
        preview_batch schema removed (a continuous session dies with the process)."""
        from music_agent.cli import main

        captured: dict = {}

        def fake_build(args, schemas=None):
            captured["schemas"] = schemas
            return FakeService(), FakeLoop()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat", "--db", "s.db", "--message", "你好",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        names = [getattr(schema, "name", None) for schema in (captured["schemas"] or ())]
        self.assertNotIn("preview_batch", names)
        self.assertIn("get_playback_context", names)  # everything else survives

    def test_session_mode_builds_the_full_tool_list(self) -> None:
        from music_agent.cli import main

        captured: dict = {}

        def fake_build(args, schemas=None):
            captured["schemas"] = schemas
            service = FakeService()
            loop = FakeLoop()
            return service, loop

        def fake_input(prompt: str = "") -> str:
            raise EOFError

        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                patch("builtins.input", fake_input):
            main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertIsNone(captured["schemas"])  # the builder itself defaults to all

    def test_ordinary_requests_still_use_the_provider_loop(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, _, loop, service, _, _ = self._run(
                ["chat-session", "--db", db, "--agent-client", f"{CLIENT_ID}:full"],
                ["推荐几首新歌", "/quit"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["推荐几首新歌"])
        self.assertEqual(loop.client.calls, [])
        self.assertIn("reply:推荐几首新歌", stdout)

    def test_session_eof_exits_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, _, _, loop, service, _, _ = self._run(
                ["chat-session", "--db", db, "--agent-client", f"{CLIENT_ID}:full"],
                ["第一条", "第二条"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["第一条", "第二条"])
        self.assertTrue(service.closed)

    def test_session_continues_after_provider_error(self) -> None:
        from music_agent.provider_contract import ProviderError

        class FlakyLoop(FakeLoop):
            def run(self, text: str):
                if "失败" in text:
                    raise ProviderError("boom")
                return super().run(text)

        service = FakeService()
        flaky = FlakyLoop()
        stdin = iter(["失败", "再来一次", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        from music_agent.cli import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, flaky)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(flaky.messages, ["再来一次"])
        self.assertIn("provider error", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_verbose_re_enables_provider_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, stderr, loop, service, _, _ = self._run(
                ["chat", "--db", db, "--message", "你好",
                 "--agent-client", f"{CLIENT_ID}:full", "--verbose"],
                [],
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("reply:你好", stdout)
        self.assertIn("[provider=deepseek", stderr)
        self.assertTrue(service.closed)

    def _run_with_leaky_loop(self, command):
        """Same wiring as _run, but the loop returns a lead-lined final_text
        (internal id, field literal, route label, self-narration sentence)."""
        from music_agent.cli import main

        leaky = SimpleNamespace(
            final_text=(
                "生成成功（rcm_58f121dd-aaaa-4aaa-8aaa-aaaaaaaaaaaa），"
                "runs_total: 2，第一条 playback.route=library，"
                "我如实告知用户这是内部状态。推荐：A — 甲"
            ),
            rounds=1,
            tool_executions=(),
            context_trimmed=False,
            rounds_capped=False,
            total_elapsed_ms=12.3,
        )

        class LeakyLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                return leaky

        loop = LeakyLoop()
        service = FakeService()

        def fake_build(args, schemas=None):
            return service, loop

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(command)
        return exit_code, stdout.getvalue(), stderr.getvalue(), service

    SCRUBBED_REPLY = (
        "生成成功（推荐编号），推荐批总数: 2，"
        "第一条 可以正式播放，推荐：A — 甲\n"
    )

    def test_chat_output_scrubs_internal_tokens_before_printing(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, stderr, service = self._run_with_leaky_loop(
                ["chat", "--db", db, "--message", "你好",
                 "--agent-client", f"{CLIENT_ID}:full"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, self.SCRUBBED_REPLY)
        self.assertNotIn("[provider=", stderr)  # trace stays hidden by default
        self.assertTrue(service.closed)

    def test_verbose_trace_is_not_scrubbed_but_stdout_still_is(self) -> None:
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, stderr, service = self._run_with_leaky_loop(
                ["chat", "--db", db, "--message", "你好",
                 "--agent-client", f"{CLIENT_ID}:full", "--verbose"],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, self.SCRUBBED_REPLY)  # --verbose leaves stdout alone
        self.assertIn("[provider=deepseek", stderr)
        self.assertTrue(service.closed)

    def test_session_exits_on_plain_words(self) -> None:
        for word in ("退出", "exit", "quit", "EXIT"):
            with self.subTest(word=word):
                with TemporaryDirectory() as tmp:
                    db = str(Path(tmp) / "s.db")
                    exit_code, _, _, loop, service, _, _ = self._run(
                        ["chat-session", "--db", db,
                         "--agent-client", f"{CLIENT_ID}:full"],
                        ["一条", word, "不该执行的"],
                    )
                self.assertEqual(exit_code, 0)
                self.assertEqual(loop.messages, ["一条"])
                self.assertTrue(service.closed)

    def test_session_ctrl_c_during_input_exits_cleanly(self) -> None:
        service = FakeService()
        loop = FakeLoop()
        stdin = iter(["第一条", "第二条"])

        def fake_input(prompt: str = "") -> str:
            try:
                text = next(stdin)
            except StopIteration:
                raise EOFError
            if text == "第二条":
                raise KeyboardInterrupt
            return text

        from music_agent.cli import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["第一条"])
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_session_ctrl_c_during_run_exits_cleanly(self) -> None:
        class InterruptedLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                raise KeyboardInterrupt

        service = FakeService()
        loop = InterruptedLoop()
        stdin = iter(["一条", "不该执行的"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        from music_agent.cli import main

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["一条"])
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)


    def test_t14e_session_play_degraded_to_preview_stops_and_answers_honestly(
        self,
    ) -> None:
        # P19-T14-E door in chat-session: 播放 degraded to a preview is
        # stopped through the session client and the reply is the contract's
        # honest sentence -- never the model's self-healed preview prose.
        from music_agent.cli import main
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        class ExecLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                result = fake_result(text)
                result.tool_executions = (
                    SimpleNamespace(name="preview_catalog_track", outcome="ok"),
                )
                return result

        service = FakeService()
        loop = ExecLoop()
        stdin = iter(["播放 Hanataba", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PLAY_PREVIEW_DOWNGRADE_FALLBACK, stdout.getvalue())
        self.assertNotIn("reply:播放 Hanataba", stdout.getvalue())
        self.assertEqual(loop.messages, ["播放 Hanataba"])
        self.assertEqual(loop.client.calls, [("stop_preview", {})])
        self.assertTrue(service.closed)

    def test_s21_followup_structured_named_play_offer_arms_session_and_accepts_exact_target(self) -> None:
        from music_agent.cli import main
        from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

        target = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        offer = OfferedAction(
            kind=PREVIEW_TRACK,
            target_canonical_id=target,
            source="structured_named_play_resolution",
            verified_title="Wendy",
            verified_artist="Test Artist",
        )
        offer_text = "《Wendy》— Test Artist 目前无法正式播放，可以试听 30 秒。需要我开始试听吗？"

        class ExecLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                result = fake_result(text)
                result.final_text = offer_text
                result.rounds = 0
                result.offered_action = offer
                return result

        service = FakeService()
        loop = ExecLoop()
        stdin = iter(["播放 Wendy", "试听吧", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn(offer_text, output)
        self.assertIn("正在试听《Wendy》— Test Artist，约 30 秒。", output)
        self.assertEqual(loop.messages, ["播放 Wendy"])
        self.assertEqual(
            loop.client.calls,
            [("preview_catalog_track", {"canonical_id": target})],
        )
        self.assertTrue(service.closed)

    def test_s21_session_target_bound_preview_continuation_bypasses_provider(self) -> None:
        from music_agent.action_attempt import (
            create_direct_action_attempt,
            mark_action_executing,
            record_action_execution,
        )
        from music_agent.cli import main
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        target = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        attempt = record_action_execution(
            mark_action_executing(
                create_direct_action_attempt(
                    target, route="preview_only", title="Wendy", artist="Test Artist"
                )
            ),
            outcome="ok",
            preview_started=True,
        )

        class ExecLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                result = fake_result(text)
                result.tool_executions = (
                    SimpleNamespace(name="preview_catalog_track", outcome="ok"),
                )
                result.action_attempt = attempt
                return result

        service = FakeService()
        loop = ExecLoop()
        stdin = iter(["播放 Wendy", "开始试听", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn(PLAY_PREVIEW_DOWNGRADE_FALLBACK, output)
        self.assertIn("正在试听《Wendy》— Test Artist，约 30 秒。", output)
        self.assertEqual(loop.messages, ["播放 Wendy"])
        self.assertEqual(
            loop.client.calls,
            [
                ("stop_preview", {}),
                ("preview_catalog_track", {"canonical_id": target}),
            ],
        )
        self.assertTrue(service.closed)

    def test_t14e_session_formal_play_only_never_touches_preview(self) -> None:
        # Formal playback available: zero stop calls, the model reply passes
        # through untouched (acceptance A/D on the CLI surface).
        from music_agent.cli import main

        class ExecLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                result = fake_result(text)
                result.tool_executions = (
                    SimpleNamespace(name="play_track", outcome="ok"),
                )
                return result

        service = FakeService()
        loop = ExecLoop()
        stdin = iter(["播放第2首", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn("reply:播放第2首", stdout.getvalue())
        self.assertEqual(loop.client.calls, [])

    def test_t14e_one_shot_chat_play_degraded_to_preview_is_guarded(self) -> None:
        # The one-shot chat command runs the same door (before the client
        # closes): stop recorded, honest sentence printed.
        from music_agent.cli import main
        from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

        class ExecLoop(FakeLoop):
            def run(self, text: str):
                self.messages.append(text)
                result = fake_result(text)
                result.tool_executions = (
                    SimpleNamespace(name="preview_batch", outcome="ok"),
                )
                return result

        service = FakeService()
        loop = ExecLoop()
        stdout = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "chat", "--db", "s.db", "--message", "播放",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PLAY_PREVIEW_DOWNGRADE_FALLBACK, stdout.getvalue())
        self.assertEqual(loop.client.calls, [("stop_preview", {})])
        self.assertTrue(service.closed)

    def test_t14fr2_session_unbound_pronouns_answer_fixed_questions(self) -> None:
        # P19-T14-F-R2 seam on chat-session: 试听他/播放她 normalize to the
        # proven 它 spellings AND bind deterministically. With no channel
        # referent the turn answers the fixed honest questions and the loop
        # never sees the line -- the live-failure drift into recommendation
        # prose is unreachable by construction from a pronoun turn.
        from music_agent.intent_router import PRONOUN_PLAY_ASK, PRONOUN_PREVIEW_ASK

        exit_code, stdout, stderr, loop, service, build_calls, prompts = self._run(
            ["chat-session", "--db", "s.db", "--agent-client", f"{CLIENT_ID}:full"],
            ["试听他", "播放她", "/exit"],
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertIn(PRONOUN_PREVIEW_ASK, stdout)
        self.assertIn(PRONOUN_PLAY_ASK, stdout)
        self.assertEqual(
            loop.client.calls,
            [("get_playback_context", {}), ("get_playback_context", {})],
        )
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(service.closed)

    def test_t14fr2_session_preview_pronoun_binds_to_the_channel_referent(self) -> None:
        # Acceptance on the chat-session seam: 试听她 binds to the service
        # channel register's canonical_id BEFORE any provider round -- one
        # context read, one preview execution with the exact id, the fixed
        # start reply, zero loop rounds, zero stop calls.
        from music_agent.cli import main
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        track = "trk_22222222-2222-4222-8222-222222222222"
        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(channel_state="preview", channel_canonical_id=track)
        stdin = iter(["试听她", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PRONOUN_PREVIEW_START_REPLY, stdout.getvalue())
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [("get_playback_context", {}), ("preview_catalog_track", {"canonical_id": track})],
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr2_session_play_pronoun_binds_to_formal_playback_only(self) -> None:
        # 播放他 on the chat-session seam: formal playback only -- the T14-E
        # contract holds by construction; zero preview/stop calls.
        from music_agent.cli import main
        from music_agent.intent_router import PRONOUN_PLAY_START_REPLY

        track = "trk_22222222-2222-4222-8222-222222222222"
        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(
            channel_state="library",
            channel_canonical_id=track,
            now_playing={
                "state": "playing",
                "persistent_id": "P-LIB-1",
                "name": "Synthetic Track",
                "artist": "Synthetic Artist",
            },
        )
        stdin = iter(["播放他", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PRONOUN_PLAY_START_REPLY, stdout.getvalue())
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": track}),
                ("get_now_playing", {}),
            ],
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr4_session_preview_pronoun_binds_the_surviving_referent(self) -> None:
        # Regression 1 on the chat-session seam: after 试听 X -> stop, the
        # channel register is none (cleared by constitution) while
        # referent_canonical_id still names X -- 试听她 binds THAT id, one
        # preview execution, zero loop rounds, never a question.
        from music_agent.cli import main
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        track = "trk_22222222-2222-4222-8222-222222222222"
        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(
            channel_state="none",
            channel_canonical_id=None,
            referent_canonical_id=track,
            now_playing={
                "state": "playing",
                "persistent_id": "P-LIB-1",
                "name": "Synthetic Track",
                "artist": "Synthetic Artist",
            },
        )
        stdin = iter(["试听她", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PRONOUN_PREVIEW_START_REPLY, stdout.getvalue())
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [
                ("get_playback_context", {}),
                ("preview_catalog_track", {"canonical_id": track}),
            ],
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr4_session_play_pronoun_binds_the_surviving_referent(self) -> None:
        # Regression 3 on the chat-session seam: 播放他 after a stop binds
        # the surviving referent through formal playback ONLY (T14-E held).
        from music_agent.cli import main
        from music_agent.intent_router import PRONOUN_PLAY_START_REPLY

        track = "trk_22222222-2222-4222-8222-222222222222"
        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(
            channel_state="none",
            channel_canonical_id=None,
            referent_canonical_id=track,
            now_playing={
                "state": "playing",
                "persistent_id": "P-LIB-1",
                "name": "Synthetic Track",
                "artist": "Synthetic Artist",
            },
        )
        stdin = iter(["播放他", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PRONOUN_PLAY_START_REPLY, stdout.getvalue())
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [
                ("get_playback_context", {}),
                ("play_track", {"canonical_id": track}),
                ("get_now_playing", {}),
            ],
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr2_session_running_continuous_session_defers_to_the_loop(self) -> None:
        # A running continuous preview session makes the referent contested
        # (single-audio rule): the binding path refuses and the provider loop
        # owns the turn -- 试听她 still arrives with the proven 它 spelling.
        from music_agent.cli import main

        track = "trk_22222222-2222-4222-8222-222222222222"
        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(
            session_state="running",
            channel_state="preview",
            channel_canonical_id=track,
        )
        stdin = iter(["试听她", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["试听它"])
        self.assertEqual(loop.client.calls, [("get_playback_context", {})])
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr2_session_offline_pronoun_falls_back_to_the_loop(self) -> None:
        # agent_runtime_offline (the real trapdoor from M5-B): the binding
        # path refuses on a non-OK truth read, so the provider loop owns the
        # turn and reports the offline state its own way.
        from music_agent.cli import main

        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(offline=True)
        stdin = iter(["试听她", "/exit"])

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat-session", "--db", "s.db",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["试听它"])
        self.assertEqual(loop.client.calls, [("get_playback_context", {})])
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr2_one_shot_preview_pronoun_binds_to_the_channel_referent(self) -> None:
        # The one-shot chat command shares the same deterministic binding:
        # with a channel referent no provider round runs at all.
        from music_agent.cli import main
        from music_agent.intent_router import PRONOUN_PREVIEW_START_REPLY

        track = "trk_22222222-2222-4222-8222-222222222222"
        service = FakeService()
        loop = FakeLoop()
        loop.client = FakeClient(channel_state="preview", channel_canonical_id=track)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat", "--db", "s.db", "--message", "试听她",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PRONOUN_PREVIEW_START_REPLY, stdout.getvalue())
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [("get_playback_context", {}), ("preview_catalog_track", {"canonical_id": track})],
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14fr2_one_shot_unbound_pronoun_answers_a_fixed_question(self) -> None:
        # The one-shot command without a referent: fixed honest question,
        # zero provider rounds (the recommendation-drift regression on the
        # one-shot seam).
        from music_agent.cli import main
        from music_agent.intent_router import PRONOUN_PREVIEW_ASK

        service = FakeService()
        loop = FakeLoop()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", return_value=(service, loop)), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main([
                "chat", "--db", "s.db", "--message", "试听她",
                "--agent-client", f"{CLIENT_ID}:full",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn(PRONOUN_PREVIEW_ASK, stdout.getvalue())
        self.assertEqual(loop.messages, [])
        self.assertEqual(loop.client.calls, [("get_playback_context", {})])
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertTrue(service.closed)

    def test_t14f_session_unrelated_pronoun_phrase_reaches_the_loop_verbatim(
        self,
    ) -> None:
        # Test 4 on the CLI seam: 试听他的歌 keeps 他 -- only the
        # sentence-final track-reference object normalizes.
        exit_code, stdout, stderr, loop, service, build_calls, prompts = self._run(
            ["chat-session", "--db", "s.db", "--agent-client", f"{CLIENT_ID}:full"],
            ["试听他的歌", "/exit"],
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["试听他的歌"])
        self.assertTrue(service.closed)

    def test_t14f_one_shot_chat_pronoun_variant_normalizes(self) -> None:
        # The one-shot chat command shares the same boundary rewrite.
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "s.db")
            exit_code, stdout, stderr, loop, service, build_calls, prompts = self._run(
                ["chat", "--db", db, "--message", "停止试听她",
                 "--agent-client", f"{CLIENT_ID}:full"],
                [],
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["停止试听它"])
        self.assertTrue(service.closed)


class PlaybackStatusFastPathTest(unittest.TestCase):
    """P15-S4-M3-B: the five V1 status phrases answer deterministically in the
    session fast path -- one authoritative get_playback_context read, zero
    provider rounds, and never a 「正在播放」 claim unless a real playing read
    (remote or local degrade) warrants it."""

    def _run_status_scenario(self, client, lines, session_args=None):
        """One session over a preconfigured FakeClient; returns the run outputs.
        ``session_args`` land right after ``chat-session`` for subparser-level
        flags like ``--trace``."""
        from music_agent.cli import main

        loop = FakeLoop()
        loop.client = client
        service = FakeService()
        stdin = iter(lines)

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        def fake_build(args, schemas=None):
            return service, loop

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                ["chat-session", *(session_args or []), "--db", "s.db",
                 "--agent-client", f"{CLIENT_ID}:full"],
            )
        return exit_code, stdout.getvalue(), stderr.getvalue(), loop, service

    @staticmethod
    def _playing_client() -> FakeClient:
        return FakeClient(
            player={"state": "playing", "name": "晴天", "artist": "周杰伦", "album": "叶惠美"}
        )

    def test_five_v1_phrases_all_hit_the_fast_path(self) -> None:
        lines = [
            "现在在播放什么",
            "现在播放的是什么",
            "当前在播放什么",
            "当前播放什么",
            "现在是什么歌",
            "/quit",
        ]
        exit_code, stdout, _, loop, service = self._run_status_scenario(
            self._playing_client(), lines
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])  # zero provider rounds
        self.assertEqual(loop.client.calls, [("get_playback_context", {})] * 5)
        self.assertEqual(stdout.count("正在播放：《晴天》— 周杰伦（叶惠美）。"), 5)
        self.assertTrue(service.closed)

    def test_trailing_punctuation_and_whitespace_still_route(self) -> None:
        lines = [
            "现在在播放什么？",
            " 现在在播放什么 ？",
            "现在播放的是什么。",
            "现在是什么歌！",
            "当前播放什么。",
            "/quit",
        ]
        exit_code, stdout, _, loop, service = self._run_status_scenario(
            self._playing_client(), lines
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(loop.client.calls, [("get_playback_context", {})] * 5)
        self.assertEqual(stdout.count("正在播放："), 5)
        self.assertTrue(service.closed)

    def test_excluded_queries_still_go_to_the_provider_loop(self) -> None:
        for line in ("这是什么歌", "播放什么", "播放周杰伦"):
            with self.subTest(line=line):
                exit_code, stdout, _, loop, service = self._run_status_scenario(
                    self._playing_client(), [line, "/quit"]
                )
                self.assertEqual(exit_code, 0)
                self.assertEqual(loop.messages, [line])  # the provider runs
                self.assertEqual(loop.client.calls, [])  # no fast-path read at all
                self.assertIn(f"reply:{line}", stdout)
                self.assertTrue(service.closed)

    def test_formal_paused_never_claims_playing(self) -> None:
        client = FakeClient(
            player={"state": "paused", "name": "晴天", "artist": "周杰伦"}
        )
        exit_code, stdout, _, loop, service = self._run_status_scenario(
            client, ["现在在播放什么", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertIn("当前暂停：《晴天》— 周杰伦。", stdout)
        self.assertNotIn("正在播放", stdout)
        self.assertTrue(service.closed)

    def test_preview_sounding_wins_the_lead(self) -> None:
        client = FakeClient(
            player={"state": "playing", "name": "晴天", "artist": "周杰伦"},
            preview_sounding=True,
            session_state="running",
        )
        exit_code, stdout, _, loop, _ = self._run_status_scenario(
            client, ["现在在播放什么", "/quit"]
        )
        self.assertEqual(loop.messages, [])
        self.assertIn("正在试听：《Synthetic Solo》（30 秒试听）", stdout)
        self.assertNotIn("正在播放", stdout)  # formal playing is not the answer

    def test_sounding_with_suspension_appends_the_suspended_note(self) -> None:
        suspended = {
            "player_state": "paused",
            "persistent_id": "P1",
            "name": "起风了",
            "pause_ok": True,
        }
        client = FakeClient(
            player={"state": "paused"},
            suspended=suspended,
            preview_sounding=True,
            session_state="running",
        )
        exit_code, stdout, _, loop, _ = self._run_status_scenario(
            client, ["现在是什么歌", "/quit"]
        )
        self.assertEqual(loop.messages, [])
        self.assertIn(
            "正在试听：《Synthetic Solo》（30 秒试听）；正式播放《起风了》已暂停"
            "（试听优先，结束后说「继续播放」恢复）",
            stdout,
        )
        self.assertNotIn("正在播放", stdout)

    def test_suspension_outranks_the_formal_state(self) -> None:
        suspended = {
            "player_state": "paused",
            "persistent_id": "P1",
            "name": "起风了",
            "pause_ok": True,
        }
        # Suspended record + player reads paused: the suspension fact leads.
        paused_client = FakeClient(
            player={"state": "paused", "name": "晴天"}, suspended=suspended
        )
        exit_code, stdout, _, loop, _ = self._run_status_scenario(
            paused_client, ["现在在播放什么", "/quit"]
        )
        self.assertEqual(loop.messages, [])
        self.assertIn(
            "当前暂停：《起风了》（因刚才的试听而暂停，说「继续播放」恢复）。", stdout
        )
        self.assertNotIn("正在播放", stdout)
        # Suspended record + player reads playing again: the audible player wins.
        playing_client = FakeClient(
            player={"state": "playing", "name": "晴天", "artist": "周杰伦"},
            suspended=suspended,
        )
        exit_code, stdout2, _, loop2, _ = self._run_status_scenario(
            playing_client, ["现在在播放什么", "/quit"]
        )
        self.assertEqual(loop2.messages, [])
        self.assertIn("正在播放：《晴天》— 周杰伦。（此前因试听暂停的播放现已恢复。）", stdout2)

    def test_stopped_and_unreadable_states_stay_honest(self) -> None:
        stopped_client = FakeClient(player={"state": "stopped"})
        exit_code, stdout, _, loop, _ = self._run_status_scenario(
            stopped_client, ["现在在播放什么", "/quit"]
        )
        self.assertEqual(loop.messages, [])
        self.assertIn("当前没有任何音乐在播放。", stdout)
        self.assertNotIn("正在播放", stdout)
        # An observation without a player read must not read as silence either.
        exit_code2, stdout2, _, _, _ = self._run_status_scenario(
            FakeClient(), ["现在在播放什么", "/quit"]
        )
        self.assertIn("当前无法确定播放状态。", stdout2)
        self.assertNotIn("正在播放", stdout2)

    def test_fast_path_writes_no_provider_instrumentation_request_line(self) -> None:
        """P15-S4-M1: an agent_request line is only ever written from a provider
        round. The deterministic fast path never starts one, so --trace must
        stay silent per line."""
        with TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.jsonl"
            exit_code, stdout, _, loop, service = self._run_status_scenario(
                self._playing_client(), ["现在在播放什么", "/quit"],
                session_args=["--trace", str(trace_path)],
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(loop.messages, [])
            if trace_path.exists():
                trace_lines = trace_path.read_text().splitlines()
                self.assertFalse(
                    any("agent_request" in ln for ln in trace_lines), trace_lines
                )
            self.assertTrue(service.closed)

    def test_offline_degrades_to_local_get_now_playing(self) -> None:
        client = FakeClient(
            offline=True,
            now_playing={"state": "playing", "name": "晴天", "artist": "周杰伦"},
        )
        exit_code, stdout, _, loop, service = self._run_status_scenario(
            client, ["现在在播放什么？", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [("get_playback_context", {}), ("get_now_playing", {})],
        )
        self.assertIn("正在播放：《晴天》— 周杰伦。", stdout)
        self.assertIn("后台运行服务暂不可达，试听与暂停恢复状态暂无法确认。", stdout)
        self.assertTrue(service.closed)

    def test_not_registered_degrades_to_local_get_now_playing(self) -> None:
        """P20-Fix01: a REACHABLE run whose gate refused the session client is
        not offline -- the status answer still degrades to the safe local
        formal read, with the client-registration note and no internal
        `unknown_client` surface anywhere."""
        client = FakeClient(
            not_registered=True,
            now_playing={"state": "playing", "name": "晴天", "artist": "周杰伦"},
        )
        exit_code, stdout, stderr, loop, service = self._run_status_scenario(
            client, ["现在在播放什么？", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [("get_playback_context", {}), ("get_now_playing", {})],
        )
        self.assertIn("正在播放：《晴天》— 周杰伦。", stdout)
        self.assertIn(
            "后台运行时未注册本会话的客户端身份，暂只显示正式播放状态；"
            "试听与暂停恢复状态无法确认。",
            stdout,
        )
        self.assertNotIn("unknown_client", stdout)
        self.assertNotIn("unknown_client", stderr)
        self.assertNotIn("not registered with the shared agent service", stdout)
        self.assertNotIn("not registered with the shared agent service", stderr)
        self.assertTrue(service.closed)

    def test_offline_fallback_marks_preview_state_unknown(self) -> None:
        client = FakeClient(
            offline=True, now_playing={"state": "paused", "name": "晴天"}
        )
        exit_code, stdout, _, loop, _ = self._run_status_scenario(
            client, ["现在在播放什么", "/quit"]
        )
        self.assertEqual(loop.messages, [])
        self.assertIn("当前暂停：《晴天》", stdout)
        self.assertIn("试听与暂停恢复状态暂无法确认", stdout)
        self.assertNotIn("正在试听", stdout)  # never a fabricated preview claim
        self.assertNotIn("正在播放", stdout)

    def test_other_read_failures_report_honestly(self) -> None:
        class BrokenClient(FakeClient):
            def call(self, tool, payload):
                if tool == "get_playback_context":
                    self.calls.append((tool, dict(payload)))
                    return AgentToolResult(
                        request_id=generate_request_id(),
                        tool=tool,
                        outcome=AgentToolOutcome.EXECUTION_ERROR,
                        payload=None,
                        error_code="playback_unavailable",
                        error_message="osascript failed",
                        completed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                    )
                return super().call(tool, payload)

        exit_code, stdout, stderr, loop, _ = self._run_status_scenario(
            BrokenClient(), ["现在在播放什么", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(loop.client.calls, [("get_playback_context", {})])
        self.assertNotIn("当前没有任何音乐在播放", stdout)
        # P20-Fix01: the raw internal code never reaches the user -- the
        # surface is the stable unreadable line, the diagnostic goes to logs.
        self.assertIn("当前无法读取播放状态。", stderr)
        self.assertNotIn("playback_unavailable", stderr)
        self.assertNotIn("osascript failed", stderr)

        class RaisingClient(FakeClient):
            def call(self, tool, payload):
                self.calls.append((tool, dict(payload)))
                raise RuntimeError("socket gone")

        exit_code2, stdout2, stderr2, _, _ = self._run_status_scenario(
            RaisingClient(), ["现在在播放什么", "/quit"]
        )
        self.assertEqual(exit_code2, 0)
        self.assertIn("无法读取播放状态", stderr2)


class PlaybackStatusFormatterTest(unittest.TestCase):
    """P15-S4-M3-B pure state-matrix tests: every branch text is pinned, and no
    branch whose truth is not a real playing read may contain 正在播放."""

    @staticmethod
    def _fmt():
        from music_agent.cli import _format_playback_status
        return _format_playback_status

    @staticmethod
    def _fmt_local():
        from music_agent.cli import _format_local_playback_status
        return _format_local_playback_status

    def test_state_matrix_templates_are_exact(self) -> None:
        f = self._fmt()
        self.assertEqual(
            f({"preview_sounding": True, "session": {"current_name": "夜曲"}}),
            "正在试听：《夜曲》（30 秒试听）。",
        )
        self.assertEqual(f({"preview_sounding": True}), "正在试听中（30 秒试听）。")
        self.assertEqual(
            f(
                {
                    "preview_sounding": True,
                    "session": {"current_name": "夜曲"},
                    "suspended": {"name": "月半圆"},
                }
            ),
            "正在试听：《夜曲》（30 秒试听）；正式播放《月半圆》已暂停"
            "（试听优先，结束后说「继续播放」恢复）。",
        )
        self.assertEqual(
            f({"preview_sounding": True, "suspended": {"name": "月半圆"}}),
            "正在试听中（30 秒试听）；正式播放《月半圆》已暂停"
            "（试听优先，结束后说「继续播放」恢复）。",
        )
        self.assertEqual(
            f({"suspended": {"name": "月半圆"}, "player": {"state": "paused"}}),
            "当前暂停：《月半圆》（因刚才的试听而暂停，说「继续播放」恢复）。",
        )
        self.assertEqual(
            f({"suspended": {}}),
            "当前暂停中（因刚才的试听而暂停，说「继续播放」恢复）。",
        )
        self.assertEqual(
            f(
                {
                    "player": {
                        "state": "playing",
                        "name": "晴天",
                        "artist": "周杰伦",
                        "album": "叶惠美",
                    }
                }
            ),
            "正在播放：《晴天》— 周杰伦（叶惠美）。",
        )
        self.assertEqual(
            f({"player": {"state": "playing", "name": "晴天"}}), "正在播放：《晴天》。"
        )
        self.assertEqual(
            f({"player": {"state": "playing"}}), "正在播放中（曲目信息暂不可读）。"
        )
        self.assertEqual(
            f({"player": {"state": "paused", "name": "晴天", "artist": "周杰伦"}}),
            "当前暂停：《晴天》— 周杰伦。",
        )
        self.assertEqual(f({"player": {"state": "paused"}}), "当前已暂停播放。")
        self.assertEqual(
            f({"player": {"state": "stopped"}}), "当前没有任何音乐在播放。"
        )
        self.assertEqual(f({}), "当前无法确定播放状态。")
        self.assertEqual(
            f({"player": {"state": "rewinding"}}), "当前无法确定播放状态。"
        )
        # preview_sounding counts only when literally True.
        self.assertEqual(
            f({"preview_sounding": "yes"}), "当前无法确定播放状态。"
        )

    def test_non_playing_branches_never_claim_playing(self) -> None:
        f = self._fmt()
        branches = [
            {"preview_sounding": True, "session": {"current_name": "夜曲"}},
            {
                "preview_sounding": True,
                "session": {"current_name": "夜曲"},
                "suspended": {"name": "月半圆"},
            },
            {"preview_sounding": True},
            {"suspended": {"name": "月半圆"}, "player": {"state": "paused"}},
            {"player": {"state": "paused", "name": "晴天"}},
            {"player": {"state": "stopped"}},
            {},
            {"player": {"state": "rewinding"}},
        ]
        for observation in branches:
            with self.subTest(observation=observation):
                self.assertNotIn("正在播放", f(observation))

    def test_local_degrade_always_carries_the_offline_note(self) -> None:
        f = self._fmt_local()
        note = "（后台运行服务暂不可达，试听与暂停恢复状态暂无法确认。）"
        self.assertEqual(
            f({"state": "playing", "name": "晴天", "artist": "周杰伦"}),
            "正在播放：《晴天》— 周杰伦。" + note,
        )
        self.assertEqual(f({"state": "paused", "name": "晴天"}), "当前暂停：《晴天》。" + note)
        self.assertEqual(f({"state": "stopped"}), "当前没有任何音乐在播放。" + note)
        self.assertEqual(f(None), "当前无法确定播放状态。" + note)
        self.assertEqual(f({"state": "rewinding"}), "当前无法确定播放状态。" + note)

    def test_local_non_playing_branches_never_claim_playing(self) -> None:
        f = self._fmt_local()
        for now in (
            None,
            {"state": "paused", "name": "晴天"},
            {"state": "stopped"},
            {"state": "rewinding"},
        ):
            with self.subTest(now=now):
                self.assertNotIn("正在播放", f(now))


class PreviewEventPresenterTest(unittest.TestCase):
    """P15-S1 cli 呈现文案: progress/completion/cancellation notices + the
    suspension-based restore hint -- local, plain, never a model round."""

    def _present(self, event):
        from music_agent.cli import _preview_event_presenter

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            _preview_event_presenter(event)
        return stdout.getvalue()

    @staticmethod
    def _session(**overrides):
        session = {
            "state": "running",
            "total": 5,
            "position": 2,
            "current_canonical_id": "trk_x",
            "current_name": "Synthetic Solo",
            "skipped": [],
        }
        session.update(overrides)
        return session

    def test_progress_prints_position_total_and_name(self) -> None:
        out = self._present(
            {"event": "progress", "session": self._session(), "suspended": None}
        )
        self.assertEqual(out, "第 2/共 5 首：Synthetic Solo\n")

    def test_progress_falls_back_without_name_or_position(self) -> None:
        out = self._present(
            {"event": "progress", "session": self._session(total="?", current_name=None), "suspended": None}
        )
        self.assertEqual(out, "正在试听：试听\n")

    def test_completed_lists_skips_and_names_the_suspended_track(self) -> None:
        session = self._session(
            skipped=[
                {"canonical_id": "trk_1", "name": "Albumless", "reason": "catalog_preview_unavailable"},
                {"canonical_id": "trk_2", "name": None, "reason": None},
            ]
        )
        suspended = {
            "player_state": "playing",
            "persistent_id": "REAL-PID-1",
            "name": "起风了 (旧版)",
            "pause_ok": True,
        }
        out = self._present(
            {"event": "completed", "session": session, "suspended": suspended}
        )
        self.assertIn("试听连播完成。", out)
        self.assertIn("  未试听：《Albumless》（catalog_preview_unavailable）", out)
        self.assertIn("  未试听：《trk_2》（不可试听）", out)
        self.assertIn('之前暂停的《起风了 (旧版)》已暂停，说“继续播放”即可恢复。', out)

    def test_completed_without_suspension_prints_no_restore_hint(self) -> None:
        """§7: no suspension memo -> the hint line is absent, not guessed."""
        out = self._present(
            {"event": "completed", "session": self._session(), "suspended": None}
        )
        self.assertIn("试听连播完成。", out)
        self.assertNotIn("恢复", out)

    def test_cancelled_warns_honestly_when_the_pause_failed(self) -> None:
        suspended = {
            "player_state": "playing",
            "persistent_id": "REAL-PID-1",
            "name": "起风了 (旧版)",
            "pause_ok": False,
        }
        out = self._present(
            {"event": "cancelled", "session": self._session(), "suspended": suspended}
        )
        self.assertIn("试听连播已停止。", out)
        self.assertIn('刚才的音乐被中断了，可以用“继续播放”恢复。', out)

    def test_failed_prints_the_interrupt_notice_without_a_restore_hint(self) -> None:
        """P15-S1 真机修复: an aborted session never invites 继续播放 -- the user
        would read it as continuing the failed batch, not restoring the old music."""
        suspended = {
            "player_state": "playing",
            "persistent_id": "REAL-PID-1",
            "name": "起风了 (旧版)",
            "pause_ok": True,
        }
        out = self._present(
            {
                "event": "failed",
                "session": self._session(
                    state="failed", failure_reason="ProgrammingError"
                ),
                "suspended": suspended,
            }
        )
        self.assertIn("试听连播已中断（ProgrammingError）。", out)
        self.assertNotIn("恢复", out)
        self.assertNotIn("继续播放", out)

    def test_failed_without_a_reason_shows_a_generic_system_error(self) -> None:
        out = self._present(
            {
                "event": "failed",
                "session": self._session(state="failed", failure_reason=None),
                "suspended": None,
            }
        )
        self.assertIn("试听连播已中断（系统错误）。", out)
        self.assertNotIn("继续播放", out)

    def test_completed_with_every_clip_skipped_suppresses_the_restore_hint(self) -> None:
        """P15-S1 真机修复: a completion that heard none of the queue is abnormal
        -- the per-skip rows tell the honest story, so no 继续播放 pointer."""
        session = self._session(
            state="completed",
            total=3,
            skipped=[
                {"canonical_id": "trk_1", "name": "A", "reason": "catalog_preview_unavailable"},
                {"canonical_id": "trk_2", "name": "B", "reason": "catalog_preview_unavailable"},
                {"canonical_id": "trk_3", "name": "C", "reason": "catalog_preview_unavailable"},
            ],
        )
        out = self._present(
            {
                "event": "completed",
                "session": session,
                "suspended": {
                    "player_state": "playing",
                    "persistent_id": "REAL-PID-1",
                    "name": "起风了 (旧版)",
                    "pause_ok": True,
                },
            }
        )
        self.assertIn("试听连播完成。", out)
        self.assertIn("未试听", out)
        self.assertNotIn("继续播放", out)

    def test_non_mapping_event_is_silent(self) -> None:
        self.assertEqual(self._present("not an event"), "")
        self.assertEqual(self._present(None), "")


class FormalPlaySelectionTest(unittest.TestCase):
    """P16-S3 pure pick tests: batch order is recommendation order, so the
    first track-target item whose playback.route is library is the pick --
    skipped items are skipped in order, never re-ranked."""

    @staticmethod
    def _select():
        from music_agent.cli import _first_library_item

        return _first_library_item

    LIBRARY = {
        "candidate_id": "cnd_lib",
        "target_kind": "track",
        "target_id": "trk_lib",
        "name": "晴天",
        "artist_name": "周杰伦",
        "score_total": 0.8,
        "playback": {"route": "library", "label": "可正式播放"},
    }
    PREVIEW = {
        "candidate_id": "cnd_prev",
        "target_kind": "track",
        "target_id": "trk_prev",
        "name": "试听曲",
        "artist_name": "某艺人",
        "score_total": 0.9,
        "playback": {"route": "preview_only", "label": "只能试听"},
    }
    UNAVAILABLE = {
        "candidate_id": "cnd_na",
        "target_kind": "track",
        "target_id": "trk_na",
        "name": "不可用曲",
        "artist_name": None,
        "score_total": 0.5,
        "playback": {"route": "unavailable", "label": "不可用"},
    }
    ARTIST_ITEM = {
        "candidate_id": "cnd_art",
        "target_kind": "artist",
        "target_id": "art_1",
        "name": "周杰伦",
        "artist_name": None,
        "score_total": 0.6,
        "playback": {"route": "library", "label": "可正式播放"},
    }

    def test_picks_the_first_library_item_in_batch_order(self) -> None:
        f = self._select()
        self.assertEqual(f([self.PREVIEW, self.LIBRARY]), ("trk_lib", "晴天"))
        self.assertEqual(f([self.LIBRARY, self.PREVIEW]), ("trk_lib", "晴天"))
        self.assertEqual(
            f([self.UNAVAILABLE, self.PREVIEW, self.LIBRARY]), ("trk_lib", "晴天")
        )
        self.assertEqual(f([self.LIBRARY]), ("trk_lib", "晴天"))

    def test_non_track_targets_are_skipped_in_order(self) -> None:
        f = self._select()
        # An artist-target item does not preempt a later library track.
        self.assertEqual(f([self.ARTIST_ITEM, self.LIBRARY]), ("trk_lib", "晴天"))

    def test_no_library_item_returns_none(self) -> None:
        f = self._select()
        for items in (
            [],
            [self.PREVIEW],
            [self.UNAVAILABLE],
            [self.ARTIST_ITEM],
            [self.PREVIEW, self.UNAVAILABLE],
        ):
            with self.subTest(items=items):
                self.assertIsNone(f(items))

    def test_non_list_inputs_and_shape_anomalies_return_none(self) -> None:
        f = self._select()
        self.assertIsNone(f(None))
        self.assertIsNone(f("items"))
        self.assertIsNone(f(()))
        self.assertIsNone(f([None, 7, "x"]))
        self.assertIsNone(f([{"target_kind": "track"}]))

    def test_missing_name_falls_back_to_the_unknown_label(self) -> None:
        f = self._select()
        item = dict(self.LIBRARY)
        item.pop("name")
        self.assertEqual(f([item]), ("trk_lib", "未知曲目"))


class FormalPlayConfirmationTest(unittest.TestCase):
    """P16-S3 pure verdict tests: 正在播放 only for a real playing read that
    matches the play_track resolution; every other branch names the actual
    state honestly and never claims 正在播放."""

    @staticmethod
    def _fmt():
        from music_agent.cli import _format_formal_play_confirmation

        return _format_formal_play_confirmation

    def test_matching_playing_read_is_the_only_success_line(self) -> None:
        f = self._fmt()
        now = {
            "state": "playing",
            "persistent_id": "P-LIB-1",
            "name": "晴天",
            "artist": "周杰伦",
            "album": "叶惠美",
        }
        self.assertEqual(f(now, "晴天", "P-LIB-1"), "正在播放：《晴天》— 周杰伦（叶惠美）。")
        self.assertEqual(
            f({"state": "playing", "persistent_id": "P1", "name": "晴天"}, "晴天", "P1"),
            "正在播放：《晴天》。",
        )

    def test_mismatching_playing_read_reports_the_actual_audible_track(self) -> None:
        f = self._fmt()
        now = {"state": "playing", "persistent_id": "P-OTHER", "name": "别的"}
        out = f(now, "晴天", "P-LIB-1")
        self.assertIn("正在播放：《别的》", out)
        self.assertIn("但这不是刚才指定的《晴天》", out)

    def test_unreadable_identities_never_claim_a_verdict(self) -> None:
        f = self._fmt()
        unconfirmed = (
            f({"state": "playing", "persistent_id": "P1", "name": "别的"}, "晴天", None),
            f({"state": "playing", "name": "别的"}, "晴天", "P-LIB-1"),
        )
        for out in unconfirmed:
            with self.subTest(out=out):
                self.assertIn("但无法确认这一首就是刚才指定的《晴天》", out)
                self.assertNotIn("不一致", out)

    def test_unknown_names_collapse_to_the_neutral_label(self) -> None:
        f = self._fmt()
        self.assertEqual(f({"state": "paused"}, "未知曲目", "P1"), "已播放刚才那首，但当前是暂停状态。")

    def test_paused_stopped_and_unreadable_never_claim_playing(self) -> None:
        f = self._fmt()
        cases = (
            ({"state": "paused"}, "已播放《晴天》，但当前是暂停状态。"),
            ({"state": "stopped"}, "已播放《晴天》，但当前播放器处于停止状态。"),
            (
                {"state": "rewinding"},
                "已播放《晴天》，但无法确认当前播放器状态"
                "（说「现在在播放什么」可查看）。",
            ),
            (
                None,
                "已播放《晴天》，但无法确认当前播放器状态"
                "（说「现在在播放什么」可查看）。",
            ),
        )
        for now, expected in cases:
            with self.subTest(now=now):
                out = f(now, "晴天", "P-LIB-1")
                self.assertEqual(out, expected)
                self.assertNotIn("正在播放", out)


class FormalPlayFastPathTest(unittest.TestCase):
    """P16-S3: the five V1 formal-playback phrases run the deterministic chain
    through the P09 client with zero provider rounds; the post-play
    get_now_playing verification stays its own step, and every pre-mutation
    branch that is not safely deterministic remits the line to the provider
    loop (nothing executed -> the fallback cannot double-play)."""

    LIBRARY_ITEM = {
        "candidate_id": "cnd_lib",
        "target_kind": "track",
        "target_id": "trk_lib",
        "name": "晴天",
        "artist_name": "周杰伦",
        "score_total": 0.8,
        "playback": {"route": "library", "label": "可正式播放"},
    }
    PREVIEW_ITEM = {
        "candidate_id": "cnd_prev",
        "target_kind": "track",
        "target_id": "trk_prev",
        "name": "试听曲",
        "artist_name": "某艺人",
        "score_total": 0.9,
        "playback": {"route": "preview_only", "label": "只能试听"},
    }
    ACTIVE_BATCH = {"run_id": "run_1", "source": "derived", "item_count": 2}
    PLAYING_NOW = {
        "state": "playing",
        "persistent_id": "P-LIB-1",
        "name": "晴天",
        "artist": "周杰伦",
        "album": "叶惠美",
    }

    def _run_scenario(self, client, lines, session_args=None):
        """One session over a preconfigured FakeClient (the status-path harness
        of PlaybackStatusFastPathTest, reused for the formal-play path)."""
        from music_agent.cli import main

        loop = FakeLoop()
        loop.client = client
        service = FakeService()
        stdin = iter(lines)

        def fake_input(prompt: str = "") -> str:
            try:
                return next(stdin)
            except StopIteration:
                raise EOFError

        def fake_build(args, schemas=None):
            return service, loop

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("music_agent.cli._build_chat_session", side_effect=fake_build), \
                patch("builtins.input", fake_input), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(
                ["chat-session", *(session_args or []), "--db", "s.db",
                 "--agent-client", f"{CLIENT_ID}:full"],
            )
        return exit_code, stdout.getvalue(), stderr.getvalue(), loop, service

    def test_five_v1_phrases_all_run_the_deterministic_chain(self) -> None:
        for phrase in (
            "播放一首正式歌曲",
            "放一首正式歌曲",
            "播放一首正式的歌",
            "来一首正式歌曲",
            "正式播放一首",
        ):
            client = FakeClient(
                active_batch=self.ACTIVE_BATCH,
                run_items=[self.PREVIEW_ITEM, self.LIBRARY_ITEM],
                now_playing=self.PLAYING_NOW,
            )
            exit_code, stdout, _, loop, service = self._run_scenario(
                client, [phrase, "/quit"]
            )
            with self.subTest(phrase=phrase):
                self.assertEqual(exit_code, 0)
                # Zero provider rounds: the loop never sees the line.
                self.assertEqual(loop.messages, [])
                # The exact chain: locate -> session gate -> batch items ->
                # play -> verify. The verification is its own sequential step.
                self.assertEqual(
                    loop.client.calls,
                    [
                        ("get_active_context", {}),
                        ("get_playback_context", {}),
                        ("get_recommendation_run", {"run_id": "run_1"}),
                        ("play_track", {"canonical_id": "trk_lib"}),
                        ("get_now_playing", {}),
                    ],
                )
                self.assertIn("正在播放《晴天》— 周杰伦。", stdout)
                self.assertTrue(service.closed)

    def test_no_recommendation_data_remits_to_the_provider(self) -> None:
        client = FakeClient()  # active_batch None: no run exists at all
        exit_code, stdout, _, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        # The line went to the provider loop, and nothing was executed.
        self.assertEqual(loop.messages, ["播放一首正式歌曲"])
        self.assertEqual(
            loop.client.calls,
            [("get_active_context", {}), ("get_playback_context", {})],
        )
        self.assertIn("reply:播放一首正式歌曲", stdout)
        self.assertTrue(service.closed)

    def test_sounding_preview_remits_before_any_mutation(self) -> None:
        client = FakeClient(
            preview_sounding=True,
            active_batch=self.ACTIVE_BATCH,
            run_items=[self.LIBRARY_ITEM],
            now_playing=self.PLAYING_NOW,
        )
        exit_code, stdout, _, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["播放一首正式歌曲"])
        # The gate stops at the audible-preview check: no batch read, no play.
        self.assertEqual(loop.client.calls, [("get_active_context", {})])
        self.assertIn("reply:播放一首正式歌曲", stdout)
        self.assertTrue(service.closed)

    def test_running_session_remits_before_any_mutation(self) -> None:
        client = FakeClient(
            session_state="running",
            active_batch=self.ACTIVE_BATCH,
            run_items=[self.LIBRARY_ITEM],
            now_playing=self.PLAYING_NOW,
        )
        exit_code, stdout, _, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, ["播放一首正式歌曲"])
        self.assertEqual(
            loop.client.calls,
            [("get_active_context", {}), ("get_playback_context", {})],
        )
        self.assertIn("reply:播放一首正式歌曲", stdout)
        self.assertTrue(service.closed)

    def test_batch_without_library_items_refuses_deterministically(self) -> None:
        client = FakeClient(
            active_batch=self.ACTIVE_BATCH,
            run_items=[self.PREVIEW_ITEM],
            now_playing=self.PLAYING_NOW,
        )
        exit_code, stdout, _, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        # Deterministic refusal: zero provider rounds, and never a play.
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [
                ("get_active_context", {}),
                ("get_playback_context", {}),
                ("get_recommendation_run", {"run_id": "run_1"}),
            ],
        )
        self.assertIn("没有可以正式播放的曲目", stdout)
        self.assertTrue(service.closed)

    def test_verification_mismatch_reports_the_actual_audible_track(self) -> None:
        client = FakeClient(
            active_batch=self.ACTIVE_BATCH,
            run_items=[self.LIBRARY_ITEM],
            now_playing={
                "state": "playing",
                "persistent_id": "P-OTHER",
                "name": "别的",
            },
        )
        exit_code, stdout, _, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertIn("暂时无法确认当前播放状态", stdout)
        self.assertNotIn("正在播放", stdout)
        self.assertTrue(service.closed)

    def test_unreadable_verification_stays_honest(self) -> None:
        client = FakeClient(
            active_batch=self.ACTIVE_BATCH,
            run_items=[self.LIBRARY_ITEM],
        )
        exit_code, stdout, stderr, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        # The mutation happened and the chain owns its answer: the provider is
        # NOT re-run over an already-executed play.
        self.assertEqual(loop.messages, [])
        self.assertIn("暂时无法确认当前播放状态", stdout)
        self.assertNotIn("正在播放", stdout)
        self.assertFalse(stderr.count("music-agent:"))
        self.assertTrue(service.closed)

    def test_play_track_failure_reports_the_typed_error(self) -> None:
        class FailingPlayClient(FakeClient):
            def call(self, tool, payload):
                if tool == "play_track":
                    self.calls.append((tool, dict(payload)))
                    return AgentToolResult(
                        request_id=generate_request_id(),
                        tool=tool,
                        outcome=AgentToolOutcome.EXECUTION_ERROR,
                        payload=None,
                        error_code="playback_unavailable",
                        error_message="play adapter absent",
                        completed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                    )

                return super().call(tool, payload)

        client = FailingPlayClient(
            active_batch=self.ACTIVE_BATCH,
            run_items=[self.LIBRARY_ITEM],
            now_playing=self.PLAYING_NOW,
        )
        exit_code, stdout, stderr, loop, service = self._run_scenario(
            client, ["播放一首正式歌曲", "/quit"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(loop.messages, [])
        self.assertEqual(
            loop.client.calls,
            [
                ("get_active_context", {}),
                ("get_playback_context", {}),
                ("get_recommendation_run", {"run_id": "run_1"}),
                ("play_track", {"canonical_id": "trk_lib"}),
            ],
        )
        self.assertIn("这首暂时无法正式播放", stdout)
        self.assertEqual(stderr, "")
        self.assertTrue(service.closed)


if __name__ == "__main__":
    unittest.main()
