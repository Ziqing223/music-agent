"""P10.1: Command-line entry point for the integrated Music Agent runtime.

Subcommands (each slice extends this parser):
  ``run``          -- start the foreground runtime; runs until SIGINT/SIGTERM, then
                      shuts down cleanly. Requires an explicit ``--db`` path.
  ``refresh``      -- run one integrated Music.app refresh cycle and print the report.
  ``library-sync`` -- one full library discovery/sync scan (enumerate real persistent
                      IDs, ingest new tracks, refresh known ones, feed P06).
  ``playback``      -- low-latency deterministic playback path: the SAME P09 playback
                      tools over the same service/permission/replay boundary, without
                      any provider round-trip (P10 daily-use latency strategy).
  ``status``       -- print the durable store/runtime status (schema, task runs, process).
  ``install-agent`` / ``uninstall-agent`` -- LaunchAgent plist install/remove (P10.6).
  ``obsidian update-status`` -- bounded projection write of the 首页 auto-sync status
                      line through the production Obsidian writer (P10.9).
  ``chat``          -- one natural-language request through the real LLM provider and
                      the P09 shared agent tools (P10.8); credentials come only from
                      the environment or --api-key-env, never from the repository.
  ``chat-session``  -- interactive chat: the provider + P09 service are built once,
                      then each input line runs through the SAME path as ``chat
                      --message`` (no cross-message history); end with /exit, /quit,
                      exit, quit, 退出, Ctrl+D or Ctrl+C (P12-C06).

The store location is always explicit: the CLI never invents a default path.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

try:
    import readline  # noqa: F401 -- enables terminal line editing (arrows, delete)
except ImportError:  # pragma: no cover -- tty defaults still work without it
    readline = None

from music_agent.final_response_boundary import (
    present_final_text,
    presentation_fallback_kind,
)
from music_agent.runtime import RuntimeConfig, RuntimeStartupError

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# P17-B: the web shell's default agent-client id (used when no
# --agent-client is given). The agent contract requires the ``agt_`` prefix
# plus a canonical UUID suffix; one stable id keeps every UI-originated
# journal row attributable to the shell across launches.
_WEB_SHELL_CLIENT_ID = "agt_00000000-0000-4000-8000-0000000000e1"


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format=_LOG_FORMAT)


def _add_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        metavar="PATH",
        help="explicit path to the SQLite durable store",
    )


def _add_agent_client_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-client",
        action="append",
        default=[],
        metavar="CLIENT_ID:POLICY",
        help="register one agent client (agt_ id with full/read_only/none policy); "
        "repeatable; an unregistered client id is refused at the P09 permission gate",
    )


def _parse_agent_clients(entries: list[str]) -> dict[str, str]:
    """Parse ``--agent-client CLIENT_ID:POLICY`` entries into a policy mapping."""
    clients: dict[str, str] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"agent client entry must be CLIENT_ID:POLICY, got {entry!r}")
        client_id, policy = entry.split(":", 1)
        if not client_id or not policy:
            raise ValueError(f"agent client entry must be CLIENT_ID:POLICY, got {entry!r}")
        if client_id in clients:
            raise ValueError(f"duplicate agent client entry for {client_id!r}")
        clients[client_id] = policy
    return clients


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-agent",
        description="Music Agent integrated runtime (P10).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="start the foreground runtime; stop with SIGINT/SIGTERM",
    )
    _add_db_argument(run_parser)
    run_parser.add_argument(
        "--music-command-timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="timeout for every osascript Music command (default: 10)",
    )
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="stdlib log level (default: INFO)",
    )
    _add_agent_client_argument(run_parser)
    run_parser.add_argument(
        "--refresh-interval",
        type=int,
        default=900,
        metavar="SECONDS",
        help="music_refresh task interval (default: 900)",
    )
    run_parser.add_argument(
        "--capability-status-interval",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="capability_status task interval (default: 3600)",
    )
    run_parser.add_argument(
        "--no-audio-safety",
        action="store_true",
        help="disable the audio-safety monitor (default: enabled)",
    )
    run_parser.add_argument(
        "--audio-safety-poll-interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="audio-safety device poll interval (default: 2.0)",
    )

    refresh_parser = subparsers.add_parser(
        "refresh",
        help="run one integrated Music.app refresh cycle over bound tracks",
    )
    _add_db_argument(refresh_parser)
    refresh_parser.add_argument(
        "--music-command-timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="timeout for every osascript Music command (default: 10)",
    )
    refresh_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="stdlib log level (default: INFO)",
    )

    library_sync_parser = subparsers.add_parser(
        "library-sync",
        help="one full Music.app library discovery/sync scan (read-only on Music.app)",
    )
    _add_db_argument(library_sync_parser)
    library_sync_parser.add_argument(
        "--music-command-timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="timeout for every osascript Music command (default: 10)",
    )
    library_sync_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="stdlib log level (default: INFO)",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="print the durable store/runtime status",
    )
    _add_db_argument(status_parser)

    storage_report_parser = subparsers.add_parser(
        "storage-report",
        help="print the read-only storage report (table row counts + store size)",
    )
    _add_db_argument(storage_report_parser)

    daily_parser = subparsers.add_parser(
        "daily",
        help="one daily tick: library-sync -> refresh -> status summary",
    )
    _add_db_argument(daily_parser)

    backup_parser = subparsers.add_parser(
        "backup",
        help="copy the store (+ verify the copy reopens) to a timestamped local backup",
    )
    _add_db_argument(backup_parser)
    backup_parser.add_argument(
        "--backup-dir", type=Path, default=None, metavar="DIR",
        help="directory for the backup file (default: next to the store)",
    )

    playback_parser = subparsers.add_parser(
        "playback",
        help="low-latency playback commands through the same P09 boundary",
    )
    playback_subparsers = playback_parser.add_subparsers(
        dest="playback_command", required=True
    )

    def _add_playback_options(command_parser: argparse.ArgumentParser) -> None:
        _add_db_argument(command_parser)
        _add_agent_client_argument(command_parser)
        command_parser.add_argument(
            "--music-command-timeout",
            type=float,
            default=10.0,
            metavar="SECONDS",
            help="timeout for every osascript Music command (default: 10)",
        )

    for command, help_text in (
        ("pause", "pause Music.app playback"),
        ("play", "start/resume Music.app playback"),
        ("next", "advance to the next track"),
        ("previous", "return to the previous track"),
        ("now-playing", "read the current playback context"),
    ):
        command_parser = playback_subparsers.add_parser(command, help=help_text)
        _add_playback_options(command_parser)
    play_track_parser = playback_subparsers.add_parser(
        "play-track", help="play one canonical track (persistent-ID binding resolved)"
    )
    _add_playback_options(play_track_parser)
    play_track_parser.add_argument(
        "track_id", metavar="TRACK_ID",
        help="the canonical track id (trk_) to play",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="restore a backup over the store (refuses to overwrite without --force)",
    )
    _add_db_argument(restore_parser)
    restore_parser.add_argument(
        "--from", dest="backup_from", type=Path, required=True, metavar="FILE",
        help="the backup file to restore",
    )
    restore_parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing store (default: refuse)",
    )

    install_parser = subparsers.add_parser(
        "install-agent",
        help="write the LaunchAgent plist (~/Library/LaunchAgents) for daily startup",
    )
    _add_db_argument(install_parser)
    _add_agent_client_argument(install_parser)
    install_parser.add_argument(
        "--refresh-interval", type=int, default=900, metavar="SECONDS",
        help="music_refresh task interval (default: 900)",
    )
    install_parser.add_argument(
        "--capability-status-interval", type=int, default=3600, metavar="SECONDS",
        help="capability_status task interval (default: 3600)",
    )
    install_parser.add_argument(
        "--audio-safety-poll-interval", type=float, default=2.0, metavar="SECONDS",
        help="audio-safety device poll interval (default: 2.0)",
    )
    install_parser.add_argument(
        "--no-audio-safety", action="store_true",
        help="disable the audio-safety monitor in the installed agent",
    )
    install_parser.add_argument(
        "--music-command-timeout", type=float, default=10.0, metavar="SECONDS",
        help="timeout for every osascript Music command (default: 10)",
    )
    install_parser.add_argument(
        "--label", default="com.musicagent.runtime", metavar="LABEL",
        help="LaunchAgent label (default: com.musicagent.runtime)",
    )
    install_parser.add_argument(
        "--program", default=None, metavar="PATH",
        help="executable for ProgramArguments (default: the current interpreter with "
        "'-m music_agent')",
    )
    install_parser.add_argument(
        "--activate", action="store_true",
        help="also run 'launchctl bootstrap' (explicit system change; default: plist only)",
    )
    install_parser.add_argument(
        "--log-dir", type=Path, default=None, metavar="DIR",
        help="directory for the agent's stdout/stderr logs (default: next to the store)",
    )

    uninstall_parser = subparsers.add_parser(
        "uninstall-agent",
        help="remove the LaunchAgent plist (and boot it out with --active)",
    )
    uninstall_parser.add_argument(
        "--label", default="com.musicagent.runtime", metavar="LABEL",
        help="LaunchAgent label (default: com.musicagent.runtime)",
    )
    uninstall_parser.add_argument(
        "--active", action="store_true",
        help="also run 'launchctl bootout' before removing the plist",
    )

    obsidian_parser = subparsers.add_parser(
        "obsidian",
        help="bounded Obsidian projection writes (P10.9; approved surfaces only)",
    )
    obsidian_subparsers = obsidian_parser.add_subparsers(dest="obsidian_command", required=True)
    status_writer = obsidian_subparsers.add_parser(
        "update-status",
        help="update exactly the one 首页 auto-sync status bullet",
    )
    status_writer.add_argument(
        "--vault", type=Path, required=True, metavar="PATH",
        help="explicit path to the Obsidian Vault root",
    )
    status_writer.add_argument(
        "--status", required=True, metavar="TEXT",
        help="the new single-line auto-sync status text",
    )

    recommend_writer = obsidian_subparsers.add_parser(
        "append-recommendation",
        help="project ONE real durable recommendation run (rcm_) into 推荐历史",
    )
    recommend_writer.add_argument(
        "--vault", type=Path, required=True, metavar="PATH",
        help="explicit path to the Obsidian Vault root",
    )
    recommend_writer.add_argument(
        "--db", type=Path, required=True, metavar="PATH",
        help="the durable store holding the run",
    )
    recommend_writer.add_argument(
        "--run-id", required=True, metavar="RCM_ID",
        help="the real recommendation run id (rcm_) to project",
    )
    recommend_writer.add_argument(
        "--theme", default="每日推荐", metavar="TEXT",
        help="record theme heading text (default: 每日推荐)",
    )

    feedback_writer = obsidian_subparsers.add_parser(
        "append-feedback",
        help="project ONE real durable feedback observation (fbk_) into 推荐反馈",
    )
    feedback_writer.add_argument(
        "--vault", type=Path, required=True, metavar="PATH",
        help="explicit path to the Obsidian Vault root",
    )
    feedback_writer.add_argument(
        "--db", type=Path, required=True, metavar="PATH",
        help="the durable store holding the observation",
    )
    feedback_writer.add_argument(
        "--feedback-id", required=True, metavar="FBK_ID",
        help="the real feedback observation id (fbk_) to project",
    )

    library_count_writer = obsidian_subparsers.add_parser(
        "update-library-count",
        help="bounded homepage projection of the synced library track count (P10.14)",
    )
    library_count_writer.add_argument(
        "--vault", type=Path, required=True, metavar="PATH",
        help="explicit path to the Obsidian Vault root",
    )
    library_count_writer.add_argument(
        "--db", type=Path, required=True, metavar="PATH",
        help="the durable store whose canonical track count is projected",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="one natural-language request through the LLM provider + P09 tools",
    )
    _add_chat_options(chat_parser)
    chat_parser.add_argument(
        "--message", required=True, metavar="TEXT",
        help="the user's natural-language request",
    )

    chat_session_parser = subparsers.add_parser(
        "chat-session",
        help="interactive chat: one provider/service session, many messages "
        "(end with /exit, /quit, or EOF)",
    )
    _add_chat_options(chat_session_parser)

    web_parser = subparsers.add_parser(
        "web",
        help="P17-B minimal local product shell: browser UI over the loopback "
        "server (chat / now playing / play-pause / recommendation cards / "
        "catalog preview / library play / open in Apple Music)",
    )
    _add_chat_options(web_parser)
    web_parser.add_argument(
        "--port", type=int, default=0, metavar="PORT",
        help="loopback listen port (default: 0 = any free port)",
    )
    web_parser.add_argument(
        "--no-browser", action="store_true",
        help="do not open the browser after startup",
    )
    web_parser.add_argument(
        "--mode", choices=("embed", "attach", "standalone"), default=None,
        help="authority mode (default: probe <db>.agent.sock -- attach when a "
        "run is serving it, else embed the authority in this process; "
        "standalone is the test/lab mode with no socket at all)",
    )
    return parser


def _refresh_command(args: argparse.Namespace) -> int:
    """One-shot refresh cycle: open store, run the cycle over bound tracks, report."""
    try:
        config = RuntimeConfig(
            database_path=args.db,
            music_command_timeout_seconds=args.music_command_timeout,
            log_level=args.log_level,
        )
    except RuntimeStartupError as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2
    _configure_logging(config.log_level)
    from music_agent.apple_music import AppleMusicSourceAdapter, OsascriptMusicRunner
    from music_agent.repository import CanonicalRepository
    from music_agent.runtime_refresh import MusicRefreshOrchestrator

    repository: CanonicalRepository | None = None
    try:
        repository = CanonicalRepository(config.database_path)
        adapter = AppleMusicSourceAdapter(
            OsascriptMusicRunner(timeout_seconds=config.music_command_timeout_seconds)
        )
        from music_agent.preference_persistence_repository import (
            PreferencePersistenceRepository,
        )

        with PreferencePersistenceRepository(config.database_path) as preference:
            report = MusicRefreshOrchestrator(
                repository, adapter, preference_repository=preference
            ).run_cycle()
    except Exception as error:
        print(f"music-agent: refresh failed: {error}", file=sys.stderr)
        return 1
    finally:
        if repository is not None:
            repository.close()

    counts = report.counts()
    print(
        f"refresh complete: bound={report.bound_track_count} "
        f"skipped_no_binding={report.skipped_no_binding} "
        f"updated={counts['updated']} unchanged={counts['unchanged']} "
        f"source_not_found={counts['source_not_found']} "
        f"lookup_failed={counts['source_lookup_failed']} "
        f"merge_failed={counts['merge_failed']} failed={counts['failed']}"
    )
    for failure in report.failures:
        print(f"  failure: {failure.canonical_id}: {failure.error}", file=sys.stderr)
    return 0


def _library_sync_command(args: argparse.Namespace) -> int:
    from music_agent.apple_music import AppleMusicSourceAdapter, OsascriptMusicRunner
    from music_agent.apple_music_library_discovery import (
        AppleMusicLibraryDiscoveryAdapter,
        LibraryDiscoveryError,
        OsascriptLibraryTrackIdsRunner,
    )
    from music_agent.library_sync import LibrarySyncOrchestrator
    from music_agent.preference_persistence_repository import PreferencePersistenceRepository
    from music_agent.repository import CanonicalRepository

    repository = None
    try:
        repository = CanonicalRepository(args.db)
        adapter = AppleMusicSourceAdapter(
            OsascriptMusicRunner(timeout_seconds=args.music_command_timeout)
        )
        from music_agent.apple_music_genre_read import (
            AppleMusicGenreReadAdapter,
            OsascriptGenreReadRunner,
        )

        discovery = AppleMusicLibraryDiscoveryAdapter(
            OsascriptLibraryTrackIdsRunner(timeout_seconds=args.music_command_timeout)
        )
        genre_adapter = AppleMusicGenreReadAdapter(
            OsascriptGenreReadRunner(timeout_seconds=args.music_command_timeout)
        )
        with PreferencePersistenceRepository(args.db) as preference:
            report = LibrarySyncOrchestrator(
                repository, adapter, discovery, preference_repository=preference,
                genre_adapter=genre_adapter,
            ).run_cycle()
    except (LibraryDiscoveryError, Exception) as error:
        print(f"music-agent: library-sync failed: {error}", file=sys.stderr)
        return 1
    finally:
        if repository is not None:
            repository.close()

    if report.enumeration_failed:
        print(f"library-sync: enumeration failed: {report.enumeration_error}")
        return 1
    counts = report.counts()
    print(
        f"library-sync complete: enumerated={report.enumerated_count} "
        f"unique={report.unique_count} "
        f"new={counts['new']} updated={counts['updated']} unchanged={counts['unchanged']} "
        f"source_not_found={counts['source_not_found']} read_failed={counts['read_failed']} "
        f"merge_failed={counts['merge_failed']} ingestion_failed={counts['ingestion_failed']} "
        f"batch_failed={counts['batch_failed']} absent_this_scan={counts['absent_this_scan']} "
        f"genres={report.genre_counts}"
    )
    for outcome in report.outcomes:
        detail = f" (error: {outcome.error})" if outcome.error else ""
        print(
            f"  {outcome.status.value} {outcome.persistent_id} -> "
            f"{outcome.canonical_id or '-'}{detail}"
        )
    for persistent_id in report.absent_this_scan:
        print(f"  absent_this_scan {persistent_id}")
    return 0


def _status_command(args: argparse.Namespace) -> int:
    from music_agent.runtime_status import StoreStatusError, build_store_status, format_store_status

    try:
        status = build_store_status(args.db)
    except StoreStatusError as error:
        print(f"music-agent: {error}", file=sys.stderr)
        return 1
    print(format_store_status(status))
    return 0


def _run_arguments_for_config(args: argparse.Namespace) -> list[str]:
    """The 'run' argument list a LaunchAgent installs (explicit store path baked in)."""
    arguments = [
        "run",
        "--db", str(args.db),
        "--refresh-interval", str(args.refresh_interval),
        "--capability-status-interval", str(args.capability_status_interval),
        "--audio-safety-poll-interval", str(args.audio_safety_poll_interval),
        "--music-command-timeout", str(args.music_command_timeout),
    ]
    if args.no_audio_safety:
        arguments.append("--no-audio-safety")
    for entry in args.agent_client:
        arguments.extend(["--agent-client", entry])
    return arguments


def _install_agent_command(args: argparse.Namespace) -> int:
    import subprocess as _subprocess

    from music_agent.runtime_launchd import (
        LaunchAgentError,
        build_launch_agent_plist,
        default_plist_path,
        serialize_plist,
    )

    try:
        clients = _parse_agent_clients(args.agent_client)
        config = RuntimeConfig(
            database_path=args.db,
            music_command_timeout_seconds=args.music_command_timeout,
            agent_clients=clients,
            refresh_interval_seconds=args.refresh_interval,
            capability_status_interval_seconds=args.capability_status_interval,
            audio_safety_enabled=not args.no_audio_safety,
            audio_safety_poll_interval_seconds=args.audio_safety_poll_interval,
        )
    except (RuntimeStartupError, ValueError) as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2

    program = [args.program] if args.program else [sys.executable, "-m", "music_agent"]
    program.extend(_run_arguments_for_config(args))
    log_dir = args.log_dir or args.db.parent
    # launchd does not inherit the interactive shell's PYTHONPATH. Resolve the
    # import root from this loaded module so a src-layout checkout remains
    # runnable without a username-specific path or an editable installation.
    import_root = Path(__file__).resolve().parents[1]
    try:
        plist = build_launch_agent_plist(
            program,
            label=args.label,
            stdout_log=log_dir / f"{args.label}.log",
            stderr_log=log_dir / f"{args.label}.err.log",
            working_directory=Path.cwd(),
            environment_variables={"PYTHONPATH": str(import_root)},
        )
        content = serialize_plist(plist)
        plist_path = Path.home() / "Library" / "LaunchAgents" / f"{args.label}.plist"
    except LaunchAgentError as error:
        print(f"music-agent: {error}", file=sys.stderr)
        return 2

    try:
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_bytes(content)
    except OSError as error:
        print(f"music-agent: could not write {plist_path}: {error}", file=sys.stderr)
        return 1
    print(f"wrote LaunchAgent plist: {plist_path}")
    print(f"program: {' '.join(program)}")

    if args.activate:
        domain = f"gui/{os.getuid()}"
        try:
            _subprocess.run(
                ["launchctl", "bootstrap", domain, str(plist_path)],
                capture_output=True, text=True, check=False,
            )
        except OSError as error:
            print(f"music-agent: could not run launchctl: {error}", file=sys.stderr)
            return 1
        print(f"activated via: launchctl bootstrap {domain} {plist_path}")
        print("check logs under the log-dir; uninstall with: music-agent uninstall-agent --active")
    else:
        print(
            "not activated; run explicitly to start at login:\n"
            f"  launchctl bootstrap gui/$(id -u) {plist_path}\n"
            "or re-run with --activate"
        )
    return 0


def _uninstall_agent_command(args: argparse.Namespace) -> int:
    import subprocess as _subprocess

    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{args.label}.plist"
    if args.active:
        domain = f"gui/{os.getuid()}"
        try:
            _subprocess.run(
                ["launchctl", "bootout", f"{domain}/{args.label}"],
                capture_output=True, text=True, check=False,
            )
        except OSError as error:
            print(f"music-agent: could not run launchctl: {error}", file=sys.stderr)
            return 1
        print(f"booted out: {domain}/{args.label}")
    try:
        plist_path.unlink()
    except FileNotFoundError:
        print(f"music-agent: no LaunchAgent plist at {plist_path}")
        return 1
    print(f"removed LaunchAgent plist: {plist_path}")
    return 0


def _obsidian_command(args: argparse.Namespace) -> int:
    from music_agent.obsidian_event_projection import ObsidianEventProjectionError
    from music_agent.obsidian_projection import (
        ObsidianProjectionConfig,
        ObsidianProjectionError,
        ObsidianProjectionWriter,
    )

    writer = None
    try:
        writer = ObsidianProjectionWriter(ObsidianProjectionConfig(vault_root=args.vault))
        if args.obsidian_command == "update-status":
            result = writer.update_homepage_autosync_status(args.status)
        elif args.obsidian_command == "append-recommendation":
            from music_agent.obsidian_event_projection import build_recommendation_record

            record = build_recommendation_record(args.db, args.run_id, theme=args.theme)
            result = writer.append_recommendation_record(record)
        elif args.obsidian_command == "append-feedback":
            from music_agent.obsidian_event_projection import build_feedback_record

            record = build_feedback_record(args.db, args.feedback_id)
            result = writer.append_feedback_record(record)
        elif args.obsidian_command == "update-library-count":
            from music_agent.repository import CanonicalRepository

            with CanonicalRepository(args.db) as repository:
                count = len(repository.load_model()["tracks"])
            result = writer.update_homepage_library_count(count)
        else:
            print(f"music-agent: unknown obsidian command: {args.obsidian_command}", file=sys.stderr)
            return 2
    except (ObsidianProjectionError, ObsidianEventProjectionError) as error:
        print(f"music-agent: obsidian write refused: {error}", file=sys.stderr)
        return 1
    print(f"obsidian: {result.action.value} {result.surface.value} {result.path} "
          f"(verified={result.verified})")
    return 0


def _add_chat_options(parser: argparse.ArgumentParser) -> None:
    """The shared option set for ``chat`` and ``chat-session`` (everything but the
    message itself)."""
    _add_db_argument(parser)
    _add_agent_client_argument(parser)
    parser.add_argument(
        "--provider",
        choices=("deepseek", "codex"),
        default="deepseek",
        help="provider selection (default: deepseek; one provider per interaction, "
        "no automatic routing or fallback)",
    )
    parser.add_argument(
        "--api-key-env", default=None, metavar="VAR",
        help="environment variable holding the HTTP provider credential "
        "(deepseek default: DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--base-url", default=None, metavar="URL",
        help="HTTP provider base URL (deepseek default: https://api.deepseek.com/v1)",
    )
    parser.add_argument(
        "--model", default=None, metavar="NAME",
        help="provider model name (deepseek default: deepseek-chat; codex: passed to "
        "the Codex CLI with --model)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, metavar="SECONDS",
        help="provider request/command timeout (default: 120)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=8, metavar="N",
        help="maximum provider tool-call rounds (default: 8)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the provider/tool trace line after each message (default: hidden)",
    )
    parser.add_argument(
        "--trace", default=None, metavar="PATH",
        help="P15-S4-M1 diagnostic: append one structured JSONL line per provider "
        "agent request (round/tool/usage measurements only; never printed to the "
        "chat surface and always off unless given)",
    )
    parser.add_argument(
        "--agent-service", default=None, metavar="PATH",
        help="run-hosted agent socket for the preview/device-safety tool family "
        "(preview_batch / preview_catalog_track / stop_preview / advance_preview / "
        "get_playback_context / play; default: <db>.agent.sock). While the socket "
        "is not served those tools refuse with agent_runtime_offline instead of "
        "degrading locally; every other tool is unaffected",
    )


def _build_chat_session(
    args: argparse.Namespace,
    schemas: object | None = None,
) -> tuple[object, object]:
    """Build (service, loop) once for a chat process.

    Used by both ``chat`` and ``chat-session``; ``schemas`` overrides the
    provider-facing tool list (``None`` = the full ``PROVIDER_TOOL_SCHEMAS``).
    Reuse across messages is safe: ``ProviderAgentLoop.run`` starts a fresh
    conversation per call, and the P09 service already serves every tool call
    of one message through the same instance -- so one message in a session
    executes exactly like ``chat --message``, with no cross-message state.
    The service registers the local preview-session presenter here (progress,
    completion, cancellation notices -- including the restore hint) so every
    chat mode presents P15-S1 events, never only the model's words.

    P15-S2-IPC S3: the loop's client is a RoutedAgentClient -- the preview/
    device-safety family (preview_batch, preview_catalog_track, stop_preview,
    advance_preview, get_playback_context, play) executes on the run-hosted
    agent socket (``--agent-service``, default ``<db>.agent.sock``) and refuses
    with ``agent_runtime_offline`` while that socket is not served (never a
    local degradation); every other tool keeps the process-local service, so
    plain chat keeps working without run. P15-S2-IPC S4: the chat side binds
    its preview-event sink (``<db>.agent-events-<pid>.sock``, lazily just
    before the first remote preview call) so run streams session events back
    to the same shared presenter; each command's ``finally`` closes the sink
    with the process.
    """
    from music_agent.agent_contract import AgentClientIdentity
    from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
    from music_agent.agent_service import SharedAgentService
    from music_agent.provider_agent import ProviderAgentLoop, ProviderLoopConfig
    from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

    clients = _parse_agent_clients(args.agent_client)
    if len(clients) != 1:
        raise ValueError("exactly one --agent-client entry is required")
    (client_id, policy), = clients.items()

    provider = _build_chat_provider(args)
    model_label = (
        provider.config.model if args.provider != "codex" else (args.model or "codex")
    )
    from music_agent.catalog_ingestion import default_catalog_search_source
    from music_agent.playback_control import (
        MusicLibraryResolver,
        MusicPlaybackAdapter,
        OsascriptLibraryResolveRunner,
        OsascriptPlaybackRunner,
    )

    playback_adapter = MusicPlaybackAdapter(OsascriptPlaybackRunner(timeout_seconds=60.0))
    playback_resolver = MusicLibraryResolver(OsascriptLibraryResolveRunner(timeout_seconds=60.0))
    registry = AgentClientRegistry({client_id: AgentClientPolicy(policy)})
    service = SharedAgentService(
        args.db,
        clients=registry,
        playback_adapter=playback_adapter,
        playback_resolver=playback_resolver,
        catalog_search_source=default_catalog_search_source(),
    )
    from music_agent.audio_safety import device_safety_trace

    device_safety_trace(
        "chat-session: service built WITHOUT an AudioOutputObserver -- the "
        "device-safety pump exists only inside Runtime('run'); no observer "
        "poll thread is started in this process"
    )
    client_identity = AgentClientIdentity(
        client_id=client_id, model_id=model_label, label=f"provider-{args.provider}"
    )
    from music_agent.agent_socket import (
        AgentEventSocketListener,
        agent_socket_path,
        preview_event_socket_path,
    )
    from music_agent.routed_client import RoutedAgentClient

    remote_socket_path = agent_socket_path(Path(args.db))
    configured_socket = getattr(args, "agent_service", None)
    if configured_socket:
        remote_socket_path = Path(configured_socket)
    # P15-S2-IPC S4: the chat-side preview-event sink, bound lazily by the
    # client just before the first remote preview call; run then streams every
    # session event back to the shared presenter (identical bytes either way).
    event_socket_path = preview_event_socket_path(Path(args.db), os.getpid())
    event_listener = AgentEventSocketListener(event_socket_path, _preview_event_presenter)
    client = RoutedAgentClient(
        client_identity,
        service,
        remote_socket_path=remote_socket_path,
        event_socket_path=event_socket_path,
        event_listener=event_listener,
    )
    device_safety_trace(
        "chat-session: RoutedAgentClient armed -- the preview/device-safety "
        f"family routes to {remote_socket_path} (agent_runtime_offline while "
        "that socket is not served); every other tool stays on the local "
        f"service; preview events return over {event_socket_path}"
    )
    service.preview_event_handler = _preview_event_presenter
    loop = ProviderAgentLoop(provider, client, schemas or PROVIDER_TOOL_SCHEMAS,
                         config=ProviderLoopConfig(
                             max_tool_rounds=args.max_rounds,
                             # P15-S4-M1: measurements only when --trace asks for
                             # them (the same loop serves every session message).
                             instrument=bool(getattr(args, "trace", None)),
                         ))
    return service, loop


_CHAT_SESSION_EXIT_WORDS = frozenset({"/exit", "/quit", "退出", "exit", "quit"})

# Routed control commands print locally after a successful tool call. stop_preview
# reports its idempotent ``stopped`` flag instead (see _run_routed_command).
_ROUTED_COMMAND_ACKS = {
    "pause": "已暂停播放。",
    "play": "已继续播放。",
    "next_track": "已切换到下一首。",
    "previous_track": "已切换到上一首。",
}


def _preview_event_presenter(event: object) -> None:
    """P15-S1 local presenter for continuous-preview session events.

    Registered on the service, so presentation happens exactly once -- from the
    reaper thread for progress/completion, from the agent loop for a stop/暂停/
    pre-emption -- and never waits on the model. A completion or cancellation
    carries the service-level suspension memo: when a preview run interrupted
    Music.app, the notice says it honestly and points at the intended restore
    (never an automatic resume, per the no-auto-resume doctrine). P15 真机修复:
    a systemic failure (event "failed") or an all-skipped completion prints no
    restore pointer -- the failure speaks for itself and must not mislead.
    """
    from collections.abc import Mapping

    if not isinstance(event, Mapping):
        return
    session = event.get("session") if isinstance(event.get("session"), Mapping) else {}
    kind = event.get("event")
    logging.getLogger("music_agent.cli").debug(
        "[preview-ipc] chat: presenter disposition kind=%s", kind
    )
    if kind == "progress":
        position, total = session.get("position"), session.get("total")
        name = session.get("current_name") or "试听"
        if isinstance(position, int) and isinstance(total, int):
            print(f"第 {position}/共 {total} 首：{name}")
        else:
            print(f"正在试听：{name}")
    elif kind == "completed":
        print("试听连播完成。")
        for record in session.get("skipped") or ():
            if not isinstance(record, Mapping):
                continue
            label = record.get("name") or record.get("canonical_id") or "未知曲目"
            print(f"  未试听：《{label}》（{record.get('reason') or '不可试听'}）")
        # P15 真机修复: a completion where the queue heard nothing is abnormal --
        # the skip rows above tell that story, and a 继续播放 pointer here would
        # read as "continue the batch", not the honest restore. Only a completion
        # that actually delivered clips gets the restore hint (restore-by-intent
        # itself is unchanged: 继续 still routes to play).
        skipped_records = session.get("skipped") or ()
        total = session.get("total")
        if isinstance(total, int) and len(skipped_records) >= total:
            return
        _print_restore_hint(event)
    elif kind == "failed":
        # P15 真机修复: a systemic failure aborts the run -- the reason is shown
        # plainly and no restore hint is invented (the failed batch is neither a
        # normal completion nor a normal stop).
        reason = session.get("failure_reason") or "系统错误"
        print(f"试听连播已中断（{reason}）。")
    elif kind == "cancelled":
        print("试听连播已停止。")  # the stop ≙ end-of-session notice (A2/A3)
        _print_restore_hint(event)


def _print_restore_hint(event: object) -> None:
    from collections.abc import Mapping

    suspended = event.get("suspended") if isinstance(event, Mapping) else None
    if not isinstance(suspended, Mapping):
        return  # nothing was interrupted -- no hint (§7: None 则不出该句)
    name = suspended.get("name")
    if suspended.get("pause_ok"):
        if name:
            print(f'之前暂停的《{name}》已暂停，说“继续播放”即可恢复。')
        else:
            print('之前暂停的音乐已暂停，说“继续播放”即可恢复。')
    else:
        print('刚才的音乐被中断了，可以用“继续播放”恢复。')


def _print_session_progress(session: object) -> None:
    """The 继续/play interception during a RUNNING session: report progress,
    never resume Music.app over sounding preview audio (single-audio-source)."""
    from collections.abc import Mapping

    details = session if isinstance(session, Mapping) else {}
    position, total = details.get("position"), details.get("total")
    if isinstance(position, int) and isinstance(total, int):
        print(f"连播进行中（第 {position}/共 {total} 首）。")
    else:
        print("连播进行中。")


def _run_routed_command(loop: object, tool: str) -> None:
    """Execute one routed control command through the same P09 client the loop uses.

    The fast path never bypasses the service: ``loop.client.call`` mints the agent
    request and runs the full SharedAgentService.execute chain (journal, replay guard,
    registry validation, permission class check) -- only the model round is skipped.
    A failing transport is reported and the session stays alive.
    """
    from collections.abc import Mapping

    from music_agent.action_attempt import (
        render_verified_playback_control_result,
        run_playback_control_attempt,
    )
    from music_agent.agent_contract import AgentToolOutcome

    if tool in {"play", "pause"}:
        attempt = run_playback_control_attempt(
            tool,
            expected_state="playing" if tool == "play" else "paused",
            invoke=loop.client.call,
        )
        print(render_verified_playback_control_result(attempt))
        return

    try:
        result = loop.client.call(tool, {})
    except Exception as error:  # osascript-family errors escape execute(); never kill the session
        print(f"music-agent: {tool} 失败：{error}", file=sys.stderr)
        return
    payload = result.payload if isinstance(result.payload, Mapping) else {}
    if result.outcome is AgentToolOutcome.OK:
        if tool == "stop_preview":
            if payload.get("preview_session_cancelled"):
                # P15-S1: the cancel notice + restore hint were already presented
                # by the session event handler during execution.
                pass
            elif payload.get("stopped"):
                print("已停止试听。")
            else:
                print("当前没有正在播放的试听。")
        elif tool == "advance_preview":
            # P15-S1 C02: the session events (the next clip's progress, or the
            # completion) are presented by the session event handler while the
            # tool runs -- a second line here would double-print.
            pass
        else:
            print(_ROUTED_COMMAND_ACKS.get(tool, f"{tool} 已执行。"))
    else:
        print(f"music-agent: {result.error_code}: {result.error_message}", file=sys.stderr)


def _format_playback_status(observation: object) -> str:
    """P15-S4-M3-B: deterministic state matrix for one get_playback_context payload.

    Pure function: observation in, fixed Chinese sentence out -- never a
    provider round, never a fabricated field. Priority is exactly the design
    matrix: preview_sounding first (the audible truth), the suspension entry
    second (a suspended formal playback must never read as 正在播放), then the
    formal player state (playing / paused / stopped / anything else unknown).
    The strict honesty rule: no branch emits 「正在播放」 unless the reads
    really show playback in progress.
    """
    from collections.abc import Mapping

    payload = observation if isinstance(observation, Mapping) else {}
    sounding = payload.get("preview_sounding") is True
    player = payload.get("player")
    player = player if isinstance(player, Mapping) else None
    session = payload.get("session")
    session = session if isinstance(session, Mapping) else None
    suspended = payload.get("suspended")
    suspended = suspended if isinstance(suspended, Mapping) else None

    if sounding:
        current_name = session.get("current_name") if session is not None else None
        if isinstance(current_name, str) and current_name:
            lead = f"正在试听：《{current_name}》（30 秒试听）"
        else:
            lead = "正在试听中（30 秒试听）"
        if suspended is not None:
            suspended_name = suspended.get("name")
            if isinstance(suspended_name, str) and suspended_name:
                lead += f"；正式播放《{suspended_name}》已暂停（试听优先，结束后说「继续播放」恢复）"
            else:
                lead += "；正式播放已暂停（试听优先，结束后说「继续播放」恢复）"
        return lead + "。"

    formal_state = player.get("state") if player is not None else None
    name = player.get("name") if player is not None else None
    artist = player.get("artist") if player is not None else None
    album = player.get("album") if player is not None else None

    if suspended is not None:
        # The recorded suspension out-ranks a playing read: a suspended formal
        # playback is never reported as still 正在播放.
        if formal_state == "playing":
            # The real player reads playing again (restore-by-intent landed, or
            # the user resumed manually): report what is now audible.
            return f"{_formal_playing_line(name, artist, album)}（此前因试听暂停的播放现已恢复。）"
        suspended_name = suspended.get("name")
        if isinstance(suspended_name, str) and suspended_name:
            return f"当前暂停：《{suspended_name}》（因刚才的试听而暂停，说「继续播放」恢复）。"
        return "当前暂停中（因刚才的试听而暂停，说「继续播放」恢复）。"

    if formal_state == "playing":
        return _formal_playing_line(name, artist, album)
    if formal_state == "paused":
        if isinstance(name, str) and name:
            line = f"当前暂停：《{name}》"
            if isinstance(artist, str) and artist:
                line += f"— {artist}"
            return line + "。"
        return "当前已暂停播放。"
    if formal_state == "stopped":
        return "当前没有任何音乐在播放。"
    # player None (no read was possible) or an unrecognized state: fail honest.
    return "当前无法确定播放状态。"


def _formal_playing_line(
    name: object, artist: object, album: object
) -> str:
    """The one factual 正在播放 line -- reachable only for a real playing read."""
    if isinstance(name, str) and name:
        line = f"正在播放：《{name}》"
        if isinstance(artist, str) and artist:
            line += f"— {artist}"
        if isinstance(album, str) and album:
            line += f"（{album}）"
        return line + "。"
    return "正在播放中（曲目信息暂不可读）。"


def _first_library_item(items: object) -> tuple[str, str] | None:
    """P16-S3: the deterministic formal-playback pick for one batch's items.

    Batch order IS the recommendation order, so the FIRST item whose
    playback.route is library is the pick -- the same 优先 library policy the
    provider prompt encodes for delegation, executed without a model round.
    Items that are not track targets or are not library-routable are skipped
    in order; None when the batch offers no formally playable track (the
    caller then answers deterministically instead of degrading to a preview
    the user explicitly ruled out with 正式). Pure: items in,
    (target_id, name) out.
    """
    from collections.abc import Mapping

    if not isinstance(items, (list, tuple)):
        return None
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if item.get("target_kind") != "track":
            continue
        playback = item.get("playback")
        if not isinstance(playback, Mapping):
            continue
        if playback.get("route") != "library":
            continue
        target_id = item.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            continue
        name = item.get("name")
        name = name if isinstance(name, str) and name else "未知曲目"
        return target_id, name
    return None


def _format_formal_play_confirmation(
    now_playing: object, intended_name: str, played_pid: object
) -> str:
    """P16-S3: the deterministic post-play answer, verdicts from the real
    verification read only.

    正在播放 appears exactly when the read shows playing AND its
    persistent_id matches the play_track resolution -- never from the
    user's intended name alone, never from the command's ok flag (the same
    honesty invariant as the M3-B status matrix). A playing read that does
    not match is reported as what is actually audible, not as success, and
    a paused/stopped/unreadable state never claims 正在播放 either.
    """
    from collections.abc import Mapping

    now = now_playing if isinstance(now_playing, Mapping) else None
    state = now.get("state") if now is not None else None
    target = f"《{intended_name}》" if intended_name != "未知曲目" else "刚才那首"

    if state == "playing":
        now_pid = now.get("persistent_id")
        line = _formal_playing_line(
            now.get("name"), now.get("artist"), now.get("album")
        )
        if played_pid is not None and now_pid is not None and played_pid == now_pid:
            return line
        if played_pid is None or now_pid is None:
            # Either identity is unreadable: report the audible truth without
            # claiming it is (or is not) the requested track.
            return f"{line[:-1]}，但无法确认这一首就是刚才指定的{target}。"
        return f"{line[:-1]}，但这不是刚才指定的{target}（与刚发出的播放指令不一致）。"
    if state == "paused":
        return f"已播放{target}，但当前是暂停状态。"
    if state == "stopped":
        return f"已播放{target}，但当前播放器处于停止状态。"
    # None (no read was possible) or an unrecognized state: fail honest.
    return f"已播放{target}，但无法确认当前播放器状态（说「现在在播放什么」可查看）。"


_PLAYBACK_STATUS_OFFLINE_NOTE = "（后台运行服务暂不可达，试听与暂停恢复状态暂无法确认。）"

# P20-Fix01: the degrade note when the runtime IS reachable but did not
# register this session's client identity -- the answer still comes from
# the safe local formal read, and the note names the configuration cause
# instead of the internal `unknown_client` registry sentence.
_PLAYBACK_STATUS_NOT_REGISTERED_NOTE = (
    "（后台运行时未注册本会话的客户端身份，暂只显示正式播放状态；"
    "试听与暂停恢复状态无法确认。）"
)


def _format_local_playback_status(
    now_playing: object, *, note: str = _PLAYBACK_STATUS_OFFLINE_NOTE
) -> str:
    """Formal-only templates for the remote-side degrade (a local get_now_playing read).

    Music.app facts are locally readable and reported honestly; preview and
    suspension truth live on the unreachable run, so every answer carries the
    degrade ``note`` (offline or client-not-registered) instead of pretending
    to know them.
    """
    from collections.abc import Mapping

    now = now_playing if isinstance(now_playing, Mapping) else None
    state = now.get("state") if now is not None else None
    name = now.get("name") if now is not None else None
    artist = now.get("artist") if now is not None else None
    album = now.get("album") if now is not None else None
    if state == "playing":
        return _formal_playing_line(name, artist, album) + note
    if state == "paused":
        if isinstance(name, str) and name:
            line = f"当前暂停：《{name}》"
            if isinstance(artist, str) and artist:
                line += f"— {artist}"
            line += "。"
        else:
            line = "当前已暂停播放。"
        return line + note
    if state == "stopped":
        return "当前没有任何音乐在播放。" + note
    # now None (local read was not possible) or an unrecognized state.
    return "当前无法确定播放状态。" + note


def _print_playback_status(loop: object) -> None:
    """P15-S4-M3-B fast path: one authoritative read + deterministic formatting.

    ``get_playback_context`` (run-hosted) carries the formal player state, the
    runner's preview truth, the live session and the suspension entry in one
    observation. On ``agent_runtime_offline`` -- and, P20-Fix01, on the
    translated ``agent_client_not_registered`` refusal -- the answer degrades
    to the local ``get_now_playing`` read: formal Music.app truth only,
    explicitly noting (with the matching note) that preview/suspension state
    is unreadable. Never a silent fail-closed "nothing is playing", never a
    fabricated preview claim; any other read failure is reported with the
    stable unreadable line (the console diagnostic names only the generic
    outcome -- raw wire codes stay out of every user-visible surface and
    live in trace logs). Zero provider rounds either way.
    """
    from collections.abc import Mapping

    from music_agent.agent_contract import AgentToolOutcome
    from music_agent.agent_socket import (
        AGENT_CLIENT_NOT_REGISTERED_CODE,
        AGENT_RUNTIME_OFFLINE_CODE,
    )

    try:
        result = loop.client.call("get_playback_context", {})
    except Exception as error:
        print(f"music-agent: 无法读取播放状态：{error}", file=sys.stderr)
        return
    if result.outcome is AgentToolOutcome.OK:
        payload = result.payload if isinstance(result.payload, Mapping) else {}
        print(_format_playback_status(payload))
        return
    error_code = getattr(result, "error_code", None)
    if error_code in (AGENT_RUNTIME_OFFLINE_CODE, AGENT_CLIENT_NOT_REGISTERED_CODE):
        note = (
            _PLAYBACK_STATUS_OFFLINE_NOTE
            if error_code == AGENT_RUNTIME_OFFLINE_CODE
            else _PLAYBACK_STATUS_NOT_REGISTERED_NOTE
        )
        try:
            local = loop.client.call("get_now_playing", {})
        except Exception as error:
            print(f"music-agent: 无法读取播放状态：{error}", file=sys.stderr)
            return
        if local.outcome is AgentToolOutcome.OK:
            payload = local.payload if isinstance(local.payload, Mapping) else {}
            print(_format_local_playback_status(payload.get("now_playing"), note=note))
            return
        print("music-agent: 当前无法读取播放状态。", file=sys.stderr)
        return
    logging.getLogger("music_agent.cli").warning(
        "playback status fast path: remote read failed with outcome=%s; "
        "showed the stable unreadable line (raw wire detail stays in traces)",
        getattr(result, "outcome", None),
    )
    print("music-agent: 当前无法读取播放状态。", file=sys.stderr)


def _run_formal_play(loop: object) -> bool:
    """P16-S3: deterministic formal-playback chain for the closed V1 phrase set.

    Replaces the provider-planned chain (batch locator reads -> selection ->
    play_track -> get_now_playing verification -> answer -- historically five
    provider rounds for 「播放一首正式歌曲」) with the SAME steps executed
    straight through ``loop.client``: the full P09 chain (journal, replay
    guard, remote-run routing) is identical to the provider path, only the
    model rounds are gone.

    Round economy is never bought with correctness or behavior changes: the
    post-mutation get_now_playing verification stays its own sequential step
    (mutate and readback are never collapsed), and before any mutation the
    chain remits to the caller's provider loop whenever it is not safely
    deterministic -- no recommendation data at all, or a sounding preview /
    running continuous-preview session that owns the single audio source.

    Returns True when the line was fully handled here (played + verified, or
    a deterministic refusal). Returns False BEFORE any mutation when no
    deterministic chain exists; the caller hands the untouched line to the
    provider loop, so the fallback can never double-play.
    """
    from collections.abc import Mapping

    from music_agent.action_attempt import (
        create_direct_action_attempt,
        render_verified_action_result,
        run_action_attempt,
    )
    from music_agent.agent_contract import AgentToolOutcome

    try:
        context = loop.client.call("get_active_context", {})
    except Exception as error:
        print(f"music-agent: 无法定位当前推荐批次：{error}", file=sys.stderr)
        return False
    if context.outcome is not AgentToolOutcome.OK:
        return False
    payload = context.payload if isinstance(context.payload, Mapping) else {}
    if payload.get("preview_sounding") is True:
        # An audible preview owns the single audio source: a deterministic
        # formal playback could race it. Remit -- the provider loop applies
        # the existing preview/suspension policy exactly as before.
        return False
    try:
        playback = loop.client.call("get_playback_context", {})
    except Exception:
        return False
    if playback.outcome is not AgentToolOutcome.OK:
        # Session truth unreadable: without audibility evidence a
        # deterministic play could collide with a running preview session.
        return False
    session = (
        playback.payload.get("session")
        if isinstance(playback.payload, Mapping)
        else None
    )
    if isinstance(session, Mapping) and session.get("state") == "running":
        # A live continuous-preview session may fire its next clip at any
        # moment: remit rather than race it with Music.app.
        return False
    batch = payload.get("active_batch")
    batch = batch if isinstance(batch, Mapping) else None
    if batch is None:
        # No active batch and no persisted run: only the model can decide
        # what to generate or ask for. Remit before any mutation.
        return False
    run_id = batch.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return False
    try:
        run = loop.client.call("get_recommendation_run", {"run_id": run_id})
    except Exception as error:
        print(f"music-agent: 无法读取当前推荐批次：{error}", file=sys.stderr)
        return False
    if run.outcome is not AgentToolOutcome.OK:
        return False
    items = run.payload.get("items") if isinstance(run.payload, Mapping) else None
    pick = _first_library_item(items)
    if pick is None:
        # Deterministic answer: a formal playback needs a library route and
        # this batch offers none. The provider path would be tempted to
        # auto-preview, which misreads the explicit 正式 constraint.
        print(
            "当前批次没有可以正式播放的曲目（都只能试听 30 秒）。"
            "说「随便播放一首」可以试听，或让我「换一组」再推荐。"
        )
        return True
    canonical_id, name = pick
    attempt = run_action_attempt(
        create_direct_action_attempt(
            canonical_id,
            route="library",
            title=name,
        ),
        loop.client.call,
    )
    print(render_verified_action_result(attempt))
    return True


def _fetch_routing_context(
    loop: object, line: str
) -> tuple[str | None, str | None, bool, object, object]:
    """P15-S1: read the live playback context for context-aware routing.

    One provider-free local read per control line -- a control command never
    pays two Music.app reads:

    - ``get_playback_context`` serves 停止/stop and the session-aware forms
      (暂停/pause + the continue family + C02's 下一首/next/下一首试听): it
      carries ``channel``, the runner truth ``preview_sounding``, the live
      ``session`` snapshot and the ``suspended`` restore target in one read.
    - ``get_active_context`` alone still carries the ownership judgment
      (``context``) that 换一首 needs (which never reads the suspension entry).

    Any failure -- a typed tool error, an exception escaping execute(), or an
    unexpected payload shape -- degrades to ``(None, None, False, None, None)``
    = unknown, which makes the router refuse the context-sensitive routes
    (宁可不路由，不误触发), makes the C02 continue gating refuse play (no
    verified restore target), and hands the line to the provider loop instead.
    """
    from collections.abc import Mapping

    exact = line.strip().casefold() if isinstance(line, str) else ""
    try:
        if exact == "换一首":
            result = loop.client.call("get_active_context", {})
        else:
            result = loop.client.call("get_playback_context", {})
    except Exception:
        return None, None, False, None, None
    payload = result.payload if isinstance(result.payload, Mapping) else {}
    channel = payload.get("channel")
    channel = channel if isinstance(channel, Mapping) else None
    if channel is None:
        return None, None, False, None, None
    state = channel.get("state")
    ownership = payload.get("context")
    sounding = payload.get("preview_sounding")
    session = payload.get("session") if isinstance(payload.get("session"), Mapping) else None
    suspended = (
        payload.get("suspended") if isinstance(payload.get("suspended"), Mapping) else None
    )
    if not isinstance(state, str):
        state = None
    if not isinstance(ownership, str):
        ownership = None
    if not isinstance(sounding, bool):
        sounding = False
    return state, ownership, sounding, session, suspended


def _show_waiting_hint() -> None:
    """One-char progress hint while a provider request runs (TTYs only)."""
    if sys.stdout.isatty():
        print("…", end="", flush=True)


def _clear_waiting_hint() -> None:
    """Erase the waiting hint before printing the result or an error."""
    if sys.stdout.isatty():
        print("\r \r", end="", flush=True)


def _arm_preview_offer(
    offered_actions: object | None,
    target_canonical_id: str,
    *,
    source: str,
    verified_title: str | None = None,
    verified_artist: str | None = None,
) -> None:
    """Arm the session-owned target-bound Preview offer when available."""
    if offered_actions is None:
        return
    from music_agent.conversation_continuation import OfferedAction, PREVIEW_TRACK

    offered_actions.arm(
        OfferedAction(
            kind=PREVIEW_TRACK,
            target_canonical_id=target_canonical_id,
            source=source,
            verified_title=verified_title,
            verified_artist=verified_artist,
        )
    )


def _run_offered_action_continuation(
    loop: object, offered_actions: object, text: str
) -> bool:
    """Resolve a pending S2.1 offer before routing or Provider planning.

    Returns True only when this function fully owns the turn (accept/decline).
    ``override`` expires the old offer and deliberately returns False so the
    existing TurnPlan/routing/provider pipeline handles the substantive request.
    """
    from music_agent.action_attempt import (
        create_direct_action_attempt,
        render_verified_action_result,
        run_action_attempt,
    )

    decision = offered_actions.resolve(text)
    if decision.outcome in {"none", "override"}:
        return False
    if decision.outcome == "declined":
        print("好的，不试听了。")
        return True

    action = decision.action
    assert action is not None
    attempt = run_action_attempt(
        create_direct_action_attempt(
            action.target_canonical_id,
            route="preview_only",
            title=action.verified_title,
            artist=action.verified_artist,
        ),
        loop.client.call,
    )
    print(render_verified_action_result(attempt))
    return True


def _run_pronoun_track_reference(
    loop: object, kind: str, offered_actions: object | None = None
) -> bool:
    """P19-T14-F-R2: deterministic pronoun binding, zero provider rounds.

    试听它/播放它 (after the T14-F spelling normalization every 它/他/她
    feedstock reaches ``kind``) bind to the ONE authoritative referent -- the
    service channel register's canonical_id -- and execute the single
    deterministic tool through the same P09 client the loop uses:

    - referent present: preview -> preview_catalog_track(id); play ->
      play_track(id) only (the T14-E contract holds by construction: the play
      form never degrades into a preview here).
    - referent absent (channel none / malformed): one fixed honest question.
    - read failed / agent offline / continuous session running: return False
      so the caller's provider loop owns the turn (its session and offline
      rules) -- the pronoun is contested or unreadable there.
    """
    from collections.abc import Mapping

    from music_agent.action_attempt import (
        ActionAttemptStatus,
        create_direct_action_attempt,
        run_action_attempt,
    )
    from music_agent.agent_contract import AgentToolOutcome
    from music_agent.intent_router import (
        PRONOUN_PLAY_ASK,
        PRONOUN_PLAY_START_REPLY,
        PRONOUN_PREVIEW_ASK,
        PRONOUN_PREVIEW_START_REPLY,
        PRONOUN_PREVIEW_UNAVAILABLE_REPLY,
        continuous_preview_session_running,
        pronoun_track_reference_target,
    )
    from music_agent.provider_agent import PLAY_PREVIEW_DOWNGRADE_FALLBACK

    log = logging.getLogger("music_agent.cli")
    try:
        result = loop.client.call("get_playback_context", {})
    except Exception:
        log.exception("pronoun track-reference context read failed")
        return False
    if result.outcome is not AgentToolOutcome.OK:
        return False
    payload = result.payload if isinstance(result.payload, Mapping) else {}
    if continuous_preview_session_running(payload):
        return False
    target_id = pronoun_track_reference_target(payload)
    if target_id is None:
        print(PRONOUN_PREVIEW_ASK if kind == "preview" else PRONOUN_PLAY_ASK)
        return True
    route = "preview_only" if kind == "preview" else "library"
    attempt = run_action_attempt(
        create_direct_action_attempt(target_id, route=route),
        loop.client.call,
    )
    if kind == "preview":
        if attempt.status is ActionAttemptStatus.COMPLETED:
            print(PRONOUN_PREVIEW_START_REPLY)
        else:
            print(PRONOUN_PREVIEW_UNAVAILABLE_REPLY)
    else:
        if attempt.status is ActionAttemptStatus.COMPLETED:
            print(PRONOUN_PLAY_START_REPLY)
        else:
            _arm_preview_offer(
                offered_actions,
                target_id,
                source="direct_track_play_unavailable",
            )
            print(PLAY_PREVIEW_DOWNGRADE_FALLBACK)
    return True


def _play_preview_guard(
    client: object,
    text: str,
    result: object,
    offered_actions: object | None = None,
) -> str | None:
    """P19-T14-E door: a play-intent turn must never leave preview audio sounding.

    Reads the run's own execution record (the same name+outcome evidence the
    T14-B reply door uses): when the user asked for formal playback and this
    run opened the preview path, the stray preview is stopped through the same
    authoritative client and -- when no formal playback happened at all -- the
    reply is replaced with the one honest sentence
    (``PLAY_PREVIEW_DOWNGRADE_FALLBACK``). Returns None (model reply unchanged)
    in every other case. The stop is best-effort; the text swap fail-closed.
    """
    from music_agent.intent_router import is_explicit_play_intent
    from music_agent.provider_agent import (
        PLAY_PREVIEW_DOWNGRADE_FALLBACK,
        action_result_play_preview_downgrade,
        action_result_preview_started,
    )

    if not is_explicit_play_intent(text):
        return None
    if not action_result_preview_started(result):
        return None
    try:
        client.call("stop_preview", {})
    except Exception:
        logging.getLogger("music_agent.cli").exception(
            "play-turn preview stop failed"
        )
    if action_result_play_preview_downgrade(text, result):
        # Preserve the exact structured target while it is still available.
        # The next turn never parses this fallback sentence or asks Provider
        # to reconstruct the canonical id.
        attempt = getattr(result, "action_attempt", None)
        target_id = getattr(attempt, "selected_canonical_id", None)
        expected_route = getattr(attempt, "expected_route", None)
        if (
            isinstance(target_id, str)
            and target_id
            and expected_route == "preview_only"
        ):
            _arm_preview_offer(
                offered_actions,
                target_id,
                source="play_preview_downgrade",
                verified_title=getattr(attempt, "selected_title", None),
                verified_artist=getattr(attempt, "selected_artist", None),
            )
        return PLAY_PREVIEW_DOWNGRADE_FALLBACK
    return None


def _print_chat_result(
    result: object,
    provider_label: str,
    *,
    verbose: bool = False,
    final_text: str | None = None,
    user_text: str | None = None,
) -> None:
    """Chat output: model text on stdout, closed by the final response boundary.

    P20-Fix08: every provider answer passes through ``present_final_text`` --
    Layer-1 replacement/prefix scrubbing (Fix04) followed by a fail-closed
    Layer-2 validator (internal ids of any length, bare internal field names,
    hard process-narration markers). A contaminated text is reduced to its
    coherent clean region or else collapses to the stable per-task fallback
    (explanation / recommendation / default, derived from ``user_text`` and
    the run's tool record); the raw provider text is never printed on any
    failure path. ``final_text`` overrides the run's text (the T14-E
    play-degradation door); the provider/tool trace stays on stderr and is
    only shown in verbose mode (default: hidden for a plain user surface)."""
    rendered = result.final_text if final_text is None else final_text
    kind = presentation_fallback_kind(user_text, result.tool_executions)
    print(present_final_text(rendered, fallback_kind=kind))
    if not verbose:
        return
    tool_line = ", ".join(
        f"{execution.name}={execution.outcome}"
        + (f"@{execution.elapsed_ms}ms" if execution.elapsed_ms is not None else "")
        for execution in result.tool_executions
    ) or "(no tools)"
    print(
        f"[provider={provider_label} rounds={result.rounds} tools={tool_line} "
        f"trimmed={result.context_trimmed} capped={result.rounds_capped} "
        f"total_ms={result.total_elapsed_ms}]",
        file=sys.stderr,
    )


def _write_trace(result: object, args: argparse.Namespace) -> None:
    """P15-S4-M1: append the run measurements as one JSONL line (``--trace PATH``).

    Diagnostic-only: nothing is printed on success, and a write failure is
    reported on stderr without failing the chat itself. A result without a
    trace (instrumentation off, or a non-loop fast path in session mode) is
    silently skipped.
    """
    trace = getattr(result, "trace", None)
    path = getattr(args, "trace", None)
    if trace is None or not path:
        return
    from music_agent.provider_instrumentation import write_trace_jsonl

    try:
        write_trace_jsonl(Path(path), trace, provider_label=args.provider)
    except OSError as error:
        print(f"music-agent: failed to write trace {path}: {error}", file=sys.stderr)


def _run_direction_shift(loop: object, line: str) -> bool:
    """P20-Fix05: deterministic direction-shift executor, zero provider rounds.

    再来一批换个方向 and the other closed shift forms (plus a mapped explicit
    direction word like 换成日系) execute through ``direction_coach`` instead of
    the provider loop: the previous batch's durable direction is read, a REAL
    different direction from the user's durable positive evidence is selected
    (or the explicit word is honored), one ``generate_inferred_recommendation``
    fires with ``genres=[new direction]`` AND the previous batch excluded, and
    the new batch is presented directly. The coach fails honest: no alternative
    direction and no unreadable state -- a fixed question or silence remits the
    line to the provider loop untouched, and nothing is ever fabricated. True =
    the line was handled here (printed deterministically); False = remit.
    """
    from collections.abc import Mapping

    from music_agent.direction_coach import run_direction_shift

    log = logging.getLogger("music_agent.cli")
    try:
        outcome = run_direction_shift(loop.client, line)
    except Exception:
        log.exception("[fix05] direction-shift executor failed")
        return False
    if outcome is None or not isinstance(outcome, Mapping):
        return False
    if outcome.get("kind") == "reply":
        print(outcome.get("text", ""))
        return True
    if outcome.get("kind") != "generated":
        return False
    print(outcome.get("note", ""))
    # P20-Fix10: the shifted batch's items render through the SAME
    # deterministic presenter as a provider-loop recommendation success
    # (evidence-carrying reasons, the Fix09 copy discipline). The note line
    # keeps its Fix05 direction framing; a payload outside the post-Fix09
    # item contract falls back to the prior name-only lines (never invented).
    from music_agent.recommendation_presenter import (
        render_recommendation_cue,
        render_recommendation_items,
    )

    lines_text = render_recommendation_items(outcome)
    if lines_text is None:
        for position, item in enumerate(outcome.get("items", []), start=1):
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            artist = item.get("artist_name")
            if isinstance(artist, str) and artist:
                print(f"{position}. {name} — {artist}")
            else:
                print(f"{position}. {name}")
        return True
    print()
    print(lines_text)
    cue = render_recommendation_cue(outcome)
    if cue is not None:
        print()
        print(cue)
    return True


def _run_explanation(loop: object, line: str) -> bool:
    """P20-Fix11: deterministic recommendation-explanation executor, zero
    provider rounds.

    The closed explanation forms (为什么推荐这些？ / 为什么这一批适合我？ / …)
    are answered from the authoritative recommendation run itself -- one
    context read, one run read, then the shared presenter renders the
    direction summary and the per-item reasons with the same evidence copy as
    the first presentation. Zero provider rounds means the whole UAT failure
    class is structurally gone: no SSL/transport traceback and no planning
    narration can ever surface for these lines. Fail-honest: no active batch
    or an unreadable run prints the fixed natural sentence; any other line
    remits to the provider loop untouched (False = remit). The rendered text
    still passes the P20-Fix08 final-response boundary (it should pass
    byte-identical; the per-task explanation fallback backs any impossible
    contamination).
    """
    from collections.abc import Mapping

    from music_agent.explanation_coach import run_recommendation_explanation
    from music_agent.final_response_boundary import present_final_text

    log = logging.getLogger("music_agent.cli")
    try:
        outcome = run_recommendation_explanation(loop.client, line)
    except Exception:
        log.exception("[fix11] recommendation-explanation executor failed")
        return False
    if outcome is None or not isinstance(outcome, Mapping):
        return False
    text = outcome.get("text")
    if not isinstance(text, str) or not text:
        return False
    print(present_final_text(text, fallback_kind="explanation"))
    return True


def _chat_command(args: argparse.Namespace) -> int:
    from music_agent.intent_router import (
        normalize_track_reference_pronouns,
        pronoun_track_reference,
    )
    from music_agent.provider_contract import ProviderError
    from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

    try:
        # P15-S1 §6: a continuous session dies with the process (the service
        # closes right after ``run``), so ``preview_batch`` would sound exactly
        # one clip and then vanish -- the one-shot tool list hides it outright
        # (the chosen one-shot interception: forbid, not a first-clip teaser).
        one_shot_schemas = tuple(
            schema for schema in PROVIDER_TOOL_SCHEMAS if schema.name != "preview_batch"
        )
        service, loop = _build_chat_session(args, schemas=one_shot_schemas)
    except (ProviderError, ValueError) as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2

    # P19-T14-F: the parse-boundary pronoun rewrite happens once here; the
    # loop, the T14-E door and the printed result all share the normalized line.
    message = normalize_track_reference_pronouns(args.message)
    # P19-T14-F-R2: the pronoun binds deterministically to the authoritative
    # referent before any provider round; only a contested/unreadable state
    # falls through to the loop.
    pronoun_ref = pronoun_track_reference(message)
    try:
        if pronoun_ref is not None and _run_pronoun_track_reference(loop, pronoun_ref):
            return 0
        # P20-Fix05: the closed direction-shift forms (and mapped explicit
        # direction words) execute deterministically BEFORE any provider round;
        # an unhandled line remits to the loop untouched.
        if _run_direction_shift(loop, message):
            return 0
        # P20-Fix11: the closed explanation forms render deterministically from
        # the authoritative recommendation run BEFORE any provider round (no
        # SSL exposure, no planning narration, no 12-tool re-query); anything
        # else remits to the loop untouched.
        if _run_explanation(loop, message):
            return 0
        _show_waiting_hint()
        try:
            try:
                result = loop.run(message)
            finally:
                _clear_waiting_hint()
            # P19-T14-E: the door runs before the client closes so the stray
            # preview stop still reaches the authority.
            override = _play_preview_guard(loop.client, message, result)
        except ProviderError as error:
            print(f"music-agent: provider error [{error.code}]: {error}", file=sys.stderr)
            return 1
    finally:
        service.close()
        loop.client.close_event_listener()

    _print_chat_result(
        result,
        args.provider,
        verbose=args.verbose,
        final_text=override,
        user_text=message,
    )
    _write_trace(result, args)
    return 0


def _chat_session_command(args: argparse.Namespace) -> int:
    """Interactive session: one provider + P09 service, many messages.

    Each non-blank line runs through the same loop as ``chat --message``. A
    provider error on one message is reported and the session continues.
    ``/exit``, ``/quit``, ``exit``, ``quit`` or ``退出`` ends the session;
    Ctrl+C and Ctrl+D exit cleanly. Exact control commands (暂停/停止试听/
    继续播放/下一首/上一首 and their English forms) take the local fast path:
    one direct tool call through loop.client, no provider round -- anything
    else is the same bounded ProviderAgentLoop.run call. P14-C06.3a/b: 停止/stop
    and 换一首 additionally route when the live active context decides them
    (preview_sounding -> stop_preview; own_queue with no channel -> next_track);
    an unreadable or ambiguous context always defers to the provider loop.
    P15-S1: the session table joins the same fast path -- while a continuous
    preview session runs, 暂停 ≙ stop_preview, 停止 covers the inter-clip gap,
    and the continue family reports progress instead of resuming Music.app;
    off a session every command keeps its pre-P15 routing verbatim.
    P15-S1 C02: 下一首/下一首试听 while the session runs route to advance_preview
    (one in-place skip, presented through the session events); off a session
    下一首 stays next_track and 下一首试听 defers to the provider loop. The
    continue family additionally demands a restore target: with no
    AudioSuspension on record, 继续播放 prints the honest refusal instead of
    fabricating a resume; with one, the plain play restore fires as before.
    P15-S2-IPC S3: the client under the loop is a RoutedAgentClient, so the
    fast path's context read (get_playback_context) is run's live truth over
    the agent socket, the routed session commands (stop_preview/advance_preview)
    execute in run, and the continue family's play lands in run too; while run
    is offline the context read fails closed into the existing
    unknown-context paths (no fabricated resume) and any provider-driven
    preview command refuses honestly instead of degrading locally.
    P15-S2-IPC S4: preview-session events stream back from run over the
    chat-owned event socket into the shared presenter (progress / completion /
    cancellation, restore hints included) -- off run, previews are refused,
    so nothing is presented that did not happen.
    P15-S4-M3-B: the five V1 status-query phrases (现在在播放什么 等) route to
    the playback_status pseudo-command -- the caller answers deterministically
    from one get_playback_context read (formal player + preview truth + live
    session + suspension entry), degrading to a local get_now_playing read
    with an explicit preview-unreadable note on agent_runtime_offline, and
    staying honest ("无法读取") on any other read failure. Zero provider
    rounds; playback_status is never forwarded as a tool call.
    P16-S3: the five V1 formal-playback phrases (播放一首正式歌曲 等) route to
    the formal_play pseudo-command -- the caller runs the deterministic chain
    (active-batch locator -> first library-routed item -> play_track ->
    get_now_playing verification -> fixed answer) through the same P09 client,
    zero provider rounds. Before any mutation the chain remits to the provider
    loop whenever it is not safely deterministic: no recommendation data, an
    unreadable context, or a sounding preview / running session that owns the
    single audio source. The post-play verification stays its own sequential
    step; formal_play is never forwarded as a tool call.
    """
    from music_agent.intent_router import (
        needs_active_context,
        normalize_track_reference_pronouns,
        pronoun_track_reference,
        route_intent,
    )
    from music_agent.provider_contract import ProviderError
    from music_agent.conversation_continuation import OfferedActionRegister

    try:
        service, loop = _build_chat_session(args)
    except (ProviderError, ValueError) as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2

    offered_actions = OfferedActionRegister()

    try:
        while True:
            try:
                line = input("你> ")
            except EOFError:
                print()
                return 0
            except KeyboardInterrupt:
                print()
                return 0
            line = line.strip()
            if not line:
                continue
            # P19-T14-F: pronoun rewrite at the parse boundary -- the routed
            # fast path, the T14-E door and the provider loop below all see
            # the same normalized line (no 它-object form is routed, so the
            # routing itself never changes).
            line = normalize_track_reference_pronouns(line)
            if line.casefold() in _CHAT_SESSION_EXIT_WORDS:
                return 0
            # P22-S2.1: the session-owned continuation slot arbitrates before
            # TurnPlan/S1.12 routing/provider.  Only a pure accept/decline
            # claims the turn; a substantive request expires the old offer and
            # falls through unchanged.
            if _run_offered_action_continuation(loop, offered_actions, line):
                continue
            # P19-T14-F-R2: the pronoun binds deterministically through the
            # authoritative channel referent BEFORE routing/loop; a contested
            # or unreadable state falls through to the existing routing and
            # the provider loop untouched.
            pronoun_ref = pronoun_track_reference(line)
            if pronoun_ref is not None and _run_pronoun_track_reference(
                loop, pronoun_ref, offered_actions
            ):
                continue
            routed_tool = route_intent(line)
            session_dict = None
            suspended_entry = None
            if needs_active_context(line):
                # P14-C06.3a/b + P15-S1: the context-sensitive forms route (or
                # un-route) with the live context; an unreadable context degrades
                # to all-unknown = no routing. The recomputation may flip a plain
                # route (暂停 session-aware -> stop_preview, C02 下一首 ->
                # advance_preview) or drop one.
                (
                    channel_state,
                    ownership,
                    sounding,
                    session_dict,
                    suspended_entry,
                ) = _fetch_routing_context(loop, line)
                routed_tool = route_intent(
                    line,
                    channel=channel_state,
                    context=ownership,
                    preview_sounding=sounding,
                    preview_session_state=(
                        session_dict.get("state") if session_dict is not None else None
                    ),
                )
            if routed_tool is not None:
                if routed_tool == "playback_status":
                    # P15-S4-M3-B: deterministic status answer -- one
                    # authoritative read + fixed templates, zero provider
                    # rounds. Never forwarded as a P09 tool call.
                    _print_playback_status(loop)
                    continue
                if routed_tool == "play" and (
                    session_dict is not None
                    and session_dict.get("state") == "running"
                ):
                    # P15-S1 §8: while the session runs, 继续/play never resumes
                    # Music.app (single audio source) -- report progress instead.
                    _print_session_progress(session_dict)
                    continue
                if routed_tool == "play" and suspended_entry is None:
                    # P15-S1 C02: no AudioSuspension = no honest restore target.
                    # 继续播放 after a preview that interrupted nothing must not
                    # fabricate a resume -- say so explicitly and stay alive.
                    # (A degraded context read also lands here: fail closed.)
                    print("没有可恢复的播放。")
                    continue
                if routed_tool == "formal_play":
                    # P16-S3: deterministic formal-playback chain (locate ->
                    # select -> play_track -> verification -> answer) through
                    # the same P09 client, zero provider rounds. False means
                    # nothing was executed and no deterministic chain exists
                    # (e.g. no recommendation data, or preview audio owning
                    # the speaker): fall through so the provider loop decides
                    # with the original line -- the fallback cannot
                    # double-play.
                    if _run_formal_play(loop):
                        continue
                else:
                    _run_routed_command(loop, routed_tool)
                    continue
            # P20-Fix05: direction-shift lines are not routing-table forms; the
            # deterministic coach claims its closed set (and mapped explicit
            # direction words) right before the provider round, zero rounds on
            # success, fail-honest remit otherwise.
            if _run_direction_shift(loop, line):
                continue
            # P20-Fix11: the closed explanation forms render deterministically
            # from the authoritative recommendation run right before the
            # provider round (zero rounds, no SSL exposure, no planning
            # narration); anything else remits to the loop untouched.
            if _run_explanation(loop, line):
                continue
            _show_waiting_hint()
            try:
                try:
                    result = loop.run(line)
                finally:
                    _clear_waiting_hint()
            except ProviderError as error:
                print(f"music-agent: provider error [{error.code}]: {error}", file=sys.stderr)
                continue
            except KeyboardInterrupt:
                print()
                return 0
            # P22-S2.1 follow-up: arm only the structured offer returned by
            # the code-owned named-play resolver. This happens before any
            # presentation override, and never reconstructs authority from
            # assistant prose.
            structured_offer = getattr(result, "offered_action", None)
            if structured_offer is not None:
                from music_agent.conversation_continuation import OfferedAction

                if isinstance(structured_offer, OfferedAction):
                    offered_actions.arm(structured_offer)
            # P19-T14-E: the play-turn door runs while the session client is
            # alive; a stray preview is stopped and a downgraded run surfaces
            # the honest sentence instead of the model's self-healed reply.
            override = _play_preview_guard(
                loop.client, line, result, offered_actions
            )
            _print_chat_result(
                result,
                args.provider,
                verbose=args.verbose,
                final_text=override,
                user_text=line,
            )
            _write_trace(result, args)
    finally:
        service.close()
        loop.client.close_event_listener()


def _build_chat_provider(args: argparse.Namespace):
    """Build exactly one provider per interaction (explicit selection, no routing)."""
    from music_agent.codex_provider import CodexCliConfig, CodexCliProvider
    from music_agent.deepseek_provider import (
        DEFAULT_DEEPSEEK_API_KEY_ENV,
        DEFAULT_DEEPSEEK_BASE_URL,
        DEFAULT_DEEPSEEK_MODEL,
        DeepSeekApiProvider,
    )
    from music_agent.provider_contract import ProviderConfig

    if args.provider == "codex":
        return CodexCliProvider(
            CodexCliConfig(model=args.model, timeout_seconds=args.timeout)
        )
    if args.provider == "deepseek":
        return DeepSeekApiProvider(
            ProviderConfig(
                base_url=args.base_url or DEFAULT_DEEPSEEK_BASE_URL,
                model=args.model or DEFAULT_DEEPSEEK_MODEL,
                api_key_env=args.api_key_env or DEFAULT_DEEPSEEK_API_KEY_ENV,
                timeout_seconds=args.timeout,
            )
        )
    raise ProviderError(f"unknown provider: {args.provider}")


def _storage_report_command(args: argparse.Namespace) -> int:
    from music_agent.storage_report import build_storage_report

    try:
        report = build_storage_report(args.db)
    except ValueError as error:
        print(f"music-agent: {error}", file=sys.stderr)
        return 1
    print(report.summary())
    return 0


def _daily_command(args: argparse.Namespace) -> int:
    from music_agent.daily_ops import run_daily

    try:
        result = run_daily(args.db)
    except Exception as error:
        print(f"music-agent: daily run failed: {error}", file=sys.stderr)
        return 1
    sync = result.library_sync_counts
    refresh = result.refresh_counts
    print(
        f"daily complete: sync new={sync['new']} updated={sync['updated']} "
        f"unchanged={sync['unchanged']} enumeration_failed={result.library_sync_counts.get('enumeration_failed', 'n/a')}; "
        f"refresh updated={refresh['updated']} unchanged={refresh['unchanged']} failed={refresh['failed']}"
    )
    return 0 if result.succeeded else 1


def _backup_command(args: argparse.Namespace) -> int:
    from music_agent.daily_ops import backup_store

    try:
        backup_path = backup_store(args.db, backup_dir=args.backup_dir)
    except (FileNotFoundError, OSError) as error:
        print(f"music-agent: backup failed: {error}", file=sys.stderr)
        return 1
    print(f"backup written and verified: {backup_path}")
    return 0


def _restore_command(args: argparse.Namespace) -> int:
    from music_agent.daily_ops import restore_store

    try:
        restore_store(args.db, args.backup_from, force=args.force)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"music-agent: restore failed: {error}", file=sys.stderr)
        return 1
    print(f"restored and verified: {args.db}")
    return 0


_PLAYBACK_TOOL_BY_COMMAND = {
    "pause": "pause",
    "play": "play",
    "next": "next_track",
    "previous": "previous_track",
    "now-playing": "get_now_playing",
}


def _playback_command(args: argparse.Namespace) -> int:
    """One deterministic playback intent through the SAME P09 boundary as chat.

    No provider round-trip, no keyword matching, no direct osascript from this layer:
    this constructs the identical SharedAgentService + AgentClient path and executes
    exactly one playback tool call (permission check, replay journal, sealed matrix,
    canonical binding resolution -- all unchanged).
    """
    import json
    import time

    from music_agent.agent_client import AgentClient
    from music_agent.agent_contract import AgentClientIdentity
    from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
    from music_agent.agent_service import SharedAgentService
    from music_agent.apple_music_catalog import AppleMusicCatalogAdapter, MusicKitTransport
    from music_agent.playback_control import MusicPlaybackAdapter, OsascriptPlaybackRunner

    try:
        clients = _parse_agent_clients(args.agent_client)
        if len(clients) != 1:
            raise ValueError("exactly one --agent-client entry is required for playback")
        (client_id, policy), = clients.items()
    except ValueError as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2

    if args.playback_command == "play-track":
        tool = "play_track"
        payload: dict = {"canonical_id": args.track_id}
    else:
        tool = _PLAYBACK_TOOL_BY_COMMAND.get(args.playback_command)
        if tool is None:
            print(f"music-agent: unknown playback command: {args.playback_command}", file=sys.stderr)
            return 2
        payload = {}

    service = None
    try:
        registry = AgentClientRegistry({client_id: AgentClientPolicy(policy)})
        playback_adapter = MusicPlaybackAdapter(
            OsascriptPlaybackRunner(timeout_seconds=args.music_command_timeout)
        )
        service = SharedAgentService(
            args.db,
            clients=registry,
            playback_adapter=playback_adapter,
            catalog_search_source=AppleMusicCatalogAdapter(MusicKitTransport()),
        )
        client = AgentClient(
            AgentClientIdentity(
                client_id=client_id, model_id="playback-cli", label="playback-cli"
            ),
            service,
        )
        started = time.monotonic()
        result = client.call(tool, payload)
        total_ms = round((time.monotonic() - started) * 1000.0, 1)
    except Exception as error:
        print(f"music-agent: playback failed: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.close()

    summary = {
        "request_id": result.request_id,
        "outcome": result.outcome.value,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "payload": dict(result.payload or {}),
        "replayed": result.replayed,
        "total_ms": total_ms,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if result.outcome.value == "ok" else 1


def _web_command(args: argparse.Namespace) -> int:
    """P17-B minimal local product shell: one process, loopback HTTP, browser UI.

    Attach-or-embed authority: probe ``<db>.agent.sock`` up front. A served
    socket means a real run is alive -- the UI and the chat loop ride the
    production routed-client IPC (preview/device-safety on run, everything
    else local). No socket means this process embeds the authority for its
    own lifetime (agent socket + audio-safety observer, no automation). The
    provider, rounds and client policy knobs are the same as ``chat``.
    """
    try:
        clients = _parse_agent_clients(args.agent_client)
        if not clients:
            # One stable canonical ``agt_`` id (the contract requires the
            # prefix + a canonical UUID suffix) so every UI-originated
            # journal row is attributable to the shell.
            clients = {_WEB_SHELL_CLIENT_ID: "full"}
        if len(clients) != 1:
            raise ValueError("exactly one --agent-client entry is required")
        ((client_id, policy),) = clients.items()
        if args.port < 0 or args.port > 65535:
            raise ValueError("--port must be between 0 and 65535")
    except ValueError as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2

    from music_agent.web_shell import ShellConfig, WebShellApp

    config = ShellConfig(
        database_path=args.db,
        provider_factory=lambda: _build_chat_provider(args),
        agent_client=(client_id, policy),
        max_rounds=args.max_rounds,
        trace_path=getattr(args, "trace", None),
        mode=args.mode,
        port=args.port,
        open_browser=not args.no_browser,
    )
    try:
        app = WebShellApp(config)
        app.start()
    except Exception as error:  # startup truth for the launcher window, never swallowed
        print(f"music-agent: 无法启动音乐助手：{error}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{app.port}"
    print(f"音乐助手已启动：{url}（Ctrl+C 退出）")
    if config.open_browser:
        import webbrowser

        webbrowser.open(url)

    serve_thread = threading.Thread(target=app.serve_forever, name="web-shell-http", daemon=True)
    stop_event = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        stop_event.set()

    previous_handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        serve_thread.start()
        # Exit on Ctrl+C/Terminal close or the browser's 退出应用 button.
        while not stop_event.is_set() and not app.shutdown_requested:
            stop_event.wait(timeout=0.5)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        app.close()
        app.stop_httpd()
        serve_thread.join(timeout=5.0)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    try:
        config = RuntimeConfig(
            database_path=args.db,
            music_command_timeout_seconds=args.music_command_timeout,
            log_level=args.log_level,
            agent_clients=_parse_agent_clients(args.agent_client),
            refresh_interval_seconds=args.refresh_interval,
            capability_status_interval_seconds=args.capability_status_interval,
            audio_safety_enabled=not args.no_audio_safety,
            audio_safety_poll_interval_seconds=args.audio_safety_poll_interval,
        )
    except (RuntimeStartupError, ValueError) as error:
        print(f"music-agent: invalid configuration: {error}", file=sys.stderr)
        return 2
    _configure_logging(config.log_level)
    from music_agent.runtime import Runtime

    runtime = Runtime(config)
    stop_event = threading.Event()

    def request_stop(signum: int, frame: object) -> None:
        logging.getLogger("music_agent.cli").info("signal %s received; stopping", signum)
        stop_event.set()

    previous_handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        runtime.start()
        runtime.run(stop_event)
    except RuntimeStartupError as error:
        print(f"music-agent: {error}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        runtime.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "refresh":
        return _refresh_command(args)
    if args.command == "library-sync":
        return _library_sync_command(args)
    if args.command == "storage-report":
        return _storage_report_command(args)
    if args.command == "daily":
        return _daily_command(args)
    if args.command == "backup":
        return _backup_command(args)
    if args.command == "restore":
        return _restore_command(args)
    if args.command == "playback":
        return _playback_command(args)
    if args.command == "status":
        return _status_command(args)
    if args.command == "install-agent":
        return _install_agent_command(args)
    if args.command == "uninstall-agent":
        return _uninstall_agent_command(args)
    if args.command == "obsidian":
        return _obsidian_command(args)
    if args.command == "chat":
        return _chat_command(args)
    if args.command == "chat-session":
        return _chat_session_command(args)
    if args.command == "web":
        return _web_command(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
