"""P10.1: Runtime composition root and lifecycle for the integrated Music Agent.

One :class:`Runtime` owns the process-level lifecycle of the integrated system:
configuration, the runtime clock, the durable store (opened and migrated through the
production :class:`CanonicalRepository`), and -- from later P10 slices -- the refresh
orchestrator, the hosted P09 :class:`SharedAgentService`, the automation engine, and the
audio-safety monitor. Components are explicit attributes, never hidden globals; each is
wired by its own slice.

Lifecycle contract: ``Runtime(config)`` → ``start()`` (open store, verify schema, start
components) → ``run(stop_event)`` (component tick loop until the event is set) → ``close()``
(closes components in reverse order, then the store). ``start`` and ``close`` are idempotent;
``run`` refuses to run on a closed runtime.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import os
import time

from music_agent.repository import CURRENT_SCHEMA_VERSION, CanonicalRepository
from music_agent.runtime_state import RuntimePidFile

logger = logging.getLogger("music_agent.runtime")

_SUPPORTED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})

# Runtime-dispatch diagnostics (trace-only): a component tick running longer
# than this breaks the 0.25s main-loop cadence for every component after it.
# With cooperative chunking one automation tick is ONE bounded step, so a
# legitimate tick spans up to the osascript subprocess cap plus model work;
# only a tick far beyond that budget counts as genuinely starved downstream
# components.
_SLOW_COMPONENT_TICK_SECONDS = 30.0


class Clock(Protocol):
    """The runtime clock: a zero-argument callable returning a timezone-aware instant."""

    def __call__(self) -> datetime: ...


def utc_now() -> datetime:
    """Default runtime clock (timezone-aware UTC)."""
    return datetime.now(timezone.utc)


class RuntimeStateError(RuntimeError):
    """Base class for runtime-level failures (store, schema, lifecycle)."""

    code = "runtime_state_error"


class RuntimeStartupError(RuntimeStateError):
    code = "runtime_startup_error"


class RuntimeLifecycleError(RuntimeStateError):
    code = "runtime_lifecycle_error"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Process-level runtime configuration.

    ``database_path`` is always explicit -- the runtime never invents a store location.
    ``music_command_timeout_seconds`` bounds every osascript Music command at the process
    level (it seeds the production runners' timeouts). ``log_level`` is one of the standard
    stdlib level names; format and destination are owned by the CLI entry point.
    ``agent_clients`` maps registered ``agt_`` client ids to their permission policy
    (``full`` / ``read_only`` / ``none``); an empty mapping hosts the shared agent service
    with every client unknown (fail closed).
    """

    database_path: Path
    music_command_timeout_seconds: float = 10.0
    log_level: str = "INFO"
    agent_clients: Mapping[str, str] = MappingProxyType({})
    refresh_interval_seconds: int = 900
    capability_status_interval_seconds: int = 3600
    audio_safety_enabled: bool = True
    audio_safety_poll_interval_seconds: float = 2.0
    library_discovery_interval_seconds: int = 21600
    agent_socket_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.database_path, Path):
            raise RuntimeStartupError("database_path must be a Path")
        if self.database_path == Path(""):
            raise RuntimeStartupError("database_path must not be empty")
        if not isinstance(self.music_command_timeout_seconds, (int, float)):
            raise RuntimeStartupError("music_command_timeout_seconds must be a number")
        if self.music_command_timeout_seconds <= 0:
            raise RuntimeStartupError("music_command_timeout_seconds must be positive")
        if not isinstance(self.log_level, str):
            raise RuntimeStartupError("log_level must be a string")
        if self.log_level not in _SUPPORTED_LOG_LEVELS:
            raise RuntimeStartupError(f"unsupported log_level: {self.log_level!r}")
        if not isinstance(self.agent_clients, Mapping):
            raise RuntimeStartupError("agent_clients must be a mapping")
        copied: dict[str, str] = {}
        for client_id, policy in self.agent_clients.items():
            if not isinstance(client_id, str) or client_id == "":
                raise RuntimeStartupError("agent client ids must be non-empty strings")
            if not isinstance(policy, str) or policy not in ("full", "read_only", "none"):
                raise RuntimeStartupError(
                    f"agent client policy for {client_id!r} must be full/read_only/none"
                )
            copied[client_id] = policy
        object.__setattr__(self, "agent_clients", MappingProxyType(copied))
        for label, value in (
            ("refresh_interval_seconds", self.refresh_interval_seconds),
            ("capability_status_interval_seconds", self.capability_status_interval_seconds),
            ("library_discovery_interval_seconds", self.library_discovery_interval_seconds),
        ):
            if not isinstance(value, int) or value <= 0:
                raise RuntimeStartupError(f"{label} must be a positive integer")
        if not isinstance(self.audio_safety_enabled, bool):
            raise RuntimeStartupError("audio_safety_enabled must be a boolean")
        if (
            not isinstance(self.audio_safety_poll_interval_seconds, (int, float))
            or self.audio_safety_poll_interval_seconds <= 0
        ):
            raise RuntimeStartupError("audio_safety_poll_interval_seconds must be positive")
        if not isinstance(self.agent_socket_enabled, bool):
            raise RuntimeStartupError("agent_socket_enabled must be a boolean")


class Runtime:
    """Composition root and lifecycle owner of one integrated Music Agent process."""

    def __init__(self, config: RuntimeConfig, *, clock: Clock = utc_now) -> None:
        if not isinstance(config, RuntimeConfig):
            raise RuntimeStartupError("config must be a RuntimeConfig")
        if not callable(clock):
            raise RuntimeStartupError("clock must be callable")
        self.config = config
        self.clock = clock
        self._canonical: CanonicalRepository | None = None
        self._agent_service = None
        self._started = False
        self._closed = False
        self._started_at: str | None = None
        self._pid_file = RuntimePidFile(config.database_path)
        # Later P10 slices wire these explicitly (refresh orchestrator, automation
        # engine, audio-safety monitor) and extend the tick loop.
        self._components: list[tuple[str, object]] = []

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def schema_version(self) -> int:
        self._require_started()
        assert self._canonical is not None
        return self._canonical.schema_version

    def _register_component(self, name: str, component: object) -> None:
        """Register a runtime component with start/tick/close hooks (wired by later slices)."""
        for hook in ("start", "tick", "close"):
            if not hasattr(component, hook):
                raise RuntimeLifecycleError(f"component {name!r} must define {hook}()")
        self._components.append((name, component))

    def start(self) -> None:
        """Open the durable store, verify schema, and start components. Idempotent."""
        if self._started:
            return
        if self._closed:
            raise RuntimeLifecycleError("cannot start a closed runtime")
        try:
            self._canonical = CanonicalRepository(self.config.database_path)
        except Exception as error:
            raise RuntimeStartupError(f"could not open durable store: {error}") from error
        if self._canonical.schema_version != CURRENT_SCHEMA_VERSION:
            store_version = self._canonical.schema_version
            self._canonical.close()
            self._canonical = None
            raise RuntimeStartupError(
                f"store schema v{store_version} does not match runtime schema "
                f"v{CURRENT_SCHEMA_VERSION}; refusing to run"
            )
        self._started = True
        try:
            playback_adapter = self._build_playback_adapter()
            self._agent_service = self._build_agent_service(playback_adapter)
            # The agent socket dispatcher registers BEFORE the automation
            # engine: the tick loop visits components in registration order,
            # and a long synchronous automation task must never delay queued
            # routed requests beyond its own tick (P15-S2 dispatch fix).
            agent_socket_server = self._build_agent_socket_server()
            if agent_socket_server is not None:
                self._register_component("agent_socket", agent_socket_server)
            self._register_component("automation", self._build_automation_engine())
            audio_observer = self._build_audio_output_observer()
            if audio_observer is not None:
                self._register_component("audio_safety", audio_observer)
            for name, component in self._components:
                component.start()
            self._started_at = self.clock().isoformat()
            try:
                self._pid_file.write(os.getpid(), self._started_at)
            except Exception:
                logger.exception("could not write runtime process-state file")
        except Exception:
            for name, component in reversed(self._components):
                try:
                    component.close()
                except Exception:
                    logger.exception("runtime component %s failed to close during startup cleanup", name)
            self._components.clear()
            if self._agent_service is not None:
                self._agent_service.close()
                self._agent_service = None
            self._started = False
            self._canonical.close()
            self._canonical = None
            raise
        logger.info(
            "runtime started: store=%s schema=v%d components=%s agent_clients=%d",
            self.config.database_path,
            self._canonical.schema_version,
            ", ".join(name for name, _ in self._components) or "(none)",
            len(self.config.agent_clients),
        )

    def _build_automation_engine(self) -> Any:
        """P10.4: the interval-task automation engine with the production task set."""
        from music_agent.apple_music import AppleMusicSourceAdapter, OsascriptMusicRunner
        from music_agent.runtime_automation import AutomationEngine, AutomationTask
        from music_agent.runtime_refresh import MusicRefreshOrchestrator

        engine = AutomationEngine(self.config.database_path, clock=self.clock)
        adapter = AppleMusicSourceAdapter(
            OsascriptMusicRunner(timeout_seconds=self.config.music_command_timeout_seconds)
        )

        def refresh_task() -> dict[str, object]:
            from music_agent.preference_persistence_repository import (
                PreferencePersistenceRepository,
            )

            # The daily refresh feeds P06: the same observations that refresh canonical
            # state also pass through the sealed preference ingestion (opened per cycle,
            # closed with the cycle -- the connection stays open for the whole run,
            # which now spans multiple runtime ticks when run cooperatively).
            with PreferencePersistenceRepository(self.config.database_path) as preference:
                report = yield from MusicRefreshOrchestrator(
                    self._canonical, adapter, clock=self.clock,
                    preference_repository=preference,
                ).iter_steps()
            return {
                "succeeded": report.succeeded,
                "bound_track_count": report.bound_track_count,
                "skipped_no_binding": report.skipped_no_binding,
                "counts": report.counts(),
            }

        def capability_status_task() -> dict[str, object]:
            summary = self.capability_summary()
            return {
                "operations": len(summary),
                "execution_ready": sum(1 for entry in summary if entry["execution_ready"]),
            }

        engine.register(
            AutomationTask(
                "music_refresh",
                self.config.refresh_interval_seconds,
                refresh_task,
            )
        )
        engine.register(
            AutomationTask(
                "capability_status",
                self.config.capability_status_interval_seconds,
                capability_status_task,
            )
        )

        def library_discovery_task() -> dict[str, object]:
            from music_agent.apple_music_library_discovery import (
                AppleMusicLibraryDiscoveryAdapter,
                OsascriptLibraryTrackIdsRunner,
            )
            from music_agent.library_sync import LibrarySyncOrchestrator
            from music_agent.preference_persistence_repository import (
                PreferencePersistenceRepository,
            )

            from music_agent.apple_music_genre_read import (
                AppleMusicGenreReadAdapter,
                OsascriptGenreReadRunner,
            )

            discovery = AppleMusicLibraryDiscoveryAdapter(
                OsascriptLibraryTrackIdsRunner(
                    timeout_seconds=self.config.music_command_timeout_seconds
                )
            )
            genre_adapter = AppleMusicGenreReadAdapter(
                OsascriptGenreReadRunner(
                    timeout_seconds=self.config.music_command_timeout_seconds
                )
            )
            with PreferencePersistenceRepository(self.config.database_path) as preference:
                report = yield from LibrarySyncOrchestrator(
                    self._canonical,
                    adapter,
                    discovery,
                    clock=self.clock,
                    preference_repository=preference,
                    genre_adapter=genre_adapter,
                ).iter_steps()
            return {
                "succeeded": report.succeeded,
                "enumeration_failed": report.enumeration_failed,
                "enumerated_count": report.enumerated_count,
                "counts": report.counts(),
            }

        engine.register(
            AutomationTask(
                "library_discovery",
                self.config.library_discovery_interval_seconds,
                library_discovery_task,
            )
        )
        return engine

    def _build_playback_adapter(self) -> Any:
        """P10.12: the transient playback adapter (shared by the service and the monitor)."""
        from music_agent.playback_control import MusicPlaybackAdapter, OsascriptPlaybackRunner

        return MusicPlaybackAdapter(
            OsascriptPlaybackRunner(
                timeout_seconds=self.config.music_command_timeout_seconds
            )
        )

    def _build_audio_output_observer(self) -> Any | None:
        """P15-S2 r3: the audio-output observer -- the safety authority's event pump.

        Detection only: every default-output transition is forwarded to the agent
        service (the single safety decision point). None when disabled or
        unavailable (fail closed). The P10.5 monitor's own playback/pause policy
        was removed in r3; the config/CLI surface (``audio_safety_*``) and the
        registered component name are deliberately unchanged.
        """
        from music_agent.audio_safety import (
            AudioOutputObserver,
            AudioSafetyUnavailableError,
            CoreAudioDefaultOutputReader,
            device_safety_trace,
        )

        if not self.config.audio_safety_enabled:
            device_safety_trace(
                "runtime: audio_safety disabled in config -> no observer attached"
            )
            return None
        try:
            device_reader = CoreAudioDefaultOutputReader()
        except AudioSafetyUnavailableError as error:
            logger.warning("audio safety observer unavailable (%s); continuing without it", error)
            device_safety_trace(f"runtime: observer unavailable, degraded: {error!r}")
            return None
        observer = AudioOutputObserver(
            device_reader,
            self._agent_service,
            poll_interval_seconds=self.config.audio_safety_poll_interval_seconds,
        )
        device_safety_trace(
            "runtime: AudioOutputObserver constructed and attached to the agent service"
        )
        return observer

    def _build_agent_socket_server(self) -> Any | None:
        """P15-S2-IPC S2: host the routed tool boundary on ``<store>.agent.sock``.

        Transport machinery only: the server executes each decoded request
        through the hosted agent service's existing ``execute`` boundary (the
        registry, journal, replay guard and permission classes are unchanged).
        None when disabled in the config.
        """
        from music_agent.agent_socket import AgentSocketServer, agent_socket_path

        if not self.config.agent_socket_enabled:
            return None
        return AgentSocketServer(agent_socket_path(self.config.database_path), self._agent_service)

    def _build_agent_service(self, playback_adapter: Any) -> Any:
        """P10.3: construct the hosted P09 shared agent service over the same store."""
        from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
        from music_agent.agent_service import SharedAgentService
        from music_agent.catalog_ingestion import default_catalog_search_source

        registry = AgentClientRegistry(
            {
                client_id: AgentClientPolicy(policy)
                for client_id, policy in self.config.agent_clients.items()
            }
        )
        # No write command adapter: live writes stay capability-gated by the sealed matrix
        # (nothing is execution-ready today); the service refuses them fail-closed.
        # The transient playback adapter (P10.12) is shared with the audio-safety monitor.
        # P11-T3: catalog discovery defaults to the credential-free iTunes Search provider
        # (MusicKit stays selectable via MUSIC_AGENT_CATALOG_PROVIDER=music_kit).
        return SharedAgentService(
            self.config.database_path,
            clients=registry,
            playback_adapter=playback_adapter,
            catalog_search_source=default_catalog_search_source(),
        )

    def run(self, stop_event: threading.Event) -> None:
        """Run the component tick loop until ``stop_event`` is set, then return.

        With no components wired (P10.1 baseline) the loop only waits on the event;
        the automation engine (P10.4) and audio-safety monitor (P10.5) add tick work.
        """
        if self._closed:
            raise RuntimeLifecycleError("cannot run a closed runtime")
        if not isinstance(stop_event, threading.Event):
            raise RuntimeLifecycleError("stop_event must be a threading.Event")
        if not self._started:
            self.start()
        logger.info("runtime loop started")
        while not stop_event.is_set():
            for name, component in self._components:
                started = time.monotonic()
                try:
                    component.tick()
                except Exception:
                    logger.exception(
                        "[runtime-dispatch] run: component %s tick raised", name
                    )
                    raise
                elapsed = time.monotonic() - started
                if elapsed > _SLOW_COMPONENT_TICK_SECONDS:
                    logger.warning(
                        "[runtime-dispatch] run: %s tick took %.1fs -- "
                        "main-loop cadence broken, downstream components starved",
                        name,
                        elapsed,
                    )
            stop_event.wait(timeout=0.25)
        logger.info("runtime loop stopped")

    def close(self) -> None:
        """Close components in reverse order, then the agent service and store. Idempotent."""
        if self._closed:
            return
        self._started = False
        for name, component in reversed(self._components):
            try:
                component.close()
            except Exception:
                logger.exception("runtime component %s failed to close cleanly", name)
        if self._agent_service is not None:
            self._agent_service.close()
            self._agent_service = None
        if self._canonical is not None:
            self._canonical.close()
            self._canonical = None
        try:
            self._pid_file.clear()
        except Exception:
            logger.exception("could not clear runtime process-state file")
        self._started_at = None
        self._closed = True
        logger.info("runtime closed")

    @property
    def agent_service(self) -> Any:
        """The hosted P09 shared agent service (available once started)."""
        self._require_started()
        return self._agent_service

    @property
    def automation(self) -> Any:
        """The daily-automation engine component (available once started)."""
        return self._component("automation")

    @property
    def audio_monitor(self) -> Any:
        """The audio-output observer (detection only; the agent service is the single
        safety authority), or None when disabled/unavailable."""
        try:
            return self._component("audio_safety")
        except RuntimeLifecycleError:
            return None

    @property
    def agent_socket(self) -> Any:
        """The routed tool boundary host (P15-S2-IPC S2), or None when disabled."""
        try:
            return self._component("agent_socket")
        except RuntimeLifecycleError:
            return None

    def _component(self, name: str) -> Any:
        self._require_started()
        for component_name, component in self._components:
            if component_name == name:
                return component
        raise RuntimeLifecycleError(f"component {name!r} is not wired")

    def capability_summary(self) -> tuple[dict[str, object], ...]:
        """The sealed read-only write-capability projection (agent-facing status surface)."""
        from music_agent.agent_permission import write_capability_summary

        return write_capability_summary()

    def status_snapshot(self) -> dict[str, object]:
        """In-process observability snapshot: startup, schema, tasks, agent, audio safety.

        The running process's full truth: in-memory task reports and audio-safety events
        that the offline ``status`` CLI (which only sees durable state) cannot report.
        """
        self._require_started()
        automation = self.automation
        task_reports: dict[str, object] = {}
        for name, report in automation.last_reports().items():
            task_reports[name] = {
                "status": report.status.value,
                "error": report.error,
                "started_at": report.started_at.isoformat(),
                "finished_at": report.finished_at.isoformat(),
                "detail": dict(report.detail),
            }
        audio_monitor = self.audio_monitor
        audio_event: dict[str, object] | None = None
        if audio_monitor is not None and audio_monitor.last_event is not None:
            event = audio_monitor.last_event
            snapshot = event.snapshot
            audio_event = {
                "kind": event.kind.value,
                "device": (
                    {
                        "device_id": snapshot.device_id,
                        "transport_type": snapshot.transport_type,
                        "uid": snapshot.device_uid,
                    }
                    if snapshot is not None
                    else None
                ),
                "action": event.action is not None,
                "error": event.error,
            }
        summary = self.capability_summary()
        return {
            "started_at": self._started_at,
            "store": str(self.config.database_path),
            "schema_version": self._canonical.schema_version,
            "agent_clients_registered": len(self.config.agent_clients),
            "tasks": task_reports,
            "audio_safety": {
                "enabled": audio_monitor is not None,
                "last_event": audio_event,
            },
            "write_capabilities": {
                "operations": len(summary),
                "execution_ready": sum(1 for entry in summary if entry["execution_ready"]),
            },
        }

    def _require_started(self) -> None:
        if not self._started or self._canonical is None:
            raise RuntimeLifecycleError("runtime is not started")
