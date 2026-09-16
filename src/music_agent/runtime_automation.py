"""P10.4: Minimal deterministic daily-automation engine.

One :class:`AutomationEngine` owns a small set of interval tasks and runs them in a
single-threaded loop driven by the runtime's component tick. Semantics:

- **Deterministic scheduling**: a task is due when the injected clock has advanced at least
  ``interval_seconds`` past its last attempt (or on the first tick after start). A failed
  task therefore retries after a full interval, never per-tick -- the engine cannot hammer a
  temporarily unavailable source. There is no wall-clock cron and no hidden timer thread;
  tests drive the clock directly.
- **Failure isolation**: one failed task never prevents other tasks from running and never
  touches their durable state; the failure is recorded and the engine stays alive. A durable
  record write that itself fails is logged and does not kill the loop.
- **Safe repetition**: tasks must be idempotent by construction (refresh merges into the
  canonical store; the capability-status task is read-only), so restarting the engine simply
  re-runs due tasks. No duplicate semantic application exists.
- **Cooperative chunking**: a task function may return an iterator of bounded steps instead
  of a final mapping. The engine resumes the active run by exactly one ``next()`` per tick,
  returning to the runtime main loop between steps, so long tasks (full-library refresh or
  discovery) never starve the other components -- the agent-socket dispatcher keeps draining
  queued requests between steps. The next due task cannot start until the active run finishes
  (single-task serialization preserved). One journal row and one ``last_run`` update happen
  only at run completion or failure, never per step; an engine restart drops the in-memory
  cursor and re-runs from scratch (idempotent by construction).
- **Durable observability**: every run appends one immutable row to ``runtime_task_runs``
  (schema v16) -- completed runs with their detail payload, failed runs with their error.
  In-memory scheduling state (last-run instants) is intentionally not persisted.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from music_agent.runtime import Clock, utc_now
from music_agent.runtime_task_run_repository import (
    RuntimeTaskRunRepository,
    TaskRunRecord,
    TaskRunRepositoryValidationError,
    TaskRunStatus,
    generate_task_run_id,
)

logger = logging.getLogger("music_agent.automation")

# Runtime-dispatch diagnostics (trace-only): a task step or one-shot run longer
# than this is the starve-the-main-loop signature -- the engine runs synchronously
# inside one tick call, so anything above it delays the other components
# (notably the agent-socket dispatcher).
_SLOW_TASK_SECONDS = 1.0


class AutomationError(ValueError):
    code = "automation_error"


class AutomationValidationError(AutomationError):
    code = "validation_error"


class AutomationLifecycleError(AutomationError):
    code = "automation_lifecycle_error"


@dataclass(frozen=True, slots=True)
class AutomationTask:
    """One interval task: a name, a cadence, and a detail-producing callable.

    ``function`` either returns the final detail mapping directly (plain,
    one-shot) or an iterator of bounded steps whose ``StopIteration.value`` is
    the final detail mapping (cooperative, resumed one step per runtime tick).
    """

    name: str
    interval_seconds: int
    function: Callable[[], Mapping[str, object] | Iterator[Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name == "":
            raise AutomationValidationError("task name must be a non-empty string")
        if not isinstance(self.interval_seconds, int) or self.interval_seconds <= 0:
            raise AutomationValidationError("interval_seconds must be a positive integer")
        if not callable(self.function):
            raise AutomationValidationError("function must be callable")


@dataclass(frozen=True, slots=True)
class _ActiveRun:
    """A cooperative (generator) run in progress.

    The cursor is resumed by exactly one ``next()`` per tick until
    ``StopIteration``; ``started_at``/``wall_started`` anchor the single
    journal row to the launch tick, not the completion tick.
    """

    task: AutomationTask
    cursor: Iterator[Any]
    started_at: datetime
    wall_started: float


@dataclass(frozen=True, slots=True)
class TaskRunReport:
    """The in-memory outcome of one task run (mirrors the durable row)."""

    run_id: str
    task_name: str
    status: TaskRunStatus
    error: str | None
    detail: Mapping[str, object]
    started_at: datetime
    finished_at: datetime


class AutomationEngine:
    """Interval task engine with durable run journaling and per-task failure isolation."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Clock = utc_now,
    ) -> None:
        self.database_path = Path(database_path)
        self._clock = clock
        self._tasks: dict[str, AutomationTask] = {}
        self._last_run: dict[str, datetime] = {}
        self._reports: dict[str, TaskRunReport] = {}
        self._active: _ActiveRun | None = None
        self._repository: RuntimeTaskRunRepository | None = None
        self._started = False
        self._closed = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def register(self, task: AutomationTask) -> None:
        """Register a task before start; duplicate names fail closed."""
        if self._started:
            raise AutomationLifecycleError("cannot register tasks after start")
        if not isinstance(task, AutomationTask):
            raise AutomationValidationError("task must be an AutomationTask")
        if task.name in self._tasks:
            raise AutomationValidationError(f"duplicate task name: {task.name!r}")
        self._tasks[task.name] = task

    def start(self) -> None:
        """Open the durable run journal. Idempotent."""
        if self._started:
            return
        if self._closed:
            raise AutomationLifecycleError("cannot start a closed automation engine")
        self._repository = RuntimeTaskRunRepository(self.database_path)
        self._started = True
        logger.info(
            "automation engine started: tasks=%s",
            ", ".join(self._tasks) or "(none)",
        )

    def tick(self) -> None:
        """Run at most ONE automation step per tick, in registration order.

        P15-S2 dispatch fix: a single tick call must not chain every due task
        back-to-back -- that would keep the runtime main loop away from the
        other components (notably the agent-socket dispatcher) for the whole
        chain. Due-ness is preserved: the next due task runs on a later tick,
        and interval cadence is unchanged.

        A cooperative (generator) run keeps priority across ticks: each tick
        resumes exactly one ``next()`` and immediately returns, so the main
        loop services the other components between bounded steps. No other
        task starts until the active run finishes (single-task serialization).
        """
        if not self._started:
            return
        if self._active is not None:
            self._advance_active()
            return
        now = self._clock()
        for task in self._tasks.values():
            last = self._last_run.get(task.name)
            if last is None or (now - last).total_seconds() >= task.interval_seconds:
                logger.info("[runtime-dispatch] auto: task due name=%s", task.name)
                self._start_task(task, now)
                return

    def run_task_now(self, task_name: str) -> TaskRunReport:
        """Force one task to run fully to completion regardless of schedule.

        Cooperative (generator) tasks drain synchronously here -- this
        validation/CLI hook returns only when the whole run has finished,
        exactly like a plain one-shot task.
        """
        self._require_started()
        task = self._tasks.get(task_name)
        if task is None:
            raise AutomationValidationError(f"unknown task: {task_name!r}")
        wall_started = time.monotonic()
        started_at = self._clock()
        try:
            outcome = task.function()
        except Exception as task_error:  # per-task isolation: never abort the caller
            logger.exception("[runtime-dispatch] auto: task %s failed", task.name)
            return self._finish_run(
                task, started_at, wall_started, None, error=str(task_error)
            )
        if isinstance(outcome, Iterator):
            cursor = iter(outcome)
            while True:
                try:
                    next(cursor)
                except StopIteration as done:
                    outcome = done.value
                    break
                except Exception as task_error:  # one bad step fails the whole run
                    logger.exception("[runtime-dispatch] auto: task %s failed", task.name)
                    return self._finish_run(
                        task, started_at, wall_started, None, error=str(task_error)
                    )
        return self._finish_run(task, started_at, wall_started, outcome, error=None)

    def run_all_now(self) -> tuple[TaskRunReport, ...]:
        """Run every registered task once, in registration order (validation hook)."""
        return tuple(self.run_task_now(name) for name in self._tasks)

    def last_reports(self) -> Mapping[str, TaskRunReport]:
        """The most recent in-memory report per task (status surface)."""
        return MappingProxyType(dict(self._reports))

    def close(self) -> None:
        """Close the durable journal. Idempotent."""
        if self._closed:
            return
        self._started = False
        if self._repository is not None:
            self._repository.close()
            self._repository = None
        self._closed = True
        logger.info("automation engine closed")

    def _start_task(self, task: AutomationTask, now: datetime) -> None:
        """Launch one due task.

        A plain result records immediately through the shared wrap-up; a
        generator becomes the active run and takes its FIRST step before this
        returns (one bounded step per tick includes the launch tick).
        """
        wall_started = time.monotonic()
        try:
            outcome = task.function()
        except Exception as task_error:  # per-task isolation: never abort the engine
            logger.exception("[runtime-dispatch] auto: task %s failed", task.name)
            self._finish_run(task, now, wall_started, None, error=str(task_error))
            return
        if isinstance(outcome, Iterator):
            self._active = _ActiveRun(
                task=task,
                cursor=iter(outcome),
                started_at=now,
                wall_started=wall_started,
            )
            self._advance_active()
            return
        self._finish_run(task, now, wall_started, outcome, error=None)

    def _advance_active(self) -> None:
        """Resume the active run by exactly one ``next()`` and return.

        ``StopIteration.value`` becomes the final detail mapping and triggers
        the single run wrap-up; a step exception fails the whole run and
        releases the cursor (retry after a full interval, as for plain tasks).
        """
        assert self._active is not None
        step_wall_started = time.monotonic()
        try:
            next(self._active.cursor)
        except StopIteration as done:
            active = self._active
            self._active = None
            self._finish_run(
                active.task,
                active.started_at,
                active.wall_started,
                done.value,
                error=None,
            )
            return
        except Exception as task_error:  # one bad step fails the whole run
            active = self._active
            self._active = None
            logger.exception("[runtime-dispatch] auto: task %s failed", active.task.name)
            self._finish_run(
                active.task,
                active.started_at,
                active.wall_started,
                None,
                error=str(task_error),
            )
            return
        step_elapsed = time.monotonic() - step_wall_started
        if step_elapsed > _SLOW_TASK_SECONDS:
            # DEBUG (not WARNING): a step up to the osascript subprocess cap
            # plus model work is NORMAL for chunked tasks -- a step is only
            # worth flagging during diagnosis.
            logger.debug(
                "[runtime-dispatch] auto: task %s step took %.1fs -- one bounded step",
                self._active.task.name,
                step_elapsed,
            )

    def _finish_run(
        self,
        task: AutomationTask,
        started_at: datetime,
        wall_started: float,
        result: Any,
        error: str | None,
    ) -> TaskRunReport:
        """The single wrap-up for one run: validate the final detail mapping,
        record the in-memory report, update ``last_run``, and append exactly
        one durable journal row.

        Called only when the whole run is done -- at completion or on failure,
        never per step. ``wall_started`` spans the whole run (one call for a
        plain task, launch through last step for a generator), so the slow-run
        diagnostic reports the true total, not a per-tick slice.
        """
        status = TaskRunStatus.COMPLETED
        detail: dict[str, object] = {}
        if error is not None:
            status = TaskRunStatus.FAILED
        else:
            try:
                if not isinstance(result, Mapping):
                    raise AutomationValidationError(
                        f"task {task.name} returned {type(result).__name__}, expected a mapping"
                    )
                detail = dict(result)
                json.dumps(detail, sort_keys=True)  # fail before recording: detail must be durable
            except Exception as task_error:  # per-task isolation: never abort the engine
                logger.exception("[runtime-dispatch] auto: task %s failed", task.name)
                status = TaskRunStatus.FAILED
                error = str(task_error)
                detail = {}
        wall_elapsed = time.monotonic() - wall_started
        if wall_elapsed > _SLOW_TASK_SECONDS:
            # INFO (not WARNING): with cooperative chunking a long wall total is
            # the expected shape (many bounded steps), not a main-loop anomaly --
            # completion timing stays visible as lifecycle information.
            logger.info(
                "[runtime-dispatch] auto: task %s finished status=%s in %.1fs",
                task.name,
                status.value,
                wall_elapsed,
            )
        finished_at = self._clock()
        report = TaskRunReport(
            run_id=generate_task_run_id(),
            task_name=task.name,
            status=status,
            error=error,
            detail=detail,
            started_at=started_at,
            finished_at=finished_at,
        )
        self._reports[task.name] = report
        self._last_run[task.name] = finished_at
        try:
            assert self._repository is not None
            self._repository.record_run(
                TaskRunRecord(
                    run_id=report.run_id,
                    task_name=report.task_name,
                    status=report.status,
                    error=report.error,
                    detail=report.detail,
                    started_at=report.started_at.isoformat(),
                    finished_at=report.finished_at.isoformat(),
                )
            )
        except Exception:  # a journal failure must not kill the automation loop
            logger.exception("could not record automation task run %s", report.run_id)
        return report

    def _require_started(self) -> None:
        if not self._started or self._repository is None:
            raise AutomationLifecycleError("automation engine is not started")
