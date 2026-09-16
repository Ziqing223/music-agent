"""P10.6: LaunchAgent generation for daily startup (macOS native mechanism).

The foreground runtime (P10.1) is the primary surface; this module only generates a
LaunchAgent plist that keeps it running at login with explicit, reversible semantics:

- ``build_launch_agent_plist`` is a pure function: plist content only, never touches
  launchctl, never installs anything. Unit tests verify the generated configuration
  without any system change.
- Installation is explicit: ``install-agent`` writes
  ``~/Library/LaunchAgents/com.musicagent.runtime.plist`` and only with ``--activate``
  runs ``launchctl bootstrap``. ``uninstall-agent`` reverses it (``bootout`` with
  ``--active``, then removes the plist). Nothing is hidden or irreversible.
- The plist bakes in the explicit store path and all runtime flags -- a LaunchAgent
  never invents configuration.
"""

from __future__ import annotations

import plistlib
from collections.abc import Mapping
from pathlib import Path

DEFAULT_LABEL = "com.musicagent.runtime"


class LaunchAgentError(ValueError):
    code = "launch_agent_error"


def build_launch_agent_plist(
    program: list[str],
    *,
    label: str = DEFAULT_LABEL,
    stdout_log: Path,
    stderr_log: Path,
    working_directory: Path | None = None,
    environment_variables: Mapping[str, str] | None = None,
) -> dict:
    """Build the LaunchAgent plist dictionary (KeepAlive, login-agent semantics).

    ``launchd`` does not inherit an interactive shell's import environment. A
    caller running a src-layout checkout may therefore supply an explicit
    ``PYTHONPATH`` (or other required variables) without relying on shell
    startup files or an editable install.
    """
    if not isinstance(program, list) or not program or not all(
        isinstance(part, str) and part for part in program
    ):
        raise LaunchAgentError("program must be a non-empty list of non-empty strings")
    if not isinstance(label, str) or label == "":
        raise LaunchAgentError("label must be a non-empty string")
    for path, name in ((stdout_log, "stdout_log"), (stderr_log, "stderr_log")):
        if not isinstance(path, Path):
            raise LaunchAgentError(f"{name} must be a Path")
    if working_directory is not None and not isinstance(working_directory, Path):
        raise LaunchAgentError("working_directory must be a Path or None")
    if environment_variables is not None:
        if not isinstance(environment_variables, Mapping) or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and value
            for key, value in environment_variables.items()
        ):
            raise LaunchAgentError(
                "environment_variables must map non-empty strings to non-empty strings"
            )
    plist: dict = {
        "Label": label,
        "ProgramArguments": list(program),
        "RunAtLoad": True,
        "KeepAlive": True,  # restart on crash: a daily runtime must survive
        "ProcessType": "Standard",
        "StandardOutPath": str(stdout_log),
        "StandardErrorPath": str(stderr_log),
    }
    if working_directory is not None:
        plist["WorkingDirectory"] = str(working_directory)
    if environment_variables is not None:
        plist["EnvironmentVariables"] = dict(environment_variables)
    return plist


def default_plist_path(home: Path | None = None) -> Path:
    """The standard per-user LaunchAgents location."""
    home = home or Path.home()
    return home / "Library" / "LaunchAgents" / f"{DEFAULT_LABEL}.plist"


def serialize_plist(plist: dict) -> bytes:
    """Serialize the plist dictionary to XML bytes (stable, testable)."""
    try:
        return plistlib.dumps(plist, sort_keys=True)
    except Exception as error:
        raise LaunchAgentError(f"could not serialize plist: {error}") from error
