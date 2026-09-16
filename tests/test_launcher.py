"""Guard the P17-B double-click launcher contract.

Real acceptance failure this pins: ``tools/MusicAgent.command`` invoked
``music_agent.cli web`` without ``--db`` — and the CLI's ``--db`` is
``required=True`` with no default (argparse exits 2,
"the following arguments are required: --db"). The launcher must always
supply a DB path itself, defaulting to the repo-conventional live store
(the same ``$HOME/MusicAgent/music_agent.db`` used by the tools/ repair
scripts), overridable via ``MUSIC_AGENT_DB`` or a trailing explicit ``--db``.
"""

import os
import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_LAUNCHER = _ROOT / "tools" / "MusicAgent.command"

_WEB_INVOCATION = re.compile(r"-m music_agent\.cli web\b")


class MusicAgentCommandLauncherTest(unittest.TestCase):
    def test_launcher_is_committed_executable(self):
        self.assertTrue(_LAUNCHER.is_file(), f"missing {_LAUNCHER}")
        self.assertTrue(
            os.access(_LAUNCHER, os.X_OK),
            "double-click needs the executable bit on tools/MusicAgent.command",
        )

    def test_every_web_invocation_carries_db_argument(self):
        text = _LAUNCHER.read_text()
        invocations = list(_WEB_INVOCATION.finditer(text))
        self.assertGreaterEqual(
            len(invocations), 1, "launcher must invoke `music_agent.cli web`"
        )
        for match in invocations:
            line_end = text.find("\n", match.end())
            rest = text[match.end():] if line_end == -1 else text[match.end():line_end]
            self.assertIn(
                "--db",
                rest,
                "every `web` invocation must pass --db, whatever else it forwards",
            )

    def test_default_db_follows_repo_live_store_convention(self):
        text = _LAUNCHER.read_text()
        self.assertIn(
            "$HOME/MusicAgent/music_agent.db",
            text,
            "default must match the tools/ repair scripts' live store path",
        )
        self.assertIn("MUSIC_AGENT_DB", text, "launcher must allow a DB override")


if __name__ == "__main__":
    unittest.main()