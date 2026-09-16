"""M1 live-process check: the real ``python -m music_agent.mcp_server`` command
serves the documented stdio flow with the production adapter chain (READ tools
never touch the osascript runners). Run as an ordinary test over a temp store.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MESSAGES = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                "clientInfo": {"name": "live-test", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
     "params": {"name": "get_agent_capabilities", "arguments": {}}},
    {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
     "params": {"name": "play", "arguments": {}}},
    {"jsonrpc": "2.0", "method": "exit"},
]


class LiveMcpProcessTest(unittest.TestCase):
    """The documented startup surface: PYTHONPATH=src python -m music_agent.mcp_server --db <db>.
    """

    def test_real_process_serves_the_documented_stdio_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "live.sqlite3"
            stdin = ("".join(json.dumps(m) + "\n" for m in MESSAGES)).encode("utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "music_agent.mcp_server",
                    "--db",
                    str(database_path),
                ],
                input=stdin,
                capture_output=True,
                timeout=120,
                cwd=Path(__file__).resolve().parent.parent,
                env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode()[-500:])
            replies = [
                json.loads(line)
                for line in completed.stdout.decode("utf-8").splitlines()
                if line.strip()
            ]
        self.assertEqual(len(replies), 4)  # initialize, tools/list, 2 calls; notification silent
        by_id = {reply["id"]: reply for reply in replies}
        self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "music-agent-core")
        tools = by_id[2]["result"]["tools"]
        self.assertEqual(len(tools), 29)
        capabilities = json.loads(by_id[3]["result"]["content"][0]["text"])
        self.assertEqual(capabilities["outcome"], "ok")
        self.assertIs(by_id[3]["result"]["isError"], False)
        offline_play = json.loads(by_id[4]["result"]["content"][0]["text"])
        self.assertEqual(offline_play["outcome"], "execution_error")
        self.assertEqual(offline_play["error_code"], "agent_runtime_offline")
        self.assertIs(by_id[4]["result"]["isError"], True)


if __name__ == "__main__":
    unittest.main()