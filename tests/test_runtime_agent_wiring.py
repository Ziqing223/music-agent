"""P10.3: Runtime-hosted shared agent wiring tests.

Proves one running process composes the full chain: canonical store -> P06 preference ->
P07 recommendation -> P08 feedback -> P09 shared agent, with the agent reading state the
refresh cycle wrote.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import AgentClientIdentity
from music_agent.apple_music import AppleMusicSourceAdapter
from music_agent.refresh import refresh_known_track
from music_agent.repository import CanonicalRepository
from music_agent.runtime import Runtime, RuntimeConfig, RuntimeLifecycleError

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
CLIENT_ID = "agt_11111111-1111-4111-8111-111111111111"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeRunner:
    def __init__(self, output: str) -> None:
        self.output = output

    def run(self, persistent_id: str) -> str:
        return self.output


def config_for(database_path: Path, **kwargs) -> RuntimeConfig:
    kwargs.setdefault("audio_safety_enabled", False)
    return RuntimeConfig(database_path=database_path, **kwargs)


class AgentClientConfigTest(unittest.TestCase):
    def test_agent_clients_default_to_empty(self) -> None:
        config = RuntimeConfig(database_path=Path("store.db"))
        self.assertEqual(dict(config.agent_clients), {})

    def test_agent_clients_are_copied_into_mapping(self) -> None:
        clients = {CLIENT_ID: "full"}
        config = RuntimeConfig(database_path=Path("store.db"), agent_clients=clients)
        clients.clear()
        self.assertEqual(dict(config.agent_clients), {CLIENT_ID: "full"})

    def test_rejects_bad_policy_value(self) -> None:
        from music_agent.runtime import RuntimeStartupError

        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path("store.db"), agent_clients={CLIENT_ID: "everything"})

    def test_rejects_empty_client_id(self) -> None:
        from music_agent.runtime import RuntimeStartupError

        with self.assertRaises(RuntimeStartupError):
            RuntimeConfig(database_path=Path("store.db"), agent_clients={"": "full"})


class RuntimeAgentWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(load_fixture())

    def _runtime(self, **kwargs) -> Runtime:
        kwargs.setdefault("audio_safety_enabled", False)
        return Runtime(
            RuntimeConfig(database_path=self.database_path, **kwargs)
        )

    def _client(self, runtime: Runtime) -> AgentClient:
        identity = AgentClientIdentity(
            client_id=CLIENT_ID, model_id="test-model", label="p10-tests"
        )
        return AgentClient(identity, runtime.agent_service)

    def test_runtime_hosts_agent_service_with_registered_client(self) -> None:
        runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        runtime.start()
        try:
            client = self._client(runtime)
            result = client.call(
                "get_canonical_entity",
                {"canonical_id": "trk_11111111-1111-4111-8111-111111111111"},
            )
            self.assertEqual(result.outcome.value, "ok")
        finally:
            runtime.close()

    def test_runtime_wires_the_real_catalog_search_source(self) -> None:
        """P11-T3: the runtime default is the credential-free iTunes adapter; MusicKit
        stays selectable through MUSIC_AGENT_CATALOG_PROVIDER."""
        from unittest.mock import patch

        from music_agent.apple_music_catalog import AppleMusicCatalogAdapter
        from music_agent.itunes_search import iTunesSearchAdapter

        runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        with patch.dict("os.environ", {}, clear=True):
            runtime.start()
        try:
            source = runtime.agent_service._catalog_search_source
            self.assertIsInstance(source, iTunesSearchAdapter)
        finally:
            runtime.close()

        music_kit_runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        with patch.dict("os.environ", {"MUSIC_AGENT_CATALOG_PROVIDER": "music_kit"}):
            music_kit_runtime.start()
        try:
            source = music_kit_runtime.agent_service._catalog_search_source
            self.assertIsInstance(source, AppleMusicCatalogAdapter)
        finally:
            music_kit_runtime.close()

    def test_unknown_client_refuses_through_runtime_hosted_service(self) -> None:
        runtime = self._runtime(agent_clients={})
        runtime.start()
        try:
            client = self._client(runtime)
            result = client.call(
                "get_canonical_entity",
                {"canonical_id": "trk_11111111-1111-4111-8111-111111111111"},
            )
            self.assertEqual(result.outcome.value, "unknown_client")
        finally:
            runtime.close()

    def test_refresh_then_agent_read_sees_the_same_state(self) -> None:
        """The integrated chain: canonical store -> refresh -> P09 agent read."""
        runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        runtime.start()
        try:
            # A refresh cycle writes through the production save path.
            found = json.dumps(
                {"status": "found", "fields": {"name": "Agent-Visible Name", "played_count": 7}}
            )
            adapter = AppleMusicSourceAdapter(FakeRunner(found))
            with CanonicalRepository(self.database_path) as repository:
                result = refresh_known_track(
                    repository, adapter, "trk_11111111-1111-4111-8111-111111111111"
                )
            self.assertEqual(result.status.value, "updated")
            # The hosted agent service reads the same durable store.
            client = self._client(runtime)
            read = client.call(
                "get_canonical_entity",
                {"canonical_id": "trk_11111111-1111-4111-8111-111111111111"},
            )
            self.assertEqual(read.outcome.value, "ok")
            entity = read.payload["entity"]
            self.assertEqual(entity["name"], "Agent-Visible Name")
            self.assertEqual(entity["library_state"]["play_count"], 7)
        finally:
            runtime.close()

    def test_preference_query_flows_through_hosted_service(self) -> None:
        runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        runtime.start()
        try:
            client = self._client(runtime)
            result = client.call(
                "query_track_preference",
                {"target_id": "trk_11111111-1111-4111-8111-111111111111"},
            )
            self.assertEqual(result.outcome.value, "ok")
        finally:
            runtime.close()

    def test_live_write_refuses_not_execution_ready(self) -> None:
        """No runtime merely by existing activates a sealed-capability live write."""
        from music_agent.identity import EntityType, ExternalIdentityKey
        from music_agent.intent_repository import PendingIntentRepository
        from music_agent.source_observation import ObservedValue
        from music_agent.write_intent import (
            WriteOperation,
            create_scalar_pending_intent,
        )

        intent = create_scalar_pending_intent(
            WriteOperation.SET_FAVORITED,
            "trk_11111111-1111-4111-8111-111111111111",
            ExternalIdentityKey("apple_music", EntityType.TRACK, "SYNTH-TRACK-001"),
            ObservedValue.value(True),
        )
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(intent)

        runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        runtime.start()
        try:
            client = self._client(runtime)
            result = client.call("execute_write_intent", {"intent_id": intent.intent_id})
            self.assertEqual(result.outcome.value, "not_execution_ready")
        finally:
            runtime.close()

    def test_replay_safety_survives_runtime_hosting(self) -> None:
        runtime = self._runtime(agent_clients={CLIENT_ID: "full"})
        runtime.start()
        try:
            client = self._client(runtime)
            payload = {"canonical_id": "trk_11111111-1111-4111-8111-111111111111"}
            request_id = "req_22222222-2222-4222-8222-222222222222"
            first = client.call("get_canonical_entity", payload, request_id=request_id)
            second = client.call("get_canonical_entity", payload, request_id=request_id)
            self.assertEqual(first.outcome.value, "ok")
            self.assertEqual(second.outcome.value, "ok")
            self.assertTrue(second.replayed)
        finally:
            runtime.close()

    def test_agent_service_unavailable_before_start(self) -> None:
        runtime = self._runtime()
        with self.assertRaises(RuntimeLifecycleError):
            _ = runtime.agent_service

    def test_capability_summary_exposes_sealed_matrix(self) -> None:
        runtime = self._runtime()
        summary = runtime.capability_summary()
        self.assertTrue(any(e["operation"] == "set_favorited" for e in summary))
        self.assertFalse(any(e["execution_ready"] for e in summary))
        self.assertEqual(
            {e["execution_ready"] for e in summary},
            {False},
        )


class AgentClientCliFlagTest(unittest.TestCase):
    def test_run_accepts_agent_client_entries(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args(
            ["run", "--db", "store.db", "--agent-client", f"{CLIENT_ID}:full"]
        )
        self.assertEqual(args.agent_client, [f"{CLIENT_ID}:full"])

    def test_malformed_agent_client_entry_rejected(self) -> None:
        from music_agent.cli import _parse_agent_clients

        with self.assertRaises(ValueError):
            _parse_agent_clients([CLIENT_ID])  # missing :POLICY
        with self.assertRaises(ValueError):
            _parse_agent_clients([f"{CLIENT_ID}:full", f"{CLIENT_ID}:read_only"])

    def test_parse_agent_clients_mapping(self) -> None:
        from music_agent.cli import _parse_agent_clients

        self.assertEqual(
            _parse_agent_clients([f"{CLIENT_ID}:read_only", "agt_99999999-9999-4999-8999-999999999999:none"]),
            {
                CLIENT_ID: "read_only",
                "agt_99999999-9999-4999-8999-999999999999": "none",
            },
        )


if __name__ == "__main__":
    unittest.main()
