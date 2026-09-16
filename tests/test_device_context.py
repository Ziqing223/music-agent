"""P15-S2 Track B: unit tests for the Device Context domain skeleton.

The domain here is deliberately policy-free: snapshots record raw observed facts,
transitions express before/after plus the observed event, and nothing interprets
a hardware fact (Bluetooth transport, a device name like "AirPods Pro") into a
personal-device claim. These tests pin exactly that surface -- construction,
transition expression, missing/unsupported properties, and the absence of any
classification behavior.
"""

from __future__ import annotations

import unittest

from music_agent.device_context import (
    AudioOutputEventType,
    AudioOutputSnapshot,
    AudioOutputTransition,
)


class AudioOutputSnapshotTest(unittest.TestCase):
    """Snapshot construction and its raw, interpretation-free surface."""

    def test_snapshot_records_the_observed_facts(self) -> None:
        snapshot = AudioOutputSnapshot(
            event_type="baseline",
            timestamp="2026-08-19T09:00:00.000Z",
            device_id="77",
            device_uid="AppleHDAEngineOutput:1B,0,1,2:0",
            device_name="MacBook Pro Speakers",
            transport_type="bltn",
            data_source_id="0",
            data_source_name="Internal Speakers",
            device_alive=True,
        )
        self.assertEqual(snapshot.event_type, AudioOutputEventType.BASELINE)
        self.assertEqual(snapshot.device_id, "77")
        self.assertEqual(snapshot.device_uid, "AppleHDAEngineOutput:1B,0,1,2:0")
        self.assertEqual(snapshot.device_name, "MacBook Pro Speakers")
        self.assertEqual(snapshot.transport_type, "bltn")
        self.assertEqual(snapshot.data_source_id, "0")
        self.assertEqual(snapshot.data_source_name, "Internal Speakers")
        self.assertIs(snapshot.device_alive, True)
        self.assertEqual(snapshot.unavailable_fields, frozenset())

    def test_missing_optional_properties_default_to_none(self) -> None:
        snapshot = AudioOutputSnapshot("baseline", "2026-08-19T09:00:00.000Z")
        self.assertIsNone(snapshot.device_id)
        self.assertIsNone(snapshot.device_uid)
        self.assertIsNone(snapshot.device_name)
        self.assertIsNone(snapshot.transport_type)
        self.assertIsNone(snapshot.data_source_id)
        self.assertIsNone(snapshot.data_source_name)
        self.assertIsNone(snapshot.device_alive)

    def test_unsupported_data_source_is_recorded_explicitly(self) -> None:
        """An unsupported property is a recorded fact, never a system failure."""
        snapshot = AudioOutputSnapshot(
            event_type="baseline",
            timestamp="2026-08-19T09:00:00.000Z",
            device_id="77",
            device_uid="virtual-output",
            data_source_id=None,
            data_source_name=None,
            unavailable_fields=frozenset({"data_source_id", "data_source_name"}),
        )
        self.assertEqual(snapshot.event_type, AudioOutputEventType.BASELINE)
        self.assertIsNone(snapshot.data_source_id)
        self.assertIsNone(snapshot.data_source_name)
        self.assertIn("data_source_id", snapshot.unavailable_fields)
        self.assertIn("data_source_name", snapshot.unavailable_fields)

    def test_unknown_event_type_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            AudioOutputSnapshot(event_type="surprise", timestamp="t")

    def test_unavailable_fields_must_name_real_optional_fields(self) -> None:
        with self.assertRaises(ValueError):
            AudioOutputSnapshot(
                event_type="baseline",
                timestamp="t",
                unavailable_fields=frozenset({"device_mood"}),
            )

    def test_bluetooth_hardware_fact_is_never_interpreted_as_private_headphones(
        self,
    ) -> None:
        """A 'bltn' transport with an AirPods-shaped name stays a raw fact.

        The skeleton exposes exactly the observed fields -- no derived role,
        no device category, no personal-device claim. Pinning the exported
        surface keeps any such interpretation from sneaking in silently.
        """
        snapshot = AudioOutputSnapshot(
            event_type="default_device_changed",
            timestamp="2026-08-19T09:00:01.000Z",
            device_id="91",
            device_uid="A0:B1:C2:D3:E4:F5",
            device_name="AirPods Pro",
            transport_type="blue",
        )
        self.assertEqual(snapshot.transport_type, "blue")  # verbatim, unmapped
        exported = snapshot.as_dict()
        self.assertEqual(
            set(exported),
            {
                "event_type",
                "timestamp",
                "device_id",
                "device_uid",
                "device_name",
                "transport_type",
                "data_source_id",
                "data_source_name",
                "device_alive",
                "unavailable_fields",
            },
        )
        # No derived interpretation surface exists to hold a claim.
        self.assertFalse(hasattr(snapshot, "is_headphones"))
        self.assertFalse(hasattr(snapshot, "is_private_device"))
        self.assertFalse(hasattr(snapshot, "device_role"))


class AudioOutputTransitionTest(unittest.TestCase):
    """Before → after transitions, same/changed-device expression included."""

    @staticmethod
    def _snapshot(*, device_id: str | None, timestamp: str) -> AudioOutputSnapshot:
        return AudioOutputSnapshot(
            event_type="baseline",
            timestamp=timestamp,
            device_id=device_id,
            device_uid=f"uid-of-{device_id}" if device_id else None,
        )

    def test_transition_holds_before_after_and_reason(self) -> None:
        before = self._snapshot(device_id="77", timestamp="t1")
        after = self._snapshot(device_id="77", timestamp="t2")
        transition = AudioOutputTransition(
            before=before, after=after, reason="kAudioHardwarePropertyDefaultOutputDevice"
        )
        self.assertIs(transition.before, before)
        self.assertIs(transition.after, after)
        self.assertEqual(
            transition.reason, "kAudioHardwarePropertyDefaultOutputDevice"
        )

    def test_same_device_is_expressed(self) -> None:
        transition = AudioOutputTransition(
            before=self._snapshot(device_id="77", timestamp="t1"),
            after=self._snapshot(device_id="77", timestamp="t2"),
            reason="default_output",
        )
        self.assertTrue(transition.same_device)
        self.assertFalse(transition.changed_device)

    def test_changed_device_is_expressed(self) -> None:
        transition = AudioOutputTransition(
            before=self._snapshot(device_id="77", timestamp="t1"),
            after=self._snapshot(device_id="91", timestamp="t2"),
            reason="default_output",
        )
        self.assertFalse(transition.same_device)
        self.assertTrue(transition.changed_device)

    def test_unknown_after_side_counts_as_changed_without_overclaiming(self) -> None:
        """known → unknown names a change (something moved); unknown → unknown
        claims neither -- the skeleton never invents identity."""
        known_to_unknown = AudioOutputTransition(
            before=self._snapshot(device_id="77", timestamp="t1"),
            after=self._snapshot(device_id=None, timestamp="t2"),
            reason="default_output",
        )
        self.assertFalse(known_to_unknown.same_device)
        self.assertTrue(known_to_unknown.changed_device)

        unknown_to_unknown = AudioOutputTransition(
            before=self._snapshot(device_id=None, timestamp="t1"),
            after=self._snapshot(device_id=None, timestamp="t2"),
            reason="default_output",
        )
        self.assertFalse(unknown_to_unknown.same_device)
        self.assertFalse(unknown_to_unknown.changed_device)

    def test_transition_reason_must_be_a_nonempty_observation(self) -> None:
        with self.assertRaises(ValueError):
            AudioOutputTransition(
                before=self._snapshot(device_id="77", timestamp="t1"),
                after=self._snapshot(device_id="77", timestamp="t2"),
                reason="   ",
            )


if __name__ == "__main__":
    unittest.main()