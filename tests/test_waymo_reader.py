"""Reader unit tests; every fixture here is synthetic test-only protobuf data.

No real driving records, Waymo data downloads or empirical research results
are created by these tests. Optional schema tests skip when not prepared.
"""

from pathlib import Path
import struct
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from delivery_risk.waymo_reader import (  # noqa: E402
    WaymoDataError,
    _masked_crc32c,
    iter_scenarios,
    iter_tfrecord,
    load_scenario_class,
    normalize_scenario,
    parse_scenario,
)


def _frame(payload: bytes) -> bytes:
    length = struct.pack("<Q", len(payload))
    return (
        length + struct.pack("<I", _masked_crc32c(length))
        + payload + struct.pack("<I", _masked_crc32c(payload))
    )


def _length_field(number: int, data: bytes) -> bytes:
    if len(data) >= 128:
        raise ValueError("This test fixture supports short fields only")
    return bytes([number * 8 + 2, len(data)]) + data


class TFRecordIntegrityTests(unittest.TestCase):
    def setUp(self):
        try:
            import google_crc32c  # noqa: F401
        except ImportError:
            self.skipTest("Optional google-crc32c is not installed")
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "synthetic_test_only.tfrecord"

    def test_reads_multiple_verified_records(self):
        self.path.write_bytes(_frame(b"first synthetic record") + _frame(b"second"))
        self.assertEqual(list(iter_tfrecord(self.path)), [b"first synthetic record", b"second"])

    def test_corrupt_length_crc_rejected(self):
        data = bytearray(_frame(b"test only"))
        data[8] ^= 1
        self.path.write_bytes(data)
        with self.assertRaisesRegex(WaymoDataError, "length CRC32C mismatch"):
            list(iter_tfrecord(self.path))

    def test_corrupt_payload_crc_rejected(self):
        data = bytearray(_frame(b"test only"))
        data[-5] ^= 1
        self.path.write_bytes(data)
        with self.assertRaisesRegex(WaymoDataError, "payload CRC32C mismatch"):
            list(iter_tfrecord(self.path))

    def test_truncation_rejected_at_each_frame_component(self):
        data = _frame(b"synthetic test only")
        for cut in (2, 9, 15, len(data) - 1):
            with self.subTest(cut=cut):
                self.path.write_bytes(data[:cut])
                with self.assertRaisesRegex(WaymoDataError, "Truncated"):
                    list(iter_tfrecord(self.path))

    def test_length_allocation_limit(self):
        self.path.write_bytes(_frame(b"test only"))
        with self.assertRaisesRegex(WaymoDataError, "exceeds limit"):
            list(iter_tfrecord(self.path, max_record_bytes=2))

    def test_tf_example_has_explicit_format_diagnostic(self):
        entry = _length_field(1, b"state/id") + _length_field(2, b"")
        example = _length_field(1, _length_field(1, entry))
        with self.assertRaisesRegex(WaymoDataError, "tf.Example"):
            parse_scenario(example)


class ScenarioProtobufTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.scenario_class = load_scenario_class()
        except (FileNotFoundError, ImportError) as exc:
            raise unittest.SkipTest(f"Prepare optional official protobuf schema first: {exc}")

    def synthetic_scenario(self):
        scenario = self.scenario_class(
            scenario_id="synthetic-unit-test-only-not-driving-data",
            current_time_index=1,
            sdc_track_index=0,
        )
        scenario.timestamps_seconds.extend([0.0, 0.1, 0.2, 0.3])
        for object_id, object_type in ((7, 1), (9, 2)):
            track = scenario.tracks.add(id=object_id, object_type=object_type)
            for index in range(4):
                track.states.add(
                    valid=True, center_x=object_id + index * 0.1,
                    center_y=2.0, center_z=0.0, length=2.0, width=1.0,
                    height=1.0, heading=0.0, velocity_x=1.0, velocity_y=0.0,
                )
        lane = scenario.map_features.add(id=100).lane
        lane.type = 2
        lane.speed_limit_mph = 25.0
        lane.polyline.add(x=0.0, y=0.0, z=0.0)
        lane.polyline.add(x=20.0, y=0.0, z=0.0)
        lane.entry_lanes.append(99)
        lane.exit_lanes.append(101)
        crosswalk = scenario.map_features.add(id=200).crosswalk
        for x, y in ((1.0, 0.0), (2.0, 0.0), (2.0, 4.0), (1.0, 4.0)):
            crosswalk.polygon.add(x=x, y=y, z=0.0)
        for index in range(4):
            light = scenario.dynamic_map_states.add().lane_states.add(lane=100)
            light.state = 4 if index <= 1 else 6
            light.stop_point.x = 3.0
        return scenario

    def test_synthetic_scenario_roundtrip(self):
        scenario = self.synthetic_scenario()
        normalized = parse_scenario(scenario.SerializeToString(), self.scenario_class)
        self.assertEqual(normalized["scenario_id"], "synthetic-unit-test-only-not-driving-data")
        self.assertEqual(normalized["current_time_index"], 1)
        self.assertEqual(normalized["sdc_track_index"], 0)
        self.assertEqual(normalized["tracks"][1]["id"], 9)
        self.assertEqual(normalized["tracks"][1]["type"], 2)
        self.assertEqual(len(normalized["tracks"][0]["states"]), 4)
        self.assertAlmostEqual(normalized["tracks"][0]["states"][1]["x"], 7.1)
        self.assertEqual(normalized["map_features"][0]["type"], "lane")
        self.assertEqual(normalized["map_features"][0]["polyline"][1], [20.0, 0.0, 0.0])
        self.assertEqual(len(normalized["map_features"][1]["polygon"]), 4)

    def test_only_current_traffic_lights_exposed(self):
        normalized = normalize_scenario(self.synthetic_scenario())
        self.assertEqual(normalized["traffic_lights"], [{"lane": 100, "state": 4, "stop_point": [3.0, 0.0, 0.0]}])
        self.assertNotIn("dynamic_map_states", normalized)

    def test_unknown_protobuf_fields_are_forward_compatible(self):
        scenario = self.synthetic_scenario()
        # Field 999, varint 1: test-only extension absent from this schema.
        payload = scenario.SerializeToString() + b"\xb8\x3e\x01"
        self.assertEqual(
            parse_scenario(payload, self.scenario_class),
            normalize_scenario(scenario),
        )

    def test_synthetic_tfrecord_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            path = Path(directory) / "synthetic_test_only.tfrecord"
            path.write_bytes(_frame(self.synthetic_scenario().SerializeToString()))
            records = list(iter_scenarios(path))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["scenario_id"], "synthetic-unit-test-only-not-driving-data")

    def test_invalid_indices_rejected(self):
        for field in ("current_time_index", "sdc_track_index"):
            with self.subTest(field=field):
                scenario = self.synthetic_scenario()
                setattr(scenario, field, 999)
                with self.assertRaises(WaymoDataError):
                    normalize_scenario(scenario)

    def test_track_time_mismatch_rejected(self):
        scenario = self.synthetic_scenario()
        del scenario.tracks[0].states[-1]
        with self.assertRaisesRegex(WaymoDataError, "length does not match"):
            normalize_scenario(scenario)

    def test_duplicate_track_id_rejected(self):
        scenario = self.synthetic_scenario()
        scenario.tracks[1].id = scenario.tracks[0].id
        with self.assertRaisesRegex(WaymoDataError, "duplicate track IDs"):
            normalize_scenario(scenario)

    def test_bad_timestamps_rejected(self):
        scenario = self.synthetic_scenario()
        scenario.timestamps_seconds[2] = 0.1
        with self.assertRaisesRegex(WaymoDataError, "strictly increasing"):
            normalize_scenario(scenario)

    def test_nonfinite_valid_state_rejected(self):
        scenario = self.synthetic_scenario()
        scenario.tracks[0].states[0].center_x = float("nan")
        with self.assertRaisesRegex(WaymoDataError, "nonfinite valid state"):
            normalize_scenario(scenario)


if __name__ == "__main__":
    unittest.main()
