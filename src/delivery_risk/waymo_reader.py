"""Read official WOMD Scenario TFRecords without a TensorFlow dependency.

This module never downloads data or accepts terms. The caller supplies files
obtained through the official Waymo access process. Generated protobuf modules
are produced separately by ``scripts/fetch_waymo_schema.py``.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterator


class WaymoDataError(ValueError):
    """Malformed, unsupported or incorrectly selected Waymo input."""


def _masked_crc32c(data: bytes) -> int:
    try:
        import google_crc32c
    except ImportError as exc:
        raise ImportError("TFRecord verification requires google-crc32c.") from exc
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord(
    path: str | Path,
    *,
    verify_crc: bool = True,
    max_record_bytes: int = 256 * 1024 * 1024,
) -> Iterator[bytes]:
    """Yield records from one uncompressed TFRecord and verify both CRCs.

    CRC checks default on. Truncation is always rejected. A size limit prevents
    a damaged length header from causing an unbounded allocation.
    """
    source = Path(path)
    if max_record_bytes < 1:
        raise ValueError("max_record_bytes must be positive")
    with source.open("rb") as handle:
        record_index = 0
        while True:
            offset = handle.tell()
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            location = f"{source.name}: record {record_index}, byte {offset}"
            if len(length_bytes) != 8:
                raise WaymoDataError(f"Truncated TFRecord length at {location}")
            length_crc = handle.read(4)
            if len(length_crc) != 4:
                raise WaymoDataError(f"Truncated TFRecord length checksum at {location}")
            if verify_crc and struct.unpack("<I", length_crc)[0] != _masked_crc32c(length_bytes):
                raise WaymoDataError(f"TFRecord length CRC32C mismatch at {location}")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > max_record_bytes:
                raise WaymoDataError(
                    f"Record length {length:,} exceeds limit at {location}. "
                    "Use uncompressed Scenario TFRecords, not gzip archives."
                )
            payload = handle.read(length)
            payload_crc = handle.read(4)
            if len(payload) != length or len(payload_crc) != 4:
                raise WaymoDataError(f"Truncated TFRecord payload at {location}")
            if verify_crc and struct.unpack("<I", payload_crc)[0] != _masked_crc32c(payload):
                raise WaymoDataError(f"TFRecord payload CRC32C mismatch at {location}")
            yield payload
            record_index += 1


def load_scenario_class(schema_root: str | Path | None = None) -> Any:
    """Load the official generated Scenario class from an explicit schema root."""
    root = (
        Path(schema_root).resolve()
        if schema_root is not None
        else Path(__file__).resolve().parents[2] / "vendor" / "waymo_schema"
    )
    expected = root / "waymo_open_dataset" / "protos" / "scenario_pb2.py"
    if not expected.is_file():
        raise FileNotFoundError(
            f"Generated Scenario schema not found: {expected}. "
            "Run python scripts/fetch_waymo_schema.py --download-schema first, "
            "or provide schema_root pointing at generated official modules."
        )
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    module = importlib.import_module("waymo_open_dataset.protos.scenario_pb2")
    actual = Path(module.__file__).resolve()
    if actual != expected.resolve():
        raise ImportError(
            f"An incompatible Waymo schema is already imported from {actual}; "
            "start a fresh Python process with the requested schema root."
        )
    return module.Scenario


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 128:
            return value, offset
    raise ValueError("invalid protobuf varint")


def _wire_fields(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    """Minimal wire reader used only to identify a misselected tf.Example."""
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if field == 0:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            _, stop = _varint(data, offset)
        elif wire == 1:
            stop = offset + 8
        elif wire == 2:
            length, offset = _varint(data, offset)
            stop = offset + length
        elif wire == 5:
            stop = offset + 4
        else:
            raise ValueError("unsupported protobuf wire type")
        if stop > len(data):
            raise ValueError("truncated protobuf field")
        yield field, wire, data[offset:stop]
        offset = stop


def _looks_like_tf_example(payload: bytes) -> bool:
    try:
        for number, wire, features in _wire_fields(payload):
            if (number, wire) != (1, 2):
                continue
            for entry_number, entry_wire, entry in _wire_fields(features):
                if (entry_number, entry_wire) != (1, 2):
                    continue
                for key_number, key_wire, key in _wire_fields(entry):
                    if (key_number, key_wire) == (1, 2) and key.startswith(
                        (b"state/", b"roadgraph_samples/", b"scenario/", b"traffic_light_state/")
                    ):
                        return True
    except ValueError:
        pass
    return False


def _point(point: Any) -> list[float]:
    return [float(point.x), float(point.y), float(point.z)]


def normalize_scenario(scenario: Any) -> dict[str, Any]:
    """Normalize trajectory and map fields; expose traffic signals at t0 only.

    Future track states are retained for label construction. Callers must slice
    feature inputs at or before current_time_index to avoid target leakage.
    Optional lidar/camera fields are intentionally ignored.
    """
    timestamps = [float(value) for value in scenario.timestamps_seconds]
    if len(timestamps) < 2 or not all(math.isfinite(value) for value in timestamps):
        raise WaymoDataError("Scenario needs at least two finite timestamps")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise WaymoDataError("Scenario timestamps must be strictly increasing")
    if not scenario.scenario_id:
        raise WaymoDataError("Scenario has no scenario_id")
    if not scenario.HasField("current_time_index"):
        raise WaymoDataError("Scenario has no current_time_index")
    current = int(scenario.current_time_index)
    if not 0 <= current < len(timestamps):
        raise WaymoDataError("current_time_index is outside the timestamps")
    if not scenario.HasField("sdc_track_index") or not 0 <= scenario.sdc_track_index < len(scenario.tracks):
        raise WaymoDataError("Scenario has no valid SDC track index")
    tracks: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for track in scenario.tracks:
        if int(track.id) in seen_ids:
            raise WaymoDataError("Scenario has duplicate track IDs")
        seen_ids.add(int(track.id))
        if len(track.states) != len(timestamps):
            raise WaymoDataError(f"Track {track.id} length does not match timestamps")
        states = []
        for state in track.states:
            normalized = {
                "valid": bool(state.valid),
                "x": float(state.center_x), "y": float(state.center_y),
                "z": float(state.center_z), "length": float(state.length),
                "width": float(state.width), "height": float(state.height),
                "heading": float(state.heading),
                "vx": float(state.velocity_x), "vy": float(state.velocity_y),
            }
            if normalized["valid"] and not all(
                math.isfinite(value) for key, value in normalized.items() if key != "valid"
            ):
                raise WaymoDataError(f"Track {track.id} has a nonfinite valid state")
            states.append(normalized)
        tracks.append({"id": int(track.id), "type": int(track.object_type), "states": states})
    features = []
    for feature in scenario.map_features:
        kind = feature.WhichOneof("feature_data")
        if kind is None:
            continue
        content = getattr(feature, kind)
        normalized_feature: dict[str, Any] = {"id": int(feature.id), "type": kind}
        if hasattr(content, "polyline"):
            normalized_feature["polyline"] = [_point(point) for point in content.polyline]
        if hasattr(content, "polygon"):
            normalized_feature["polygon"] = [_point(point) for point in content.polygon]
        if hasattr(content, "type"):
            normalized_feature["subtype"] = int(content.type)
        if kind == "lane":
            normalized_feature.update(
                speed_limit_mph=float(content.speed_limit_mph),
                entry_lanes=list(content.entry_lanes), exit_lanes=list(content.exit_lanes),
            )
        if kind == "stop_sign":
            normalized_feature.update(position=_point(content.position), lanes=list(content.lane))
        features.append(normalized_feature)
    lights = []
    if scenario.dynamic_map_states:
        if len(scenario.dynamic_map_states) != len(timestamps):
            raise WaymoDataError("Dynamic map states do not match timestamps")
        for light in scenario.dynamic_map_states[current].lane_states:
            lights.append({"lane": int(light.lane), "state": int(light.state), "stop_point": _point(light.stop_point)})
    return {
        "scenario_id": str(scenario.scenario_id),
        "current_time_index": current,
        "timestamps": timestamps,
        "sdc_track_index": int(scenario.sdc_track_index),
        "tracks": tracks,
        "map_features": features,
        "traffic_lights": lights,
    }


def parse_scenario(payload: bytes, scenario_class: Any = None, *, schema_root: str | Path | None = None) -> dict[str, Any]:
    """Parse a Scenario record with an explicit tf.Example format diagnostic."""
    if _looks_like_tf_example(payload):
        raise WaymoDataError(
            "Input is a WOMD tf.Example record. This reader supports only official "
            "Scenario-proto TFRecords (v1.2+); select the scenario/ directory."
        )
    if scenario_class is None:
        scenario_class = load_scenario_class(schema_root)
    from google.protobuf.message import DecodeError
    try:
        scenario = scenario_class.FromString(payload)
    except DecodeError as exc:
        raise WaymoDataError("Cannot decode Scenario protobuf; choose Scenario-format TFRecords") from exc
    return normalize_scenario(scenario)


def iter_scenarios(path: str | Path, *, schema_root: str | Path | None = None, verify_crc: bool = True) -> Iterator[dict[str, Any]]:
    """Iterate one official Scenario TFRecord; provenance remains caller-owned."""
    scenario_class = load_scenario_class(schema_root)
    for payload in iter_tfrecord(path, verify_crc=verify_crc):
        yield parse_scenario(payload, scenario_class)
