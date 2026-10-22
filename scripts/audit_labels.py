"""Independent scalar audit of a fixed, outcome-unselected real-data prefix.

This script does not train models or change labels/protocol. Raw identifiers
and coordinates remain in process memory; the saved report is aggregate only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))
from delivery_risk.geometry import clearance
from delivery_risk.waymo_reader import iter_tfrecord, load_scenario_class, normalize_scenario

SAMPLE_SIZE = 20
THRESHOLDS = (1.0, 2.0, 3.0)
FUTURE_STEPS = 30
HISTORY_STEPS = 10
RADIUS = 50.0
MAX_NEIGHBORS = 16
TOLERANCE_M = 1e-8


def file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def valid(state):
    keys = ("x", "y", "length", "width", "heading", "vx", "vy")
    return bool(state["valid"]) and all(math.isfinite(state[key]) for key in keys) and state["length"] > 0 and state["width"] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official training Scenario TFRecord shard to audit")
    parser.add_argument("--cache", type=Path, required=True, help="Frozen per-shard CSV; the first 20 retained scenes are audited")
    parser.add_argument("--metadata", type=Path, help="Per-shard JSON metadata; defaults to the cache path with .json suffix")
    parser.add_argument("--schema-root", type=Path, help="Prepared official Python protobuf schema root")
    parser.add_argument("--output", type=Path, default=project / "artifacts" / "audit_labels.json")
    args = parser.parse_args()
    source, cache = args.source.resolve(), args.cache.resolve()
    metadata_path = args.metadata.resolve() if args.metadata else cache.with_suffix(".json")
    report_path = args.output.resolve()
    for path in (source, cache, metadata_path):
        if not path.is_file():
            parser.error(f"Required local audit input is missing: {path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with cache.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:SAMPLE_SIZE]
    if len(rows) != SAMPLE_SIZE:
        raise SystemExit("Need exactly the predetermined first 20 complete cached scenes")

    sample_ids = [row["scenario_id"] for row in rows]
    cache_lookup = {row["scenario_id"]: row for row in rows}
    source_sha = file_sha256(source)
    cache_sha = file_sha256(cache)
    failures = Counter()
    if source_sha != metadata["source_sha256"]:
        failures["source_sha256_mismatch"] += 1
    if cache_sha != metadata["csv_sha256"]:
        failures["cache_sha256_mismatch"] += 1

    class_scenario = load_scenario_class(args.schema_root)
    read_count = 0
    audited_ids = []
    audited_counts = Counter()
    scalar_pair_comparisons = 0
    max_error_current = 0.0
    max_error_future = 0.0
    max_error_neighbor_current = 0.0
    threshold_stats = {f"{d:g}m": {"label_mismatches": 0, "eligibility_mismatches": 0, "event_time_mismatches": 0, "positive_count_in_audit": 0, "eligible_count_in_audit": 0} for d in THRESHOLDS}

    for payload in iter_tfrecord(source, verify_crc=True):
        read_count += 1
        raw = class_scenario.FromString(payload)
        if raw.scenario_id not in cache_lookup:
            continue
        scene = normalize_scenario(raw)
        row = cache_lookup[scene["scenario_id"]]
        audited_ids.append(scene["scenario_id"])
        current = scene["current_time_index"]
        ego_states = scene["tracks"][scene["sdc_track_index"]]["states"]
        ego = ego_states[current]
        # Independent reimplementation of the current-time selection rule. Future
        # outcomes and labels do not enter selection or the ordering of this audit.
        candidates = []
        for index, track in enumerate(scene["tracks"]):
            if index == scene["sdc_track_index"] or track["type"] not in (1, 2, 3):
                continue
            state = track["states"][current]
            if valid(state):
                distance = math.hypot(state["x"] - ego["x"], state["y"] - ego["y"])
                if distance <= RADIUS:
                    candidates.append((distance, track["id"], index))
        chosen = [entry[2] for entry in sorted(candidates)[:MAX_NEIGHBORS]]
        if not chosen:
            failures["retained_scene_has_no_current_neighbors"] += 1
            continue
        if int(float(row["neighbor_count"])) != len(chosen):
            failures["selected_object_count_mismatch"] += 1

        retained_valid = True
        for track_index in [scene["sdc_track_index"], *chosen]:
            future = scene["tracks"][track_index]["states"][current+1:current+FUTURE_STEPS+1]
            if len(future) != FUTURE_STEPS or not all(valid(state) for state in future):
                retained_valid = False
        hist = ego_states[current-HISTORY_STEPS:current+1]
        retained_valid = retained_valid and len(hist) == HISTORY_STEPS+1 and all(valid(state) for state in hist)
        timestamps = scene["timestamps"][current-HISTORY_STEPS:current+FUTURE_STEPS+1]
        if not all(math.isclose(b-a, 0.1, abs_tol=0.002, rel_tol=1e-5) for a,b in zip(timestamps,timestamps[1:])):
            retained_valid = False
        if not retained_valid:
            failures["retained_scene_has_invalid_required_coverage"] += 1
            continue
        audited_counts["complete_coverage_verified"] += 1

        current_distances = []
        for slot, index in enumerate(chosen):
            state = scene["tracks"][index]["states"][current]
            distance = clearance(ego, state)
            scalar_pair_comparisons += 1
            current_distances.append(distance)
            error = abs(distance - float(row[f"n{slot}_clearance"]))
            max_error_neighbor_current = max(max_error_neighbor_current, error)
            if error > TOLERANCE_M:
                failures["selected_object_current_clearance_mismatch"] += 1
            if int(float(row[f"n{slot}_type"])) != scene["tracks"][index]["type"]:
                failures["selected_object_type_mismatch"] += 1
            # Verifies the current selection/order, beyond merely comparing counts.
            c, s = math.cos(ego["heading"]), math.sin(ego["heading"])
            dx, dy = state["x"] - ego["x"], state["y"] - ego["y"]
            if abs(c*dx+s*dy-float(row[f"n{slot}_x"])) > TOLERANCE_M or abs(-s*dx+c*dy-float(row[f"n{slot}_y"])) > TOLERANCE_M:
                failures["selected_object_order_or_position_mismatch"] += 1
        scalar_current = min(current_distances)
        error = abs(scalar_current - float(row["current_min_clearance"]))
        max_error_current = max(max_error_current, error)
        if error > TOLERANCE_M:
            failures["current_min_clearance_mismatch"] += 1

        scalar_future_by_step = []
        for step in range(1, FUTURE_STEPS+1):
            step_min = math.inf
            for index in chosen:
                other = scene["tracks"][index]["states"][current+step]
                step_min = min(step_min, clearance(ego_states[current+step], other))
                scalar_pair_comparisons += 1
            scalar_future_by_step.append(step_min)
        scalar_future = min(scalar_future_by_step)
        error = abs(scalar_future - float(row["future_min_clearance"]))
        max_error_future = max(max_error_future, error)
        if error > TOLERANCE_M:
            failures["future_min_clearance_mismatch"] += 1
        for threshold in THRESHOLDS:
            key = f"{threshold:g}m"
            stats = threshold_stats[key]
            eligible = scalar_current > threshold
            label = scalar_future <= threshold
            event_time = next((round((step+1)*0.1, 10) for step,distance in enumerate(scalar_future_by_step) if distance <= threshold), None)
            cache_time = None if row[f"event_time_{key}"] == "" else float(row[f"event_time_{key}"])
            stats["eligibility_mismatches"] += int(eligible != bool(int(row[f"eligible_{key}"])))
            stats["label_mismatches"] += int(label != bool(int(row[f"label_{key}"])))
            stats["event_time_mismatches"] += int((event_time is None) != (cache_time is None) or (event_time is not None and cache_time is not None and not math.isclose(event_time,cache_time,abs_tol=1e-10)))
            stats["positive_count_in_audit"] += int(label)
            stats["eligible_count_in_audit"] += int(eligible)

    if audited_ids != sample_ids:
        failures["audit_sample_missing_duplicated_or_out_of_source_order"] += 1
    if read_count != metadata["counts"]["raw"]:
        failures["source_record_count_mismatch"] += 1
    if any(stats[metric] for stats in threshold_stats.values() for metric in ("label_mismatches", "eligibility_mismatches", "event_time_mismatches")):
        failures["threshold_comparison_mismatch"] += 1

    report = {
        "status": "pass" if not failures else "fail",
        "purpose": "independent_scalar_geometry_label_audit_no_model_tuning",
        "source_file": source.name,
        "sample_selection": "first_20_complete_retained_scenes_in_cached_source_order_before_outcome_inspection",
        "audited_scene_count": len(audited_ids),
        "source_sha256": source_sha,
        "source_sha256_matches_frozen_cache": source_sha == metadata["source_sha256"],
        "cache_sha256_matches_metadata": cache_sha == metadata["csv_sha256"],
        "source_records_crc32c_verified": read_count,
        "source_crc_verification": "both_length_and_payload_CRC32C_for_every_record_in_shard",
        "scalar_rectangle_pair_comparisons": scalar_pair_comparisons,
        "complete_coverage_verified_count": audited_counts["complete_coverage_verified"],
        "object_selection_verified": "nearest_16_current_valid_type123_within_50m_sorted_center_distance_then_id",
        "clearance_absolute_tolerance_m": TOLERANCE_M,
        "max_abs_current_min_clearance_error_m": max_error_current,
        "max_abs_future_min_clearance_error_m": max_error_future,
        "max_abs_selected_object_current_clearance_error_m": max_error_neighbor_current,
        "thresholds": threshold_stats,
        "failure_counts": dict(failures),
        "protocol_changed": False,
        "model_trained_or_tuned": False,
        "public_report_contains_raw_ids_or_coordinates": False,
        "elapsed_seconds": round(time.monotonic()-started, 3),
    }
    report_path.write_text(json.dumps(report,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report,indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
