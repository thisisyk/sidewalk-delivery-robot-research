"""Run the frozen learned model on synthetic urban presets and visibility pairs.

No raw scene export, weights, retraining selection, real safety or cost claim.
"""
from datetime import datetime, timezone
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delivery_risk.demo_model import load_frozen_demo_model
from delivery_risk.urban_simulation import UrbanConfig, simulate_urban
from delivery_risk.urban_worlds import SCENARIO_CATALOG
from serve_demo import information


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "results/urban")
    p.add_argument("--gallery", type=Path, default=ROOT / "demo/scene_gallery.js")
    p.add_argument("--skip-visibility-pairs", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_frozen_demo_model(args.samples)
    started = time.monotonic()
    runs, rows, checks = {}, [], []
    for scenario in SCENARIO_CATALOG:
        for occlusion in ([True] if args.skip_visibility_pairs else [True, False]):
            cfg = UrbanConfig(scenario=scenario, domain="road", fleet=6, threshold=model.dev_policy["cutoff"],
                              operators=1, seed=42, density=1.0, occlusion=occlusion)
            result = simulate_urban(model, cfg, compare=True)
            if occlusion:
                runs[scenario] = result
            summary = result["summary"]
            row = {"scenario": scenario, "occlusion": occlusion, **summary}
            rows.append(row)
            assert result["metadata"].get("synthetic") is True
            assert len(result["frames"]) > 0
            assert summary["completed"] + summary["unfinished"] == cfg.fleet
            assert result["frames"][-1]["t"] == cfg.duration_s
            for frame in result["frames"]:
                served = {o["vehicle_id"] for o in frame["operators"] if o["vehicle_id"] is not None}
                queued = {q["id"] for q in frame["queue"]}
                assert served.isdisjoint(queued) and len(queued) == len(frame["queue"])
                for vehicle in frame["vehicles"]:
                    if vehicle["score"] is not None:
                        assert 0 <= vehicle["score"] <= 1
            checks.append({"scenario": scenario, "occlusion": occlusion, "resource_and_score_bounds": "pass"})
            print(json.dumps({"scenario": scenario, "occlusion": occlusion, "summary": summary,
                              "elapsed_s": round(time.monotonic()-started, 1)}, ensure_ascii=False), flush=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (args.output / "scenario_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    args.gallery.parent.mkdir(parents=True, exist_ok=True)
    args.gallery.write_text("window.DELIVERY_SCENE_GALLERY="+json.dumps({"info": information(model), "runs": runs}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+";\n", encoding="utf-8")
    summary = {"status": "complete_synthetic_mechanism_study", "real_model": "original_development_selected_hist_gradient_boosting",
        "model_fit_train_rows": model.fit_audit["training_rows"], "model_fit_eval_rows": 0,
        "scenarios": len(runs), "runs": len(rows), "gallery_config": asdict(UrbanConfig(fleet=6, seed=42, threshold=model.dev_policy["cutoff"])),
        "new_real_data": False, "simulated_obstacles_are_real_training_labels": False,
        "unsupported_objects_mapped_to_vehicle": False, "visibility_pair_interpretation": "conditional_on_one_seed_and_toy_world_not_real_sensor_evaluation",
        "checks": checks, "publication_approval": "pending_user_review"}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "input_samples_sha256": sha(args.samples),
        "core_model_code_unchanged": True, "no_weights_saved": True, "no_real_scene_coordinates_exported": True,
        "script_sha256": sha(Path(__file__)), "source_sha256": {name: sha(ROOT / "src/delivery_risk" / name) for name in ("urban_worlds.py", "urban_simulation.py", "demo_model.py")},
        "outputs_sha256": {name: sha(args.output / name) for name in ("scenario_metrics.csv", "summary.json")},
        "gallery_sha256": sha(args.gallery), "elapsed_s": round(time.monotonic()-started, 2), "user_publication_approval": "pending"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"complete": True, "scenarios": len(runs), "paired_runs": len(rows), "gallery_bytes": args.gallery.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
