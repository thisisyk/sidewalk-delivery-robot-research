"""Run the frozen learned model on synthetic sidewalk-robot presets and visibility pairs.

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
from delivery_risk.sidewalk_simulation import SidewalkConfig, simulate_sidewalk
from delivery_risk.sidewalk_worlds import SIDEWALK_CATALOG
from serve_demo import information


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", required=True, type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "results/sidewalk")
    p.add_argument("--gallery", type=Path, default=ROOT / "demo/sidewalk_gallery.js")
    p.add_argument("--skip-visibility-pairs", action="store_true")
    p.add_argument("--legacy-baseline", type=Path, default=ROOT / "results/sidewalk_baseline/scenario_metrics.csv")
    p.add_argument("--reuse-counterfactuals", type=Path,
        help="Prior folder containing gallery, comparison and manifest; requires unchanged input/model/world/controller")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_frozen_demo_model(args.samples)
    started = time.monotonic()
    runs, rows, checks, pairs = {}, [], [], []
    reuse_pairs, reuse_gallery, reuse_audit = {}, {}, None
    if args.reuse_counterfactuals:
        prior = args.reuse_counterfactuals
        prior_manifest = json.loads((prior/"manifest.json").read_text(encoding="utf-8"))
        assert prior_manifest["input_samples_sha256"] == sha(args.samples)
        for name in ("sidewalk_worlds.py", "sidewalk_simulation.py", "demo_model.py"):
            assert prior_manifest["source_sha256"][name] == sha(ROOT/"src/delivery_risk"/name), (name,"counterfactual_reuse_requires_unchanged_controller")
        assert prior_manifest["gallery_sha256"] == sha(prior/"sidewalk_gallery.js")
        assert prior_manifest["outputs_sha256"]["fallback_comparison.json"] == sha(prior/"fallback_comparison.json")
        prior_comparison = json.loads((prior/"fallback_comparison.json").read_text(encoding="utf-8"))
        reuse_pairs = {(p["scenario"],p["occlusion"]):p for p in prior_comparison["pairs"]}
        reuse_gallery = json.loads((prior/"sidewalk_gallery.js").read_text(encoding="utf-8").split("=",1)[1].strip().removesuffix(";"))["runs"]
        reuse_audit = {"input_model_world_controller_unchanged":True,
            "only_primary_operator_helper_revised":True,
            "prior_comparison_sha256":sha(prior/"fallback_comparison.json"),
            "prior_gallery_sha256":sha(prior/"sidewalk_gallery.js"),
            "prior_manifest_sha256":sha(prior/"manifest.json"),
            "reused_policy_names":["learned","fallback_only","all","no_assist"],
            "off_visibility_unused_all_and_no_assist_summaries_not_exported":True}
    baseline = {}
    if args.legacy_baseline.is_file():
        with args.legacy_baseline.open(encoding="utf-8", newline="") as handle:
            baseline = {(r["scenario"], r["occlusion"].lower()=="true"): r for r in csv.DictReader(handle)}
    compare_metrics = ("completed", "unfinished", "requests", "mean_wait_s", "p95_wait_s",
        "physical_overlap_frames", "static_contact_frames", "off_walkway_frames", "cars_off_road_frames",
        "pedestrians_off_walkway_frames", "waiting_at_end", "route_replans", "failed_replans")
    for scenario in SIDEWALK_CATALOG:
        for occlusion in ([True] if args.skip_visibility_pairs else [True, False]):
            cfg = SidewalkConfig(scenario=scenario, domain="robot", fleet=6, threshold=model.dev_policy["cutoff"],
                              operators=1, seed=42, density=1.0, occlusion=occlusion)
            if reuse_pairs:
                result = simulate_sidewalk(model, cfg, compare=False)
                old = reuse_pairs[(scenario,occlusion)]
                prior_cfg = reuse_gallery[scenario]["config"]
                assert all(prior_cfg[key]==value for key,value in asdict(cfg).items() if key!="occlusion")
                other = [old["legacy"], old["fallback_only"]]
                if occlusion:
                    other += [p for p in reuse_gallery[scenario]["policies"] if p["policy"] in ("all","no_assist")]
                result["policies"] = [{"name":"模型 + 停滞兜底 + 协助指令", **result["summary"]}] + other
            else:
                result = simulate_sidewalk(model, cfg, compare=True)
            if occlusion:
                runs[scenario] = result
            summary = result["summary"]
            policies = {r["policy"]: r for r in result["policies"]}
            legacy, fallback = policies["learned"], policies["fallback_only"]
            assert summary["policy"] == "learned_with_fallback"
            baseline_row = baseline.get((scenario, occlusion))
            if baseline_row:
                for metric in compare_metrics:
                    assert abs(float(baseline_row[metric])-float(legacy[metric])) < 1e-8, (scenario, occlusion, metric, "legacy_regression")
            pair = {"scenario": scenario, "occlusion": occlusion, "legacy": legacy,
                "hybrid": summary, "fallback_only": fallback,
                "delta": {metric: summary[metric]-legacy[metric] for metric in compare_metrics},
                "legacy_saved_baseline_matches": bool(baseline_row)}
            if reuse_pairs:
                pair["static_only_operator"] = reuse_pairs[(scenario,occlusion)]["hybrid"]
            pairs.append(pair)
            row = {"scenario": scenario, "occlusion": occlusion, **summary}
            rows.append(row)
            assert result["metadata"].get("synthetic") is True
            assert len(result["frames"]) > 0
            for metric in ("off_walkway_frames", "cars_off_road_frames", "pedestrians_off_walkway_frames"):
                assert summary[metric] == 0, (scenario, occlusion, metric, summary[metric])
            assert cfg.domain == "robot" and result["world"].get("walkways")
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
    args.gallery.write_text("window.DELIVERY_SIDEWALK_GALLERY="+json.dumps({"info": information(model), "runs": runs}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+";\n", encoding="utf-8")
    summary = {"status": "complete_synthetic_mechanism_study", "real_model": "original_development_selected_hist_gradient_boosting",
        "model_fit_train_rows": model.fit_audit["training_rows"], "model_fit_eval_rows": 0,
        "scenarios": len(runs), "runs": len(rows), "gallery_config": asdict(SidewalkConfig(fleet=6, seed=42, threshold=model.dev_policy["cutoff"])),
        "new_real_data": False, "simulated_obstacles_are_real_training_labels": False,
        "counterfactual_reuse": reuse_audit,
        "unsupported_objects_mapped_to_vehicle": False, "visibility_pair_interpretation": "conditional_on_one_seed_and_toy_world_not_real_sensor_evaluation",
        "checks": checks, "publication_approval": "pending_user_review"}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    total_metrics = ("completed", "unfinished", "requests", "physical_overlap_frames", "static_contact_frames",
        "off_walkway_frames", "cars_off_road_frames", "pedestrians_off_walkway_frames", "route_replans", "failed_replans")
    comparison = {"status": "complete_paired_synthetic_comparison", "paired_settings": len(pairs),
        "baseline_source": "pure learned gate re-run on identical settings; checked against saved pre-fix aggregates",
        "comparison_scope": "engineering mechanism bundle; fallback_only isolates request fallback with old handler",
        "legacy_baseline_sha256": sha(args.legacy_baseline) if args.legacy_baseline.is_file() else None,
        "same_model_world_seed_density_and_visibility_within_each_pair": True,
        "new_real_samples": 0, "independent_replicates": False,
        "counterfactual_reuse":reuse_audit,
        "totals": {key: {metric: sum(p[key][metric] for p in pairs) for metric in total_metrics}
                   for key in ("legacy", "hybrid", "fallback_only")}, "pairs": pairs,
        "publication_approval": "pending_user_review"}
    (args.output / "fallback_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    comparison_rows = [{"scenario": pair["scenario"], "occlusion": pair["occlusion"],
        **{f"{key}_{metric}": pair[key][metric] for key in ("legacy", "hybrid", "fallback_only") for metric in compare_metrics},
        **{f"delta_{metric}": pair["delta"][metric] for metric in compare_metrics}} for pair in pairs]
    with (args.output / "fallback_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0])); writer.writeheader(); writer.writerows(comparison_rows)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "input_samples_sha256": sha(args.samples),
        "core_model_code_unchanged": True, "no_weights_saved": True, "no_real_scene_coordinates_exported": True,
        "script_sha256": sha(Path(__file__)), "source_sha256": {name: sha(ROOT / "src/delivery_risk" / name) for name in ("sidewalk_worlds.py", "sidewalk_simulation.py", "demo_model.py", "assistance.py")},
        "outputs_sha256": {name: sha(args.output / name) for name in ("scenario_metrics.csv", "summary.json", "fallback_comparison.csv", "fallback_comparison.json")},
        "gallery_sha256": sha(args.gallery), "elapsed_s": round(time.monotonic()-started, 2), "user_publication_approval": "pending"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"complete": True, "scenarios": len(runs), "paired_runs": len(rows), "gallery_bytes": args.gallery.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
