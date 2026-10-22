"""Read-only independent audit of frozen WOMD preparation and aggregate results.

Writes only its aggregate audit JSON. It never trains models, changes samples,
uses credentials, changes the protocol, or publishes anything. Raw-file SHA256
rechecking is optional because it reads the complete downloaded files again.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE
EXPECTED_PROTOCOL = {"history_steps": 10, "future_steps": 30, "dt": .1,
                     "radius_m": 50., "max_neighbors": 16,
                     "thresholds_m": [1., 2., 3.]}
MODEL_ORDER = ("logistic_regression", "random_forest", "hist_gradient_boosting")
BUDGETS = (.05, .1, .2)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_ap(labels, scores):
    """Independent AP using equal-score groups, without sklearn/project imports."""
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    boundaries = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    cumulative = np.cumsum(y)[boundaries]
    positive_in_group = np.diff(np.r_[0, cumulative])
    return float(np.sum(positive_in_group * cumulative / (boundaries + 1)) / labels.sum())


def expected_auc(labels, scores):
    labels = np.asarray(labels, int)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if not positives or not negatives:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def rank_metrics(labels, scores, budget):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    capacity = int(math.floor(len(scores) * budget))
    weights = np.zeros(len(scores))
    if capacity:
        cutoff = np.sort(scores)[-capacity]
        greater, tied = scores > cutoff, scores == cutoff
        weights[greater] = 1
        weights[tied] = (capacity - greater.sum()) / tied.sum()
    return weighted_metrics(labels, weights)


def weighted_metrics(labels, weights):
    labels, weights = np.asarray(labels, int), np.asarray(weights, float)
    selected, true_positive = weights.sum(), np.dot(labels, weights)
    return {"request_rate": float(selected / len(labels)),
            "recall": float(true_positive / labels.sum()) if labels.sum() else float("nan"),
            "precision": float(true_positive / selected) if selected else float("nan")}


def independent_bootstrap(labels, scores, cv_scores, groups, draws, seed):
    groups = np.asarray(groups)
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    indices_by_group = {name: np.flatnonzero(groups == name) for name in names}
    values = []
    for _ in range(draws):
        if len(names) >= 5:
            selected = rng.choice(names, size=len(names), replace=True)
            indices = np.concatenate([indices_by_group[name] for name in selected])
        else:
            indices = rng.integers(0, len(labels), size=len(labels))
        y, a, b = labels[indices], scores[indices], cv_scores[indices]
        if len(np.unique(y)) != 2:
            continue
        values.append((expected_ap(y, a) - expected_ap(y, b),
                       rank_metrics(y, a, .1)["recall"] - rank_metrics(y, b, .1)["recall"]))
    array = np.asarray(values)
    if not len(values):
        return {"valid_draws": 0,
                "unit": "source_shard" if len(names) >= 5 else "scenario_iid_exploratory",
                "ap_difference": {"ci95_low": None, "ci95_high": None},
                "recall_at_10pct_difference": {"ci95_low": None, "ci95_high": None}}
    return {"valid_draws": len(values),
            "unit": "source_shard" if len(names) >= 5 else "scenario_iid_exploratory",
            **{name: {"ci95_low": float(np.quantile(array[:, i], .025)),
                      "ci95_high": float(np.quantile(array[:, i], .975))}
               for i, name in enumerate(("ap_difference", "recall_at_10pct_difference"))}}


class Audit:
    def __init__(self):
        self.errors, self.warnings = [], []

    def check(self, condition, message):
        if not condition:
            self.errors.append(message)

    def close(self, actual, expected, message, atol=1e-9):
        if actual is None or expected is None:
            self.check(actual is None and expected is None, message)
            return
        self.check(bool(np.isclose(actual, expected, atol=atol, rtol=1e-8, equal_nan=True)), message)


def audit_preparation(prepared, raw_root, audit):
    provenance = read_json(prepared / "provenance.json")
    split_manifest = read_json(prepared / "split_manifest.json")
    samples = prepared / "samples.csv"
    frame = pd.read_csv(samples, dtype={"scenario_id": str, "source_file": str,
                                       "source_sha256": str, "trajectory_sha256": str})
    audit.check(provenance.get("synthetic") is False, "provenance_not_real_data")
    audit.check(provenance.get("source_type") == "official_womd_scenario", "unexpected_source_type")
    audit.check(provenance.get("samples_sha256") == digest(samples), "prepared_csv_hash_mismatch")
    audit.check(provenance.get("split_manifest_sha256") == digest(prepared / "split_manifest.json"), "split_manifest_hash_mismatch")
    audit.check(provenance.get("protocol") == EXPECTED_PROTOCOL, "frozen_protocol_changed")
    expected_protocol_hash = hashlib.sha256(json.dumps(EXPECTED_PROTOCOL, sort_keys=True).encode()).hexdigest()
    audit.check(provenance.get("protocol_sha256") == expected_protocol_hash, "protocol_fingerprint_mismatch")
    audit.check(split_manifest.get("protocol_sha256") == expected_protocol_hash, "split_protocol_fingerprint_mismatch")
    code_paths = [PROJECT / "scripts" / "prepare.py", *[PROJECT / "src" / "delivery_risk" / name
                  for name in ("samples.py", "geometry.py", "waymo_reader.py")]]
    extraction_hash = hashlib.sha256(json.dumps({p.name: digest(p) for p in code_paths}, sort_keys=True).encode()).hexdigest()
    audit.check(not frame.scenario_id.duplicated().any(), "duplicate_scenario_ids")
    for column in ("source_file", "source_sha256", "trajectory_sha256"):
        audit.check(not frame.groupby(column)["split"].nunique().gt(1).any(), f"cross_split_overlap_{column}")
    audit.check(set(frame.split.unique()) == {"train", "dev", "eval"}, "missing_or_unexpected_splits")
    audit.check(frame.neighbor_count.between(1, 16).all(), "invalid_retained_neighbor_count")
    aggregates, counts_by_split = Counter(), {}
    raw_verified = 0
    for meta in provenance["sources"]:
        filename, split, counts = meta["source_file"], meta["split"], meta["counts"]
        audit.check(meta.get("extraction_code_sha256") == extraction_hash, f"extraction_code_changed:{filename}")
        audit.check(meta.get("protocol_sha256") == expected_protocol_hash, f"source_protocol_mismatch:{filename}")
        audit.check(filename in split_manifest["files"].get(split, []), f"source_missing_from_frozen_split:{filename}")
        cache = Path(meta["csv"]).resolve()
        audit.check(cache.is_relative_to(prepared.resolve()), f"cache_outside_prepared_directory:{filename}")
        if cache.is_relative_to(prepared.resolve()):
            audit.check(cache.exists() and digest(cache) == meta["csv_sha256"], f"cached_csv_changed:{filename}")
        subset = frame[(frame.source_file == filename) & (frame.split == split)]
        audit.check(len(subset) == counts.get("retained_complete", 0), f"retained_count_mismatch:{filename}")
        audit.check(subset.source_sha256.eq(meta["source_sha256"]).all(), f"source_hash_in_rows_mismatch:{filename}")
        exclusions = sum(n for name, n in counts.items() if name not in ("raw", "retained_complete")
                         and not name.startswith(("eligible_", "positive_", "current_event_")))
        audit.check(counts.get("raw", 0) == counts.get("retained_complete", 0) + exclusions,
                    f"raw_count_reconciliation_failed:{filename}")
        for threshold in (1, 2, 3):
            eligible = subset[f"eligible_{threshold}m"].eq(1)
            audit.check(int(eligible.sum()) == counts.get(f"eligible_{threshold}m", 0), f"eligible_count_mismatch:{filename}:{threshold}m")
            audit.check(int(subset.loc[eligible, f"label_{threshold}m"].sum()) == counts.get(f"positive_{threshold}m", 0), f"positive_count_mismatch:{filename}:{threshold}m")
            audit.check(int((~eligible).sum()) == counts.get(f"current_event_{threshold}m", 0), f"current_event_count_mismatch:{filename}:{threshold}m")
        aggregates.update(counts)
        counts_by_split.setdefault(split, Counter()).update(counts)
        if raw_root:
            folder = "training" if split == "train" else "validation"
            raw = raw_root / folder / filename
            audit.check(raw.exists(), f"raw_file_missing:{filename}")
            if raw.exists():
                audit.check(raw.stat().st_size == meta["source_bytes"], f"raw_bytes_mismatch:{filename}")
                audit.check(digest(raw) == meta["source_sha256"], f"raw_sha256_mismatch:{filename}")
                raw_verified += 1
    cohort_report = {}
    for threshold in (1, 2, 3):
        eligible = frame[f"eligible_{threshold}m"].astype(int)
        label = frame[f"label_{threshold}m"].astype(int)
        audit.check(eligible.eq(frame.current_min_clearance.gt(threshold).astype(int)).all(), f"eligibility_rule_mismatch:{threshold}m")
        audit.check(label.eq(frame.future_min_clearance.le(threshold).astype(int)).all(), f"label_rule_mismatch:{threshold}m")
        event = pd.to_numeric(frame[f"event_time_{threshold}m"], errors="coerce")
        audit.check(event[label == 1].between(.09999, 3.00001).all(), f"positive_event_time_invalid:{threshold}m")
        audit.check(event[label == 0].isna().all(), f"negative_has_event_time:{threshold}m")
        cohorts = {}
        for split in ("train", "dev", "eval"):
            part = frame[(frame.split == split) & eligible.eq(1)]
            positives = int(part[f"label_{threshold}m"].sum())
            cohorts[split] = {"n": len(part), "positive": positives, "negative": len(part) - positives,
                              "source_shards": part.source_file.nunique(),
                              "positive_rate": positives / len(part) if len(part) else None}
        cohort_report[f"{threshold}m"] = cohorts
    audit.check((frame.label_1m <= frame.label_2m).all() and (frame.label_2m <= frame.label_3m).all(), "nonmonotone_sensitivity_labels")
    audit.check((frame.eligible_1m >= frame.eligible_2m).all() and (frame.eligible_2m >= frame.eligible_3m).all(), "nonmonotone_eligibility")
    expected_sources = {name for names in split_manifest["files"].values() for name in names}
    audit.check(expected_sources == {m["source_file"] for m in provenance["sources"]}, "source_manifest_set_mismatch")
    audit.check(len(provenance["sources"]) == 22, "expected_fixed_22_source_files")
    audit.check({k: len(v) for k, v in split_manifest["files"].items()} == {"train": 12, "dev": 5, "eval": 5}, "expected_fixed_12_5_5_source_split")
    audit.warnings.append("原始日志分组未知；精确轨迹哈希与文件分离不能证明所有重叠窗口或地点完全独立。")
    result = {"rows": len(frame), "source_files": len(provenance["sources"]),
              "raw_sha256_rechecked_files": raw_verified, "raw_sha256_recheck_requested": bool(raw_root),
              "extraction_code_fingerprint_matches_current_files": extraction_hash,
              "stage_counts_total": dict(aggregates), "stage_counts_by_split": {k: dict(v) for k, v in counts_by_split.items()},
              "cohorts": cohort_report, "retained_zero_neighbor_rows": int(frame.neighbor_count.eq(0).sum()),
              "raw_complete_retention_fraction": len(frame) / aggregates["raw"] if aggregates["raw"] else None,
              "future_censor_fraction_of_raw": aggregates["future_censored"] / aggregates["raw"] if aggregates["raw"] else None}
    return frame, provenance, result


def audit_results(results, frame, provenance, audit, recompute_bootstrap):
    if not (results / "summary.json").exists():
        return {"status": "awaiting_completed_evaluation"}
    summary = read_json(results / "summary.json")
    manifest = read_json(results / "run_manifest.json")
    selection = read_json(results / "primary_selection.json")
    metrics, policies = pd.read_csv(results / "metrics.csv"), pd.read_csv(results / "dev_fixed_policies.csv")
    audit.check(manifest.get("status") == "complete", "experiment_manifest_not_complete")
    audit.check(manifest.get("input_sha256") == provenance["samples_sha256"], "experiment_input_hash_mismatch")
    audit.check(selection.get("frozen_before_eval_prediction") is True, "selection_not_recorded_as_frozen")
    audit.check(selection.get("seed") == 42 and selection.get("threshold_m") == 2 and selection.get("ablation") == "full", "primary_selection_protocol_changed")
    for name, value in manifest["code_sha256"].items():
        audit.check(digest(PROJECT / "src" / "delivery_risk" / name) == value, f"experiment_code_changed:{name}")
    dev = metrics[(metrics.threshold_m == 2) & (metrics.split == "dev") & (metrics.ablation == "full") & (metrics.seed == 42)]
    dev_scores = {name: float(dev.loc[dev.model.eq(name), "ap"].iloc[0]) for name in MODEL_ORDER}
    expected_choice = max(MODEL_ORDER, key=lambda name: dev_scores[name])
    audit.check(selection["model"] == expected_choice, "model_not_dev_ap_choice")
    primary = None
    checked_rows = 0
    for threshold in (1, 2, 3):
        file = results.parent / "private" / results.name / f"predictions_{threshold}m.csv"
        if not file.exists():
            audit.warnings.append(f"No private prediction file for {threshold}m; inspect skipped cohort status.")
            continue
        predictions = pd.read_csv(file, dtype={"scenario_id": str, "source_file": str})
        cohort = frame[(frame.split == "eval") & frame[f"eligible_{threshold}m"].eq(1)].copy()
        audit.check(not predictions.scenario_id.duplicated().any(), f"duplicate_prediction_ids:{threshold}m")
        audit.check(set(predictions.scenario_id) == set(cohort.scenario_id), f"evaluation_sample_set_mismatch:{threshold}m")
        merged = predictions.merge(cohort[["scenario_id", "source_file", f"label_{threshold}m", "cv_min_clearance"]],
                                   on="scenario_id", validate="one_to_one", suffixes=("", "_prepared"))
        audit.check(merged.source_file.eq(merged.source_file_prepared).all(), f"prediction_source_mismatch:{threshold}m")
        audit.check(merged[f"label_{threshold}m"].eq(merged[f"label_{threshold}m_prepared"]).all(), f"prediction_labels_mismatch:{threshold}m")
        audit.check(np.allclose(merged.constant_velocity, -merged.cv_min_clearance), f"cv_scores_not_past_only_geometry:{threshold}m")
        y = merged[f"label_{threshold}m"].to_numpy(int)
        score_columns = [c for c in predictions if c not in ("scenario_id", "source_file", f"label_{threshold}m")]
        for column in score_columns:
            if column == "constant_velocity":
                ablation, model, seed = "full", "constant_velocity", 42
            else:
                ablation, model, seed = column.split("__")
                seed = int(seed)
            rows = metrics[(metrics.threshold_m == threshold) & (metrics.split == "eval") &
                           (metrics.ablation == ablation) & (metrics.model == model) & (metrics.seed == seed)]
            audit.check(len(rows) == 1, f"metric_row_missing_or_duplicate:{threshold}:{column}")
            if len(rows) != 1:
                continue
            row = rows.iloc[0]
            score = merged[column].to_numpy(float)
            audit.close(row.ap, expected_ap(y, score), f"ap_disagrees:{threshold}:{column}")
            audit.close(row.roc_auc, expected_auc(y, score), f"roc_disagrees:{threshold}:{column}")
            if model != "constant_velocity":
                audit.close(row.brier, np.mean((y - score) ** 2), f"brier_disagrees:{threshold}:{column}")
            for budget in BUDGETS:
                recalculated = rank_metrics(y, score, budget)
                for key, value in recalculated.items():
                    audit.close(row[f"{key}_budget_{round(budget * 100):02d}pct"], value,
                                f"budget_metric_disagrees:{threshold}:{column}:{budget}:{key}")
            policy_rows = policies[(policies.threshold_m == threshold) & (policies.ablation == ablation) &
                                   (policies.model == model) & (policies.seed == seed)]
            for _, policy in policy_rows.iterrows():
                weights = (score > policy.cutoff).astype(float) + (score == policy.cutoff) * policy.tie_probability
                audit.check(0 <= policy.tie_probability <= 1, f"invalid_tie_probability:{threshold}:{column}")
                for key, value in weighted_metrics(y, weights).items():
                    audit.close(policy[key], value, f"fixed_dev_policy_disagrees:{threshold}:{column}:{key}")
            checked_rows += 1
        if threshold == 2:
            primary = merged
    outcome = {"status": summary["primary_status"], "eval_metric_rows_independently_recomputed": checked_rows,
               "selected_model": selection["model"], "dev_ap_scores": dev_scores,
               "primary_comparison": summary.get("primary_comparison"),
               "stored_uncertainty": summary.get("paired_bootstrap"),
               "seed_stability": summary.get("seed_stability"),
               "curve_artifact_present": (results / "primary_curves.png").is_file(),
               "case_review_record_present": (PROJECT / "docs" / "case_review.md").is_file(),
               "upload_approval": "pending_user_review"}
    if primary is not None and summary.get("primary_comparison"):
        y = primary.label_2m.to_numpy(int)
        selected = primary[f"full__{selection['model']}__42"].to_numpy(float)
        cv = primary.constant_velocity.to_numpy(float)
        delta_ap = expected_ap(y, selected) - expected_ap(y, cv)
        delta_recall = rank_metrics(y, selected, .1)["recall"] - rank_metrics(y, cv, .1)["recall"]
        audit.close(summary["primary_comparison"]["ap_difference"], delta_ap, "primary_ap_difference_mismatch")
        audit.close(summary["primary_comparison"]["recall_at_10pct_difference"], delta_recall, "primary_recall_difference_mismatch")
        shards = primary.source_file.nunique()
        expected_unit = "source_shard" if shards >= 5 else "scenario_iid_exploratory"
        audit.check(summary["paired_bootstrap"]["unit"] == expected_unit, "bootstrap_unit_mismatch")
        if summary["primary_status"] == "eligible_for_limited_empirical_comparison":
            audit.check(shards >= 5, "empirical_status_with_too_few_eval_shards")
            for split, minimum in (("train", 200), ("dev", 100), ("eval", 100)):
                part = frame[(frame.split == split) & frame.eligible_2m.eq(1)]
                positives = part.label_2m.sum()
                audit.check(len(part) >= minimum and min(positives, len(part) - positives) >= 10,
                            f"empirical_status_below_minimum_counts:{split}")
        if recompute_bootstrap:
            independent = independent_bootstrap(y, selected, cv, primary.source_file.to_numpy(),
                int(manifest["config"]["bootstrap_draws"]), int(manifest["config"]["bootstrap_seed"]))
            stored = summary["paired_bootstrap"]
            audit.check(independent["valid_draws"] == stored["valid_draws"], "bootstrap_valid_draws_disagree")
            for name in ("ap_difference", "recall_at_10pct_difference"):
                for end in ("ci95_low", "ci95_high"):
                    audit.close(independent[name][end], stored[name][end], f"bootstrap_ci_disagrees:{name}:{end}")
            outcome["independently_recomputed_bootstrap"] = independent
        policy = policies[(policies.threshold_m == 2) & (policies.ablation == "full") &
            (policies.model == selection["model"]) & (policies.seed == 42) & np.isclose(policies.nominal_dev_budget, .1)]
        if len(policy):
            outcome["dev_fixed_10pct_policy_on_eval"] = {key: float(policy.iloc[0][key])
                for key in ("request_rate", "recall", "precision", "tie_probability")}
    elif primary is not None:
        outcome["comparison_unavailable"] = "缺少可定义的主比较；检查评估类别和summary状态，不作优势结论。"
    audit.warnings.append("AP优势与固定预算召回优势应分别解释；代理事件离线排序不能推出真实人工接管、外卖成本或事故效益。")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PROJECT / "artifacts" / "prepared")
    parser.add_argument("--results", type=Path, default=PROJECT / "results" / "real")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts" / "audit_results.json")
    parser.add_argument("--skip-bootstrap-recompute", action="store_true")
    args = parser.parse_args()
    for name in ("provenance.json", "samples.csv", "split_manifest.json"):
        if not (args.prepared / name).is_file():
            parser.error(f"Preparation is not complete: missing {name}; this auditor must wait.")
    audit = Audit()
    frame, provenance, prep_result = audit_preparation(args.prepared.resolve(), args.raw_root, audit)
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "preparation": prep_result,
              "evaluation": audit_results(args.results, frame, provenance, audit, not args.skip_bootstrap_recompute),
              "errors": audit.errors, "warnings": audit.warnings,
              "status": "pass_with_stated_limits" if not audit.errors else "audit_failed",
              "changes_to_data_or_models": False, "publication_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": len(audit.errors),
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
