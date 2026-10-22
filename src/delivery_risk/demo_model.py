"""Inference adapter for the already selected 2 m model, never a controller.

The adapter re-fits the frozen HistGradientBoosting pipeline on exactly the
original eligible training cohort. Development and evaluation rows are not
used by ``fit``. ``verify_parity`` can subsequently compare held-out scores to
the original run. No weight file is written or loaded.

Scores are uncalibrated classifier outputs for a future-clearance proxy on
filtered road-vehicle data. They are not probabilities of danger, intervention,
or delivery failure. Synthetic animation is a distribution-shift demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn

from .experiment import make_model, metric_row, sha256_file
from .samples import Protocol, extract_features, feature_columns


SCORE_SEMANTICS = (
    "uncalibrated model score for the filtered 3-second / 2-metre clearance "
    "proxy; not intervention probability, collision probability, or a control action"
)
PRIMARY_MODEL = "hist_gradient_boosting"
PRIMARY_SEED = 42
PRIMARY_THRESHOLD_M = 2
FULL_FEATURE_COUNT = 194


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_or_none(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


@dataclass
class FrozenDemoModel:
    """In-memory fitted pipeline with an explicit, ordered feature boundary."""

    estimator: Any
    columns: tuple[str, ...]
    protocol: Protocol
    training_feature_ranges: dict[str, dict[str, float | None]]
    dev_policy: dict[str, float | int]
    fit_audit: dict[str, Any]

    def _matrix(self, rows: Sequence[dict[str, float]]) -> pd.DataFrame:
        expected = set(self.columns)
        for row in rows:
            absent, extra = expected - set(row), set(row) - expected
            if absent or extra:
                raise ValueError(
                    f"exact_feature_schema_required: missing={sorted(absent)}; "
                    f"unexpected={sorted(extra)}"
                )
        matrix = pd.DataFrame(rows, columns=self.columns).apply(pd.to_numeric, errors="raise")
        if np.isinf(matrix.to_numpy(float)).any():
            raise ValueError("infinite_demo_features")
        return matrix

    def predict_features_many(self, rows: Sequence[dict[str, float]]) -> list[dict]:
        """Batch inference; row dictionaries are aligned to the frozen 194 columns.

        Range flags compare individual features to training min/max and 1st/99th
        percentiles. They are descriptive warnings, not a validated OOD detector.
        No score is corrected or replaced when a warning is present.
        """
        if not rows:
            return []
        matrix = self._matrix(rows)
        scores = np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=float)
        if len(scores) != len(rows) or not np.isfinite(scores).all():
            raise ValueError("invalid_model_scores")
        outputs = []
        for features, score in zip(rows, scores):
            outside_range, outside_central = [], []
            for name in self.columns:
                value = float(features[name])
                support = self.training_feature_ranges.get(name, {})
                if np.isfinite(value):
                    if support.get("min") is not None and not support["min"] <= value <= support["max"]:
                        outside_range.append(name)
                    if support.get("q01") is not None and not support["q01"] <= value <= support["q99"]:
                        outside_central.append(name)
            current = float(features["current_min_clearance"])
            eligible = bool(np.isfinite(current) and current > PRIMARY_THRESHOLD_M)
            outputs.append({
                "score": float(score),
                "current_min_clearance": current,
                "cv_min_clearance": float(features["cv_min_clearance"]),
                "current_eligible": eligible,
                "score_applicable": eligible,
                "feature_count": len(self.columns),
                "score_semantics": SCORE_SEMANTICS,
                "outside_training_range_features": outside_range,
                "outside_training_central98_features": outside_central,
                "range_check_is_validated_ood_detector": False,
            })
        return outputs

    def predict_features(self, features: dict[str, float]) -> dict:
        return self.predict_features_many([features])[0]

    def features_from_scene(self, scene: dict) -> dict[str, float]:
        """Reuse the study's past-only boundary, without calling future_labels."""
        features, _ = extract_features(scene, self.protocol)
        return features

    def predict_scene(self, scene: dict) -> dict:
        return self.predict_features(self.features_from_scene(scene))

    def predict_scenes(self, scenes: Sequence[dict]) -> list[dict]:
        return self.predict_features_many([self.features_from_scene(scene) for scene in scenes])


def load_frozen_demo_model(
    samples_csv: str | Path,
    feature_columns_json: str | Path | None = None,
    primary_selection_json: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> FrozenDemoModel:
    """Fit only the frozen eligible training rows, after checking original hashes.

    The optional explicit artifact paths support callers with copied read-only
    run metadata. Other frozen artifacts are read from ``results_dir`` (default:
    this project's results/real directory). This function does not select models,
    evaluate performance, tune thresholds, or save weights. Training uses the
    original future-derived target; inference inputs remain strictly past-only.
    """
    result = Path(results_dir) if results_dir else Path(__file__).resolve().parents[2] / "results" / "real"
    feature_path = Path(feature_columns_json) if feature_columns_json else result / "feature_columns.json"
    selection_path = Path(primary_selection_json) if primary_selection_json else result / "primary_selection.json"
    columns = tuple(_read_json(feature_path)["full"])
    selection = _read_json(selection_path)
    required_selection = {"model": PRIMARY_MODEL, "seed": PRIMARY_SEED,
                          "ablation": "full", "threshold_m": PRIMARY_THRESHOLD_M,
                          "frozen_before_eval_prediction": True}
    if any(selection.get(k) != v for k, v in required_selection.items()):
        raise ValueError("unexpected_frozen_primary_model_selection")
    if len(columns) != FULL_FEATURE_COUNT or len(set(columns)) != len(columns):
        raise ValueError("frozen_full_feature_count_must_be_194")
    manifest = _read_json(result / "run_manifest.json")
    source = Path(samples_csv)
    if sha256_file(source) != manifest["input_sha256"]:
        raise ValueError("demo_training_input_differs_from_original_run")
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"frozen_study_code_changed: {name}")
    frame = pd.read_csv(source, dtype={"scenario_id": str})
    if columns != tuple(feature_columns(frame.columns, "full")):
        raise ValueError("frozen_feature_order_differs_from_prepared_csv")
    train = frame.loc[(frame.split == "train") & (frame.eligible_2m.astype(int) == 1)].copy()
    y_train = train.label_2m.to_numpy(int)
    expected_train = _read_json(result / "summary.json")["cohorts"]["2m"]["counts"]["train"]
    if len(train) != expected_train["n"] or int(y_train.sum()) != expected_train["positive"]:
        raise ValueError("demo_training_cohort_differs_from_original_run")
    estimator = make_model(PRIMARY_MODEL, PRIMARY_SEED)
    estimator.fit(train[list(columns)], y_train)
    quantiles = train[list(columns)].quantile([0, .01, .99, 1])
    ranges = {name: {key: _finite_or_none(quantiles.loc[q, name])
                     for key, q in (("min", 0), ("q01", .01), ("q99", .99), ("max", 1))}
              for name in columns}
    policies = pd.read_csv(result / "dev_fixed_policies.csv")
    chosen = policies.loc[(policies.threshold_m == PRIMARY_THRESHOLD_M) &
        (policies.ablation == "full") & (policies.model == PRIMARY_MODEL) &
        (policies.seed == PRIMARY_SEED) & np.isclose(policies.nominal_dev_budget, .1)]
    if len(chosen) != 1:
        raise ValueError("single_frozen_development_policy_required")
    row = chosen.iloc[0]
    policy = {"cutoff": float(row.cutoff), "tie_probability": float(row.tie_probability),
              "capacity": int(row.capacity), "nominal_dev_budget": .1}
    audit = {
        "model": PRIMARY_MODEL, "seed": PRIMARY_SEED, "ablation": "full",
        "threshold_m": PRIMARY_THRESHOLD_M, "feature_count": len(columns),
        "input_sha256": manifest["input_sha256"], "training_rows": len(train),
        "training_positive": int(y_train.sum()), "fit_splits": ["train"],
        "development_rows_used_for_fit": 0, "evaluation_rows_used_for_fit": 0,
        "selection_repeated": False, "weights_written": False,
        "original_feature_list_sha256": sha256_file(feature_path),
        "original_selection_sha256": sha256_file(selection_path),
        "frozen_model_code_sha256": manifest["code_sha256"],
        "sklearn_version": sklearn.__version__,
        "feature_order_sha256": hashlib.sha256(json.dumps(columns).encode()).hexdigest(),
    }
    return FrozenDemoModel(estimator, columns, Protocol(), ranges, policy, audit)


def verify_parity(
    model: FrozenDemoModel,
    samples_csv: str | Path,
    frozen_metrics_csv: str | Path,
    frozen_private_predictions_csv: str | Path | None = None,
) -> dict:
    """Post-fit audit only. Does not change the estimator or any frozen result."""
    if sha256_file(Path(samples_csv)) != model.fit_audit["input_sha256"]:
        raise ValueError("parity_input_differs_from_training_input")
    frame = pd.read_csv(samples_csv, dtype={"scenario_id": str})
    evaluation = frame.loc[(frame.split == "eval") & (frame.eligible_2m.astype(int) == 1)].copy()
    scores = model.estimator.predict_proba(evaluation[list(model.columns)])[:, 1]
    actual = metric_row(evaluation.label_2m.to_numpy(int), scores, True)
    metrics = pd.read_csv(frozen_metrics_csv)
    expected = metrics.loc[(metrics.threshold_m == 2) & (metrics.split == "eval") &
        (metrics.ablation == "full") & (metrics.model == PRIMARY_MODEL) & (metrics.seed == 42)]
    if len(expected) != 1:
        raise ValueError("single_frozen_primary_metric_row_required")
    differences = {name: abs(float(value) - float(expected.iloc[0][name]))
                   for name, value in actual.items()}
    maximum = max(differences.values())
    prediction_difference = None
    predictions_match = None
    if frozen_private_predictions_csv is not None:
        saved = pd.read_csv(frozen_private_predictions_csv, dtype={"scenario_id": str})
        if list(saved.scenario_id) != list(evaluation.scenario_id):
            raise ValueError("original_prediction_row_order_mismatch")
        prediction_difference = float(np.max(np.abs(scores - saved["full__hist_gradient_boosting__42"].to_numpy(float))))
        predictions_match = prediction_difference <= 1e-12
    passed = maximum <= 1e-12 and predictions_match is not False
    result = {**model.fit_audit, "audit": "post_fit_reproduction_not_new_model_selection",
        "evaluation_rows": len(evaluation), "evaluation_positive": int(evaluation.label_2m.sum()),
        "expected_eval_ap": float(expected.iloc[0].ap), "reproduced_eval_ap": float(actual["ap"]),
        "all_aggregate_metrics_max_absolute_difference": maximum,
        "original_prediction_max_absolute_difference": prediction_difference,
        "original_prediction_parity_within_tolerance": predictions_match,
        "tolerance": 1e-12, "passed": passed,
        "development_policy": model.dev_policy,
        "calibration": "not_recalibrated; score is not a validated probability on synthetic scenes",
        "synthetic_domain_warning": "road vehicle sizes and observation quality differ from sidewalk delivery robots",
        "score_semantics": SCORE_SEMANTICS}
    if not passed:
        raise ValueError(f"retrained_model_failed_frozen_parity: {json.dumps(result)}")
    return result
