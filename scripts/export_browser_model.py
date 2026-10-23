"""Convert already published HistGB weights to browser-readable, data-only JSON.

This is an inference-format conversion. It never calls fit or uses study rows.
Only a trusted, digest-checked model artifact is read by the existing loader.
The browser file retains the model's non-commercial dataset-derived license.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delivery_risk.browser_model import BrowserFrozenModel, FORMAT, NODE_LAYOUT, SCORE_SEMANTICS
from delivery_risk.model_artifact import load_model_artifact, sha256_file


def export_payload(model, source_artifact_sha256: str) -> dict:
    """Export numerical tree nodes and imputer metadata, without fitted records."""
    pipeline = model.estimator
    if (not isinstance(pipeline, Pipeline) or len(pipeline.steps) != 2
            or not isinstance(pipeline.steps[0][1], SimpleImputer)
            or not isinstance(pipeline.steps[1][1], HistGradientBoostingClassifier)):
        raise ValueError("browser_export_requires_imputer_histgb_pipeline")
    imputer, estimator = (step[1] for step in pipeline.steps)
    if (imputer.strategy != "median" or not imputer.keep_empty_features
            or imputer.add_indicator or not np.isnan(imputer.missing_values)
            or not np.isfinite(imputer.statistics_).all()):
        raise ValueError("unsupported_browser_export_imputer")
    if (list(estimator.classes_) != [0, 1] or estimator.n_trees_per_iteration_ != 1
            or estimator._preprocessor is not None
            or (estimator.is_categorical_ is not None and estimator.is_categorical_.any())):
        raise ValueError("browser_export_requires_binary_numerical_histgb")
    trees = []
    for iteration in estimator._predictors:
        if len(iteration) != 1:
            raise ValueError("browser_export_requires_one_tree_per_iteration")
        predictor = iteration[0]
        if predictor.nodes["is_categorical"].any():
            raise ValueError("browser_export_does_not_support_categorical_nodes")
        trees.append([[float(node[name]) if name in ("value", "num_threshold") else int(node[name])
                       for name in NODE_LAYOUT] for node in predictor.nodes])
    payload = {"format": FORMAT, "columns": list(model.columns),
        "protocol": asdict(model.protocol), "training_feature_ranges": model.training_feature_ranges,
        "dev_policy": model.dev_policy, "fit_audit": model.fit_audit,
        "score_semantics": SCORE_SEMANTICS, "source_artifact_sha256": source_artifact_sha256,
        "inference": {"classes": [0, 1], "link": "logistic",
            "imputer_statistics": [float(value) for value in imputer.statistics_],
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "node_layout": list(NODE_LAYOUT), "leaf_values_include_learning_rate": True,
            "trees": trees},
        "license": {"name": "Waymo Dataset License for Non-Commercial Use (March 2025)",
            "agreement_url": "https://github.com/thisisyk/sidewalk-delivery-robot-research/blob/main/licenses/WAYMO_DATASET_LICENSE_NON_COMMERCIAL_MARCH_2025.txt",
            "terms_url": "https://waymo.com/open/terms/",
            "notice_url": "https://github.com/thisisyk/sidewalk-delivery-robot-research/blob/main/models/NOTICE.txt", "non_commercial_research_only": True,
            "raw_records_included": False, "training_records_included": False}}
    # Validate the exact JSON round-trip that the browser will consume.
    normalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    BrowserFrozenModel.from_dict(normalized)
    return normalized


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "models/frozen_histgb_seed42.joblib")
    parser.add_argument("--output", type=Path, default=root / "demo/frozen_model.json")
    parser.add_argument("--trust-model-artifact", action="store_true", required=True,
                        help="Required: the input uses pickle; only convert trusted publisher weights.")
    args = parser.parse_args()
    model = load_model_artifact(args.model, trusted=args.trust_model_artifact)
    payload = export_payload(model, sha256_file(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"format": FORMAT, "sha256": sha256_file(args.output),
                      "bytes": args.output.stat().st_size, "features": len(model.columns),
                      "trees": len(payload["inference"]["trees"]), "training_repeated": False}))


if __name__ == "__main__":
    main()
