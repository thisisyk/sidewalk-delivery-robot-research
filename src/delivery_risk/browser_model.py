"""Data-only, NumPy inference for the published frozen HistGB model.

The JSON contains the existing median imputer, baseline and numerical decision
trees. Leaf values already include scikit-learn's learning rate. Nothing here
fits a model, selects a threshold, or changes the original feature extractor.
This module deliberately imports neither pandas, scikit-learn nor pickle, so
the same inference boundary can run inside a Pyodide browser worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from .samples import Protocol, extract_features


FORMAT = "delivery-risk-browser-histgb-v1"
NODE_LAYOUT = ("value", "feature_idx", "num_threshold", "missing_go_to_left",
               "left", "right", "is_leaf")
FULL_FEATURE_COUNT = 194
PRIMARY_THRESHOLD_M = 2
SCORE_SEMANTICS = (
    "uncalibrated model score for the filtered 3-second / 2-metre clearance "
    "proxy; not intervention probability, collision probability, or a control action"
)
_FIT_KEYS = {
    "model", "seed", "ablation", "threshold_m", "feature_count", "input_sha256",
    "training_rows", "training_positive", "fit_splits",
    "development_rows_used_for_fit", "evaluation_rows_used_for_fit",
    "selection_repeated", "original_feature_list_sha256", "original_selection_sha256",
    "frozen_model_code_sha256", "sklearn_version", "feature_order_sha256",
}
_PAYLOAD_KEYS = {
    "format", "columns", "protocol", "training_feature_ranges", "dev_policy",
    "fit_audit", "score_semantics", "source_artifact_sha256", "inference", "license",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)


@dataclass
class BrowserFrozenModel:
    """FrozenDemoModel-compatible inference using only the exported tree data."""

    columns: tuple[str, ...]
    protocol: Protocol
    training_feature_ranges: dict[str, dict[str, float | None]]
    dev_policy: dict[str, float | int]
    fit_audit: dict[str, Any]
    source_artifact_sha256: str
    _imputer_statistics: np.ndarray
    _baseline: float
    _trees: tuple

    @classmethod
    def from_json(cls, value: str | bytes | dict) -> "BrowserFrozenModel":
        """Accept fetched JSON data or decoded text, without executing code."""
        return cls.from_dict(value if isinstance(value, dict) else json.loads(value))

    @classmethod
    def from_dict(cls, payload: dict) -> "BrowserFrozenModel":
        if (not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS
                or payload.get("format") != FORMAT):
            raise ValueError("unsupported_browser_model_schema")
        columns = payload["columns"]
        if (not isinstance(columns, list) or len(columns) != FULL_FEATURE_COUNT
                or any(not isinstance(name, str) for name in columns)
                or len(set(columns)) != FULL_FEATURE_COUNT):
            raise ValueError("browser_model_requires_194_unique_ordered_features")
        order_hash = hashlib.sha256(json.dumps(tuple(columns)).encode()).hexdigest()
        audit = payload["fit_audit"]
        expected = {"model": "hist_gradient_boosting", "seed": 42, "ablation": "full",
                    "threshold_m": PRIMARY_THRESHOLD_M, "feature_count": FULL_FEATURE_COUNT,
                    "fit_splits": ["train"], "development_rows_used_for_fit": 0,
                    "evaluation_rows_used_for_fit": 0, "selection_repeated": False}
        if (not isinstance(audit, dict) or set(audit) - _FIT_KEYS
                or any(audit.get(key) != value for key, value in expected.items())
                or audit.get("feature_order_sha256") != order_hash):
            raise ValueError("browser_model_frozen_selection_or_feature_order_mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload["source_artifact_sha256"])):
            raise ValueError("browser_model_requires_source_artifact_digest")
        protocol = dict(payload["protocol"])
        protocol["thresholds_m"] = tuple(protocol["thresholds_m"])
        cfg = Protocol(**protocol)
        if cfg != Protocol():
            raise ValueError("browser_model_protocol_differs_from_frozen_study")
        ranges = payload["training_feature_ranges"]
        if not isinstance(ranges, dict) or set(ranges) != set(columns):
            raise ValueError("browser_model_feature_range_schema_mismatch")
        for support in ranges.values():
            if (not isinstance(support, dict) or set(support) != {"min", "q01", "q99", "max"}
                    or any(value is not None and not _finite_number(value) for value in support.values())):
                raise ValueError("browser_model_invalid_feature_ranges")
        policy = payload["dev_policy"]
        if not isinstance(policy, dict) or set(policy) != {"cutoff", "tie_probability", "capacity", "nominal_dev_budget"}:
            raise ValueError("browser_model_development_policy_schema_mismatch")
        if payload["score_semantics"] != SCORE_SEMANTICS:
            raise ValueError("browser_model_score_semantics_mismatch")
        inference = payload["inference"]
        if (not isinstance(inference, dict) or set(inference) != {
                "classes", "link", "imputer_statistics", "baseline", "node_layout",
                "leaf_values_include_learning_rate", "trees"}
                or inference["classes"] != [0, 1] or inference["link"] != "logistic"
                or inference["node_layout"] != list(NODE_LAYOUT)
                or inference["leaf_values_include_learning_rate"] is not True):
            raise ValueError("unsupported_browser_model_inference_layout")
        medians = inference["imputer_statistics"]
        if (not isinstance(medians, list) or len(medians) != FULL_FEATURE_COUNT
                or not all(_finite_number(value) for value in medians)
                or not _finite_number(inference["baseline"])):
            raise ValueError("browser_model_invalid_imputer_or_baseline")
        trees = inference["trees"]
        if not isinstance(trees, list) or not 1 <= len(trees) <= 1000:
            raise ValueError("browser_model_invalid_tree_count")
        for tree in trees:
            if not isinstance(tree, list) or not 1 <= len(tree) <= 10000:
                raise ValueError("browser_model_invalid_nodes")
            for index, node in enumerate(tree):
                if (not isinstance(node, list) or len(node) != len(NODE_LAYOUT)
                        or not _finite_number(node[0]) or not _finite_number(node[2])
                        or not all(type(node[i]) is int for i in (1, 3, 4, 5, 6))
                        or node[3] not in (0, 1) or node[6] not in (0, 1)):
                    raise ValueError("browser_model_invalid_node_fields")
                if not node[6] and (not 0 <= node[1] < FULL_FEATURE_COUNT
                        or not index < node[4] < len(tree) or not index < node[5] < len(tree)):
                    raise ValueError("browser_model_invalid_node_children_or_feature")
        return cls(tuple(columns), cfg, ranges, dict(policy), dict(audit),
                   payload["source_artifact_sha256"], np.asarray(medians, dtype=float),
                   float(inference["baseline"]), tuple(tuple(tuple(node) for node in tree) for tree in trees))

    def _matrix(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        expected = set(self.columns)
        for row in rows:
            absent, extra = expected - set(row), set(row) - expected
            if absent or extra:
                raise ValueError(f"exact_feature_schema_required: missing={sorted(absent)}; unexpected={sorted(extra)}")
        matrix = np.asarray([[row[name] for name in self.columns] for row in rows], dtype=float)
        if np.isinf(matrix).any():
            raise ValueError("infinite_demo_features")
        return matrix

    def _scores(self, matrix: np.ndarray) -> np.ndarray:
        # SimpleImputer's published statistics are all finite; keep all 194
        # columns, including empty training columns, in their original order.
        matrix = np.where(np.isnan(matrix), self._imputer_statistics, matrix)
        raw = np.full(len(matrix), self._baseline, dtype=float)
        # Match sklearn's iteration order. Summing leaf values in one vectorized
        # reduction could change rounding, so each tree is accumulated in turn.
        for tree in self._trees:
            for row_index, row in enumerate(matrix):
                index = 0
                while not tree[index][6]:
                    node = tree[index]
                    value = row[node[1]]
                    go_left = bool(node[3]) if np.isnan(value) else value <= node[2]
                    index = node[4] if go_left else node[5]
                raw[row_index] += tree[index][0]
        scores = np.empty_like(raw)
        positive = raw >= 0
        scores[positive] = 1. / (1. + np.exp(-raw[positive]))
        exponent = np.exp(raw[~positive])
        scores[~positive] = exponent / (1. + exponent)
        return scores

    def predict_features_many(self, rows: Sequence[dict[str, float]]) -> list[dict]:
        if not rows:
            return []
        scores = self._scores(self._matrix(rows))
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
            outputs.append({"score": float(score), "current_min_clearance": current,
                "cv_min_clearance": float(features["cv_min_clearance"]),
                "current_eligible": eligible, "score_applicable": eligible,
                "feature_count": len(self.columns), "score_semantics": SCORE_SEMANTICS,
                "outside_training_range_features": outside_range,
                "outside_training_central98_features": outside_central,
                "range_check_is_validated_ood_detector": False})
        return outputs

    def predict_features(self, features: dict[str, float]) -> dict:
        return self.predict_features_many([features])[0]

    def features_from_scene(self, scene: dict) -> dict[str, float]:
        features, _ = extract_features(scene, self.protocol)
        return features

    def predict_scene(self, scene: dict) -> dict:
        return self.predict_features(self.features_from_scene(scene))

    def predict_scenes(self, scenes: Sequence[dict]) -> list[dict]:
        return self.predict_features_many([self.features_from_scene(scene) for scene in scenes])


def load_browser_model(path: str | Path, *, expected_sha256: str | None = None) -> BrowserFrozenModel:
    """Read inert JSON and optionally check bytes against a pinned manifest."""
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("browser_model_sha256_mismatch")
    return BrowserFrozenModel.from_dict(json.loads(raw))
