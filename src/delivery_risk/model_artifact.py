"""Publish and load the fixed study model without retaining training records.

Joblib uses pickle and can execute code while loading. A SHA256 check detects
changed bytes; it does not make an untrusted publisher safe. Only explicitly
trusted, self-published artifacts should be loaded with this module.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .demo_model import FrozenDemoModel, FULL_FEATURE_COUNT, SCORE_SEMANTICS
from .samples import Protocol


FORMAT = "delivery-risk-frozen-histgb-v1"
_FIT_KEYS = {
    "model", "seed", "ablation", "threshold_m", "feature_count", "input_sha256",
    "training_rows", "training_positive", "fit_splits",
    "development_rows_used_for_fit", "evaluation_rows_used_for_fit",
    "selection_repeated", "original_feature_list_sha256", "original_selection_sha256",
    "frozen_model_code_sha256", "sklearn_version", "feature_order_sha256",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__}


def _default_manifest(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _validate_payload(payload: Any) -> FrozenDemoModel:
    required = {"format", "estimator", "columns", "protocol",
                "training_feature_ranges", "dev_policy", "fit_audit", "software"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != FORMAT:
        raise ValueError("unsupported_model_artifact_schema")
    columns = payload["columns"]
    if (not isinstance(columns, (tuple, list)) or len(columns) != FULL_FEATURE_COUNT
            or any(not isinstance(name, str) for name in columns)
            or len(set(columns)) != FULL_FEATURE_COUNT):
        raise ValueError("model_requires_194_unique_ordered_features")
    estimator = payload["estimator"]
    if (not isinstance(estimator, Pipeline)
            or not isinstance(estimator.steps[-1][1], HistGradientBoostingClassifier)):
        raise ValueError("model_requires_hist_gradient_boosting_pipeline")
    check_is_fitted(estimator)
    if estimator.n_features_in_ != FULL_FEATURE_COUNT:
        raise ValueError("estimator_feature_count_mismatch")
    if hasattr(estimator, "feature_names_in_") and list(estimator.feature_names_in_) != list(columns):
        raise ValueError("estimator_feature_order_mismatch")
    audit = payload["fit_audit"]
    if not isinstance(audit, dict) or set(audit) - _FIT_KEYS:
        raise ValueError("unexpected_public_fit_audit_fields")
    expected = {"model": "hist_gradient_boosting", "seed": 42, "ablation": "full",
                "threshold_m": 2, "feature_count": FULL_FEATURE_COUNT,
                "fit_splits": ["train"], "development_rows_used_for_fit": 0,
                "evaluation_rows_used_for_fit": 0, "selection_repeated": False}
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("model_is_not_the_fixed_train_only_selection")
    order_hash = hashlib.sha256(json.dumps(tuple(columns)).encode()).hexdigest()
    if audit.get("feature_order_sha256") != order_hash:
        raise ValueError("model_feature_order_hash_mismatch")
    protocol = dict(payload["protocol"])
    protocol["thresholds_m"] = tuple(protocol["thresholds_m"])
    cfg = Protocol(**protocol)
    if cfg != Protocol():
        raise ValueError("model_protocol_differs_from_frozen_study")
    ranges = payload["training_feature_ranges"]
    if not isinstance(ranges, dict) or set(ranges) != set(columns):
        raise ValueError("model_feature_range_schema_mismatch")
    policy = payload["dev_policy"]
    if not isinstance(policy, dict) or set(policy) != {"cutoff", "tie_probability", "capacity", "nominal_dev_budget"}:
        raise ValueError("model_development_policy_schema_mismatch")
    return FrozenDemoModel(estimator, tuple(columns), cfg, ranges, policy, dict(audit))


def save_model_artifact(
    model: FrozenDemoModel, path: str | Path, *, manifest_path: str | Path | None = None,
    parity_audit: dict | None = None, supporting_files: list[str | Path] | None = None,
) -> dict:
    """Serialize the fitted model and necessary inference metadata only.

    This function does not fit, select, or evaluate a model. The publication
    script separately checks parity before and after this serialization step.
    The historical fit audit remains a train-only reproduction record.
    """
    target = Path(path)
    manifest_target = Path(manifest_path) if manifest_path else _default_manifest(target)
    if target.exists() or manifest_target.exists():
        raise FileExistsError("model_artifact_output_already_exists")
    payload = {"format": FORMAT, "estimator": model.estimator, "columns": tuple(model.columns),
               "protocol": asdict(model.protocol),
               "training_feature_ranges": model.training_feature_ranges,
               "dev_policy": model.dev_policy,
               "fit_audit": {key: value for key, value in model.fit_audit.items() if key in _FIT_KEYS},
               "software": runtime_versions()}
    _validate_payload(payload)
    supporting = []
    for value in supporting_files or []:
        source = Path(value).resolve()
        relative = os.path.relpath(source, target.parent.resolve()).replace("\\", "/")
        # The model bundle may reference its sibling licenses/ directory.
        if not source.is_relative_to(target.parent.resolve().parent):
            raise ValueError("supporting_file_outside_model_bundle_root")
        supporting.append({"path": relative, "sha256": sha256_file(source)})
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target, compress=3, protocol=5)
    manifest = {"format": FORMAT, "artifact": {"filename": target.name,
                "sha256": sha256_file(target), "bytes": target.stat().st_size},
                "software": payload["software"], "feature_count": len(model.columns),
                "feature_order_sha256": model.fit_audit["feature_order_sha256"],
                "training_provenance": "Re-fitted the frozen train-only selection; not an originally saved weight file.",
                "original_run_saved_weights": False, "raw_records_included": False,
                "fit_audit": payload["fit_audit"], "score_semantics": SCORE_SEMANTICS,
                "supporting_files": supporting,
                "serialization": "joblib/pickle; load only a trusted publisher's artifact with verified SHA256",
                "parity_audit": parity_audit}
    manifest_target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_model_artifact(
    path: str | Path, *, manifest_path: str | Path | None = None,
    expected_sha256: str | None = None, trusted: bool = False,
) -> FrozenDemoModel:
    """Load explicitly trusted weights after integrity and version checks.

    Obtain the expected digest through a trusted channel. A manifest downloaded
    alongside an unknown pickle is an integrity record, not proof of trust.
    """
    if trusted is not True:
        raise PermissionError("joblib_can_execute_code; explicitly_trusted_artifact_required")
    source = Path(path)
    manifest_source = Path(manifest_path) if manifest_path else _default_manifest(source)
    manifest = None
    if manifest_source.exists():
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest["artifact"]["filename"] != source.name:
            raise ValueError("model_manifest_schema_or_filename_mismatch")
        recorded = manifest["artifact"]["sha256"]
        if expected_sha256 is not None and recorded != expected_sha256:
            raise ValueError("model_manifest_expected_sha256_mismatch")
        expected_sha256 = recorded
        if source.stat().st_size != manifest["artifact"]["bytes"]:
            raise ValueError("model_artifact_size_mismatch")
        for entry in manifest.get("supporting_files", []):
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ":" in str(relative) or "\\" in str(relative):
                raise ValueError("invalid_supporting_file_path")
            supporting = (source.parent / str(relative)).resolve()
            if not supporting.is_relative_to(source.parent.resolve().parent):
                raise ValueError("supporting_file_outside_model_bundle_root")
            if sha256_file(supporting) != entry["sha256"]:
                raise ValueError("model_supporting_file_sha256_mismatch")
        software = manifest["software"]
        if software["scikit_learn"] != sklearn.__version__:
            raise ValueError("model_requires_recorded_scikit_learn_version")
        if software["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
            raise ValueError("model_requires_recorded_python_major_minor")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("model_requires_sha256_manifest_or_expected_digest")
    if sha256_file(source) != expected_sha256:
        raise ValueError("model_artifact_sha256_mismatch")
    # Unpickling occurs only after explicit trust and all pre-load checks.
    payload = joblib.load(source)
    model = _validate_payload(payload)
    if payload["software"]["scikit_learn"] != sklearn.__version__:
        raise ValueError("model_requires_recorded_scikit_learn_version")
    if manifest is not None and (manifest["feature_order_sha256"] != model.fit_audit["feature_order_sha256"]
                                 or manifest["feature_count"] != len(model.columns)):
        raise ValueError("model_manifest_feature_schema_mismatch")
    return model
