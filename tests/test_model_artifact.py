"""Integrity checks precede unpickling, and serialization preserves predictions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from delivery_risk.demo_model import FrozenDemoModel
from delivery_risk.model_artifact import load_model_artifact, save_model_artifact, sha256_file
from delivery_risk.samples import Protocol


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns = tuple(["current_min_clearance", "cv_min_clearance"] +
                            [f"synthetic_feature_{i}" for i in range(192)])
        rng = np.random.default_rng(42)
        cls.frame = pd.DataFrame(rng.normal(size=(32, 194)), columns=cls.columns)
        cls.frame.loc[0, "cv_min_clearance"] = np.nan
        estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingClassifier(max_iter=3, min_samples_leaf=2,
                max_leaf_nodes=3, early_stopping=False, random_state=42))
        estimator.fit(cls.frame, (cls.frame.iloc[:, 0] > 0).astype(int))
        audit = {"model": "hist_gradient_boosting", "seed": 42, "ablation": "full",
            "threshold_m": 2, "feature_count": 194, "fit_splits": ["train"],
            "development_rows_used_for_fit": 0, "evaluation_rows_used_for_fit": 0,
            "selection_repeated": False,
            "feature_order_sha256": hashlib.sha256(json.dumps(cls.columns).encode()).hexdigest()}
        cls.model = FrozenDemoModel(estimator, cls.columns, Protocol(),
            {name: {"min": -10., "q01": -5., "q99": 5., "max": 10.} for name in cls.columns},
            {"cutoff": .5, "tie_probability": 1., "capacity": 20, "nominal_dev_budget": .1}, audit)

    def test_serialization_keeps_exact_predictions_and_order_with_missing_features(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixed.joblib"
            manifest = save_model_artifact(self.model, target)
            loaded = load_model_artifact(target, trusted=True,
                                        expected_sha256=manifest["artifact"]["sha256"])
            self.assertEqual(loaded.columns, self.columns)
            np.testing.assert_array_equal(loaded.estimator.predict_proba(self.frame),
                                          self.model.estimator.predict_proba(self.frame))
            self.assertNotIn("samples", joblib.load(target))

    def test_explicit_trust_is_required_before_any_unpickle(self):
        with patch("delivery_risk.model_artifact.joblib.load") as unpickle:
            with self.assertRaisesRegex(PermissionError, "explicitly_trusted"):
                load_model_artifact("missing.joblib")
            unpickle.assert_not_called()

    def test_tampered_model_is_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixed.joblib"
            save_model_artifact(self.model, target)
            raw = bytearray(target.read_bytes())
            raw[len(raw) // 2] ^= 1
            target.write_bytes(raw)
            with patch("delivery_risk.model_artifact.joblib.load") as unpickle:
                with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
                    load_model_artifact(target, trusted=True)
                unpickle.assert_not_called()

    def test_explicit_digest_cannot_disagree_with_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixed.joblib"
            save_model_artifact(self.model, target)
            with self.assertRaisesRegex(ValueError, "expected_sha256_mismatch"):
                load_model_artifact(target, expected_sha256="0" * 64, trusted=True)

    def test_digest_is_required_when_manifest_is_absent(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixed.joblib"
            save_model_artifact(self.model, target)
            target.with_suffix(".manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "requires_sha256"):
                load_model_artifact(target, trusted=True)
            loaded = load_model_artifact(target, expected_sha256=sha256_file(target), trusted=True)
            np.testing.assert_array_equal(loaded.estimator.predict_proba(self.frame),
                                          self.model.estimator.predict_proba(self.frame))

    def test_changed_license_is_rejected_before_unpickle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "licenses").mkdir()
            license_path = root / "licenses" / "NOTICE.txt"
            license_path.write_text("original notice", encoding="utf-8")
            target = root / "models" / "fixed.joblib"
            save_model_artifact(self.model, target, supporting_files=[license_path])
            license_path.write_text("changed notice", encoding="utf-8")
            with patch("delivery_risk.model_artifact.joblib.load") as unpickle:
                with self.assertRaisesRegex(ValueError, "supporting_file_sha256_mismatch"):
                    load_model_artifact(target, trusted=True)
                unpickle.assert_not_called()

    def test_changed_feature_order_is_rejected_even_with_recomputed_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixed.joblib"
            save_model_artifact(self.model, target)
            payload = joblib.load(target)
            payload["columns"] = tuple(reversed(payload["columns"]))
            joblib.dump(payload, target, compress=3)
            target.with_suffix(".manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "feature_order_mismatch"):
                load_model_artifact(target, expected_sha256=sha256_file(target), trusted=True)

    def test_recorded_sklearn_version_is_checked_before_unpickle(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "fixed.joblib"
            save_model_artifact(self.model, target)
            manifest_path = target.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["software"]["scikit_learn"] = "0.0.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("delivery_risk.model_artifact.joblib.load") as unpickle:
                with self.assertRaisesRegex(ValueError, "requires_recorded_scikit_learn"):
                    load_model_artifact(target, trusted=True)
                unpickle.assert_not_called()


class PublishedModelSmokeTests(unittest.TestCase):
    def test_public_weights_score_without_raw_or_prepared_records(self):
        root = Path(__file__).resolve().parents[1]
        target = root / "models/frozen_histgb_seed42.joblib"
        model = load_model_artifact(target, trusted=True)
        self.assertEqual(model.fit_audit["training_rows"], 857)
        self.assertEqual(len(model.columns), 194)
        features = {name: 0.0 for name in model.columns}
        features.update(current_min_clearance=3.0, cv_min_clearance=3.0)
        prediction = model.predict_features(features)
        self.assertTrue(prediction["score_applicable"])
        self.assertTrue(np.isfinite(prediction["score"]))
        self.assertGreaterEqual(prediction["score"], 0.0)
        self.assertLessEqual(prediction["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
