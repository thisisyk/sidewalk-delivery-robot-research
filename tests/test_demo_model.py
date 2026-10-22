"""Inference boundary tests; synthetic fixtures are not performance evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from delivery_risk.demo_model import FrozenDemoModel, load_frozen_demo_model
from delivery_risk.samples import Protocol, extract_features, feature_columns


def fixture():
    def state(x, vx):
        return dict(valid=True, x=float(x), y=0., length=4., width=2.,
                    heading=0., vx=float(vx), vy=0.)
    return {"current_time_index": 10, "sdc_track_index": 0,
            "timestamps": [i * .1 for i in range(41)], "map_features": [],
            "tracks": [{"id": 0, "type": 1,
                        "states": [state(i * .5, 5.) for i in range(41)]},
                       {"id": 1, "type": 1,
                        "states": [state(20., 0.) for _ in range(41)]}]}


class CapturingEstimator:
    def __init__(self):
        self.columns = None
    def predict_proba(self, matrix):
        self.columns = tuple(matrix.columns)
        probability = np.clip(matrix["ego_speed"].to_numpy(float) / 20, 0, 1)
        return np.column_stack([1 - probability, probability])


def adapter():
    features, _ = extract_features(fixture(), Protocol())
    columns = tuple(feature_columns(features, "full"))
    return FrozenDemoModel(CapturingEstimator(), columns, Protocol(), {},
                           {"cutoff": .2, "tie_probability": 1., "capacity": 20}, {})


class DemoBoundaryTests(unittest.TestCase):
    def test_feature_order_is_frozen_in_single_and_batch_inference(self):
        model = adapter()
        features = model.features_from_scene(fixture())
        reversed_features = dict(reversed(list(features.items())))
        values = model.predict_features_many([features, reversed_features])
        self.assertEqual(model.estimator.columns, model.columns)
        self.assertEqual(len(model.columns), 194)
        self.assertEqual(values[0]["score"], values[1]["score"])

    def test_future_labels_or_extra_features_cannot_enter_estimator(self):
        model = adapter()
        features = model.features_from_scene(fixture())
        features["label_2m"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected=.*label_2m"):
            model.predict_features(features)
        features.pop("label_2m")
        features.pop("ego_speed")
        with self.assertRaisesRegex(ValueError, "missing=.*ego_speed"):
            model.predict_features(features)

    def test_future_can_be_removed_or_corrupted_without_changing_prediction(self):
        model = adapter()
        full = fixture()
        corrupt = copy.deepcopy(full)
        for track in corrupt["tracks"]:
            for state in track["states"][11:]:
                state.update(x=999999., y=-888888., valid=False)
        past_only = copy.deepcopy(full)
        past_only["timestamps"] = past_only["timestamps"][:11]
        for track in past_only["tracks"]:
            track["states"] = track["states"][:11]
        self.assertEqual(model.predict_scene(full), model.predict_scene(corrupt))
        self.assertEqual(model.predict_scene(full), model.predict_scene(past_only))

    def test_current_event_score_is_marked_outside_the_study_cohort(self):
        model = adapter()
        scene = fixture()
        for state in scene["tracks"][1]["states"]:
            state["x"] = 10.
        result = model.predict_scene(scene)
        self.assertFalse(result["current_eligible"])
        self.assertFalse(result["score_applicable"])
        self.assertEqual(result["score"], .25)

    def test_unapproved_model_reselection_is_rejected_before_fit(self):
        model = adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feature_columns.json").write_text(json.dumps({"full": model.columns}), encoding="utf-8")
            (root / "primary_selection.json").write_text(json.dumps({"model": "random_forest"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected_frozen_primary"):
                load_frozen_demo_model(root / "not_even_read.csv", results_dir=root)


if __name__ == "__main__":
    unittest.main()
