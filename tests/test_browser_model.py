"""Actual published weights retain inference under the browser JSON conversion."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from delivery_risk.browser_model import BrowserFrozenModel, load_browser_model
from delivery_risk.model_artifact import load_model_artifact, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_browser_model", ROOT / "scripts/export_browser_model.py")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def synthetic_scene(seed):
    """New independent past-only synthetic histories; no study records."""
    rng = np.random.default_rng(seed)
    times = [i * .1 for i in range(11)]
    heading = float(rng.uniform(-np.pi, np.pi))
    speed = float(rng.uniform(0, 12))
    ego_vx, ego_vy = speed * np.cos(heading), speed * np.sin(heading)
    def history(x, y, vx, vy, length, width, angle):
        return [dict(valid=True, x=float(x + (t-1)*vx), y=float(y + (t-1)*vy),
                     vx=float(vx), vy=float(vy), length=length, width=width, heading=float(angle))
                for t in times]
    tracks = [{"id": 0, "type": 1, "states": history(0., 0., ego_vx, ego_vy, .9, .65, heading)}]
    for index in range(1, 1 + int(rng.integers(1, 17))):
        angle, distance = rng.uniform(-np.pi, np.pi), rng.uniform(2, 45)
        kind = int(rng.integers(1, 4))
        velocity_angle, velocity = rng.uniform(-np.pi, np.pi), rng.uniform(0, 10)
        vx, vy = velocity*np.cos(velocity_angle), velocity*np.sin(velocity_angle)
        dimensions = {1: (4.5, 1.8), 2: (.6, .6), 3: (1.7, .6)}[kind]
        tracks.append({"id": index, "type": kind,
                       "states": history(distance*np.cos(angle), distance*np.sin(angle),
                                         vx, vy, *dimensions, velocity_angle)})
    return {"current_time_index": 10, "sdc_track_index": 0, "timestamps": times,
            "tracks": tracks, "map_features": [
                {"type": "crosswalk", "polygon": [[5., -3.], [8., -3.], [8., 5.], [5., 5.]]},
                {"type": "road_edge", "polyline": [[-20., -4.], [20., -4.]]}]}


class BrowserModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "demo/frozen_model.json"
        cls.payload = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.original = load_model_artifact(ROOT / "models/frozen_histgb_seed42.joblib", trusted=True)
        cls.browser = load_browser_model(cls.path)
        cls.zero = {name: 0. for name in cls.browser.columns}
        cls.zero.update(current_min_clearance=3., cv_min_clearance=3.)

    def assert_parity(self, rows):
        original = self.original.predict_features_many(rows)
        converted = self.browser.predict_features_many(rows)
        np.testing.assert_allclose([row["score"] for row in converted],
                                   [row["score"] for row in original], atol=1e-12, rtol=0)
        for a, b in zip(original, converted):
            for name in set(a) - {"score", "current_min_clearance", "cv_min_clearance"}:
                self.assertEqual(a[name], b[name])

    def test_exported_nodes_imputer_baseline_and_metadata_equal_actual_weights(self):
        regenerated = EXPORTER.export_payload(self.original, sha256_file(ROOT / "models/frozen_histgb_seed42.joblib"))
        self.assertEqual(regenerated, self.payload)
        self.assertEqual(len(self.payload["inference"]["trees"]), 100)
        self.assertEqual(len(self.browser.columns), 194)
        self.assertEqual(self.browser.fit_audit["training_rows"], 857)

    def test_random_extreme_and_missing_values_preserve_original_pipeline_scores(self):
        rng = np.random.default_rng(821)
        rows = [dict(zip(self.browser.columns, values)) for values in rng.normal(0, 25, (64, 194))]
        rows += [{name: value for name in self.browser.columns} for value in (-1e12, 0., 1e12, float("nan"))]
        for column in self.browser.columns:
            row = dict(self.zero)
            row[column] = float("nan")
            rows.append(row)
        self.assert_parity(rows)

    def test_exact_numeric_split_threshold_and_nextafter_neighbors_preserve_decisions(self):
        rows = []
        for tree in self.payload["inference"]["trees"][:4]:
            for node in tree:
                if node[6]:
                    continue
                for value in (np.nextafter(node[2], -np.inf), node[2], np.nextafter(node[2], np.inf)):
                    row = dict(self.zero)
                    row[self.browser.columns[node[1]]] = value
                    rows.append(row)
        self.assert_parity(rows)

    def test_new_multiactor_synthetic_scenes_reuse_original_past_only_features(self):
        scenes = [synthetic_scene(seed) for seed in range(25)]
        rows = []
        for scene in scenes:
            features = self.original.features_from_scene(scene)
            self.assertEqual(features, self.browser.features_from_scene(scene))
            rows.append(features)
        self.assert_parity(rows)
        np.testing.assert_allclose([p["score"] for p in self.browser.predict_scenes(scenes)],
                                   [p["score"] for p in self.original.predict_scenes(scenes)], atol=1e-12, rtol=0)
        self.assertEqual(self.browser.predict_scene(scenes[0]), self.browser.predict_scenes(scenes[:1])[0])

    def test_single_batch_empty_and_reordered_dicts_have_same_boundary(self):
        reversed_features = dict(reversed(list(self.zero.items())))
        self.assertEqual(self.browser.predict_features(self.zero), self.browser.predict_features(reversed_features))
        self.assertEqual(self.browser.predict_features_many([]), [])
        self.assertEqual(self.browser.predict_features_many([self.zero])[0], self.browser.predict_features(self.zero))

    def test_missing_schema_extra_schema_and_infinity_rejected(self):
        absent = dict(self.zero)
        absent.pop(self.browser.columns[0])
        for row in (absent, {**self.zero, "future_min_clearance": 1.}):
            with self.assertRaisesRegex(ValueError, "exact_feature_schema_required"):
                self.browser.predict_features(row)
        for value in (float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "infinite_demo_features"):
                self.browser.predict_features({**self.zero, "ego_speed": value})

    def test_feature_order_protocol_and_cyclic_tree_changes_rejected(self):
        changed = copy.deepcopy(self.payload)
        changed["columns"] = list(reversed(changed["columns"]))
        with self.assertRaisesRegex(ValueError, "feature_order_mismatch"):
            BrowserFrozenModel.from_dict(changed)
        changed = copy.deepcopy(self.payload)
        changed["protocol"]["future_steps"] = 40
        with self.assertRaisesRegex(ValueError, "protocol_differs"):
            BrowserFrozenModel.from_dict(changed)
        changed = copy.deepcopy(self.payload)
        changed["inference"]["trees"][0][0][4] = 0
        with self.assertRaisesRegex(ValueError, "invalid_node_children"):
            BrowserFrozenModel.from_dict(changed)

    def test_loader_checks_digest_and_accepts_only_data_json(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(load_browser_model(self.path, expected_sha256=digest).columns, self.browser.columns)
        with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
            load_browser_model(self.path, expected_sha256="0" * 64)
        self.assertEqual(BrowserFrozenModel.from_json(self.payload).columns, self.browser.columns)
        self.assertEqual(BrowserFrozenModel.from_json(self.path.read_text(encoding="utf-8")).columns, self.browser.columns)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "malicious.json"
            path.write_text("__import__('os').system('false')", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_browser_model(path)

    def test_browser_source_dependency_boundary_has_no_training_or_pickle_stack(self):
        for filename in ("browser_model.py", "samples.py", "geometry.py"):
            source = ROOT / "src/delivery_risk" / filename
            parsed = ast.parse(source.read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imports.append((node.module or "").split(".")[0])
            self.assertFalse(set(imports) & {"pandas", "sklearn", "pickle", "joblib", "scipy"})


if __name__ == "__main__":
    unittest.main()
