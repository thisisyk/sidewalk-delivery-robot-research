"""Causality, unsupported classes and geometry checks for the urban toy world."""
import math
import copy
import unittest
from collections import deque
from unittest.mock import patch

from delivery_risk.geometry import clearance
from delivery_risk.simulation import state
from delivery_risk.urban_simulation import (ObservationHistory, Route, UrbanConfig,
    line_of_sight, plan_static_detour, simulate_urban)


def box(ident, x, y, length=4., width=2., kind="building", **more):
    return {"id": ident, "kind": kind, **state(x, y, length=length, width=width), **more}


class CheckingModel:
    """Test double for software checks only; no performance claims use it."""
    def __init__(self, score=0.):
        self.score, self.calls = score, 0
    def features_from_scene(self, scene):
        self.calls += 1
        assert len(scene["timestamps"]) == 11
        assert scene["current_time_index"] == 10
        assert all(len(t["states"]) == 11 and t["type"] in (1, 2, 3) for t in scene["tracks"])
        assert not any("future" in k for k in scene)
        if len(scene["tracks"]) == 1:
            raise ValueError("no_current_neighbors")
        return {"test_only": 1.}
    def predict_features_many(self, rows):
        return [{"score": self.score, "score_applicable": True,
                 "outside_training_range_features": []} for _ in rows]


class UrbanTests(unittest.TestCase):
    def test_building_los_and_rotated_box(self):
        ego, target = box(0, 0, 0), box(1, 20, 0)
        self.assertFalse(line_of_sight(ego, target, [box(2, 10, 0)]))
        self.assertTrue(line_of_sight(ego, target, [box(2, 10, 6)]))
        self.assertTrue(line_of_sight(ego, target, [box(2, 10, 0)], enabled=False))
        self.assertFalse(line_of_sight(ego, target, [box(2, 10, 0, heading=math.pi / 4)]))

    def test_new_and_reappearing_targets_never_receive_hidden_history(self):
        ego = box(0, 0, 0)
        actor = box(800, 20, 0, type=2)
        wall = box(700, 10, 0)
        obs = ObservationHistory()
        for _ in range(11):
            obs.update(ego, [actor], [wall])
        self.assertEqual(obs.visible_ids, [])
        self.assertEqual(obs.hidden_ids, [800])
        obs.update(ego, [actor], [])
        self.assertTrue(obs.records[800][-1]["valid"])
        self.assertFalse(obs.records[800][0]["valid"])
        for _ in range(11):
            obs.update(ego, [actor], [])
        self.assertTrue(obs.records[800][0]["valid"])
        obs.update(ego, [actor], [wall])
        obs.update(ego, [actor], [])
        self.assertFalse(obs.records[800][-2]["valid"])

    def test_unsupported_geometry_never_becomes_model_track(self):
        ego = box(0, 0, 0)
        ego["history"] = deque([state(0, 0) for _ in range(11)], maxlen=11)
        targets = [box(700, 20, 0, kind="building"), box(701, 22, 0, kind="trash"),
                   box(702, 24, 0, kind="barrier"), box(800, 25, 2, kind="parked_car", type=1)]
        obs = ObservationHistory(); obs.update(ego, targets, [])
        scene = obs.scene(ego, {"map_features": [{"type": "lane", "polyline": [[0, 0], [40, 0]]}]}, 10.)
        self.assertEqual([t["id"] for t in scene["tracks"]], [0, 800])
        self.assertEqual(scene["timestamps"][-1], 10.)
        self.assertEqual(scene["map_features"][0]["polyline"], [[0, 0], [40, 0]])

    def test_route_turn_updates_heading_and_rounds_corner(self):
        route = Route([[0, 0], [20, 0], [20, 20]])
        headings = [route.pose(s)[2] for s in (0, 18, 20, 30)]
        self.assertAlmostEqual(headings[0], 0.)
        self.assertGreater(headings[2], .2)
        self.assertAlmostEqual(headings[-1], math.pi / 2)
        # A rounded turn visits intermediate orientations, not a fixed heading.
        self.assertTrue(any(.1 < route.pose(s)[2] < 1.4 for s in range(16, 23)))

    def test_detour_stays_outside_known_buildings_and_trash(self):
        world = {"width": 60, "height": 30, "roads": [{"points": [[0, 15], [60, 15]], "width": 24}],
                 "buildings": [box(700, 30, 2, length=18, width=2)]}
        trash = box(900, 30, 15, length=6, width=4, kind="trash")
        ego = {**state(6, 15), "goal": [54, 15]}
        route = plan_static_detour(ego, world, [trash])
        self.assertIsNotNone(route)
        for s in range(int(route.total * 5) + 1):
            x, y, heading = route.pose(s / 5)
            body = state(x, y, heading=heading)
            self.assertGreater(clearance(body, trash), 0.)
            self.assertGreater(clearance(body, world["buildings"][0]), 0.)
        # A complete wall across the road must remain a failed route.
        wall = box(901, 30, 15, length=4, width=30, kind="barrier")
        self.assertIsNone(plan_static_detour(ego, world, [wall]))

    def test_geometry_guard_does_not_drive_through_static_obstacle(self):
        obstacle = box(900, 24, 15, length=4, width=20, kind="trash")
        world = {"width": 60, "height": 30, "roads": [{"points": [[0, 15], [60, 15]], "width": 24}],
            "routes": [[[5, 15], [55, 15]]], "buildings": [], "obstacles": [obstacle], "map_features": []}
        with patch("delivery_risk.urban_simulation.make_urban_world", return_value=world), patch("delivery_risk.urban_simulation.urban_actors", return_value=[]):
            result = simulate_urban(CheckingModel(0.), UrbanConfig(fleet=6, duration_s=12), compare=False)
        self.assertEqual(result["summary"]["physical_overlap_frames"], 0)
        self.assertEqual(result["summary"]["requests"], 0)
        self.assertGreater(result["summary"]["waiting_at_end"], 0)
        for frame in result["frames"]:
            for vehicle in frame["vehicles"]:
                self.assertGreater(clearance(vehicle, obstacle), 0.)

    def test_settings_are_bounded(self):
        for kwargs in ({"scenario": "bad"}, {"density": 99}, {"occlusion": "false"}, {"fleet": 1000}):
            with self.assertRaises(ValueError):
                UrbanConfig(**kwargs).validate()

    def test_different_future_actor_motion_cannot_change_prior_inputs(self):
        world = {"width": 60, "height": 30, "roads": [{"points": [[0, 15], [60, 15]], "width": 24}],
            "routes": [[[5, 15], [55, 15]]], "buildings": [], "obstacles": [], "map_features": []}
        class RecordingModel(CheckingModel):
            def __init__(self):
                super().__init__(0.)
                self.inputs = []
            def features_from_scene(self, scene):
                if scene["timestamps"][-1] <= 5.:
                    self.inputs.append(copy.deepcopy(scene))
                return super().features_from_scene(scene)
        results, inputs = [], []
        for changed_future in (False, True):
            def actors(t, scenario, seed, density):
                # Two worlds have identical observations through t=5 but
                # radically different traffic afterwards.
                y = 15 if changed_future and t > 5 else 23
                return [box(800, 25, y, length=.6, width=.6, kind="pedestrian", type=2)]
            model = RecordingModel()
            with patch("delivery_risk.urban_simulation.make_urban_world", return_value=world), patch("delivery_risk.urban_simulation.urban_actors", side_effect=actors):
                result = simulate_urban(model, UrbanConfig(fleet=6, duration_s=10), compare=False)
            inputs.append(model.inputs)
            results.append([frame for frame in result["frames"] if frame["t"] <= 5.])
        self.assertTrue(inputs[0])
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
