"""Hard separation of robot walking space and motor-road space."""
import math
import unittest
from unittest.mock import patch

from delivery_risk.geometry import clearance
from delivery_risk.simulation import state
from delivery_risk.sidewalk_simulation import (SidewalkConfig, plan_sidewalk_detour, simulate_sidewalk)
from delivery_risk.sidewalk_worlds import footprint_in_walkable, footprint_in_road


class TestOnlyModel:
    def features_from_scene(self, scene):
        assert len(scene["timestamps"]) == 11
        assert all(len(t["states"]) == 11 for t in scene["tracks"])
        if len(scene["tracks"]) == 1:
            raise ValueError("no_current_neighbors")
        return {"test_only": 1.}
    def predict_features_many(self, rows):
        return [{"score": 0., "score_applicable": True,
            "outside_training_range_features": []} for row in rows]


def straight_world():
    return {"width": 40, "height": 25, "roads": [{"points": [[0, 12.5], [40, 12.5]], "width": 15}],
        "walkways": [{"points": [[0, 2.5], [40, 2.5]], "width": 5}], "crosswalks": [],
        "walkable_rectangles": [(0, 0, 40, 5)], "road_rectangles": [(0, 5, 40, 20)],
        "routes": [[[3, 2.5], [37, 2.5]]], "buildings": [], "obstacles": [], "map_features": []}


class SidewalkSimulationTests(unittest.TestCase):
    def test_road_domain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires_delivery_robot"):
            SidewalkConfig(domain="road").validate()

    def test_planner_cannot_use_adjacent_road_to_bypass_full_walkway_block(self):
        world = straight_world()
        barrier = {"id": 900, "kind": "barrier", **state(20, 2.5, length=3, width=6)}
        robot = {**state(5, 2.5, length=.9, width=.65), "goal": [35, 2.5]}
        self.assertIsNone(plan_sidewalk_detour(robot, world, [barrier]))

    def test_partial_garbage_can_be_bypassed_without_entering_road(self):
        world = straight_world()
        trash = {"id": 900, "kind": "trash", **state(20, 2.5, length=4, width=1)}
        robot = {**state(5, 2.5, length=.9, width=.65), "goal": [35, 2.5]}
        route = plan_sidewalk_detour(robot, world, [trash])
        self.assertIsNotNone(route)
        for i in range(math.ceil(route.total * 10) + 1):
            x, y, heading = route.pose(i / 10)
            body = state(x, y, length=.9, width=.65, heading=heading)
            self.assertTrue(footprint_in_walkable(body, world))
            self.assertFalse(footprint_in_road(body, world))
            self.assertGreater(clearance(body, trash), 0.)

    def test_entire_footprint_not_only_center_is_constrained(self):
        world = straight_world()
        self.assertFalse(footprint_in_walkable(state(10, 4.9, length=.9, width=.65), world))
        # Route deliberately heads across the motor road without a crosswalk.
        # The hard domain guard must stop the robot rather than honor this route.
        world["routes"] = [[[10, 2.5], [10, 12.5]]]
        with patch("delivery_risk.sidewalk_simulation.make_sidewalk_world", return_value=world), patch("delivery_risk.sidewalk_simulation.sidewalk_actors", return_value=[]):
            result = simulate_sidewalk(TestOnlyModel(), SidewalkConfig(fleet=6, duration_s=10), compare=False)
        self.assertEqual(result["summary"]["off_walkway_frames"], 0)
        self.assertGreater(result["summary"]["domain_guard_stop_steps"], 0)
        self.assertEqual(result["summary"]["completed"], 0)
        for frame in result["frames"]:
            for robot in frame["vehicles"]:
                self.assertEqual(robot["kind"], "delivery_robot")
                self.assertTrue(footprint_in_walkable(robot, world))

    def test_static_cars_are_included_in_off_road_audit(self):
        world = straight_world()
        # Intentionally invalid fixture proves audit does not hide static cars.
        world["obstacles"] = [{"id": 950, "kind": "parked_car", "type": 1,
                               **state(25, 2.5, length=4.5, width=1.8)}]
        with patch("delivery_risk.sidewalk_simulation.make_sidewalk_world", return_value=world), patch("delivery_risk.sidewalk_simulation.sidewalk_actors", return_value=[]):
            result = simulate_sidewalk(TestOnlyModel(), SidewalkConfig(fleet=6, duration_s=10), compare=False)
        self.assertEqual(result["summary"]["cars_off_road_frames"], 20)

    def test_full_blockage_remains_unfinished_without_static_penetration(self):
        world = straight_world()
        trash = {"id": 900, "kind": "trash", **state(14, 2.5, length=3, width=6)}
        world["obstacles"] = [trash]
        with patch("delivery_risk.sidewalk_simulation.make_sidewalk_world", return_value=world), patch("delivery_risk.sidewalk_simulation.sidewalk_actors", return_value=[]):
            result = simulate_sidewalk(TestOnlyModel(), SidewalkConfig(fleet=6, duration_s=15), compare=True)
        self.assertEqual(result["summary"]["off_walkway_frames"], 0)
        self.assertEqual(result["summary"]["static_contact_frames"], 0)
        self.assertEqual(result["summary"]["completed"], 0)
        baseline = next(row for row in result["policies"] if row["policy"] == "all")
        self.assertGreater(baseline["requests"], 0)
        self.assertEqual(baseline["completed"], 0)
        for frame in result["frames"]:
            for robot in frame["vehicles"]:
                self.assertGreater(clearance(robot, trash), 0.)


if __name__ == "__main__":
    unittest.main()
