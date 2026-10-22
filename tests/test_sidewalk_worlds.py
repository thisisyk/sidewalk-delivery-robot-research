"""Traffic-space invariants for synthetic sidewalk scenarios, not safety claims."""
import math
import unittest

from delivery_risk.geometry import clearance_many
from delivery_risk.sidewalk_worlds import (
    SIDEWALK_CATALOG, make_sidewalk_world, sidewalk_actors,
    point_in_walkable, footprint_in_walkable, footprint_in_road,
)


class SidewalkWorldTests(unittest.TestCase):
    def test_full_robot_footprint_stays_in_sidewalk_crosswalk_union(self):
        for key in SIDEWALK_CATALOG:
            with self.subTest(scenario=key):
                world = make_sidewalk_world(key)
                self.assertGreaterEqual(len(world["routes"]), 3)
                robots, buildings = [], []
                for route in world["routes"]:
                    for a, b in zip(route, route[1:]):
                        heading = math.atan2(b[1]-a[1], b[0]-a[0])
                        count = max(2, math.ceil(math.dist(a, b)/.1))
                        for i in range(count):
                            robot = {"x": a[0]+(b[0]-a[0])*i/(count-1),
                                     "y": a[1]+(b[1]-a[1])*i/(count-1),
                                     "length": .9, "width": .65, "heading": heading}
                            self.assertTrue(footprint_in_walkable(robot, world), (key, robot))
                            robots.extend([robot]*len(world["buildings"]))
                            buildings.extend(world["buildings"])
                self.assertTrue((clearance_many(robots, buildings) > 0).all())

    def test_all_dynamic_actors_obey_their_whole_footprint_domain(self):
        for key in SIDEWALK_CATALOG:
            with self.subTest(scenario=key):
                world = make_sidewalk_world(key)
                static = world["buildings"]+world["obstacles"]
                actors, obstacles = [], []
                for step in range(900):
                    for actor in sidewalk_actors(step*.2, key, density=3.):
                        legal = footprint_in_road(actor, world) if actor["type"] == 1 else footprint_in_walkable(actor, world)
                        self.assertTrue(legal, (key, step*.2, actor))
                        actors.extend([actor]*len(static))
                        obstacles.extend(static)
                self.assertTrue((clearance_many(actors, obstacles) > 0).all())

    def test_exact_union_check_rejects_gap_even_if_corners_are_inside(self):
        # Four corners are legal, yet the middle crosses a forbidden road gap.
        world = {"walkable_rectangles": [(0., 0., 10., 1.), (0., 3., 10., 4.)]}
        for point in ((4.75, 0.), (5.25, 0.), (4.75, 4.), (5.25, 4.)):
            self.assertTrue(point_in_walkable(point, world))
        spanning_gap = {"x": 5., "y": 2., "length": 4., "width": .5, "heading": math.pi/2}
        self.assertFalse(footprint_in_walkable(spanning_gap, world))
        world["walkable_rectangles"] = [(0., 0., 10., 4.)]
        self.assertTrue(footprint_in_walkable(spanning_gap, world))

    def test_road_center_is_not_a_robot_corridor_and_crosswalk_is(self):
        world = make_sidewalk_world("intersection")
        robot = {"x": 50., "y": 35., "length": .9, "width": .65, "heading": 0.}
        self.assertTrue(footprint_in_road(robot, world))
        self.assertFalse(footprint_in_walkable(robot, world))
        robot.update(y=44.5)
        self.assertTrue(footprint_in_walkable(robot, world))
        # Northern crosswalk crosses the vertical road; permission comes from
        # the explicit crosswalk, not from treating the carriageway as walkway.
        self.assertTrue(footprint_in_road(robot, world))
        robot.update(x=20., y=28.5)
        self.assertFalse(footprint_in_walkable(robot, world))

    def test_parked_and_moving_cars_cannot_occupy_sidewalk(self):
        for key in SIDEWALK_CATALOG:
            world = make_sidewalk_world(key)
            for obj in world["buildings"]+world["obstacles"]:
                if obj["kind"] == "parked_car":
                    self.assertTrue(footprint_in_road(obj, world))
                    self.assertFalse(footprint_in_walkable(obj, world))
                    self.assertEqual(obj["type"], 1)
                else:
                    self.assertNotIn("type", obj)
            for actor in sidewalk_actors(42., key):
                self.assertIn(actor["type"], (1, 2))
                self.assertNotIn("points", actor)
        with self.assertRaises(ValueError):
            make_sidewalk_world("intersection", domain="road")


if __name__ == "__main__":
    unittest.main()
