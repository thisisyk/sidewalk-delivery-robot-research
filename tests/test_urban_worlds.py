"""Urban geometry invariants prevent visual routes through buildings.

These are simulator implementation checks, not evidence of real vehicle safety.
"""
import math
import unittest

from delivery_risk.geometry import clearance_many
from delivery_risk.urban_worlds import SCENARIO_CATALOG, make_urban_world, urban_actors


class UrbanWorldTests(unittest.TestCase):
    def test_routes_keep_the_whole_road_car_outside_buildings(self):
        for key in SCENARIO_CATALOG:
            with self.subTest(scenario=key):
                world = make_urban_world(key)
                self.assertGreaterEqual(len(world["routes"]), 3)
                cars, buildings = [], []
                for route in world["routes"]:
                    for a, b in zip(route, route[1:]):
                        heading = math.atan2(b[1]-a[1], b[0]-a[0])
                        count = max(2, math.ceil(math.dist(a, b)*2))
                        for i in range(count):
                            car = {"x": a[0]+(b[0]-a[0])*i/(count-1),
                                   "y": a[1]+(b[1]-a[1])*i/(count-1),
                                   "length": 4.5, "width": 1.8, "heading": heading}
                            cars.extend([car]*len(world["buildings"]))
                            buildings.extend(world["buildings"])
                self.assertTrue((clearance_many(cars, buildings) > .1).all())

    def test_analytical_actors_do_not_walk_or_drive_through_static_objects(self):
        for key in SCENARIO_CATALOG:
            with self.subTest(scenario=key):
                world = make_urban_world(key)
                static = world["buildings"]+world["obstacles"]
                actors, obstacles = [], []
                for step in range(720):
                    for actor in urban_actors(step*.25, key, density=3.):
                        actors.extend([actor]*len(static))
                        obstacles.extend(static)
                self.assertTrue((clearance_many(actors, obstacles) > 0).all())

    def test_non_vehicle_obstacles_cannot_silently_become_vehicle_tracks(self):
        for key in SCENARIO_CATALOG:
            world = make_urban_world(key)
            for item in world["buildings"]+world["obstacles"]:
                if item["kind"] == "parked_car":
                    self.assertEqual(item["type"], 1)
                else:
                    self.assertNotIn("type", item)
            actors = urban_actors(20., key)
            static_ids = {item["id"] for item in world["obstacles"]}
            self.assertTrue(static_ids.isdisjoint({a["id"] for a in actors}))
            self.assertTrue(all(a["type"] in (1, 2, 3) for a in actors))
            self.assertFalse(any("points" in a or "future" in a for a in actors))

    def test_deterministic_history_and_density_zero(self):
        self.assertEqual(urban_actors(-.1, "intersection"), [])
        self.assertEqual(urban_actors(50, "intersection", density=0), [])
        for key in SCENARIO_CATALOG:
            self.assertEqual(urban_actors(19.6, key, seed=7), urban_actors(19.6, key, seed=7))
            # An analytical current-state query does not mutate subsequent or
            # preceding queries, so an adapter can build real past observations.
            before = urban_actors(19.5, key, seed=7)
            urban_actors(100., key, seed=7)
            self.assertEqual(before, urban_actors(19.5, key, seed=7))
        a, b = make_urban_world("mixed"), make_urban_world("mixed")
        a["roads"][0]["width"] = -1
        self.assertGreater(b["roads"][0]["width"], 0)


if __name__ == "__main__":
    unittest.main()
