"""End-to-end assistance commands retain the actual robot and motion guards."""
import math
import unittest
from unittest.mock import patch

from delivery_risk.simulation import state
from delivery_risk.sidewalk_simulation import SidewalkConfig, simulate_sidewalk
from test_sidewalk_simulation import TestOnlyModel, straight_world


class AssistanceExecutionTests(unittest.TestCase):
    def blocked_run(self, action="auto", module=True):
        world = straight_world()
        world["obstacles"] = [{"id": 900, "kind": "trash", **state(14, 2.5, length=3, width=6)}]
        cfg = SidewalkConfig(fleet=6, duration_s=30, service_s=1,
            assistance_module_enabled=module, assistance_action=action)
        with patch("delivery_risk.sidewalk_simulation.make_sidewalk_world", return_value=world), patch("delivery_risk.sidewalk_simulation.sidewalk_actors", return_value=[]):
            return simulate_sidewalk(TestOnlyModel(), cfg, compare=False)

    def test_unscorable_blocked_robot_reaches_operator_and_stops_for_onsite_help(self):
        run = self.blocked_run()
        first = next(i for i in run["assistance_incidents"] if i["robot_id"] == 0)
        self.assertIsNone(first["score_at_request"])
        self.assertFalse(first["eligible_at_request"])
        self.assertIsNotNone(first["started_at"])
        self.assertEqual(first["action"], "onsite_required")
        self.assertIsNotNone(first["plan_failure"])
        self.assertEqual(first["execution_state"], "onsite_required")
        saved = [next(v for v in f["vehicles"] if v["id"] == 0)
                 for f in run["frames"] if f["t"] >= first["finished_at"]+.5]
        self.assertTrue(saved)
        self.assertTrue(all(v["status"] == "onsite_required" and not v["done"] for v in saved))
        self.assertTrue(all(math.hypot(v["x"]-saved[0]["x"], v["y"]-saved[0]["y"]) < 1e-8 for v in saved))
        self.assertEqual(run["summary"]["off_walkway_frames"], 0)
        self.assertEqual(run["summary"]["static_contact_frames"], 0)

    def test_wait_command_holds_position_then_rechecks_without_removing_obstacle(self):
        run = self.blocked_run("hold_then_resume")
        first = next(i for i in run["assistance_incidents"] if i["robot_id"] == 0)
        self.assertEqual(first["action"], "hold_then_resume")
        self.assertAlmostEqual(first["hold_until"]-first["finished_at"], 3.)
        during = [next(v for v in f["vehicles"] if v["id"] == 0)
                  for f in run["frames"] if first["finished_at"]+.1 < f["t"] < first["hold_until"]]
        self.assertGreaterEqual(len(during), 4)
        self.assertTrue(all(v["status"] == "holding" and math.hypot(v["vx"],v["vy"]) < 1e-8 for v in during))
        self.assertTrue(any(e["state"] == "blocked_after_instruction" and e["t"] >= first["hold_until"] for e in first["execution_events"]))
        self.assertEqual(run["summary"]["completed"], 0)
        self.assertEqual(run["summary"]["static_contact_frames"], 0)

    def test_user_selected_onsite_instruction_is_executed_as_a_stop(self):
        run = self.blocked_run("onsite_required")
        first = run["assistance_incidents"][0]
        self.assertEqual(first["decision_source"], "user_selected_simulation_command")
        self.assertEqual(first["action"], "onsite_required")
        self.assertIsNone(first["plan_failure"])
        self.assertEqual(first["route_points"], [])
        self.assertGreater(run["summary"]["onsite_required_robots"], 0)
        self.assertEqual(run["summary"]["completed"], 0)


if __name__ == "__main__":
    unittest.main()
