"""A simulated operator cannot authorize shortcuts or fake successful motion."""
import unittest

from delivery_risk.assistance import plan_assistance, new_incident, record_incident_state


class AssistanceTests(unittest.TestCase):
    def setUp(self):
        self.robot = {"id": 0, "x": 5., "y": 2., "score": None, "eligible": False, "request_count": 1}

    def test_invalid_score_does_not_prevent_review_or_an_incident(self):
        incident = new_incident(self.robot, "prolonged_stall", 5.)
        route = object()
        decision = plan_assistance(self.robot, {}, [], [], lambda *args: route)
        self.assertIs(decision.route, route)
        self.assertIsNone(incident["score_at_request"])
        self.assertFalse(incident["eligible_at_request"])
        self.assertEqual(incident["execution_state"], "queued")

    def test_no_planned_route_stays_unresolved_and_requires_onsite_help(self):
        decision = plan_assistance(self.robot, {}, [], [], lambda *args: None)
        self.assertEqual(decision.action, "onsite_required")
        self.assertIsNone(decision.route)
        self.assertIsNotNone(decision.plan_failure)
        self.assertEqual((self.robot["x"], self.robot["y"]), (5., 2.))

    def test_visible_moving_pedestrian_causes_timed_yield_not_a_teleport(self):
        actor = {"x": 6., "y": 2., "type": 2, "vx": 1., "vy": 0.}
        def forbidden(*args):
            self.fail("Timed yield must not call route planner")
        decision = plan_assistance(self.robot, {}, [], [actor], forbidden)
        self.assertEqual(decision.action, "hold_then_resume")
        self.assertEqual(decision.hold_s, 3.)
        self.assertIsNone(decision.route)

    def test_nearby_robot_calls_planner_instead_of_treating_all_blocks_as_pedestrians(self):
        people = [{"x": 6., "y": 2., "type": 2, "vx": 1., "vy": 0.},
                  {"x": 5.5, "y": 2., "type": 1, "kind": "delivery_robot"}]
        received = []
        def planner(robot, world, obstacles):
            received.extend(obstacles)
            return "route"
        known = [{"id": 900}]
        decision = plan_assistance(self.robot, {}, known, people, planner)
        self.assertEqual(decision.action, "replan_walkway")
        self.assertEqual(received, known+[people[-1]])

    def test_operator_plans_around_visible_stopped_robots_without_freezing_moving_or_hidden_actors(self):
        stopped = {"id": 1, "kind": "delivery_robot", "x": 6., "y": 2., "vx": 0., "vy": 0.}
        moving = {**stopped, "id": 2, "vx": .5}
        own = {**stopped, "id": 0}
        supplied = [{"id": 900}]
        received = []
        def planner(robot, world, obstacles):
            received.extend(obstacles)
            return "route"
        plan_assistance(self.robot, {}, supplied, [stopped, moving, own], planner, "replan_walkway")
        self.assertEqual([a["id"] for a in received], [900, 1])
        self.assertEqual(supplied, [{"id": 900}])

    def test_user_selected_onsite_action_does_not_call_planner_or_move_robot(self):
        decision = plan_assistance(self.robot, {}, [], [], lambda *args: self.fail("No remote plan"), "onsite_required")
        self.assertEqual(decision.action, "onsite_required")
        self.assertIsNone(decision.route)

    def test_incident_states_are_timestamped_and_do_not_fake_success_at_command_issue(self):
        incident = new_incident(self.robot, "prolonged_stall", 5.)
        record_incident_state(incident, 6., "reviewing", "review")
        record_incident_state(incident, 21., "route_executing", "command issued")
        self.assertNotEqual(incident["execution_state"], "completed")
        with self.assertRaisesRegex(ValueError, "monotonic"):
            record_incident_state(incident, 20., "completed", "invalid earlier result")


if __name__ == "__main__":
    unittest.main()
