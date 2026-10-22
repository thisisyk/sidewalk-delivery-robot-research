"""Independent fallback triggers, queue conservation and bounded retries."""
import unittest
from unittest.mock import patch

from delivery_risk.simulation import state
from delivery_risk.sidewalk_simulation import (FALLBACK_RULES, SidewalkConfig,
    assistance_request_reason, simulate_sidewalk)
from test_sidewalk_simulation import straight_world


class UnscorableModel:
    def features_from_scene(self, scene):
        if len(scene["tracks"]) == 1:
            raise ValueError("no_current_neighbors")
        return {"test_only": 1.}
    def predict_features_many(self, rows):
        return [{"score": .001, "score_applicable": False,
                 "outside_training_range_features": []} for row in rows]


def candidate(**changes):
    robot = {"done": False, "requested": False, "score": None, "eligible": False,
        "blocked_s": 5., "stalled_s": 5., "robot_blockage_s": 0., "request_count": 0,
        "next_request_allowed_t": 0., "consecutive_planning_failures": 0}
    robot.update(changes)
    return robot


class SidewalkFallbackTests(unittest.TestCase):
    def test_ineligible_or_low_score_stall_still_requests(self):
        cfg = SidewalkConfig()
        self.assertEqual(assistance_request_reason(candidate(), "learned_with_fallback", cfg, 10.), "persistent_stall")
        self.assertEqual(assistance_request_reason(candidate(score=.01, eligible=True), "learned_with_fallback", cfg, 10.), "persistent_stall")
        self.assertEqual(assistance_request_reason(candidate(stalled_s=2., robot_blockage_s=2.), "learned_with_fallback", cfg, 10.), "robot_blockage")
        self.assertEqual(assistance_request_reason(candidate(stalled_s=.1, consecutive_planning_failures=2), "learned_with_fallback", cfg, 30.), "repeated_planning_failure")

    def test_short_wait_pending_and_cooldown_do_not_generate_requests(self):
        cfg = SidewalkConfig()
        self.assertIsNone(assistance_request_reason(candidate(stalled_s=4.9), "learned_with_fallback", cfg, 10.))
        self.assertIsNone(assistance_request_reason(candidate(), "learned_with_fallback", cfg, 10., pending=True))
        self.assertIsNone(assistance_request_reason(candidate(next_request_allowed_t=25.), "learned_with_fallback", cfg, 10.))
        self.assertIsNone(assistance_request_reason(candidate(request_count=3), "learned_with_fallback", cfg, 100.))

    def test_disabling_fallback_retains_legacy_score_gate(self):
        cfg = SidewalkConfig(fallback_enabled=False)
        self.assertIsNone(assistance_request_reason(candidate(), "learned", cfg, 100.))
        self.assertIsNone(assistance_request_reason(candidate(), "learned_with_fallback", cfg, 100.))
        self.assertEqual(assistance_request_reason(candidate(score=.999, eligible=True), "learned", cfg, 100.), "model_threshold")
        self.assertIsNone(assistance_request_reason(candidate(score=.999, eligible=True, requested=True), "learned", cfg, 100.))

    def test_actual_queue_retries_are_bounded_and_never_leave_walkway(self):
        world = straight_world()
        world["obstacles"] = [{"id": 900, "kind": "trash", **state(14, 2.5, length=3, width=6)}]
        with patch("delivery_risk.sidewalk_simulation.make_sidewalk_world", return_value=world), patch("delivery_risk.sidewalk_simulation.sidewalk_actors", return_value=[]):
            # Isolate retry behavior with the old handler: the new explicit
            # operator module correctly escalates this full blockage onsite.
            run = simulate_sidewalk(UnscorableModel(), SidewalkConfig(fleet=6, duration_s=100, service_s=1, assistance_module_enabled=False), compare=False)
            legacy = simulate_sidewalk(UnscorableModel(), SidewalkConfig(fleet=6, duration_s=100, service_s=1, fallback_enabled=False), compare=False)
        self.assertGreater(run["summary"]["fallback_requests"], 0)
        self.assertEqual(run["summary"]["model_requests"], 0)
        self.assertGreater(run["summary"]["retry_requests"], 0)
        self.assertEqual(legacy["summary"]["requests"], 0)
        self.assertEqual(legacy["summary"]["policy"], "learned")
        self.assertEqual(run["summary"]["completed"], 0)
        self.assertEqual(run["summary"]["off_walkway_frames"], 0)
        self.assertEqual(run["summary"]["static_contact_frames"], 0)
        self.assertLessEqual(run["summary"]["max_requests_per_robot"], 3)
        self.assertLessEqual(run["summary"]["max_queue_length"], 2)
        self.assertGreater(run["summary"]["planning_failure_attempts"], 0)
        per_robot = {}
        request_times = []
        for event in run["events"]:
            if "request_reason" not in event:
                continue
            self.assertFalse(event["eligible"])
            self.assertIsNone(event["score"])
            per_robot.setdefault(event["id"], []).append(event["t"])
            request_times.append(event["t"])
        for times in per_robot.values():
            self.assertLessEqual(len(times), 3)
            for before, after in zip(times, times[1:]):
                self.assertGreaterEqual(after - before, FALLBACK_RULES["retry_cooldown_after_service_s"])
        for before, after in zip(request_times, request_times[1:]):
            self.assertGreaterEqual(after - before + 1e-8, FALLBACK_RULES["minimum_fleet_request_interval_s"])
        for frame in run["frames"]:
            queued = [q["id"] for q in frame["queue"]]
            serving = [o["vehicle_id"] for o in frame["operators"] if o["vehicle_id"] is not None]
            self.assertEqual(len(queued), len(set(queued)))
            self.assertEqual(len(serving), len(set(serving)))
            self.assertTrue(set(queued).isdisjoint(serving))
            for robot in frame["vehicles"]:
                if robot["id"] in queued + serving:
                    self.assertEqual(robot["blocked_s"], 0.)


if __name__ == "__main__":
    unittest.main()
