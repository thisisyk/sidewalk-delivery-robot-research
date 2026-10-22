"""Resource/causality checks; test doubles are never empirical evidence."""
import unittest
from delivery_risk.simulation import SimulationConfig, simulate


class TestOnlyModel:
    def __init__(self, score):
        self.score = score
        self.calls = 0
    def features_from_scene(self, scene):
        assert scene["current_time_index"] == 10
        assert len(scene["timestamps"]) == 11
        assert all(len(track["states"]) == 11 for track in scene["tracks"])
        self.calls += 1
        return {"test_only": 1}
    def predict_features_many(self, rows):
        return [{"score": self.score, "score_applicable": True,
                 "outside_training_range_features": []} for row in rows]


class SimulationTests(unittest.TestCase):
    def test_parameter_limits(self):
        for kwargs in ({"threshold": 2}, {"operators": 100}, {"fleet": 10000}, {"scenario": "unknown"}):
            with self.assertRaises(ValueError):
                SimulationConfig(**kwargs).validate()

    def test_scores_actually_gate_assistance_and_no_future_is_sent(self):
        low = TestOnlyModel(0.)
        a = simulate(low, SimulationConfig(fleet=6, duration_s=60), compare=False)
        high = TestOnlyModel(1.)
        b = simulate(high, SimulationConfig(fleet=6, duration_s=60), compare=False)
        self.assertGreater(low.calls, 0)
        self.assertGreater(high.calls, 0)
        self.assertEqual(a["summary"]["requests"], 0)
        self.assertGreater(b["summary"]["requests"], 0)
        self.assertGreaterEqual(b["summary"]["completed"], a["summary"]["completed"])
        self.assertTrue(b["metadata"]["synthetic"])

    def test_operator_conservation_and_single_request(self):
        result = simulate(TestOnlyModel(1.), SimulationConfig(fleet=6, duration_s=60), compare=False)
        self.assertLessEqual(result["summary"]["requests"], 6)
        self.assertLessEqual(result["summary"]["operator_busy_fraction"], 1.)
        for frame in result["frames"]:
            self.assertEqual(len(frame["operators"]), 1)
            serving = {o["vehicle_id"] for o in frame["operators"] if o["vehicle_id"] is not None}
            queued = {v["id"] for v in frame["queue"]}
            self.assertTrue(serving.isdisjoint(queued))
            self.assertEqual(len(frame["queue"]), len(queued))
        self.assertEqual(result["summary"]["completed"]+result["summary"]["unfinished"], 6)


if __name__ == "__main__":
    unittest.main()
