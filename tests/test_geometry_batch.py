"""Independent geometry invariants; fixtures are software tests, not research data."""
import unittest

import numpy as np

from delivery_risk.geometry import clearance, clearance_many


def rectangle(x, y, heading, length=4.0, width=2.0):
    return {"x": float(x), "y": float(y), "heading": float(heading),
            "length": float(length), "width": float(width)}


def rigid_transform(state, angle, translation):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    point = rotation @ np.array([state["x"], state["y"]]) + translation
    return {**state, "x": point[0], "y": point[1], "heading": state["heading"] + angle}


class BatchGeometryTests(unittest.TestCase):
    def test_randomized_scalar_agreement(self):
        rng = np.random.default_rng(723)
        a, b = [], []
        for index in range(240):
            xy = rng.uniform(-20, 20, 2)
            other = xy + rng.uniform(-3, 3, 2) if index % 2 else rng.uniform(-20, 20, 2)
            a.append(rectangle(*xy, rng.uniform(-np.pi, np.pi), *rng.uniform(.3, 8, 2)))
            b.append(rectangle(*other, rng.uniform(-np.pi, np.pi), *rng.uniform(.3, 8, 2)))
        expected = np.array([clearance(left, right) for left, right in zip(a, b)])
        np.testing.assert_allclose(clearance_many(a, b), expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(clearance_many(b, a), expected, atol=1e-10, rtol=1e-10)
        self.assertTrue(np.any(expected == 0))
        self.assertTrue(np.any(expected > 0))

    def test_rigid_transform_invariance(self):
        a = [rectangle(1, 2, .1), rectangle(-3, 4, -.7), rectangle(0, 0, 0)]
        b = [rectangle(12, -3, 1.4), rectangle(-2, 4, 1.3), rectangle(8, 8, .5)]
        expected = clearance_many(a, b)
        for angle in (0., .8, -2.7, 2 * np.pi):
            shifted_a = [rigid_transform(state, angle, [1234., -789.]) for state in a]
            shifted_b = [rigid_transform(state, angle, [1234., -789.]) for state in b]
            np.testing.assert_allclose(clearance_many(shifted_a, shifted_b), expected,
                                       atol=1e-9, rtol=1e-9)

    def test_parallel_rectangles_have_analytic_distance(self):
        angle = .73
        origin = rectangle(0, 0, angle)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        a, b, expected = [], [], []
        for dx, dy in ((0, 0), (4, 0), (4, 2), (7, 6), (-7, -6), (0, 5)):
            xy = rotation @ [dx, dy]
            a.append(origin)
            b.append(rectangle(*xy, angle))
            expected.append(np.hypot(max(abs(dx) - 4, 0), max(abs(dy) - 2, 0)))
        np.testing.assert_allclose(clearance_many(a, b), expected, atol=1e-10)

    def test_containment_and_touching_are_zero(self):
        a = [rectangle(0, 0, .8, 10, 10), rectangle(0, 0, 0)]
        b = [rectangle(.1, .2, -.5, .4, .7), rectangle(4, 0, 0)]
        np.testing.assert_allclose(clearance_many(a, b), [0, 0], atol=1e-12)

    def test_empty_mismatch_and_invalid_geometry(self):
        self.assertEqual(clearance_many([], []).shape, (0,))
        with self.assertRaises(ValueError):
            clearance_many([rectangle(0, 0, 0)], [])
        for bad in (rectangle(0, 0, 0, 0, 2), rectangle(np.nan, 0, 0)):
            with self.assertRaises(ValueError):
                clearance_many([bad], [rectangle(0, 0, 0)])


if __name__ == "__main__":
    unittest.main()
