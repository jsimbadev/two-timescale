from __future__ import annotations

import unittest

import numpy as np

from two_timescale.core import (
    EMARecursion,
    LearningRate,
    Recursion,
    RecursionContext,
    ScheduledLearningRate,
    VectorField,
)


class CoreRecursionTests(unittest.TestCase):
    def test_ema_recursion_tracks_targets_with_per_step_smoothing(self) -> None:
        ema = EMARecursion(np.array([0.0, 2.0]), step_size=0.25)

        ema.advance(np.array([4.0, -2.0]))
        np.testing.assert_allclose(ema.state, np.array([1.0, 1.0]))

        ema.advance(np.array([3.0, 5.0]), step_size=0.5)
        np.testing.assert_allclose(ema.state, np.array([2.0, 3.0]))
        self.assertEqual(ema.iteration, 2)
        self.assertEqual(ema.recursion.iteration, 2)

    def test_vector_field_can_be_scaled_by_a_control_recursion(self) -> None:
        control = Recursion(0.5, VectorField(lambda state, obs, ctx: 0.1, name="constant"), step_size=1.0)
        field = VectorField(lambda state, obs, ctx: obs - state, name="tracking") * control
        tracked = Recursion(np.array([0.0]), field, step_size=LearningRate(1.0))

        tracked.advance(np.array([2.0]))
        np.testing.assert_allclose(tracked.state, np.array([1.0]))

        control.advance(None)
        tracked.advance(np.array([2.0]))
        np.testing.assert_allclose(tracked.state, np.array([1.6]))

    def test_scheduled_learning_rate_uses_recursion_context_iteration(self) -> None:
        schedule = ScheduledLearningRate(LearningRate(0.8), decay=0.5, power=2.0)

        self.assertEqual(schedule.value(), 0.8)
        self.assertAlmostEqual(schedule.value(RecursionContext(iteration=2)), 0.2)


if __name__ == "__main__":
    unittest.main()
