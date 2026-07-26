from __future__ import annotations

import unittest

import numpy as np

from two_timescale import LearningRate, Recursion, VectorField
from two_timescale.gmm import OnlineGaussianMixture
from two_timescale.ngpca import NeuralGasPCA


class RecursionSmokeTests(unittest.TestCase):
    def test_vector_field_can_be_scaled_by_a_control_recursion(self) -> None:
        control = Recursion(0.5, VectorField(lambda state, obs, ctx: 0.1, name="constant"), step_size=1.0)
        field = VectorField(lambda state, obs, ctx: obs - state, name="tracking") * control
        tracked = Recursion(np.array([0.0]), field, step_size=LearningRate(1.0))

        tracked.advance(np.array([2.0]))
        self.assertTrue(np.allclose(tracked.state, np.array([1.0])))

        control.advance(None)
        tracked.advance(np.array([2.0]))
        self.assertTrue(np.allclose(tracked.state, np.array([1.6])))

    def test_ngpca_smoke(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(64, 2))
        model = NeuralGasPCA(4, rng=rng).initialize(data).fit(data, iterations=20)
        self.assertEqual(model.centers().shape, (4, 2))
        self.assertTrue(np.isfinite(model.centers()).all())

    def test_gmm_smoke(self) -> None:
        rng = np.random.default_rng(1)
        data = np.vstack([
            rng.normal(loc=-2.0, scale=0.4, size=(64, 2)),
            rng.normal(loc=2.0, scale=0.4, size=(64, 2)),
        ])
        model = OnlineGaussianMixture(2, 2, learning_rate=LearningRate(0.05), rng=rng).fit(data, iterations=100)
        weights, means, covariances = model.parameters()
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(means.shape, (2, 2))
        self.assertEqual(covariances.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(means).all())


if __name__ == "__main__":
    unittest.main()
