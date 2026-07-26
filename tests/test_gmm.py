from __future__ import annotations

import unittest

import numpy as np

from two_timescale.core import LearningRate
from two_timescale.gmm import GMMMoments, OnlineGaussianMixture, moments_to_parameters


class GaussianMixtureTests(unittest.TestCase):
    def test_moments_to_parameters_normalizes_weights_and_regularizes_covariances(self) -> None:
        state = GMMMoments(
            mass=np.array([2.0, 1.0]),
            first=np.array([[2.0, 0.0], [0.0, 1.0]]),
            second=np.array([
                [[3.0, 0.0], [0.0, 2.0]],
                [[2.0, 0.5], [0.5, 3.0]],
            ]),
        )

        weights, means, covariances = moments_to_parameters(state, regularization=1e-3)

        np.testing.assert_allclose(weights, np.array([2.0 / 3.0, 1.0 / 3.0]))
        np.testing.assert_allclose(means, np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(covariances.shape, (2, 2, 2))
        for covariance in covariances:
            np.testing.assert_allclose(covariance, covariance.T)
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))

    def test_gmm_smoke_matches_empirical_mixture_moments(self) -> None:
        rng = np.random.default_rng(1)
        data = np.vstack([
            rng.normal(loc=-2.0, scale=0.4, size=(64, 2)),
            rng.normal(loc=2.0, scale=0.4, size=(64, 2)),
        ])

        model = OnlineGaussianMixture(2, 2, learning_rate=LearningRate(0.05), rng=rng).fit(data, iterations=100)
        weights, means, covariances = model.parameters()

        fitted_mean = weights @ means
        fitted_second_moment = sum(
            weight * (covariance + np.outer(mean, mean))
            for weight, mean, covariance in zip(weights, means, covariances)
        )
        fitted_covariance = fitted_second_moment - np.outer(fitted_mean, fitted_mean)

        empirical_mean = data.mean(axis=0)
        empirical_covariance = np.cov(data, rowvar=False, bias=True)

        self.assertEqual(weights.shape, (2,))
        self.assertEqual(means.shape, (2, 2))
        self.assertEqual(covariances.shape, (2, 2, 2))
        self.assertTrue(np.isfinite(means).all())

        np.testing.assert_allclose(fitted_mean, empirical_mean, atol=0.25)
        np.testing.assert_allclose(fitted_covariance, empirical_covariance, rtol=0.1, atol=0.25)

        for covariance in covariances:
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))


if __name__ == "__main__":
    unittest.main()
