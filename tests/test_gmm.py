from __future__ import annotations

import unittest

import numpy as np

from two_timescale.core import LearningRate
from two_timescale.gmm import GMMRecursions, GMMMoments, GaussianMixtureRecursion, moments_to_parameters
from two_timescale.gmm_controls import AdaptiveGMMGain, SmoothedCovarianceControl, SmoothedMomentResidualControl
from two_timescale.recursions import GMMMomentRecursion


class GaussianMixtureTests(unittest.TestCase):
    def test_gmm_recursions_construct_named_moment_recursion(self) -> None:
        moments = GMMMoments(
            mass=np.array([0.5, 0.5]),
            first=np.array([[0.0, 0.0], [1.0, 1.0]]),
            second=np.tile(np.eye(2), (2, 1, 1)),
        )

        recursions = GMMRecursions.new(
            moments,
            step_size=LearningRate(0.2),
            regularization=1e-4,
        )

        self.assertIsInstance(recursions.moments, GMMMomentRecursion)
        self.assertEqual(recursions.moments.name, "gmm_moments")
        self.assertEqual(float(recursions.moments.step_size), 0.2)
        self.assertEqual(recursions.moments.regularization, 1e-4)

        moments.mass[:] = 0.0
        np.testing.assert_allclose(recursions.moments.state.mass, np.array([0.5, 0.5]))

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

        model = GaussianMixtureRecursion(2, 2, step_size=LearningRate(0.05), rng=rng)
        stream = data[[rng.integers(0, len(data)) for _ in range(100)]]

        model.advance(stream)
        weights, means, covariances = model.parameters()

        self.assertFalse(hasattr(model, "learning_rate"))
        self.assertFalse(hasattr(model, "fit"))
        self.assertFalse(hasattr(model, "predict"))
        self.assertFalse(hasattr(model, "predict_proba"))
        self.assertIsInstance(model.recursions.moments, GMMMomentRecursion)
        self.assertEqual(model.recursions.moments.iteration, 100)
        self.assertEqual(model.recursions.moments.recursion.name, "gmm_moments")

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

    def test_gmm_recursion_scores_observations_without_predict_api(self) -> None:
        rng = np.random.default_rng(3)
        data = np.vstack([
            rng.normal(loc=-1.0, scale=0.2, size=(8, 2)),
            rng.normal(loc=1.0, scale=0.2, size=(8, 2)),
        ])

        model = GaussianMixtureRecursion(2, 2, step_size=LearningRate(0.1), rng=rng)
        model.advance(data)

        responsibilities = model.responsibilities(data[:3])
        assignments = model.assignments(data[:3])

        self.assertEqual(responsibilities.shape, (3, 2))
        np.testing.assert_allclose(responsibilities.sum(axis=1), np.ones(3))
        self.assertEqual(assignments.shape, (3,))

    def test_gmm_recursion_accepts_responsibility_adaptive_gain_control(self) -> None:
        gain = AdaptiveGMMGain.new(n_components=2, initial_gain=0.05)
        model = GaussianMixtureRecursion(2, 2, step_size=gain, rng=np.random.default_rng(5))
        observation = np.array([0.5, -0.25])

        gain.update(model.recursions.moments, observation)
        model.advance(observation)

        self.assertGreaterEqual(gain.value(), 0.0)
        self.assertLessEqual(gain.value(), gain.upper)
        self.assertEqual(model.recursions.moments.iteration, 1)

    def test_gmm_recursion_accepts_smoothed_covariance_control(self) -> None:
        gain = SmoothedCovarianceControl.new(n_components=2, n_features=2, initial_gain=0.05)
        model = GaussianMixtureRecursion(2, 2, step_size=gain, rng=np.random.default_rng(5))
        observation = np.array([0.5, -0.25])

        gain.update(model.recursions.moments, observation)
        model.advance(observation)

        self.assertGreaterEqual(gain.value(), 0.0)
        self.assertLessEqual(gain.value(), gain.upper)
        self.assertEqual(gain.statistic_average.name, "covariance_average")
        self.assertEqual(gain.statistic_average.state.shape, (2, 2, 2))
        self.assertEqual(model.recursions.moments.iteration, 1)

    def test_gmm_recursion_accepts_smoothed_moment_residual_control(self) -> None:
        gain = SmoothedMomentResidualControl.new(n_components=2, n_features=2, initial_gain=0.05)
        model = GaussianMixtureRecursion(2, 2, step_size=gain, rng=np.random.default_rng(5))
        observation = np.array([0.5, -0.25])

        gain.update(model.recursions.moments, observation)
        model.advance(observation)

        self.assertGreaterEqual(gain.value(), 0.0)
        self.assertLessEqual(gain.value(), gain.upper)
        self.assertEqual(gain.statistic_average.name, "moment_residual_average")
        self.assertEqual(gain.statistic_average.state.mass.shape, (2,))
        self.assertEqual(gain.statistic_average.state.first.shape, (2, 2))
        self.assertEqual(gain.statistic_average.state.second.shape, (2, 2, 2))
        self.assertEqual(model.recursions.moments.iteration, 1)


if __name__ == "__main__":
    unittest.main()
