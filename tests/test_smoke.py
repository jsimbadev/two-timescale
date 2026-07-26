from __future__ import annotations

import unittest

import numpy as np

from two_timescale import EMARecursion, LearningRate, Recursion, VectorField
from two_timescale.gmm import OnlineGaussianMixture
from two_timescale.ngpca import EFORRLSARecursion, NeuralGasPCA
from two_timescale.potentials import LocalPCAGeometry, NamedPotential, as_potential


class RecursionSmokeTests(unittest.TestCase):
    def test_ema_recursion_tracks_targets_with_per_step_smoothing(self) -> None:
        ema = EMARecursion(np.array([0.0, 2.0]), step_size=0.25)

        ema.advance(np.array([4.0, -2.0]))
        np.testing.assert_allclose(ema.state, np.array([1.0, 1.0]))

        ema.advance(np.array([3.0, 5.0]), step_size=0.5)
        np.testing.assert_allclose(ema.state, np.array([2.0, 3.0]))
        self.assertEqual(ema.iteration, 2)

    def test_named_and_custom_potentials_are_callable(self) -> None:
        geometry = LocalPCAGeometry(
            x_c=np.array([2.0, 1.0]),
            y=np.array([2.0]),
            eigenvalue=np.array([4.0]),
            sigma_sqr=1.0,
            activity=1.0,
            pca_dimensionality=1,
            data_dimensionality=2,
        )

        named = NamedPotential("N")
        custom = as_potential(lambda unit_geometry: unit_geometry.activity + unit_geometry.y.sum())

        self.assertGreater(named(geometry), 0.0)
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom(geometry), 3.0)

    def test_eforrlsa_recursion_updates_pca_shape(self) -> None:
        shape = EFORRLSARecursion.from_shape(
            weight=np.eye(2),
            eigenvalue=np.array([1.0, 0.5]),
        )

        state = shape.advance(
            x_c=np.array([1.0, 2.0]),
            y=np.array([1.0, 2.0]),
            alpha=0.2,
        )

        self.assertEqual(state.weight.shape, (2, 2))
        self.assertEqual(state.eigenvalue.shape, (2,))
        self.assertTrue(np.isfinite(state.weight).all())
        self.assertTrue(np.all(state.eigenvalue > 0.0))
        self.assertEqual(shape.iteration, 1)

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

        model = NeuralGasPCA(4, rng=rng).initialize(data)
        stream = data[[rng.integers(0, len(data)) for _ in range(20)]]
        model.advance(stream)

        centers = model.centers()
        eigenvalues = np.vstack([unit.eigenvalue for unit in model.units])

        # The fitted centers should remain in the neighborhood of the sampled
        # data cloud, and each local PCA unit should land near observed mass.
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_span = data_max - data_min
        data_scale = np.linalg.norm(data_span)

        nearest_observation_distances = np.min(
            np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2),
            axis=0,
        )

        self.assertEqual(model.centers().shape, (4, 2))
        self.assertTrue(np.isfinite(centers).all())

        self.assertTrue(np.all(centers >= data_min - 0.1 * data_span))
        self.assertTrue(np.all(centers <= data_max + 0.1 * data_span))
        self.assertLess(nearest_observation_distances.max(), 0.1 * data_scale)

        # Local variances should stay positive and remain on the scale induced
        # by the finite sample, rather than exploding during adaptation.
        self.assertTrue(np.all(eigenvalues > 0.0))
        self.assertLess(eigenvalues.max(), data_scale**2)

    def test_gmm_smoke(self) -> None:
        rng = np.random.default_rng(1)
        data = np.vstack([
            rng.normal(loc=-2.0, scale=0.4, size=(64, 2)),
            rng.normal(loc=2.0, scale=0.4, size=(64, 2)),
        ])

        model = OnlineGaussianMixture(2, 2, learning_rate=LearningRate(0.05), rng=rng).fit(data, iterations=100)

        weights, means, covariances = model.parameters()

        # Compare mixture-level moments against the empirical moments of the
        # generated data, instead of pinning the test to component ordering.
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

        # Each component covariance should be a valid Gaussian covariance.
        for covariance in covariances:
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))


if __name__ == "__main__":
    unittest.main()
