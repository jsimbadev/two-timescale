from __future__ import annotations

import unittest

import numpy as np

from examples.ngpca_example import to_matlab_json_dict
from two_timescale import DEFAULT_NGPCA_INITIAL_GAIN
from two_timescale.ngpca import NGPCAParameters, NGPCAUnitRecursions, NeuralGasPCA
from two_timescale.potentials import LocalPCAGeometry


class NeuralGasPCATests(unittest.TestCase):
    def test_unit_recursions_copy_inputs_and_initialize_gain_state(self) -> None:
        center = np.array([1.0, -1.0])
        weight = np.eye(2)
        eigenvalue = np.array([2.0, 0.5])

        recursions = NGPCAUnitRecursions.new(
            center=center,
            weight=weight,
            eigenvalue=eigenvalue,
            residual_variance=3.0,
            initial_gain=0.7,
        )

        self.assertEqual(recursions.initial_gain, 0.7)
        self.assertEqual(recursions.adaptive_gain.gain, 0.7)

        np.testing.assert_allclose(recursions.adaptive_gain.innovation_average.state, eigenvalue**2 + eigenvalue)
        np.testing.assert_allclose(recursions.adaptive_gain.signal_average.state, eigenvalue**2)
        np.testing.assert_allclose(recursions.adaptive_gain.eigenvalue_average.state, eigenvalue)

        center[:] = 0.0
        weight[:] = 0.0
        eigenvalue[:] = 0.0
        np.testing.assert_allclose(recursions.center.state, np.array([1.0, -1.0]))
        np.testing.assert_allclose(recursions.shape.state.weight, np.eye(2))
        np.testing.assert_allclose(recursions.shape.state.eigenvalue, np.array([2.0, 0.5]))

    def test_ngpca_smoke_keeps_units_and_recursions_in_sync(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(64, 2))

        model = NeuralGasPCA(4, rng=rng).initialize(data)
        np.testing.assert_allclose(
            [recursions.initial_gain for recursions in model.recursions],
            DEFAULT_NGPCA_INITIAL_GAIN,
        )
        np.testing.assert_allclose(
            [unit.alpha for unit in model.units],
            [recursions.adaptive_gain.gain for recursions in model.recursions],
        )

        stream = data[[rng.integers(0, len(data)) for _ in range(20)]]
        model.advance(stream)

        centers = model.centers()
        eigenvalues = np.vstack([unit.eigenvalue for unit in model.units])
        recursion_centers = np.vstack([recursions.center.state for recursions in model.recursions])

        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_span = data_max - data_min
        data_scale = np.linalg.norm(data_span)
        nearest_observation_distances = np.min(
            np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2),
            axis=0,
        )

        self.assertEqual(model.centers().shape, (4, 2))
        self.assertEqual(len(model.recursions), 4)
        self.assertTrue(np.isfinite(centers).all())
        np.testing.assert_allclose(recursion_centers, centers)

        self.assertTrue(np.all(centers >= data_min - 0.1 * data_span))
        self.assertTrue(np.all(centers <= data_max + 0.1 * data_span))
        self.assertLess(nearest_observation_distances.max(), 0.1 * data_scale)

        self.assertTrue(np.all(eigenvalues > 0.0))
        self.assertLess(eigenvalues.max(), data_scale**2)

    def test_ngpca_accepts_custom_potential_function(self) -> None:
        calls: list[tuple[float, float]] = []

        def potential(geometry: LocalPCAGeometry) -> float:
            calls.append((float(geometry.activity), float(geometry.y @ geometry.y)))
            return float(geometry.y @ geometry.y)

        rng = np.random.default_rng(2)
        data = rng.normal(size=(8, 2))
        model = NeuralGasPCA(2, params=NGPCAParameters(potential_function=potential), rng=rng)

        model.initialize(data)
        model.advance(data[:3])

        self.assertGreater(len(calls), 0)
        self.assertTrue(np.isfinite(model.ranking).all())

    def test_ngpca_json_exports_recursion_state(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(size=(16, 2))

        model = NeuralGasPCA(2, rng=rng).initialize(data)
        model.advance(data[:5])
        payload = to_matlab_json_dict(model)

        first_unit = payload["units"][0]
        first_gain = model.recursions[0].adaptive_gain

        self.assertEqual(payload["learningRate"], DEFAULT_NGPCA_INITIAL_GAIN)
        self.assertEqual(len(payload["units"]), 2)
        np.testing.assert_allclose(first_unit["eta_bar"], first_gain.signal_average.state)
        np.testing.assert_allclose(first_unit["l_bar"], first_gain.eigenvalue_average.state)
        np.testing.assert_allclose(first_unit["gamma_bar"], first_gain.innovation_average.state)
        self.assertEqual(first_unit["epsilon"], first_gain.gain)


if __name__ == "__main__":
    unittest.main()
