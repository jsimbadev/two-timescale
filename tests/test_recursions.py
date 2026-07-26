from __future__ import annotations

import unittest

import numpy as np

import two_timescale
import two_timescale.recursions as recursions
from two_timescale.recursions.adaptive_gain import AdaptiveGainRecursions
from two_timescale.recursions.eforrlsa import EFORRLSAObservation, EFORRLSARecursion, EFORRLSAState
from two_timescale.recursions.scalar_tracking import ScalarTrackingRecursion
from two_timescale.recursions.state_tracking import StateTrackingRecursion


class SplitRecursionModuleTests(unittest.TestCase):
    def test_recursion_package_reexports_split_module_objects(self) -> None:
        self.assertIs(recursions.AdaptiveGainRecursions, AdaptiveGainRecursions)
        self.assertIs(recursions.EFORRLSAObservation, EFORRLSAObservation)
        self.assertIs(recursions.EFORRLSARecursion, EFORRLSARecursion)
        self.assertIs(recursions.EFORRLSAState, EFORRLSAState)
        self.assertIs(recursions.ScalarTrackingRecursion, ScalarTrackingRecursion)
        self.assertIs(recursions.StateTrackingRecursion, StateTrackingRecursion)
        self.assertIs(two_timescale.AdaptiveGainRecursions, AdaptiveGainRecursions)
        self.assertIs(two_timescale.EFORRLSARecursion, EFORRLSARecursion)
        self.assertIs(two_timescale.ScalarTrackingRecursion, ScalarTrackingRecursion)
        self.assertIs(two_timescale.StateTrackingRecursion, StateTrackingRecursion)

    def test_adaptive_gain_recursions_copy_inputs_and_update_all_averages(self) -> None:
        innovation_average = np.array([2.0, -1.0])
        signal_average = np.array([1.0, 3.0])
        eigenvalue_average = np.array([1.0, 1.0])

        gains = AdaptiveGainRecursions.new(
            innovation_average=innovation_average,
            signal_average=signal_average,
            eigenvalue_average=eigenvalue_average,
            gain=0.5,
        )
        innovation_average[:] = 99.0
        signal_average[:] = 99.0
        eigenvalue_average[:] = 99.0

        gain = gains.update(
            signal=np.array([4.0, 2.0]),
            eigenvalue=np.array([3.0, 5.0]),
            step_size=0.25,
        )

        expected_innovation = np.array([1.75, -1.5])
        expected_signal = np.array([1.75, 2.75])
        expected_eigenvalue = np.array([1.5, 2.0])
        expected_gain = np.mean(np.abs(expected_innovation / (expected_signal + expected_eigenvalue)))

        np.testing.assert_allclose(gains.innovation_average.state, expected_innovation)
        np.testing.assert_allclose(gains.signal_average.state, expected_signal)
        np.testing.assert_allclose(gains.eigenvalue_average.state, expected_eigenvalue)
        self.assertAlmostEqual(gain, expected_gain)
        self.assertEqual(gains.innovation_average.name, "innovation_average")
        self.assertEqual(gains.signal_average.name, "signal_average")
        self.assertEqual(gains.eigenvalue_average.name, "eigenvalue_average")
        self.assertEqual(gains.innovation_average.iteration, 1)
        self.assertEqual(gains.signal_average.iteration, 1)
        self.assertEqual(gains.eigenvalue_average.iteration, 1)

    def test_state_tracking_recursion_updates_a_vector_state(self) -> None:
        center = StateTrackingRecursion.new(np.array([1.0, -1.0]), name="center_field")

        updated = center.update(innovation=np.array([2.0, 4.0]), step_size=0.25)

        np.testing.assert_allclose(updated, np.array([1.5, 0.0]))
        np.testing.assert_allclose(center.recursion.state, center.state)
        self.assertEqual(center.name, "center_field")
        self.assertEqual(center.iteration, 1)

    def test_scalar_tracking_recursion_tracks_a_target(self) -> None:
        sigma = ScalarTrackingRecursion.new(2.0, name="residual_variance")

        updated = sigma.update(target=6.0, step_size=0.25)

        self.assertEqual(updated, 3.0)
        self.assertEqual(sigma.recursion.state, 3.0)
        self.assertEqual(sigma.name, "residual_variance")
        self.assertEqual(sigma.iteration, 1)

    def test_eforrlsa_recursion_updates_pca_shape_without_aliasing_inputs(self) -> None:
        weight = np.eye(2)
        eigenvalue = np.array([1.0, 0.5])
        shape = EFORRLSARecursion.new(EFORRLSAState(weight=weight, eigenvalue=eigenvalue))
        weight[:] = 0.0
        eigenvalue[:] = 0.0

        state = shape.update(
            x_c=np.array([1.0, 2.0]),
            y=np.array([1.0, 2.0]),
            alpha=0.2,
        )

        self.assertIs(state, shape.state)
        self.assertEqual(state.weight.shape, (2, 2))
        self.assertEqual(state.eigenvalue.shape, (2,))
        self.assertTrue(np.isfinite(state.weight).all())
        self.assertTrue(np.all(state.eigenvalue > 0.0))
        self.assertEqual(shape.recursion.name, "eforrlsa_shape")
        self.assertEqual(shape.iteration, 1)


if __name__ == "__main__":
    unittest.main()
