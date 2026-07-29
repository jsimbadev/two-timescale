from __future__ import annotations

import unittest

import numpy as np

import two_timescale
import two_timescale.recursions as recursions
from two_timescale.recursions.adaptive_gain import AdaptiveGainRecursions
from two_timescale.recursions.eforrlsa import EFORRLSAObservation, EFORRLSARecursion, EFORRLSAState
from two_timescale.recursions.gmm_moments import (
    GMMMomentRecursion,
    GMMMoments,
    StepSize,
    gmm_moment_residual,
    gmm_moment_target,
    moments_to_parameters,
    responsibilities,
)
from two_timescale.recursions.scalar_tracking import ScalarTrackingRecursion
from two_timescale.recursions.smoothed_statistics import SmoothedStatisticControl
from two_timescale.recursions.state_tracking import StateTrackingRecursion


class SplitRecursionModuleTests(unittest.TestCase):
    def test_recursion_package_reexports_split_module_objects(self) -> None:
        self.assertIs(recursions.AdaptiveGainRecursions, AdaptiveGainRecursions)
        self.assertIs(recursions.EFORRLSAObservation, EFORRLSAObservation)
        self.assertIs(recursions.EFORRLSARecursion, EFORRLSARecursion)
        self.assertIs(recursions.EFORRLSAState, EFORRLSAState)
        self.assertIs(recursions.GMMMomentRecursion, GMMMomentRecursion)
        self.assertIs(recursions.GMMMoments, GMMMoments)
        self.assertIs(recursions.StepSize, StepSize)
        self.assertIs(recursions.ScalarTrackingRecursion, ScalarTrackingRecursion)
        self.assertIs(recursions.SmoothedStatisticControl, SmoothedStatisticControl)
        self.assertIs(recursions.StateTrackingRecursion, StateTrackingRecursion)
        self.assertIs(two_timescale.AdaptiveGainRecursions, AdaptiveGainRecursions)
        self.assertIs(two_timescale.EFORRLSARecursion, EFORRLSARecursion)
        self.assertIs(two_timescale.GMMMomentRecursion, GMMMomentRecursion)
        self.assertIs(two_timescale.GMMMoments, GMMMoments)
        self.assertIs(two_timescale.ScalarTrackingRecursion, ScalarTrackingRecursion)
        self.assertIs(two_timescale.SmoothedStatisticControl, SmoothedStatisticControl)
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

    def test_gmm_moment_recursion_updates_sufficient_statistics(self) -> None:
        initial = GMMMoments(
            mass=np.array([0.5, 0.5]),
            first=np.array([[-0.5, 0.0], [0.5, 0.0]]),
            second=np.tile(np.eye(2), (2, 1, 1)),
        )
        moments = GMMMomentRecursion.new(initial, regularization=1e-4)

        updated = moments.update(np.array([-1.0, 0.25]))

        self.assertIs(updated, moments.state)
        self.assertEqual(moments.recursion.name, "gmm_moments")
        self.assertEqual(moments.iteration, 1)
        self.assertTrue(np.isfinite(updated.mass).all())
        self.assertTrue(np.isfinite(updated.first).all())
        self.assertTrue(np.isfinite(updated.second).all())
        self.assertAlmostEqual(updated.mass.sum(), 1.0)

        weights, means, covariances = moments.parameters()
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(means.shape, (2, 2))
        self.assertEqual(covariances.shape, (2, 2, 2))

    def test_gmm_moment_target_and_residual_match_vector_field_target(self) -> None:
        state = GMMMoments(
            mass=np.array([0.5, 0.5]),
            first=np.array([[-0.5, 0.0], [0.5, 0.0]]),
            second=np.tile(np.eye(2), (2, 1, 1)),
        )
        observation = np.array([-1.0, 0.25])

        target = gmm_moment_target(state, observation, regularization=1e-4)
        residual = gmm_moment_residual(state, observation, regularization=1e-4)

        np.testing.assert_allclose(residual.mass, target.mass - state.mass)
        np.testing.assert_allclose(residual.first, target.first - state.first)
        np.testing.assert_allclose(residual.second, target.second - state.second)
        np.testing.assert_allclose(target.mass.sum(), 1.0)

    def test_smoothed_statistic_control_updates_average_and_clips_gain(self) -> None:
        control = SmoothedStatisticControl.new(
            initial_statistic=np.array([0.0, 0.0]),
            diagnostic=lambda state, observation, context: observation - state,
            gain_map=lambda statistic, state, observation, context: float(np.linalg.norm(statistic)),
            initial_gain=0.5,
            smoothing=0.5,
            upper=0.6,
            name="test_statistic",
        )

        gain = control.update(np.array([1.0, 1.0]), np.array([3.0, 1.0]))

        np.testing.assert_allclose(control.statistic_average.state, np.array([1.0, 0.0]))
        self.assertEqual(control.statistic_average.name, "test_statistic")
        self.assertEqual(gain, 0.6)

    def test_gmm_responsibilities_are_probabilities(self) -> None:
        state = GMMMoments(
            mass=np.array([0.25, 0.75]),
            first=np.array([[-0.5, 0.0], [1.5, 0.0]]),
            second=np.array([
                [[1.25, 0.0], [0.0, 0.25]],
                [[3.75, 0.0], [0.0, 0.75]],
            ]),
        )
        weights, means, covariances = moments_to_parameters(state)

        responsibility = responsibilities(np.array([2.0, 0.0]), weights, means, covariances)

        self.assertEqual(responsibility.shape, (2,))
        self.assertAlmostEqual(float(responsibility.sum()), 1.0)
        self.assertTrue(np.all(responsibility >= 0.0))
        self.assertGreater(responsibility[1], responsibility[0])


if __name__ == "__main__":
    unittest.main()
