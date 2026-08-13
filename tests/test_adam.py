from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from two_timescale import (
    Adam,
    AdamMoments,
    AdamPreconditioner,
    AdamStepSize,
    EMARecursion,
    Recursion,
    SmoothedStatisticTransform,
    VectorField,
)


class AdamTests(unittest.TestCase):
    def test_adam_matches_reference_updates(self) -> None:
        params = AdamParameters()
        adam = Adam.new(
            _initial_state(),
            step_size=params.step_size,
            first_step_size=params.first_step_size,
            second_step_size=params.second_step_size,
            epsilon=params.epsilon,
        )

        for expected, innovation in zip(_reference_states(params), _innovations()):
            np.testing.assert_allclose(adam.update(innovation), expected)

        first, second = _reference_moments(params)
        np.testing.assert_allclose(adam.first_average.state, first)
        np.testing.assert_allclose(adam.second_average.state, second)

    def test_composed_recursion_matches_adam_wrapper(self) -> None:
        params = AdamParameters()
        wrapped = Adam.new(
            _initial_state(),
            step_size=params.step_size,
            first_step_size=params.first_step_size,
            second_step_size=params.second_step_size,
            epsilon=params.epsilon,
        )
        composed = Recursion(
            _initial_state(),
            _adam_preconditioner(params).precondition(
                VectorField(lambda state, innovation, ctx: innovation, name="innovation")
            ),
            step_size=AdamStepSize(params.step_size),
            name="composed_adam",
        )

        for innovation in _innovations():
            np.testing.assert_allclose(composed.advance(innovation), wrapped.update(innovation))

    def test_preconditioner_wraps_state_dependent_vector_field(self) -> None:
        params = AdamParameters(step_size=0.1, first_step_size=0.1, second_step_size=0.001)
        field = _adam_preconditioner(params).precondition(
            VectorField(lambda state, observation, ctx: -2.0 * state, name="negative_quadratic_gradient")
        )
        recursion = Recursion(
            np.array([8.0, -6.0]),
            field,
            step_size=AdamStepSize(params.step_size),
            name="preconditioned_quadratic",
        )

        initial_loss = _quadratic_loss(recursion.state)
        for _ in range(300):
            recursion.advance(None)

        self.assertLess(_quadratic_loss(recursion.state), initial_loss * 1e-6)
        np.testing.assert_allclose(recursion.state, np.zeros(2), atol=1e-4)

    def test_composed_smoothed_statistics_own_moment_step_sizes(self) -> None:
        first_average = EMARecursion(np.zeros(2), step_size=0.2, name="first")
        second_average = EMARecursion(np.zeros(2), step_size=0.1, name="second")
        preconditioner = AdamPreconditioner(
            first_moment=SmoothedStatisticTransform(
                first_average,
                lambda state, delta, ctx: delta,
                lambda statistic, state, delta, ctx: statistic,
            ),
            second_moment=SmoothedStatisticTransform(
                second_average,
                lambda state, delta, ctx: delta * delta,
                lambda statistic, state, delta, ctx: statistic,
            ),
            first_bias=EMARecursion(0.0, step_size=0.2, name="first_bias"),
            second_bias=EMARecursion(0.0, step_size=0.1, name="second_bias"),
            epsilon=1e-6,
        )

        direction = preconditioner.update(np.array([1.0, -2.0]))

        expected_direction = np.array([1.0, -2.0]) / (np.sqrt(np.array([1.0, 4.0])) + 1e-6)
        np.testing.assert_allclose(direction, expected_direction)
        np.testing.assert_allclose(first_average.state, np.array([0.2, -0.4]))
        np.testing.assert_allclose(second_average.state, np.array([0.1, 0.4]))

    def test_invalid_parameters_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            Adam.new(np.zeros(2), AdamMoments(first=np.zeros(3), second=np.zeros(2)))
        with self.assertRaises(ValueError):
            Adam.new(np.zeros(2), step_size=-0.1)
        with self.assertRaises(ValueError):
            Adam.new(np.zeros(2), first_step_size=0.0)
        with self.assertRaises(ValueError):
            Adam.new(np.zeros(2), second_step_size=1.1)
        with self.assertRaises(ValueError):
            Adam.new(np.zeros(2), epsilon=0.0)

        adam = Adam.new(np.zeros(2))
        with self.assertRaises(ValueError):
            adam.update(np.zeros(3))


@dataclass(frozen=True)
class AdamParameters:
    step_size: float = 0.05
    first_step_size: float = 0.2
    second_step_size: float = 0.1
    epsilon: float = 1e-6


def _initial_state() -> np.ndarray:
    return np.array([0.25, -0.5])


def _innovations() -> list[np.ndarray]:
    return [
        np.array([1.0, -2.0]),
        np.array([0.5, -1.0]),
        np.array([-0.25, 0.5]),
    ]


def _adam_preconditioner(params: AdamParameters) -> AdamPreconditioner:
    return AdamPreconditioner.new(
        (2,),
        first_step_size=params.first_step_size,
        second_step_size=params.second_step_size,
        epsilon=params.epsilon,
    )


def _reference_states(params: AdamParameters) -> list[np.ndarray]:
    state = _initial_state()
    states: list[np.ndarray] = []
    first = np.zeros(2)
    second = np.zeros(2)
    first_bias = 0.0
    second_bias = 0.0
    for innovation in _innovations():
        first = first + params.first_step_size * (innovation - first)
        second = second + params.second_step_size * (innovation * innovation - second)
        first_bias = first_bias + params.first_step_size * (1.0 - first_bias)
        second_bias = second_bias + params.second_step_size * (1.0 - second_bias)
        direction = (first / first_bias) / (np.sqrt(second / second_bias) + params.epsilon)
        state = state + params.step_size * direction
        states.append(state.copy())
    return states


def _reference_moments(params: AdamParameters) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros(2)
    second = np.zeros(2)
    for innovation in _innovations():
        first = first + params.first_step_size * (innovation - first)
        second = second + params.second_step_size * (innovation * innovation - second)
    return first, second


def _quadratic_loss(x: np.ndarray) -> float:
    return float(x @ x)


if __name__ == "__main__":
    unittest.main()
