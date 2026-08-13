"""Adam recursions composed from smoothed statistic transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import Control, EMARecursion, Recursion, RecursionContext, VectorField
from .smoothed_statistics import SmoothedStatisticTransform


StepSize = float | Control | Callable[[RecursionContext | None], float]


@dataclass
class AdamMoments:
    """First and second Adam moment states."""

    first: np.ndarray
    second: np.ndarray


@dataclass
class AdamStepSize:
    """Named scalar step size for Adam's outer recursion."""

    step_size: StepSize = 0.001
    name: str = "adam_step_size"

    def value(self, context: RecursionContext | None = None) -> float:
        value = _step_value(self.step_size, context)
        if value < 0.0:
            raise ValueError("step_size must be non-negative")
        return value


@dataclass
class AdamPreconditioner:
    """Coordinate-wise Adam transform built from smoothed gradient statistics."""

    first_moment: SmoothedStatisticTransform[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    second_moment: SmoothedStatisticTransform[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    first_bias: EMARecursion[float]
    second_bias: EMARecursion[float]
    epsilon: float = 1e-8
    name: str = "adam_preconditioner"

    @classmethod
    def new(
        cls,
        shape: tuple[int, ...],
        moments: AdamMoments | None = None,
        first_step_size: StepSize = 0.1,
        second_step_size: StepSize = 0.001,
        epsilon: float = 1e-8,
        name: str = "adam_preconditioner",
        iteration: int = 0,
    ) -> "AdamPreconditioner":
        if moments is None:
            moments = AdamMoments(
                first=np.zeros(shape, dtype=float),
                second=np.zeros(shape, dtype=float),
            )
        first = np.array(moments.first, dtype=float, copy=True)
        second = np.array(moments.second, dtype=float, copy=True)
        expected = np.zeros(shape, dtype=float)
        _validate_matching_shape(expected, first, "shape", "first moment")
        _validate_matching_shape(expected, second, "shape", "second moment")
        return cls(
            first_moment=SmoothedStatisticTransform(
                statistic_average=EMARecursion(
                    first,
                    step_size=first_step_size,
                    name=f"{name}_first",
                    iteration=iteration,
                ),
                diagnostic=_gradient_statistic,
                statistic_map=_identity_statistic,
            ),
            second_moment=SmoothedStatisticTransform(
                statistic_average=EMARecursion(
                    second,
                    step_size=second_step_size,
                    name=f"{name}_second",
                    iteration=iteration,
                ),
                diagnostic=_squared_gradient_statistic,
                statistic_map=_identity_statistic,
            ),
            first_bias=EMARecursion(
                0.0,
                step_size=first_step_size,
                name=f"{name}_first_bias",
                iteration=iteration,
            ),
            second_bias=EMARecursion(
                0.0,
                step_size=second_step_size,
                name=f"{name}_second_bias",
                iteration=iteration,
            ),
            epsilon=epsilon,
            name=name,
        )

    def __post_init__(self) -> None:
        self.epsilon = float(self.epsilon)
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        _validate_matching_shape(
            np.asarray(self.first_average.state),
            np.asarray(self.second_average.state),
            "first moment",
            "second moment",
        )
        _validate_step_size(self.first_average.recursion.step_size, "first_step_size", self.iteration)
        _validate_step_size(self.second_average.recursion.step_size, "second_step_size", self.iteration)
        _validate_matching_iteration(self.first_average.iteration, self.second_average.iteration)
        self.first_bias.recursion.step_size = self.first_average.recursion.step_size
        self.second_bias.recursion.step_size = self.second_average.recursion.step_size
        _validate_matching_iteration(self.first_bias.iteration, self.iteration)
        _validate_matching_iteration(self.second_bias.iteration, self.iteration)

    @property
    def first_average(self) -> EMARecursion[np.ndarray]:
        return self.first_moment.statistic_average

    @property
    def second_average(self) -> EMARecursion[np.ndarray]:
        return self.second_moment.statistic_average

    @property
    def iteration(self) -> int:
        return self.first_average.iteration

    @property
    def moments(self) -> AdamMoments:
        return AdamMoments(
            first=np.asarray(self.first_average.state),
            second=np.asarray(self.second_average.state),
        )

    def corrected_moments(self) -> AdamMoments:
        first_bias = max(float(self.first_bias.state), np.finfo(float).tiny)
        second_bias = max(float(self.second_bias.state), np.finfo(float).tiny)
        return AdamMoments(
            first=np.asarray(self.first_average.state) / first_bias,
            second=np.asarray(self.second_average.state) / second_bias,
        )

    def update(
        self,
        delta: np.ndarray,
        context: RecursionContext | None = None,
    ) -> np.ndarray:
        delta = np.asarray(delta, dtype=float)
        _validate_matching_shape(
            np.asarray(self.first_average.state),
            delta,
            "Adam moments",
            "delta",
        )
        if context is None:
            context = RecursionContext(iteration=self.iteration)
        first_step_size = self.first_average.recursion.step_size
        second_step_size = self.second_average.recursion.step_size
        _validate_step_size(first_step_size, "first_step_size", context.iteration)
        _validate_step_size(second_step_size, "second_step_size", context.iteration)

        self.first_moment.update(np.asarray(self.first_average.state), delta, context=context)
        self.second_moment.update(np.asarray(self.second_average.state), delta, context=context)
        self.first_bias.recursion.step_size = first_step_size
        self.second_bias.recursion.step_size = second_step_size
        self.first_bias.advance(1.0, context=context)
        self.second_bias.advance(1.0, context=context)
        _validate_sync(
            self.first_average.iteration,
            self.second_average.iteration,
            self.first_bias.iteration,
            self.second_bias.iteration,
        )

        corrected = self.corrected_moments()
        return corrected.first / (np.sqrt(corrected.second) + self.epsilon)

    def precondition(
        self,
        field: VectorField,
        name: str | None = None,
    ) -> VectorField:
        def preconditioned(state: object, observation: object, context: RecursionContext) -> np.ndarray:
            return self.update(field(state, observation, context), context)

        return VectorField(preconditioned, name=name or f"{self.name}*{field.name}")


@dataclass
class Adam:
    """Track a state with Adam-scaled stochastic-approximation innovations."""

    state: np.ndarray
    preconditioner: AdamPreconditioner
    step_size: StepSize = 0.001
    name: str = "adam"
    iteration: int = 0

    @classmethod
    def new(
        cls,
        state: np.ndarray,
        moments: AdamMoments | None = None,
        step_size: StepSize = 0.001,
        first_step_size: StepSize = 0.1,
        second_step_size: StepSize = 0.001,
        epsilon: float = 1e-8,
        name: str = "adam",
        iteration: int = 0,
    ) -> "Adam":
        state_array = np.array(state, dtype=float, copy=True)
        return cls(
            state=state_array,
            preconditioner=AdamPreconditioner.new(
                state_array.shape,
                moments=moments,
                first_step_size=first_step_size,
                second_step_size=second_step_size,
                epsilon=epsilon,
                name=f"{name}_preconditioner",
                iteration=iteration,
            ),
            step_size=step_size,
            name=name,
            iteration=iteration,
        )

    def __post_init__(self) -> None:
        self.state = np.array(self.state, dtype=float, copy=True)
        self.iteration = int(self.iteration)
        _validate_outer_step_size(self.step_size, self.iteration)
        _validate_matching_shape(
            self.state,
            np.asarray(self.preconditioner.first_average.state),
            "state",
            "Adam moments",
        )
        _validate_matching_iteration(self.preconditioner.iteration, self.iteration)
        self.recursion = Recursion(
            self.state,
            _adam_tracking_field(self.preconditioner, self.name),
            step_size=AdamStepSize(self.step_size, name=f"{self.name}_step_size"),
            name=self.name,
            iteration=self.iteration,
        )

    @property
    def first_average(self) -> EMARecursion[np.ndarray]:
        return self.preconditioner.first_average

    @property
    def second_average(self) -> EMARecursion[np.ndarray]:
        return self.preconditioner.second_average

    @property
    def first_bias(self) -> EMARecursion[float]:
        return self.preconditioner.first_bias

    @property
    def second_bias(self) -> EMARecursion[float]:
        return self.preconditioner.second_bias

    @property
    def moments(self) -> AdamMoments:
        return self.preconditioner.moments

    def corrected_moments(self) -> AdamMoments:
        return self.preconditioner.corrected_moments()

    def update(
        self,
        innovation: np.ndarray,
        context: RecursionContext | None = None,
    ) -> np.ndarray:
        innovation = np.asarray(innovation, dtype=float)
        _validate_matching_shape(self.state, innovation, "state", "innovation")
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(innovation, context=context)
        self.iteration = self.recursion.iteration
        if self.iteration != self.preconditioner.iteration:
            raise RuntimeError("Adam state and preconditioner recursions are out of sync")
        return self.state


def _adam_tracking_field(
    preconditioner: AdamPreconditioner,
    name: str,
) -> VectorField[np.ndarray, np.ndarray, np.ndarray]:
    return VectorField(lambda state, innovation, ctx: preconditioner.update(innovation, ctx), name=name)


def _gradient_statistic(
    state: np.ndarray,
    delta: np.ndarray,
    context: RecursionContext | None,
) -> np.ndarray:
    return np.asarray(delta, dtype=float)


def _squared_gradient_statistic(
    state: np.ndarray,
    delta: np.ndarray,
    context: RecursionContext | None,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return delta * delta


def _identity_statistic(
    statistic: np.ndarray,
    state: np.ndarray,
    delta: np.ndarray,
    context: RecursionContext | None,
) -> np.ndarray:
    return np.asarray(statistic)


def _step_value(step_size: StepSize, context: RecursionContext | None = None) -> float:
    if hasattr(step_size, "value"):
        return float(step_size.value(context))  # type: ignore[union-attr]
    if callable(step_size):
        return float(step_size(context))
    return float(step_size)


def _validate_step_size(
    step_size: StepSize,
    name: str,
    iteration: int = 0,
) -> None:
    value = _step_value(step_size, RecursionContext(iteration=iteration))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")


def _validate_outer_step_size(step_size: StepSize, iteration: int = 0) -> None:
    if _step_value(step_size, RecursionContext(iteration=iteration)) < 0.0:
        raise ValueError("step_size must be non-negative")


def _validate_sync(*iterations: int) -> None:
    if len(set(iterations)) != 1:
        raise RuntimeError("Adam moment recursions are out of sync")


def _validate_matching_iteration(left: int, right: int) -> None:
    if left != right:
        raise RuntimeError("Adam recursions are out of sync")


def _validate_matching_shape(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise ValueError(f"{left_name} shape {left.shape} must match {right_name} shape {right.shape}")
