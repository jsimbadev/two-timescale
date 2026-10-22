r"""Algebraic primitives for stochastic recursions.

This module gives the rest of the package a small common language for online
updates.  The central object is the stochastic-approximation recursion

.. math::

    z_{t+1} = z_t + a_t F(z_t, x_t),

where ``z_t`` is the recursion state, ``x_t`` is the current observation,
``F`` is a :class:`VectorField`, and ``a_t`` is a scalar step size.  The same
object also represents auxiliary recursions, so an adaptive step size can be
the state of another recursion rather than a fixed float.

References:
    Kushner and Yin (2003):
        Kushner, H. J., & Yin, G. (2003). *Stochastic Approximation and
        Recursive Algorithms and Applications*. Springer New York.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Generic, Protocol, TypeVar

import numpy as np

State = TypeVar("State")
Observation = TypeVar("Observation")
Delta = TypeVar("Delta")


class Control(Protocol):
    """Protocol for scalar step-size sources that can scale a vector field.

    The protocol is intentionally narrow: it only has to expose the scalar value
    used by a recursion at the supplied context.  Implementations may be fixed
    parameters, deterministic schedules, or stateful recursions.
    """

    def value(self, context: "RecursionContext | None" = None) -> float:
        ...


class SupportsArray(Protocol):
    """Protocol for objects NumPy can coerce with ``np.asarray``."""

    def __array__(self) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RecursionContext:
    """Runtime context supplied during one recursion update.

    Attributes:
        iteration: Zero-based iteration index of the recursion invoking the
            vector field.
        rng: Optional random number generator shared with fields or step-size
            sources that need reproducible stochastic choices.
        metadata: Optional caller-defined data for algorithms that need to pass
            rank, component, or diagnostic information through the recursion
            stack.
    """

    iteration: int = 0
    rng: np.random.Generator | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FreeParameter:
    """Chosen scalar parameter with optional validation bounds.

    Args:
        value_: Numeric value of the parameter.
        name: Name used in validation errors and diagnostic displays.
        lower: Inclusive lower bound.  ``None`` disables the lower-bound check.
        upper: Inclusive upper bound.  ``None`` disables the upper-bound check.

    Raises:
        ValueError: If ``value_`` falls outside the supplied bounds.
    """

    value_: float
    name: str = "parameter"
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        value = float(self.value_)
        if self.lower is not None and value < self.lower:
            raise ValueError(f"{self.name}={value} is below lower bound {self.lower}")
        if self.upper is not None and value > self.upper:
            raise ValueError(f"{self.name}={value} is above upper bound {self.upper}")
        object.__setattr__(self, "value_", value)

    def value(self, context: RecursionContext | None = None) -> float:
        return self.value_

    def __float__(self) -> float:
        return self.value_


@dataclass(frozen=True)
class LearningRate(FreeParameter):
    """Free parameter specialized to a stochastic-approximation step size."""

    name: str = "learning_rate"
    lower: float | None = 0.0
    upper: float | None = 1.0


@dataclass(frozen=True)
class ScheduledLearningRate:
    r"""Deterministic polynomial-decay learning-rate schedule.

    The schedule evaluates

    .. math::

        a_t = a_0 (1 + d t)^{-p},

    where ``a_0`` is ``initial``, ``d`` is ``decay``, and ``p`` is ``power``.

    Args:
        initial: Initial learning rate.
        decay: Non-negative decay multiplier applied to the iteration index.
        power: Polynomial decay exponent.
    """

    initial: LearningRate
    decay: float = 0.0
    power: float = 1.0

    def value(self, context: RecursionContext | None = None) -> float:
        iteration = 0 if context is None else context.iteration
        return float(self.initial) / (1.0 + self.decay * iteration) ** self.power


def _ema_field() -> "VectorField[Any, Any, Any]":
    return VectorField(lambda state, target, ctx: add_delta(target, scale_delta(state, -1.0)), name="ema_tracking")


def _as_scalar(control: float | Control | Callable[[RecursionContext | None], float],
               context: RecursionContext | None) -> float:
    """Evaluate a scalar step-size source."""
    if isinstance(control, Recursion):
        return float(np.asarray(control.state))
    if hasattr(control, "value"):
        return float(control.value(context))  # type: ignore[union-attr]
    if callable(control):
        return float(control(context))
    return float(control)


def scale_delta(delta: Any, scalar: float) -> Any:
    """Scale a structured recursion delta by a scalar.

    Args:
        delta: Numeric object, dataclass, tuple, list, or dictionary whose
            leaves support scalar multiplication.
        scalar: Scalar multiplier.

    Returns:
        Object with the same structure as ``delta`` and each leaf multiplied by
        ``scalar``.
    """

    if is_dataclass(delta):
        return replace(delta, **{f.name: scale_delta(getattr(delta, f.name), scalar) for f in fields(delta)})
    if isinstance(delta, dict):
        return {key: scale_delta(value, scalar) for key, value in delta.items()}
    if isinstance(delta, tuple):
        return tuple(scale_delta(value, scalar) for value in delta)
    if isinstance(delta, list):
        return [scale_delta(value, scalar) for value in delta]
    return delta * scalar


def add_delta(state: Any, delta: Any) -> Any:
    """Add a structured delta to a recursion state.

    Args:
        state: Current recursion state.
        delta: Increment with the same structure as ``state``.

    Returns:
        Updated state with ``delta`` added leafwise.
    """

    if is_dataclass(state):
        return replace(state, **{f.name: add_delta(getattr(state, f.name), getattr(delta, f.name)) for f in fields(state)})
    if isinstance(state, dict):
        return {key: add_delta(value, delta[key]) for key, value in state.items()}
    if isinstance(state, tuple):
        return tuple(add_delta(value, step) for value, step in zip(state, delta))
    if isinstance(state, list):
        return [add_delta(value, step) for value, step in zip(state, delta)]
    return state + delta


@dataclass(frozen=True)
class VectorField(Generic[State, Observation, Delta]):
    """Callable vector field with algebraic composition.

    A vector field is the ``F`` in ``z <- z + a F(z, x)``.  It can be added to
    another vector field or scaled by a float, callable schedule, or
    :class:`Control`.  Scaling is evaluated at call time, which lets a field be
    premultiplied by a stateful recursion.

    Args:
        fn: Callable implementing ``fn(state, observation, context)``.
        name: Human-readable field name used in composed field names.
    """

    fn: Callable[[State, Observation, RecursionContext], Delta]
    name: str = "vector_field"

    def __call__(self, state: State, observation: Observation, context: RecursionContext) -> Delta:
        return self.fn(state, observation, context)

    def __mul__(self, control: float | Control | Callable[[RecursionContext | None], float]) -> "VectorField":
        return self.scaled(control)

    def __rmul__(self, control: float | Control | Callable[[RecursionContext | None], float]) -> "VectorField":
        return self.scaled(control)

    def __add__(self, other: "VectorField") -> "VectorField":
        def summed(state: State, observation: Observation, context: RecursionContext) -> Any:
            return add_delta(self(state, observation, context), other(state, observation, context))

        return VectorField(summed, name=f"({self.name}+{other.name})")

    def scaled(self, control: float | Control | Callable[[RecursionContext | None], float]) -> "VectorField":
        def scaled_field(state: State, observation: Observation, context: RecursionContext) -> Any:
            return scale_delta(self(state, observation, context), _as_scalar(control, context))

        control_name = getattr(control, "name", control.__class__.__name__)
        return VectorField(scaled_field, name=f"{control_name}*{self.name}")


@dataclass
class Recursion(Generic[State, Observation]):
    r"""Stateful Euler recursion driven by a vector field and step size.

    The update is

    .. math::

        z_{t+1} = z_t + a_t F(z_t, x_t).

    Args:
        state: Initial recursion state.
        field: Vector field evaluated at each update.
        step_size: Fixed scalar, callable schedule, :class:`Control`, or scalar
            recursion state used as ``a_t``.
        name: Human-readable recursion name.
        iteration: Initial iteration counter.
    """

    state: State
    field: VectorField[State, Observation, Any]
    step_size: float | Control | Callable[[RecursionContext | None], float] = 1.0
    name: str = "recursion"
    iteration: int = 0

    def step_value(self, context: RecursionContext | None = None) -> float:
        """Evaluate the scalar step size for one update context."""
        return _as_scalar(self.step_size, context)

    def advance(
        self,
        observation: Observation,
        context: RecursionContext | None = None,
    ) -> State:
        """Advance the recursion by one observation.

        Args:
            observation: Current observation or target consumed by ``field``.
            context: Optional runtime context.  If omitted, one is created from
                this recursion's current iteration.

        Returns:
            Updated recursion state.
        """
        if context is None:
            context = RecursionContext(iteration=self.iteration)
        delta = self.field(self.state, observation, context)
        self.state = add_delta(self.state, scale_delta(delta, self.step_value(context)))
        self.iteration += 1
        return self.state

    def control(self, selector: Callable[[State], float] | None = None) -> Callable[[RecursionContext | None], float]:
        """Expose the current recursion state as a scalar step-size function.

        Args:
            selector: Optional function selecting a scalar from a structured
                recursion state.

        Returns:
            Callable compatible with :class:`Recursion` step sizes.
        """

        def current_value(context: RecursionContext | None = None) -> float:
            selected = self.state if selector is None else selector(self.state)
            return float(np.asarray(selected))

        return current_value


@dataclass
class EMARecursion(Generic[State]):
    r"""Exponential moving average expressed through the recursion algebra.

    The update is

    .. math::

        \bar{s}_{t+1} = \bar{s}_t + a_t (s_t - \bar{s}_t).

    The optional per-call ``step_size`` lets algorithms use observation- or
    rank-dependent smoothing while keeping the moving average as a persistent
    recursion object.

    Args:
        state: Initial average.
        step_size: Default smoothing coefficient.
        name: Human-readable recursion name.
        iteration: Initial iteration counter.
    """

    state: State
    step_size: float | Control | Callable[[RecursionContext | None], float] = 1.0
    name: str = "ema"
    iteration: int = 0

    def __post_init__(self) -> None:
        self.recursion = Recursion(
            self.state,
            _ema_field(),
            step_size=self.step_size,
            name=self.name,
            iteration=self.iteration,
        )

    def advance(
        self,
        target: State,
        step_size: float | Control | Callable[[RecursionContext | None], float] | None = None,
        context: RecursionContext | None = None,
    ) -> State:
        """Update the moving average toward a target.

        Args:
            target: New statistic or target state.
            step_size: Optional smoothing coefficient for this update only.
            context: Optional runtime context.

        Returns:
            Updated average state.
        """
        if step_size is not None:
            self.recursion.step_size = step_size
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(target, context)
        self.iteration = self.recursion.iteration
        return self.state
