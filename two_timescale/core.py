"""Algebraic primitives for stochastic recursions.

The central object is a recursion

    z[n + 1] = z[n] + a[n] * F(z[n], observation[n]).

`F` is a :class:`VectorField`; `a` is a typed control such as a
:class:`LearningRate`. Vector fields can be scaled by parameters or by other
recursions, which is the main hook for adaptive controls.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Generic, Protocol, TypeVar

import numpy as np

State = TypeVar("State")
Observation = TypeVar("Observation")
Delta = TypeVar("Delta")


class Control(Protocol):
    """A scalar value used to scale a vector field."""

    def value(self, context: "RecursionContext | None" = None) -> float:
        ...


class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RecursionContext:
    """Runtime context available to vector fields and controls."""

    iteration: int = 0
    rng: np.random.Generator | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FreeParameter:
    """A chosen scalar parameter with optional validation bounds."""

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
    """A typed free parameter for stochastic approximation step sizes."""

    name: str = "learning_rate"
    lower: float | None = 0.0
    upper: float | None = 1.0


@dataclass(frozen=True)
class ScheduledLearningRate:
    """Deterministic schedule a_n = initial / (1 + decay * n)^power."""

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
    if isinstance(control, Recursion):
        return float(np.asarray(control.state))
    if hasattr(control, "value"):
        return float(control.value(context))  # type: ignore[union-attr]
    if callable(control):
        return float(control(context))
    return float(control)


def scale_delta(delta: Any, scalar: float) -> Any:
    """Scale numeric, array, dataclass, tuple/list, or dict deltas."""

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
    """Add a delta to a state with the same structure."""

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
    """Callable vector field with scalar multiplication."""

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
    """Stateful Euler recursion driven by a vector field and scalar control."""

    state: State
    field: VectorField[State, Observation, Any]
    step_size: float | Control | Callable[[RecursionContext | None], float] = 1.0
    name: str = "recursion"
    iteration: int = 0

    def step_value(self, context: RecursionContext | None = None) -> float:
        return _as_scalar(self.step_size, context)

    def advance(
        self,
        observation: Observation,
        context: RecursionContext | None = None,
    ) -> State:
        if context is None:
            context = RecursionContext(iteration=self.iteration)
        delta = self.field(self.state, observation, context)
        self.state = add_delta(self.state, scale_delta(delta, self.step_value(context)))
        self.iteration += 1
        return self.state

    def control(self, selector: Callable[[State], float] | None = None) -> Callable[[RecursionContext | None], float]:
        """Expose the current recursion state as a scalar control."""

        def current_value(context: RecursionContext | None = None) -> float:
            selected = self.state if selector is None else selector(self.state)
            return float(np.asarray(selected))

        return current_value


@dataclass
class EMARecursion(Generic[State]):
    """Exponential moving average expressed through the recursion algebra.

    The update is ``state <- state + step_size * (target - state)``.  The
    optional per-call ``step_size`` lets algorithms use rank-dependent smoothing
    while keeping the moving average as a persistent recursion object.
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
        if step_size is not None:
            self.recursion.step_size = step_size
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(target, context)
        self.iteration = self.recursion.iteration
        return self.state
