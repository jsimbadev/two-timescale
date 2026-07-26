"""Scalar stochastic-approximation recursions."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import EMARecursion, RecursionContext


@dataclass
class ScalarTrackingRecursion:
    """Track a scalar target with update state <- state + step_size * (target - state)."""

    state: float
    name: str = "scalar_tracking"
    iteration: int = 0

    @classmethod
    def new(cls, state: float, name: str = "scalar_tracking") -> "ScalarTrackingRecursion":
        return cls(state, name=name)

    def __post_init__(self) -> None:
        self.recursion = EMARecursion(float(self.state), name=self.name, iteration=self.iteration)

    def update(
        self,
        target: float,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> float:
        self.state = float(self.recursion.advance(float(target), step_size=float(step_size), context=context))
        self.iteration = self.recursion.iteration
        return self.state
