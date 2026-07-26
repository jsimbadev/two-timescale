"""Residual variance recursions for local PCA units."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import EMARecursion, RecursionContext


@dataclass
class ResidualVarianceRecursion:
    """EMA for variance outside the represented PCA subspace."""

    state: float
    iteration: int = 0

    @classmethod
    def new(cls, state: float) -> "ResidualVarianceRecursion":
        return cls(state)

    def __post_init__(self) -> None:
        self.recursion = EMARecursion(float(self.state), name="residual_variance", iteration=self.iteration)

    def update(
        self,
        residual: float,
        alpha: float,
        context: RecursionContext | None = None,
    ) -> float:
        self.state = float(self.recursion.advance(float(residual), step_size=float(alpha), context=context))
        self.iteration = self.recursion.iteration
        return self.state
