r"""Scalar stochastic-approximation recursions.

This module wraps :class:`two_timescale.core.EMARecursion` for scalar targets:

.. math::

    z_{t+1} = z_t + a_t(s_t - z_t).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import EMARecursion, RecursionContext


@dataclass
class ScalarTrackingRecursion:
    """Track a scalar target by exponential smoothing.

    Args:
        state: Initial scalar state.
        name: Human-readable recursion name.
        iteration: Initial iteration counter.
    """

    state: float
    name: str = "scalar_tracking"
    iteration: int = 0

    @classmethod
    def new(cls, state: float, name: str = "scalar_tracking") -> "ScalarTrackingRecursion":
        """Construct a scalar tracking recursion."""
        return cls(state, name=name)

    def __post_init__(self) -> None:
        self.recursion = EMARecursion(float(self.state), name=self.name, iteration=self.iteration)

    def update(
        self,
        target: float,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> float:
        """Advance the scalar state toward one target.

        Args:
            target: Scalar target value.
            step_size: EMA smoothing coefficient.
            context: Optional recursion context.

        Returns:
            Updated scalar state.
        """
        self.state = float(self.recursion.advance(float(target), step_size=float(step_size), context=context))
        self.iteration = self.recursion.iteration
        return self.state
