r"""State-tracking stochastic approximation recursions.

This module provides the simplest vector-valued recursion used by NGPCA:

.. math::

    z_{t+1} = z_t + a_t \Delta_t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Recursion, RecursionContext, VectorField


@dataclass
class StateTrackingRecursion:
    """Track a vector state with additive innovations.

    Args:
        state: Initial vector state.
        name: Human-readable recursion and vector-field name.
        iteration: Initial iteration counter.
    """

    state: np.ndarray
    name: str = "state_tracking"
    iteration: int = 0

    @classmethod
    def new(cls, state: np.ndarray, name: str = "state_tracking") -> "StateTrackingRecursion":
        """Construct a vector tracking recursion with copied state."""
        return cls(state, name=name)

    def __post_init__(self) -> None:
        self.state = np.array(self.state, dtype=float, copy=True)
        self.recursion = Recursion(
            self.state,
            _tracking_field(self.name),
            step_size=1.0,
            name=self.name,
            iteration=self.iteration,
        )

    def update(
        self,
        innovation: np.ndarray,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> np.ndarray:
        """Advance the vector state by one scaled innovation.

        Args:
            innovation: Vector delta before step-size scaling.
            step_size: Scalar update coefficient.
            context: Optional recursion context.

        Returns:
            Updated vector state.
        """
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.recursion.step_size = float(step_size)
        self.state = self.recursion.advance(innovation, context)
        self.iteration = self.recursion.iteration
        return self.state


def _tracking_field(name: str) -> VectorField[np.ndarray, np.ndarray, np.ndarray]:
    return VectorField(lambda state, innovation, ctx: innovation, name=name)
