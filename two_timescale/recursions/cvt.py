"""Centroidal vector quantization recursions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Recursion, RecursionContext, VectorField


@dataclass
class CVTRecursion:
    """Centroidal vector quantization update for a local PCA center."""

    state: np.ndarray
    iteration: int = 0

    @classmethod
    def new(cls, state: np.ndarray) -> "CVTRecursion":
        return cls(state)

    def __post_init__(self) -> None:
        self.state = np.array(self.state, dtype=float, copy=True)
        self.recursion = Recursion(
            self.state,
            _cvt_field(),
            step_size=1.0,
            name="cvt_center",
            iteration=self.iteration,
        )

    def update(
        self,
        x_c: np.ndarray,
        alpha: float,
        context: RecursionContext | None = None,
    ) -> np.ndarray:
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.recursion.step_size = float(alpha)
        self.state = self.recursion.advance(x_c, context)
        self.iteration = self.recursion.iteration
        return self.state


def _cvt_field() -> VectorField[np.ndarray, np.ndarray, np.ndarray]:
    return VectorField(lambda center, x_c, ctx: x_c, name="cvt")
