"""Error-forgetting robust recursive least-squares PCA recursion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Recursion, RecursionContext, VectorField


@dataclass
class EFORRLSAState:
    weight: np.ndarray
    eigenvalue: np.ndarray


@dataclass(frozen=True)
class EFORRLSAObservation:
    x_c: np.ndarray
    y: np.ndarray
    alpha: float


@dataclass
class EFORRLSARecursion:
    """Robust recursive least-squares PCA shape update as a recursion."""

    state: EFORRLSAState
    iteration: int = 0

    @classmethod
    def new(cls, state: EFORRLSAState) -> "EFORRLSARecursion":
        return cls(state)

    def __post_init__(self) -> None:
        self.state = EFORRLSAState(
            weight=np.array(self.state.weight, dtype=float, copy=True),
            eigenvalue=np.array(self.state.eigenvalue, dtype=float, copy=True),
        )
        self.recursion = Recursion(
            self.state,
            _eforrlsa_field(),
            step_size=1.0,
            name="eforrlsa_shape",
            iteration=self.iteration,
        )

    def update(
        self,
        x_c: np.ndarray,
        y: np.ndarray,
        alpha: float,
        context: RecursionContext | None = None,
    ) -> EFORRLSAState:
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(EFORRLSAObservation(x_c=x_c, y=y, alpha=float(alpha)), context)
        self.iteration = self.recursion.iteration
        return self.state


def _eforrlsa_field() -> VectorField[EFORRLSAState, EFORRLSAObservation, EFORRLSAState]:
    def field(state: EFORRLSAState, observation: EFORRLSAObservation, context: RecursionContext) -> EFORRLSAState:
        return _eforrlsa_delta(state, observation)

    return VectorField(field, name="eforrlsa")


def _eforrlsa_delta(state: EFORRLSAState, observation: EFORRLSAObservation) -> EFORRLSAState:
    next_state = _eforrlsa_next_state(state, observation)
    return EFORRLSAState(
        weight=next_state.weight - state.weight,
        eigenvalue=next_state.eigenvalue - state.eigenvalue,
    )


def _eforrlsa_next_state(state: EFORRLSAState, observation: EFORRLSAObservation) -> EFORRLSAState:
    weight = state.weight.copy()
    old_weight = state.weight.copy()
    eigenvalue = state.eigenvalue
    x_c = observation.x_c
    y = observation.y
    alpha = observation.alpha
    m = eigenvalue.size

    l2 = np.zeros(m, dtype=float)
    p = np.zeros(m, dtype=float)
    q = np.zeros((m, m), dtype=float)
    r = np.zeros((m, m), dtype=float)
    t = 0.0
    d = float(x_c @ x_c)

    for i in range(m):
        alpha_l = (1.0 - alpha) * eigenvalue[i]
        beta_y = alpha * y[i]
        if i > 0:
            t += p[i - 1] * p[i - 1]
            d = max(0.0, d - y[i - 1] * y[i - 1])

        s = (alpha_l + alpha * d) * y[i]
        l2[i] = alpha_l * alpha_l + beta_y * (alpha_l * y[i] + s)
        n2 = max(float(l2[i] - s * s * t), np.finfo(float).eps)
        n = np.sqrt(n2)
        p[i] = (beta_y - s * t) / n

        column = p[i] * x_c
        for j in range(i + 1):
            if j < i:
                r[i, j] = r[i - 1, j] + p[i - 1] * q[i - 1, j]
                q[i, j] = -(beta_y * y[j] + s * r[i, j]) / n
            else:
                q[i, j] = alpha_l / n
            column = column + q[i, j] * old_weight[:, j]
        weight[:, i] = column

    return EFORRLSAState(weight=weight, eigenvalue=np.sqrt(np.maximum(l2, 0.0)))
