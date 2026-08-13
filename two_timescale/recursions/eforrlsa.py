r"""Error-forgetting robust recursive least-squares eigensystem recursion.

This module implements the local PCA shape update used by NGPCA.  The update
recursively maintains an orthonormal basis ``W`` and principal eigenvalue vector
``lambda`` from a centered observation ``xi`` and projection ``y = W.T @ xi``.
The paper describes the recursive eigenvector form

.. math::

    w_i = \sum_{j=1}^{i}(q_{i,j} w_j + p_i x),

and the eigenvalue update

.. math::

    \lambda_i^2
      := (\epsilon\lambda_i)^2
         + (1 - \epsilon)y_i
           (2\epsilon\lambda_i y_i + (1 - \epsilon)d_i y_i).

References:
    Migenda02/2_RGHQJ3PP:
        Migenda, N., Moeller, R., & Schenck, W. (2024). Adaptive local
        Principal Component Analysis improves the clustering of
        high-dimensional data. *Pattern Recognition*, 146, 110030. Equations
        (6)--(8). Zotero item key ``RGHQJ3PP``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Recursion, RecursionContext, VectorField


@dataclass
class EFORRLSAState:
    """Local PCA shape state.

    Args:
        weight: Basis matrix ``W`` with one eigenvector per column.
        eigenvalue: Principal eigenvalue vector ``lambda``.
    """

    weight: np.ndarray
    eigenvalue: np.ndarray


@dataclass(frozen=True)
class EFORRLSAObservation:
    """Observation passed to the EFORRLSA shape recursion.

    Args:
        x_c: Centered observation ``x - c``.
        y: Projection ``W.T @ (x - c)`` computed before the shape update.
        alpha: Effective local learning rate for this unit.
    """

    x_c: np.ndarray
    y: np.ndarray
    alpha: float


@dataclass
class EFORRLSARecursion:
    """Robust recursive least-squares eigensystem update as a recursion.

    Args:
        state: Initial local PCA shape state.
        iteration: Initial iteration counter.
    """

    state: EFORRLSAState
    iteration: int = 0

    @classmethod
    def new(cls, state: EFORRLSAState) -> "EFORRLSARecursion":
        """Construct a shape recursion from an initial basis and eigenvalues."""
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
        """Update the PCA basis and eigenvalue estimates.

        Args:
            x_c: Centered observation ``x - c``.
            y: Projection of ``x_c`` onto the pre-update basis.
            alpha: Effective local learning rate.
            context: Optional recursion context.

        Returns:
            Updated shape state.
        """
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
        # The paper writes the eigenvalue update in terms of epsilon; here
        # ``alpha_l`` and ``beta_y`` are the two scalar factors reused by the
        # Gram-Schmidt-like interlocked update.
        alpha_l = (1.0 - alpha) * eigenvalue[i]
        beta_y = alpha * y[i]
        if i > 0:
            # ``d`` is the remaining squared norm after deflating already
            # processed principal directions.
            t += p[i - 1] * p[i - 1]
            d = max(0.0, d - y[i - 1] * y[i - 1])

        s = (alpha_l + alpha * d) * y[i]
        l2[i] = alpha_l * alpha_l + beta_y * (alpha_l * y[i] + s)
        n2 = max(float(l2[i] - s * s * t), np.finfo(float).eps)
        n = np.sqrt(n2)
        p[i] = (beta_y - s * t) / n

        column = p[i] * x_c
        for j in range(i + 1):
            # ``q`` and ``r`` encode the triangular correction coefficients
            # used to keep the updated basis orthonormal.
            if j < i:
                r[i, j] = r[i - 1, j] + p[i - 1] * q[i - 1, j]
                q[i, j] = -(beta_y * y[j] + s * r[i, j]) / n
            else:
                q[i, j] = alpha_l / n
            column = column + q[i, j] * old_weight[:, j]
        weight[:, i] = column

    return EFORRLSAState(weight=weight, eigenvalue=np.sqrt(np.maximum(l2, 0.0)))
