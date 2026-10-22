r"""Gaussian-mixture moment recursions as stochastic approximation.

The recursion state stores sufficient statistics rather than natural Gaussian
parameters.  For component responsibilities ``r_k(x_t)``, one stochastic target
is

.. math::

    S_t^k = \left(r_k,\; r_k x_t,\; r_k x_t x_t^\top\right),

and the online recursion updates ``state <- state + a_t * (S_t - state)``.
The parameters exposed to callers are recovered as

.. math::

    \pi_k = s_k / \sum_j s_j,\qquad
    \mu_k = m_k / s_k,\qquad
    \Sigma_k = Q_k / s_k - \mu_k\mu_k^\top.

References:
    Kushner and Yin (2003):
        Kushner, H. J., & Yin, G. (2003). *Stochastic Approximation and
        Recursive Algorithms and Applications*. Springer New York. General
        stochastic-approximation background.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import Control, LearningRate, Recursion, RecursionContext, VectorField

StepSize = float | Control | Callable[[RecursionContext | None], float]


@dataclass
class GMMMoments:
    """Sufficient statistics for an online Gaussian mixture.

    Args:
        mass: Per-component zeroth moments ``s_k``.
        first: Per-component first moments ``m_k = s_k * mu_k``.
        second: Per-component second raw moments ``Q_k = s_k * E[x x^T]``.
    """

    mass: np.ndarray
    first: np.ndarray
    second: np.ndarray


@dataclass
class GMMMomentRecursion:
    """Online moment recursion over Gaussian-mixture sufficient statistics.

    Args:
        state: Initial sufficient-statistic state.
        step_size: Fixed, scheduled, or adaptive stochastic-approximation step
            size.
        regularization: Positive diagonal covariance regularizer and minimum
            component mass used when converting moments to parameters.
        name: Human-readable recursion name.
        iteration: Initial iteration counter.
    """

    state: GMMMoments
    step_size: StepSize = LearningRate(0.05)
    regularization: float = 1e-6
    name: str = "gmm_moments"
    iteration: int = 0

    @classmethod
    def new(
        cls,
        state: GMMMoments,
        step_size: StepSize = LearningRate(0.05),
        regularization: float = 1e-6,
        name: str = "gmm_moments",
    ) -> "GMMMomentRecursion":
        """Construct a GMM moment recursion.

        Args:
            state: Initial sufficient statistics.
            step_size: Recursion step size.
            regularization: Covariance and mass regularization constant.
            name: Human-readable recursion name.

        Returns:
            Configured GMM moment recursion.
        """
        return cls(
            state=state,
            step_size=step_size,
            regularization=regularization,
            name=name,
        )

    def __post_init__(self) -> None:
        self.state = GMMMoments(
            mass=np.array(self.state.mass, dtype=float, copy=True),
            first=np.array(self.state.first, dtype=float, copy=True),
            second=np.array(self.state.second, dtype=float, copy=True),
        )
        self.regularization = float(self.regularization)
        self.recursion = Recursion(
            self.state,
            _gmm_moment_field(self.regularization),
            step_size=self.step_size,
            name=self.name,
            iteration=self.iteration,
        )

    def update(
        self,
        observation: np.ndarray,
        context: RecursionContext | None = None,
    ) -> GMMMoments:
        """Advance the sufficient statistics by one observation.

        Args:
            observation: Observation vector.
            context: Optional runtime context.

        Returns:
            Updated sufficient-statistic state.
        """
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(np.asarray(observation, dtype=float), context)
        self.iteration = self.recursion.iteration
        return self.state

    def parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mixture weights, means, and covariance matrices."""
        return moments_to_parameters(self.state, self.regularization)

    def responsibility(self, observation: np.ndarray) -> np.ndarray:
        """Return posterior component probabilities for one observation."""
        weights, means, covariances = self.parameters()
        return responsibilities(np.asarray(observation, dtype=float), weights, means, covariances)


def moments_to_parameters(
    state: GMMMoments,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert sufficient statistics to Gaussian-mixture parameters.

    Args:
        state: Moment state to convert.
        regularization: Minimum mass and diagonal covariance regularizer.

    Returns:
        Tuple ``(weights, means, covariances)``.
    """
    mass = np.maximum(state.mass, regularization)
    weights = mass / mass.sum()
    means = state.first / mass[:, None]
    covariances = state.second / mass[:, None, None] - means[:, :, None] * means[:, None, :]
    eye = np.eye(means.shape[1])
    covariances = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
    covariances = covariances + regularization * eye[None, :, :]
    return weights, means, covariances


def responsibilities(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """Compute Gaussian-mixture responsibilities for one observation.

    Args:
        x: Observation vector.
        weights: Mixture weights.
        means: Component mean matrix.
        covariances: Component covariance tensor.

    Returns:
        Normalized responsibility vector.
    """
    logp = np.log(np.maximum(weights, np.finfo(float).tiny))
    for k in range(len(weights)):
        logp[k] += _log_gaussian(x, means[k], covariances[k])
    logp -= logp.max()
    p = np.exp(logp)
    total = p.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights))
    return p / total


def _gmm_moment_field(regularization: float) -> VectorField[GMMMoments, np.ndarray, GMMMoments]:
    def field(state: GMMMoments, x: np.ndarray, context: RecursionContext) -> GMMMoments:
        return gmm_moment_residual(state, x, regularization)

    return VectorField(field, name="gmm_stochastic_em")


def gmm_moment_target(
    state: GMMMoments,
    observation: np.ndarray,
    regularization: float = 1e-6,
) -> GMMMoments:
    """Return the one-sample responsibility-weighted target moments.

    Args:
        state: Current moment state used to compute responsibilities.
        observation: Observation vector.
        regularization: Covariance and mass regularization constant.

    Returns:
        Moment target implied by the current responsibility vector.
    """
    x = np.asarray(observation, dtype=float)
    weights, means, covariances = moments_to_parameters(state, regularization)
    responsibility = responsibilities(x, weights, means, covariances)
    return GMMMoments(
        mass=responsibility,
        first=responsibility[:, None] * x[None, :],
        second=responsibility[:, None, None] * np.outer(x, x)[None, :, :],
    )


def gmm_moment_residual(
    state: GMMMoments,
    observation: np.ndarray,
    regularization: float = 1e-6,
) -> GMMMoments:
    """Return the residual between target and current GMM moments."""
    target = gmm_moment_target(state, observation, regularization)
    return GMMMoments(
        mass=target.mass - state.mass,
        first=target.first - state.first,
        second=target.second - state.second,
    )


def _log_gaussian(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    n_features = x.size
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = covariance + 1e-6 * np.eye(n_features)
        sign, logdet = np.linalg.slogdet(covariance)
    diff = x - mean
    solved = np.linalg.solve(covariance, diff)
    return float(-0.5 * (n_features * np.log(2.0 * np.pi) + logdet + diff @ solved))
