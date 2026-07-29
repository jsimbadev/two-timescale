"""Gaussian-mixture moment recursions for stochastic EM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import Control, LearningRate, Recursion, RecursionContext, VectorField

StepSize = float | Control | Callable[[RecursionContext | None], float]


@dataclass
class GMMMoments:
    """Sufficient statistics for an online Gaussian mixture."""

    mass: np.ndarray
    first: np.ndarray
    second: np.ndarray


@dataclass
class GMMMomentRecursion:
    """Stochastic-EM recursion over Gaussian-mixture sufficient statistics."""

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
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(np.asarray(observation, dtype=float), context)
        self.iteration = self.recursion.iteration
        return self.state

    def parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return moments_to_parameters(self.state, self.regularization)

    def responsibility(self, observation: np.ndarray) -> np.ndarray:
        weights, means, covariances = self.parameters()
        return responsibilities(np.asarray(observation, dtype=float), weights, means, covariances)


def moments_to_parameters(
    state: GMMMoments,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        weights, means, covariances = moments_to_parameters(state, regularization)
        responsibility = responsibilities(x, weights, means, covariances)
        target_first = responsibility[:, None] * x[None, :]
        target_second = responsibility[:, None, None] * np.outer(x, x)[None, :, :]
        return GMMMoments(
            mass=responsibility - state.mass,
            first=target_first - state.first,
            second=target_second - state.second,
        )

    return VectorField(field, name="gmm_stochastic_em")


def _log_gaussian(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    n_features = x.size
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = covariance + 1e-6 * np.eye(n_features)
        sign, logdet = np.linalg.slogdet(covariance)
    diff = x - mean
    solved = np.linalg.solve(covariance, diff)
    return float(-0.5 * (n_features * np.log(2.0 * np.pi) + logdet + diff @ solved))
