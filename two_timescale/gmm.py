"""Online Gaussian mixture recursions built from the generic algebra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import LearningRate, Recursion, RecursionContext, VectorField


@dataclass
class GMMMoments:
    """Sufficient statistics for an online Gaussian mixture."""

    mass: np.ndarray
    first: np.ndarray
    second: np.ndarray


def _gmm_moment_field(regularization: float) -> VectorField[GMMMoments, np.ndarray, GMMMoments]:
    def field(state: GMMMoments, x: np.ndarray, context: RecursionContext) -> GMMMoments:
        weights, means, covariances = moments_to_parameters(state, regularization)
        responsibility = _responsibilities(x, weights, means, covariances)
        target_first = responsibility[:, None] * x[None, :]
        target_second = responsibility[:, None, None] * np.outer(x, x)[None, :, :]
        return GMMMoments(
            mass=responsibility - state.mass,
            first=target_first - state.first,
            second=target_second - state.second,
        )

    return VectorField(field, name="gmm_stochastic_em")


class OnlineGaussianMixture:
    """Stochastic-approximation Gaussian mixture using moment recursions."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        learning_rate: LearningRate = LearningRate(0.05),
        regularization: float = 1e-6,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.n_features = int(n_features)
        self.learning_rate = learning_rate
        self.regularization = float(regularization)
        self.rng = rng or np.random.default_rng()
        self.recursion = Recursion(
            self._initial_moments(),
            _gmm_moment_field(self.regularization),
            step_size=self.learning_rate,
            name="online_gmm",
        )

    def fit(self, data: np.ndarray, iterations: int | None = None) -> "OnlineGaussianMixture":
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("data must have shape (n_samples, n_features)")
        count = len(data) if iterations is None else int(iterations)
        for _ in range(count):
            x = data[self.rng.integers(0, len(data))]
            self.recursion.advance(x, RecursionContext(iteration=self.recursion.iteration, rng=self.rng))
        return self

    @property
    def weights(self) -> np.ndarray:
        return self.parameters()[0]

    @property
    def means(self) -> np.ndarray:
        return self.parameters()[1]

    @property
    def covariances(self) -> np.ndarray:
        return self.parameters()[2]

    def parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return moments_to_parameters(self.recursion.state, self.regularization)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        weights, means, covariances = self.parameters()
        data = np.asarray(data, dtype=float)
        return np.vstack([_responsibilities(x, weights, means, covariances) for x in data])

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.predict_proba(data).argmax(axis=1)

    def _initial_moments(self) -> GMMMoments:
        mass = np.full(self.n_components, 1.0 / self.n_components, dtype=float)
        means = self.rng.normal(scale=0.1, size=(self.n_components, self.n_features))
        covariances = np.tile(np.eye(self.n_features), (self.n_components, 1, 1))
        first = mass[:, None] * means
        second = mass[:, None, None] * (covariances + means[:, :, None] * means[:, None, :])
        return GMMMoments(mass=mass, first=first, second=second)


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


def _responsibilities(
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


def _log_gaussian(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    n_features = x.size
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = covariance + 1e-6 * np.eye(n_features)
        sign, logdet = np.linalg.slogdet(covariance)
    diff = x - mean
    solved = np.linalg.solve(covariance, diff)
    return float(-0.5 * (n_features * np.log(2.0 * np.pi) + logdet + diff @ solved))
