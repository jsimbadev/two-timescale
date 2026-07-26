"""Online Gaussian mixture model composed from named stochastic recursions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import LearningRate, RecursionContext
from .recursions import GMMMomentRecursion, GMMMoments, moments_to_parameters


@dataclass
class GMMRecursions:
    moments: GMMMomentRecursion

    @classmethod
    def new(
        cls,
        moments: GMMMoments,
        learning_rate: LearningRate = LearningRate(0.05),
        regularization: float = 1e-6,
    ) -> "GMMRecursions":
        return cls(
            moments=GMMMomentRecursion.new(
                moments,
                learning_rate=learning_rate,
                regularization=regularization,
                name="gmm_moments",
            )
        )


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
        self.regularization = float(regularization)
        self.rng = rng or np.random.default_rng()
        self.recursions = GMMRecursions.new(
            self._initial_moments(),
            learning_rate=learning_rate,
            regularization=self.regularization,
        )

    def fit(self, data: np.ndarray, iterations: int | None = None) -> "OnlineGaussianMixture":
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("data must have shape (n_samples, n_features)")
        count = len(data) if iterations is None else int(iterations)
        for _ in range(count):
            x = data[self.rng.integers(0, len(data))]
            self.recursions.moments.update(
                x,
                RecursionContext(iteration=self.recursions.moments.iteration, rng=self.rng),
            )
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
        return self.recursions.moments.parameters()

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        return np.vstack([self.recursions.moments.responsibility(x) for x in data])

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.predict_proba(data).argmax(axis=1)

    def _initial_moments(self) -> GMMMoments:
        mass = np.full(self.n_components, 1.0 / self.n_components, dtype=float)
        means = self.rng.normal(scale=0.1, size=(self.n_components, self.n_features))
        covariances = np.tile(np.eye(self.n_features), (self.n_components, 1, 1))
        first = mass[:, None] * means
        second = mass[:, None, None] * (covariances + means[:, :, None] * means[:, None, :])
        return GMMMoments(mass=mass, first=first, second=second)
