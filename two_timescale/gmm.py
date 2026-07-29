"""Online Gaussian mixture model composed from named stochastic recursions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import LearningRate, RecursionContext
from .recursions import GMMMomentRecursion, GMMMoments, StepSize, moments_to_parameters


@dataclass
class GMMRecursions:
    moments: GMMMomentRecursion

    @classmethod
    def new(
        cls,
        moments: GMMMoments,
        step_size: StepSize = LearningRate(0.05),
        regularization: float = 1e-6,
    ) -> "GMMRecursions":
        return cls(
            moments=GMMMomentRecursion.new(
                moments,
                step_size=step_size,
                regularization=regularization,
                name="gmm_moments",
            )
        )


class GaussianMixtureRecursion:
    """Stochastic-approximation Gaussian mixture recursion."""

    def __init__(
        self,
        n_components: int,
        n_features: int,
        step_size: StepSize = LearningRate(0.05),
        regularization: float = 1e-6,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_components = int(n_components)
        self.n_features = int(n_features)
        self.regularization = float(regularization)
        self.rng = rng or np.random.default_rng()
        self.recursions = GMMRecursions.new(
            self._initial_moments(),
            step_size=step_size,
            regularization=self.regularization,
        )

    def advance(self, observations: np.ndarray) -> "GaussianMixtureRecursion":
        data = self._as_observations(observations)
        for x in data:
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

    def responsibilities(self, observations: np.ndarray) -> np.ndarray:
        data = self._as_observations(observations)
        return np.vstack([self.recursions.moments.responsibility(x) for x in data])

    def assignments(self, observations: np.ndarray) -> np.ndarray:
        return self.responsibilities(observations).argmax(axis=1)

    def _initial_moments(self) -> GMMMoments:
        mass = np.full(self.n_components, 1.0 / self.n_components, dtype=float)
        means = self.rng.normal(scale=0.1, size=(self.n_components, self.n_features))
        covariances = np.tile(np.eye(self.n_features), (self.n_components, 1, 1))
        first = mass[:, None] * means
        second = mass[:, None, None] * (covariances + means[:, :, None] * means[:, None, :])
        return GMMMoments(mass=mass, first=first, second=second)

    def _as_observations(self, observations: np.ndarray) -> np.ndarray:
        data = np.asarray(observations, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("observations must have shape (n_observations, n_features)")
        return data
