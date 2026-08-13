"""Online Gaussian mixture model composed from named stochastic recursions.

This module is a thin user-facing wrapper around
``recursions.gmm_moments.GMMMomentRecursion``.  It initializes Gaussian-mixture
moments, feeds observations through the online moment recursion, and exposes
weights, means, covariances, responsibilities, and hard assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import LearningRate, RecursionContext
from .recursions import GMMMomentRecursion, GMMMoments, StepSize, moments_to_parameters


@dataclass
class GMMRecursions:
    """Owned recursions for a Gaussian mixture model.

    Args:
        moments: Sufficient-statistic recursion for the mixture.
    """

    moments: GMMMomentRecursion

    @classmethod
    def new(
        cls,
        moments: GMMMoments,
        step_size: StepSize = LearningRate(0.05),
        regularization: float = 1e-6,
    ) -> "GMMRecursions":
        """Construct GMM recursions from initial sufficient statistics.

        Args:
            moments: Initial sufficient statistics.
            step_size: Stochastic-approximation step size.
            regularization: Covariance and mass regularization constant.

        Returns:
            Container holding the configured moment recursion.
        """
        return cls(
            moments=GMMMomentRecursion.new(
                moments,
                step_size=step_size,
                regularization=regularization,
                name="gmm_moments",
            )
        )


class GaussianMixtureRecursion:
    """Stochastic-approximation Gaussian mixture recursion.

    Args:
        n_components: Number of mixture components.
        n_features: Observation dimensionality.
        step_size: Fixed, scheduled, or adaptive step size for moment updates.
        regularization: Covariance and mass regularization constant.
        rng: Optional random generator used for initial means.
    """

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
        """Update the model with one observation or a batch.

        Args:
            observations: Array with shape ``(n_features,)`` or
                ``(n_observations, n_features)``.

        Returns:
            ``self`` after processing every observation.
        """
        data = self._as_observations(observations)
        for x in data:
            self.recursions.moments.update(
                x,
                RecursionContext(iteration=self.recursions.moments.iteration, rng=self.rng),
            )
        return self

    @property
    def weights(self) -> np.ndarray:
        """Current mixture weights."""
        return self.parameters()[0]

    @property
    def means(self) -> np.ndarray:
        """Current component means."""
        return self.parameters()[1]

    @property
    def covariances(self) -> np.ndarray:
        """Current component covariance matrices."""
        return self.parameters()[2]

    def parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current mixture weights, means, and covariances."""
        return self.recursions.moments.parameters()

    def responsibilities(self, observations: np.ndarray) -> np.ndarray:
        """Return component responsibilities for one observation or a batch."""
        data = self._as_observations(observations)
        return np.vstack([self.recursions.moments.responsibility(x) for x in data])

    def assignments(self, observations: np.ndarray) -> np.ndarray:
        """Return maximum-responsibility component labels."""
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
