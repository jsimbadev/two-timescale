"""Adaptive step-size sources for Gaussian-mixture moment recursions.

The step-size sources in this module are exploratory compositions of the same
smoothed-statistic vocabulary used by NGPCA.  They are deliberately framed as
diagnostics of a running GMM recursion: each object observes the current model
and the next sample, updates an internal statistic, and exposes a scalar value
that can be used as the model's next step size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import EMARecursion, RecursionContext
from .recursions import AdaptiveGainRecursions, GMMMomentRecursion, GMMMoments
from .recursions.gmm_moments import gmm_moment_residual
from .recursions.smoothed_statistics import SmoothedStatisticControl


@dataclass
class AdaptiveGMMGain:
    """NGPCA-inspired scalar gain from component-membership disagreement.

    This adapts the NGPCA variance-matching structure to mixture weights by
    comparing the one-sample responsibility vector with the current mixture
    weights.

    Args:
        recursions: Adaptive-gain statistic recursions.
        smoothing: EMA smoothing coefficient for the diagnostics.
        upper: Upper bound for the returned scalar gain.
    """

    recursions: AdaptiveGainRecursions
    smoothing: float = 0.03
    upper: float = 0.18

    @classmethod
    def new(
        cls,
        n_components: int,
        initial_gain: float = 0.03,
        smoothing: float = 0.03,
        upper: float = 0.18,
    ) -> "AdaptiveGMMGain":
        """Construct a responsibility-disagreement gain.

        Args:
            n_components: Number of mixture components.
            initial_gain: Initial scalar gain.
            smoothing: EMA smoothing coefficient.
            upper: Upper bound for the returned gain.

        Returns:
            Configured adaptive GMM gain.
        """
        mixture_mass = np.full(n_components, 1.0 / n_components)
        return cls(
            recursions=AdaptiveGainRecursions.new(
                innovation_average=np.zeros(n_components),
                signal_average=mixture_mass,
                eigenvalue_average=mixture_mass,
                gain=initial_gain,
            ),
            smoothing=float(smoothing),
            upper=float(upper),
        )

    def value(self, context: RecursionContext | None = None) -> float:
        """Return the current clipped gain."""
        return min(max(float(self.recursions.gain), 0.0), self.upper)

    def update(
        self,
        moments: GMMMomentRecursion,
        observation: np.ndarray,
        context: RecursionContext | None = None,
    ) -> float:
        """Update the gain from a model state and observation.

        Args:
            moments: Running GMM moment recursion.
            observation: Observation about to update the GMM.
            context: Optional recursion context.

        Returns:
            Clipped scalar gain for the outer GMM recursion.
        """
        weights, _, _ = moments.parameters()
        responsibility = moments.responsibility(observation)
        self.recursions.update(
            signal=responsibility,
            eigenvalue=weights,
            step_size=self.smoothing,
            context=context,
        )
        return self.value(context)


class SmoothedCovarianceControl(SmoothedStatisticControl[GMMMomentRecursion, np.ndarray, np.ndarray]):
    """Scalar gain from a smoothed covariance mismatch diagnostic.

    The diagnostic tracks componentwise sample covariance matrices around the
    current means.  The gain map compares the smoothed diagnostic with the
    current covariance estimates and weights the relative mismatch by the
    observation responsibilities.
    """

    @classmethod
    def new(
        cls,
        n_components: int,
        n_features: int,
        initial_gain: float = 0.03,
        smoothing: float = 0.03,
        upper: float = 0.18,
    ) -> "SmoothedCovarianceControl":
        """Construct a covariance-mismatch gain."""
        initial_covariance = np.tile(np.eye(n_features), (n_components, 1, 1))
        return cls(
            statistic_average=EMARecursion(initial_covariance, name="covariance_average"),
            diagnostic=_component_sample_covariances,
            gain_map=_covariance_mismatch_gain,
            gain=float(initial_gain),
            smoothing=float(smoothing),
            lower=0.0,
            upper=float(upper),
        )


class SmoothedMomentResidualControl(SmoothedStatisticControl[GMMMomentRecursion, np.ndarray, GMMMoments]):
    """Scalar gain from the smoothed full GMM moment residual.

    The diagnostic is the same residual that drives the GMM moment recursion.
    Its norm is scaled by the current moment norm to produce a dimensionless
    gain.
    """

    @classmethod
    def new(
        cls,
        n_components: int,
        n_features: int,
        initial_gain: float = 0.03,
        smoothing: float = 0.03,
        upper: float = 0.18,
    ) -> "SmoothedMomentResidualControl":
        """Construct a moment-residual gain."""
        initial_residual = GMMMoments(
            mass=np.zeros(n_components),
            first=np.zeros((n_components, n_features)),
            second=np.zeros((n_components, n_features, n_features)),
        )
        return cls(
            statistic_average=EMARecursion(initial_residual, name="moment_residual_average"),
            diagnostic=_moment_residual,
            gain_map=_relative_moment_residual_gain,
            gain=float(initial_gain),
            smoothing=float(smoothing),
            lower=0.0,
            upper=float(upper),
        )


def _component_sample_covariances(
    moments: GMMMomentRecursion,
    observation: np.ndarray,
    context: RecursionContext | None = None,
) -> np.ndarray:
    _, means, _ = moments.parameters()
    centered = np.asarray(observation, dtype=float)[None, :] - means
    return centered[:, :, None] * centered[:, None, :]


def _covariance_mismatch_gain(
    covariance_average: np.ndarray,
    moments: GMMMomentRecursion,
    observation: np.ndarray,
    context: RecursionContext | None = None,
) -> float:
    responsibility = moments.responsibility(observation)
    _, _, covariances = moments.parameters()
    residual = covariance_average - covariances
    residual_norm = np.linalg.norm(residual, axis=(1, 2))
    scale = np.linalg.norm(covariance_average, axis=(1, 2)) + np.linalg.norm(covariances, axis=(1, 2))
    relative_mismatch = residual_norm / np.maximum(scale, np.finfo(float).eps)
    return float(responsibility @ relative_mismatch)


def _moment_residual(
    moments: GMMMomentRecursion,
    observation: np.ndarray,
    context: RecursionContext | None = None,
) -> GMMMoments:
    return gmm_moment_residual(
        moments.state,
        observation,
        regularization=moments.regularization,
    )


def _relative_moment_residual_gain(
    residual_average: GMMMoments,
    moments: GMMMomentRecursion,
    observation: np.ndarray,
    context: RecursionContext | None = None,
) -> float:
    scale = max(float(np.linalg.norm(_flatten_moments(moments.state))), np.finfo(float).eps)
    return float(np.linalg.norm(_flatten_moments(residual_average)) / scale)


def _flatten_moments(moments: GMMMoments) -> np.ndarray:
    return np.concatenate([
        moments.mass.ravel(),
        moments.first.ravel(),
        moments.second.ravel(),
    ])
