r"""Adaptive gain recursion built from smoothed NGPCA statistics.

Migenda, Moeller, and Schenck define a local PCA learning-rate estimate by
tracking how well each squared projection ``y_i^2`` matches the current
eigenvalue estimate ``lambda_i``.  The implemented statistics are the low-pass
filters

.. math::

    \bar{\gamma}_{j,t+1}
        = \bar{\gamma}_{j,t} + \mu_j(\gamma_{j,t} - \bar{\gamma}_{j,t}),
        \quad \gamma_{j,t} = y_{j,t}^2 - \lambda_{j,t},

.. math::

    \bar{\eta}_{j,t+1}
        = \bar{\eta}_{j,t} + \mu_j(y_{j,t}^2 - \bar{\eta}_{j,t}),
        \quad
    \bar{\lambda}_{j,t+1}
        = \bar{\lambda}_{j,t} + \mu_j(\lambda_{j,t} - \bar{\lambda}_{j,t}).

The scalar gain is then

.. math::

    \epsilon_j
      = \frac{1}{m}\sum_{i=1}^m
        \left|\frac{\bar{\gamma}_{j,i}}
        {\bar{\eta}_{j,i} + \bar{\lambda}_{j,i}}\right|.

References:
    Migenda, Moeller, and Schenck (2024):
        Migenda, N., Moeller, R., & Schenck, W. (2024). Adaptive local
        Principal Component Analysis improves the clustering of
        high-dimensional data. *Pattern Recognition*, 146, 110030.
        Equations (17)--(20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import EMARecursion, RecursionContext
from .smoothed_statistics import SmoothedStatisticTransform


@dataclass(frozen=True)
class AdaptiveGainObservation:
    """Observation passed to the NGPCA adaptive-gain statistics.

    Args:
        signal: Squared projected signal ``y_j ** 2`` for one local PCA unit.
        eigenvalue: Current local PCA eigenvalue vector ``lambda_j``.
    """

    signal: np.ndarray
    eigenvalue: np.ndarray


@dataclass
class AdaptiveGainRecursions:
    """Orchestrate the three smoothed statistics defining an NGPCA gain.

    The object owns the statistic recursions for
    ``innovation_average`` (``gamma_bar``), ``signal_average`` (``eta_bar``),
    and ``eigenvalue_average`` (``lambda_bar``).  The public average properties
    are intentionally kept in the terminology used by the NGPCA MATLAB export.

    Args:
        innovation_statistic: Smoothed statistic for ``signal - eigenvalue``.
        signal_statistic: Smoothed statistic for ``signal``.
        eigenvalue_statistic: Smoothed statistic for ``eigenvalue``.
        gain: Current scalar gain before neighborhood scaling.
    """

    innovation_statistic: SmoothedStatisticTransform[
        "AdaptiveGainRecursions",
        AdaptiveGainObservation,
        np.ndarray,
        np.ndarray,
    ]
    signal_statistic: SmoothedStatisticTransform[
        "AdaptiveGainRecursions",
        AdaptiveGainObservation,
        np.ndarray,
        np.ndarray,
    ]
    eigenvalue_statistic: SmoothedStatisticTransform[
        "AdaptiveGainRecursions",
        AdaptiveGainObservation,
        np.ndarray,
        np.ndarray,
    ]
    gain: float

    @classmethod
    def new(
        cls,
        innovation_average: np.ndarray,
        signal_average: np.ndarray,
        eigenvalue_average: np.ndarray,
        gain: float,
    ) -> "AdaptiveGainRecursions":
        """Construct NGPCA adaptive-gain statistic recursions.

        Args:
            innovation_average: Initial ``gamma_bar`` statistic.
            signal_average: Initial ``eta_bar`` statistic.
            eigenvalue_average: Initial ``lambda_bar`` statistic.
            gain: Initial scalar gain ``epsilon``.

        Returns:
            Adaptive-gain recursion bundle with copied initial arrays.
        """
        return cls(
            innovation_statistic=SmoothedStatisticTransform.new(
                initial_statistic=np.array(innovation_average, dtype=float, copy=True),
                diagnostic=_innovation_statistic,
                statistic_map=_identity_statistic,
                name="innovation_average",
            ),
            signal_statistic=SmoothedStatisticTransform.new(
                initial_statistic=np.array(signal_average, dtype=float, copy=True),
                diagnostic=_signal_statistic,
                statistic_map=_identity_statistic,
                name="signal_average",
            ),
            eigenvalue_statistic=SmoothedStatisticTransform.new(
                initial_statistic=np.array(eigenvalue_average, dtype=float, copy=True),
                diagnostic=_eigenvalue_statistic,
                statistic_map=_identity_statistic,
                name="eigenvalue_average",
            ),
            gain=float(gain),
        )

    @property
    def innovation_average(self) -> EMARecursion[np.ndarray]:
        """EMA recursion storing the smoothed variance mismatch."""
        return self.innovation_statistic.statistic_average

    @property
    def signal_average(self) -> EMARecursion[np.ndarray]:
        """EMA recursion storing the smoothed squared projected signal."""
        return self.signal_statistic.statistic_average

    @property
    def eigenvalue_average(self) -> EMARecursion[np.ndarray]:
        """EMA recursion storing the smoothed eigenvalue estimate."""
        return self.eigenvalue_statistic.statistic_average

    def update(
        self,
        signal: np.ndarray,
        eigenvalue: np.ndarray,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> float:
        """Update all NGPCA gain statistics and return the new gain.

        Args:
            signal: Squared projected signal ``y_j ** 2``.
            eigenvalue: Current eigenvalue vector for the same unit.
            step_size: Low-pass filter coefficient ``mu_j`` for this update.
            context: Optional recursion context.

        Returns:
            Mean absolute normalized variance mismatch.
        """
        observation = AdaptiveGainObservation(
            signal=np.asarray(signal, dtype=float),
            eigenvalue=np.asarray(eigenvalue, dtype=float),
        )

        self.innovation_statistic.update(self, observation, step_size=step_size, context=context)
        self.signal_statistic.update(self, observation, step_size=step_size, context=context)
        self.eigenvalue_statistic.update(self, observation, step_size=step_size, context=context)

        denominator = self.signal_average.state + self.eigenvalue_average.state
        normalized = self.innovation_average.state / np.maximum(denominator, np.finfo(float).eps)
        self.gain = float(np.mean(np.abs(normalized)))
        return self.gain


def _innovation_statistic(
    state: AdaptiveGainRecursions,
    observation: AdaptiveGainObservation,
    context: RecursionContext | None,
) -> np.ndarray:
    return observation.signal - observation.eigenvalue


def _signal_statistic(
    state: AdaptiveGainRecursions,
    observation: AdaptiveGainObservation,
    context: RecursionContext | None,
) -> np.ndarray:
    return observation.signal


def _eigenvalue_statistic(
    state: AdaptiveGainRecursions,
    observation: AdaptiveGainObservation,
    context: RecursionContext | None,
) -> np.ndarray:
    return observation.eigenvalue


def _identity_statistic(
    statistic: np.ndarray,
    state: AdaptiveGainRecursions,
    observation: AdaptiveGainObservation,
    context: RecursionContext | None,
) -> np.ndarray:
    return statistic
