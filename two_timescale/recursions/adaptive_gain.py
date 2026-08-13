"""Adaptive gain recursion built from smoothed NGPCA statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import EMARecursion, RecursionContext
from .smoothed_statistics import SmoothedStatisticTransform


@dataclass(frozen=True)
class AdaptiveGainObservation:
    signal: np.ndarray
    eigenvalue: np.ndarray


@dataclass
class AdaptiveGainRecursions:
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
        return self.innovation_statistic.statistic_average

    @property
    def signal_average(self) -> EMARecursion[np.ndarray]:
        return self.signal_statistic.statistic_average

    @property
    def eigenvalue_average(self) -> EMARecursion[np.ndarray]:
        return self.eigenvalue_statistic.statistic_average

    def update(
        self,
        signal: np.ndarray,
        eigenvalue: np.ndarray,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> float:
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
