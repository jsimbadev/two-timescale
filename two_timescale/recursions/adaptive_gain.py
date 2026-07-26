"""Adaptive gain recursions built from stochastic-approximation averages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import EMARecursion, RecursionContext


@dataclass
class AdaptiveGainRecursions:
    innovation_average: EMARecursion[np.ndarray]
    signal_average: EMARecursion[np.ndarray]
    scale_average: EMARecursion[np.ndarray]
    gain: float

    @classmethod
    def new(
        cls,
        innovation_average: np.ndarray,
        signal_average: np.ndarray,
        scale_average: np.ndarray,
        gain: float,
    ) -> "AdaptiveGainRecursions":
        return cls(
            innovation_average=EMARecursion(
                np.array(innovation_average, dtype=float, copy=True),
                name="innovation_average",
            ),
            signal_average=EMARecursion(
                np.array(signal_average, dtype=float, copy=True),
                name="signal_average",
            ),
            scale_average=EMARecursion(
                np.array(scale_average, dtype=float, copy=True),
                name="scale_average",
            ),
            gain=float(gain),
        )

    def update(
        self,
        signal: np.ndarray,
        scale: np.ndarray,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> float:
        innovation = signal - scale

        self.innovation_average.advance(innovation, step_size=step_size, context=context)
        self.signal_average.advance(signal, step_size=step_size, context=context)
        self.scale_average.advance(scale, step_size=step_size, context=context)

        denominator = self.signal_average.state + self.scale_average.state
        normalized = self.innovation_average.state / np.maximum(denominator, np.finfo(float).eps)
        self.gain = float(np.mean(np.abs(normalized)))
        return self.gain
