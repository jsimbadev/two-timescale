"""Adaptive learning-rate recursions for local PCA units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import EMARecursion, RecursionContext


@dataclass
class LearningRateRecursions:
    gamma_bar: EMARecursion[np.ndarray]
    eta_bar: EMARecursion[np.ndarray]
    lambda_bar: EMARecursion[np.ndarray]
    epsilon: float

    @classmethod
    def new(
        cls,
        gamma_bar: np.ndarray,
        eta_bar: np.ndarray,
        lambda_bar: np.ndarray,
        epsilon: float,
    ) -> "LearningRateRecursions":
        return cls(
            gamma_bar=EMARecursion(np.array(gamma_bar, dtype=float, copy=True), name="gamma_bar"),
            eta_bar=EMARecursion(np.array(eta_bar, dtype=float, copy=True), name="eta_bar"),
            lambda_bar=EMARecursion(np.array(lambda_bar, dtype=float, copy=True), name="lambda_bar"),
            epsilon=float(epsilon),
        )

    def update(
        self,
        y: np.ndarray,
        eigenvalue: np.ndarray,
        step_size: float,
        context: RecursionContext | None = None,
    ) -> float:
        eta = y * y
        gamma = eta - eigenvalue

        self.gamma_bar.advance(gamma, step_size=step_size, context=context)
        self.eta_bar.advance(eta, step_size=step_size, context=context)
        self.lambda_bar.advance(eigenvalue, step_size=step_size, context=context)

        denominator = self.eta_bar.state + self.lambda_bar.state
        normalized = self.gamma_bar.state / np.maximum(denominator, np.finfo(float).eps)
        self.epsilon = float(np.mean(np.abs(normalized)))
        return self.epsilon
