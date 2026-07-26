"""Neural Gas PCA recursions after Migenda, Moeller, and Schenck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .core import FreeParameter, LearningRate, RecursionContext
from .potentials import LocalPCAGeometry, PotentialSpec, as_potential
from .recursions import (
    AdaptiveGainRecursions,
    EFORRLSARecursion,
    EFORRLSAState,
    ScalarTrackingRecursion,
    StateTrackingRecursion,
)

WeightInitialization = Literal["normal", "uniform"]


@dataclass(frozen=True)
class NGPCAParameters:
    """Free parameters matching the MATLAB NGPCA defaults."""

    potential_function: PotentialSpec = "AR"
    softhard: FreeParameter = FreeParameter(0.0, name="softhard", lower=0.0, upper=1.0)
    learning_rate: LearningRate = LearningRate(0.99)
    activity: FreeParameter = FreeParameter(1.0, name="activity", lower=0.0, upper=1.0)
    rho_init: FreeParameter = FreeParameter(0.99, name="rho_init", lower=0.0, upper=1.0)
    rho_final: FreeParameter = FreeParameter(0.22, name="rho_final", lower=0.0, upper=1.0)
    mu: LearningRate = LearningRate(0.005, name="mu")
    rmax: FreeParameter = FreeParameter(200.0, name="rmax", lower=0.0)
    initial_eigenvalue: FreeParameter = FreeParameter(1.0, name="lambda", lower=0.0)


@dataclass
class LocalPCAUnit:
    m: int
    center: np.ndarray
    weight: np.ndarray
    eigenvalue: np.ndarray
    sigma_sqr: float
    activity: float
    x_c: np.ndarray
    y: np.ndarray
    alpha: float


@dataclass
class NGPCAUnitRecursions:
    learning_rate: AdaptiveGainRecursions
    center: StateTrackingRecursion
    shape: EFORRLSARecursion
    residual_variance: ScalarTrackingRecursion


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q


def _orthonormal_columns_svd(matrix: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return u


class NeuralGasPCA:
    """Online neural-gas clustering with local PCA units.

    The adaptive learning rate follows the paper's variance-matching algorithm:
    gamma_bar, eta_bar, and lambda_bar are each tracked by their own EMA
    recursion, then combined into a per-unit learning rate.
    """

    def __init__(
        self,
        number_units: int,
        pca_dimensionality: int = 2,
        params: NGPCAParameters | None = None,
        rng: Any | None = None,
        weight_initialization: WeightInitialization = "normal",
    ) -> None:
        self.number_units = int(number_units)
        self.pca_dimensionality = int(pca_dimensionality)
        self.params = params or NGPCAParameters()
        self.potential = as_potential(self.params.potential_function, rmax=float(self.params.rmax))
        self.rng = rng or np.random.default_rng()
        self.weight_initialization = weight_initialization
        self.units: list[LocalPCAUnit] = []
        self.recursions: list[NGPCAUnitRecursions] = []
        self.rho = float(self.params.rho_init)
        self.ranking: list[list[float]] = []
        self.iteration = 0

    def initialize(self, data: np.ndarray) -> "NeuralGasPCA":
        data = _as_2d(data)
        n_samples, n_features = data.shape
        if n_features < 2:
            raise ValueError("data dimensionality must be at least two")
        if self.pca_dimensionality > n_features:
            raise ValueError("PCA dimensionality must not exceed data dimensionality")

        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        span = np.maximum(maxs - mins, 1e-12)
        initial_lambda = float(self.params.initial_eigenvalue)

        self.units = []
        self.recursions = []
        for _ in range(self.number_units):
            if n_samples == 1:
                center = data[0] * self.rng.random(n_features)
            else:
                center = 2.0 * span * self.rng.random(n_features) + 2.0 * mins
            if self.weight_initialization == "uniform":
                weight_matrix = self.rng.random((n_features, self.pca_dimensionality))
                weight = _orthonormal_columns_svd(weight_matrix)
            else:
                weight = _orthonormal_columns(self.rng.normal(size=(n_features, self.pca_dimensionality)))
            eigenvalue = np.full(self.pca_dimensionality, initial_lambda, dtype=float)
            self.units.append(
                LocalPCAUnit(
                    m=self.pca_dimensionality,
                    center=center.astype(float),
                    weight=weight.astype(float),
                    eigenvalue=eigenvalue,
                    sigma_sqr=initial_lambda,
                    activity=float(self.params.activity),
                    x_c=np.zeros(n_features, dtype=float),
                    y=np.zeros(self.pca_dimensionality, dtype=float),
                    alpha=float(self.params.learning_rate),
                )
            )
            self.recursions.append(
                NGPCAUnitRecursions(
                    learning_rate=self._initial_learning_rate_recursions(eigenvalue),
                    center=StateTrackingRecursion.new(center, name="cvt_center"),
                    shape=EFORRLSARecursion.new(EFORRLSAState(weight, eigenvalue)),
                    residual_variance=ScalarTrackingRecursion.new(initial_lambda, name="residual_variance"),
                )
            )
        return self

    def _initial_learning_rate_recursions(self, eigenvalue: np.ndarray) -> AdaptiveGainRecursions:
        return AdaptiveGainRecursions.new(
            innovation_average=eigenvalue**2 + eigenvalue,
            signal_average=eigenvalue**2,
            eigenvalue_average=eigenvalue,
            gain=float(self.params.learning_rate),
        )

    def advance(self, observations: np.ndarray) -> "NeuralGasPCA":
        data = _as_observations(observations)
        if not self.units:
            self.initialize(data)
        for x in data:
            self._advance_one(x)
        return self

    def _advance_one(self, x: np.ndarray) -> None:
        observation = self._validate_observation(x)
        n_features = observation.size

        self._project_units(observation)
        self.rho = self._neighborhood_range()
        rankings = self._rank_units(n_features)
        neighborhoods = self._neighborhood_weights()

        self._adapt_ranked_units(rankings, neighborhoods, n_features)
        self._normalize_activity()
        self.iteration += 1

    def _validate_observation(self, x: np.ndarray) -> np.ndarray:
        if not self.units:
            self.initialize(np.asarray(x, dtype=float)[None, :])
        observation = np.asarray(x, dtype=float).reshape(-1)
        if self.units[0].center.size != observation.size:
            raise ValueError("observation dimensionality must match initialized units")
        return observation

    def _project_units(self, x: np.ndarray) -> None:
        for unit in self.units:
            unit.x_c = x - unit.center
            unit.y = unit.weight.T @ unit.x_c

    def _neighborhood_range(self) -> float:
        average_epsilon = sum(recursions.learning_rate.gain for recursions in self.recursions) / self.number_units
        return (
            (float(self.params.rho_init) - float(self.params.rho_final))
            * average_epsilon
            + float(self.params.rho_final)
        )

    def _rank_units(self, n_features: int) -> list[tuple[int, float]]:
        rankings = [
            (index, self._score_unit(unit, n_features))
            for index, unit in enumerate(self.units)
        ]
        rankings.sort(key=lambda item: item[1])
        self.ranking = [[float(index + 1), float(potential)] for index, potential in rankings]
        return rankings

    def _neighborhood_weights(self) -> np.ndarray:
        ranks = np.arange(1, self.number_units + 1)
        return np.exp((1.0 - ranks) / max(self.rho, 1e-12))

    def _adapt_ranked_units(
        self,
        rankings: list[tuple[int, float]],
        neighborhoods: np.ndarray,
        n_features: int,
    ) -> None:
        for rank, (index, _) in enumerate(rankings):
            h = float(neighborhoods[rank])
            self.units[index].activity += float(self.params.mu) * h
            if h < float(self.params.softhard):
                break
            self._adapt_unit(self.units[index], self.recursions[index], h=h, n_features=n_features)

    def _normalize_activity(self) -> None:
        activity_sum = sum(unit.activity for unit in self.units)
        if activity_sum > 0.0:
            for unit in self.units:
                unit.activity *= self.number_units / activity_sum

    def centers(self) -> np.ndarray:
        return np.vstack([unit.center for unit in self.units])

    def labels(self, data: np.ndarray) -> np.ndarray:
        data = _as_2d(data)
        return np.array([min(self._rank_for(x), key=lambda item: item[1])[0] for x in data], dtype=int)

    def _rank_for(self, x: np.ndarray) -> list[tuple[int, float]]:
        result: list[tuple[int, float]] = []
        n_features = x.size
        for index, unit in enumerate(self.units):
            result.append((index, self._score_observation(unit, x, n_features)))
        return result

    def _score_observation(self, unit: LocalPCAUnit, x: np.ndarray, n_features: int) -> float:
        x_c = x - unit.center
        y = unit.weight.T @ x_c
        return self.potential(self._geometry(unit, x_c, y, n_features))

    def _score_unit(self, unit: LocalPCAUnit, n_features: int) -> float:
        return self.potential(self._geometry(unit, unit.x_c, unit.y, n_features))

    def _geometry(self, unit: LocalPCAUnit, x_c: np.ndarray, y: np.ndarray, n_features: int) -> LocalPCAGeometry:
        return LocalPCAGeometry(
            x_c=x_c,
            y=y,
            eigenvalue=unit.eigenvalue,
            sigma_sqr=unit.sigma_sqr,
            activity=unit.activity,
            pca_dimensionality=unit.m,
            data_dimensionality=n_features,
        )

    def _adapt_unit(
        self,
        unit: LocalPCAUnit,
        recursions: NGPCAUnitRecursions,
        h: float,
        n_features: int,
    ) -> None:
        a_lowpass = float(self.params.mu) * h
        context = RecursionContext(iteration=self.iteration, rng=self.rng)
        unit.alpha = recursions.learning_rate.update(
            signal=unit.y * unit.y,
            eigenvalue=unit.eigenvalue.copy(),
            step_size=a_lowpass,
            context=context,
        ) * h

        unit.center = recursions.center.update(unit.x_c, unit.alpha, context)

        shape = recursions.shape.update(unit.x_c, unit.y, unit.alpha, context)
        unit.weight = shape.weight
        unit.eigenvalue = shape.eigenvalue

        if n_features != unit.m:
            residual = float(unit.x_c @ unit.x_c - unit.y @ unit.y)
            unit.sigma_sqr = recursions.residual_variance.update(residual, unit.alpha, context)


def _as_2d(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    return data


def _as_observations(observations: np.ndarray) -> np.ndarray:
    data = np.asarray(observations, dtype=float)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    raise ValueError("observations must be a one- or two-dimensional array")
