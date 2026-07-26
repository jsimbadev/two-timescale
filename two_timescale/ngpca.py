"""Neural Gas PCA recursions after Migenda, Moeller, and Schenck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .core import EMARecursion, FreeParameter, LearningRate, Recursion, RecursionContext, VectorField
from .potentials import LocalPCAGeometry, PotentialSpec, as_potential

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
class LearningRateRecursions:
    gamma_bar: EMARecursion[np.ndarray]
    eta_bar: EMARecursion[np.ndarray]
    lambda_bar: EMARecursion[np.ndarray]
    epsilon: float

    @classmethod
    def from_initial_lambda(cls, pca_dimensionality: int, initial_lambda: float, epsilon: float) -> "LearningRateRecursions":
        eigenvalue = np.full(pca_dimensionality, initial_lambda, dtype=float)
        return cls(
            gamma_bar=EMARecursion(
                np.full(pca_dimensionality, initial_lambda**2 + initial_lambda, dtype=float),
                name="gamma_bar",
            ),
            eta_bar=EMARecursion(
                np.full(pca_dimensionality, initial_lambda**2, dtype=float),
                name="eta_bar",
            ),
            lambda_bar=EMARecursion(eigenvalue.copy(), name="lambda_bar"),
            epsilon=float(epsilon),
        )

    def advance(self, y: np.ndarray, eigenvalue: np.ndarray, step_size: float, context: RecursionContext) -> float:
        eta = y * y
        gamma = eta - eigenvalue

        self.gamma_bar.advance(gamma, step_size=step_size, context=context)
        self.eta_bar.advance(eta, step_size=step_size, context=context)
        self.lambda_bar.advance(eigenvalue, step_size=step_size, context=context)

        denominator = self.eta_bar.state + self.lambda_bar.state
        normalized = self.gamma_bar.state / np.maximum(denominator, np.finfo(float).eps)
        self.epsilon = float(np.mean(np.abs(normalized)))
        return self.epsilon


@dataclass
class CVTRecursion:
    """Centroidal vector quantization update for a local PCA center."""

    state: np.ndarray
    iteration: int = 0

    def __post_init__(self) -> None:
        self.recursion = Recursion(
            self.state.copy(),
            _cvt_field(),
            step_size=1.0,
            name="cvt_center",
            iteration=self.iteration,
        )

    @classmethod
    def from_center(cls, center: np.ndarray) -> "CVTRecursion":
        return cls(center.copy())

    def advance(
        self,
        x_c: np.ndarray,
        alpha: float,
        context: RecursionContext | None = None,
    ) -> np.ndarray:
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.recursion.step_size = float(alpha)
        self.state = self.recursion.advance(x_c, context)
        self.iteration = self.recursion.iteration
        return self.state


@dataclass
class ResidualVarianceRecursion:
    """EMA for variance outside the represented PCA subspace."""

    state: float
    iteration: int = 0

    def __post_init__(self) -> None:
        self.recursion = EMARecursion(float(self.state), name="residual_variance", iteration=self.iteration)

    @classmethod
    def from_sigma_sqr(cls, sigma_sqr: float) -> "ResidualVarianceRecursion":
        return cls(float(sigma_sqr))

    def advance(
        self,
        residual: float,
        alpha: float,
        context: RecursionContext | None = None,
    ) -> float:
        self.state = float(self.recursion.advance(float(residual), step_size=float(alpha), context=context))
        self.iteration = self.recursion.iteration
        return self.state


@dataclass
class EFORRLSAState:
    weight: np.ndarray
    eigenvalue: np.ndarray


@dataclass(frozen=True)
class EFORRLSAObservation:
    x_c: np.ndarray
    y: np.ndarray
    alpha: float


@dataclass
class EFORRLSARecursion:
    """Robust recursive least-squares PCA shape update as a recursion."""

    state: EFORRLSAState
    iteration: int = 0

    def __post_init__(self) -> None:
        self.recursion = Recursion(
            self.state,
            _eforrlsa_field(),
            step_size=1.0,
            name="eforrlsa_shape",
            iteration=self.iteration,
        )

    @classmethod
    def from_shape(cls, weight: np.ndarray, eigenvalue: np.ndarray) -> "EFORRLSARecursion":
        return cls(EFORRLSAState(weight=weight.copy(), eigenvalue=eigenvalue.copy()))

    def advance(
        self,
        x_c: np.ndarray,
        y: np.ndarray,
        alpha: float,
        context: RecursionContext | None = None,
    ) -> EFORRLSAState:
        if context is None:
            context = RecursionContext(iteration=self.recursion.iteration)
        self.state = self.recursion.advance(EFORRLSAObservation(x_c=x_c, y=y, alpha=float(alpha)), context)
        self.iteration = self.recursion.iteration
        return self.state


@dataclass
class LocalPCAUnit:
    m: int
    center: np.ndarray
    weight: np.ndarray
    eigenvalue: np.ndarray
    sigma_sqr: float
    activity: float
    lr_state: LearningRateRecursions
    x_c: np.ndarray
    y: np.ndarray
    alpha: float


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q


def _orthonormal_columns_svd(matrix: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return u


def _cvt_field() -> VectorField[np.ndarray, np.ndarray, np.ndarray]:
    return VectorField(lambda center, x_c, ctx: x_c, name="cvt")


def _eforrlsa_field() -> VectorField[EFORRLSAState, EFORRLSAObservation, EFORRLSAState]:
    def field(state: EFORRLSAState, observation: EFORRLSAObservation, context: RecursionContext) -> EFORRLSAState:
        return _eforrlsa_delta(state, observation)

    return VectorField(field, name="eforrlsa")


def _eforrlsa_delta(state: EFORRLSAState, observation: EFORRLSAObservation) -> EFORRLSAState:
    next_state = _eforrlsa_next_state(state, observation)
    return EFORRLSAState(
        weight=next_state.weight - state.weight,
        eigenvalue=next_state.eigenvalue - state.eigenvalue,
    )


def _eforrlsa_next_state(state: EFORRLSAState, observation: EFORRLSAObservation) -> EFORRLSAState:
    weight = state.weight.copy()
    old_weight = state.weight.copy()
    eigenvalue = state.eigenvalue
    x_c = observation.x_c
    y = observation.y
    alpha = observation.alpha
    m = eigenvalue.size

    l2 = np.zeros(m, dtype=float)
    p = np.zeros(m, dtype=float)
    q = np.zeros((m, m), dtype=float)
    r = np.zeros((m, m), dtype=float)
    t = 0.0
    d = float(x_c @ x_c)

    for i in range(m):
        alpha_l = (1.0 - alpha) * eigenvalue[i]
        beta_y = alpha * y[i]
        if i > 0:
            t += p[i - 1] * p[i - 1]
            d = max(0.0, d - y[i - 1] * y[i - 1])

        s = (alpha_l + alpha * d) * y[i]
        l2[i] = alpha_l * alpha_l + beta_y * (alpha_l * y[i] + s)
        n2 = max(float(l2[i] - s * s * t), np.finfo(float).eps)
        n = np.sqrt(n2)
        p[i] = (beta_y - s * t) / n

        column = p[i] * x_c
        for j in range(i + 1):
            if j < i:
                r[i, j] = r[i - 1, j] + p[i - 1] * q[i - 1, j]
                q[i, j] = -(beta_y * y[j] + s * r[i, j]) / n
            else:
                q[i, j] = alpha_l / n
            column = column + q[i, j] * old_weight[:, j]
        weight[:, i] = column

    return EFORRLSAState(weight=weight, eigenvalue=np.sqrt(np.maximum(l2, 0.0)))


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
            lr_state = LearningRateRecursions.from_initial_lambda(
                self.pca_dimensionality,
                initial_lambda,
                epsilon=float(self.params.learning_rate),
            )
            self.units.append(
                LocalPCAUnit(
                    m=self.pca_dimensionality,
                    center=center.astype(float),
                    weight=weight.astype(float),
                    eigenvalue=eigenvalue,
                    sigma_sqr=initial_lambda,
                    activity=float(self.params.activity),
                    lr_state=lr_state,
                    x_c=np.zeros(n_features, dtype=float),
                    y=np.zeros(self.pca_dimensionality, dtype=float),
                    alpha=float(self.params.learning_rate),
                )
            )
        return self

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
        average_epsilon = sum(unit.lr_state.epsilon for unit in self.units) / self.number_units
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
            self._adapt_unit(self.units[index], h=h, n_features=n_features)

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

    def _adapt_unit(self, unit: LocalPCAUnit, h: float, n_features: int) -> None:
        a_lowpass = float(self.params.mu) * h
        context = RecursionContext(iteration=self.iteration, rng=self.rng)
        unit.alpha = unit.lr_state.advance(unit.y.copy(), unit.eigenvalue.copy(), a_lowpass, context) * h

        center_recursion = CVTRecursion.from_center(unit.center)
        unit.center = center_recursion.advance(unit.x_c, unit.alpha, context)

        shape_recursion = EFORRLSARecursion.from_shape(unit.weight, unit.eigenvalue)
        shape = shape_recursion.advance(unit.x_c, unit.y, unit.alpha, context)
        unit.weight = shape.weight
        unit.eigenvalue = shape.eigenvalue

        if n_features != unit.m:
            residual = float(unit.x_c @ unit.x_c - unit.y @ unit.y)
            residual_variance = ResidualVarianceRecursion.from_sigma_sqr(unit.sigma_sqr)
            unit.sigma_sqr = residual_variance.advance(residual, unit.alpha, context)


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
