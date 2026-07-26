"""Neural Gas PCA recursions after Migenda, Moeller, and Schenck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .core import FreeParameter, LearningRate, Recursion, RecursionContext, VectorField

PotentialName = Literal["AR", "N", "H", "AV", "VRV", "VRR"]
WeightInitialization = Literal["normal", "uniform"]


@dataclass(frozen=True)
class NGPCAParameters:
    """Free parameters matching the MATLAB NGPCA defaults."""

    potential_function: PotentialName = "AR"
    softhard: FreeParameter = FreeParameter(0.0, name="softhard", lower=0.0, upper=1.0)
    learning_rate: LearningRate = LearningRate(0.99)
    activity: FreeParameter = FreeParameter(1.0, name="activity", lower=0.0, upper=1.0)
    rho_init: FreeParameter = FreeParameter(0.99, name="rho_init", lower=0.0, upper=1.0)
    rho_final: FreeParameter = FreeParameter(0.22, name="rho_final", lower=0.0, upper=1.0)
    mu: LearningRate = LearningRate(0.005, name="mu")
    rmax: FreeParameter = FreeParameter(200.0, name="rmax", lower=0.0)
    initial_eigenvalue: FreeParameter = FreeParameter(1.0, name="lambda", lower=0.0)


@dataclass
class LearningRateState:
    gamma_bar: np.ndarray
    eta_bar: np.ndarray
    l_bar: np.ndarray
    epsilon: float


@dataclass
class LocalPCAUnit:
    m: int
    center: np.ndarray
    weight: np.ndarray
    eigenvalue: np.ndarray
    sigma_sqr: float
    activity: float
    lr_state: LearningRateState
    x_c: np.ndarray
    y: np.ndarray
    alpha: float


def _orthonormal_columns(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix)
    return q


def _orthonormal_columns_svd(matrix: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(matrix, full_matrices=False)
    return u


def _learning_rate_field() -> VectorField[LearningRateState, dict[str, np.ndarray], LearningRateState]:
    def field(state: LearningRateState, observation: dict[str, np.ndarray], context: RecursionContext) -> LearningRateState:
        y = observation["y"]
        eigenvalue = observation["eigenvalue"]
        gamma = y * y - eigenvalue
        return LearningRateState(
            gamma_bar=gamma - state.gamma_bar,
            eta_bar=y * y - state.eta_bar,
            l_bar=eigenvalue - state.l_bar,
            epsilon=0.0,
        )

    return VectorField(field, name="ngpca_learning_rate_control")


class NeuralGasPCA:
    """Online neural-gas clustering with local PCA units.

    The implementation follows the update equations in the reference MATLAB
    code. The adaptive learning rate is represented as its own recursion and
    is then used as the control for the center/PCA recursions.
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
            lr_state = LearningRateState(
                gamma_bar=np.full(self.pca_dimensionality, initial_lambda**2 + initial_lambda, dtype=float),
                eta_bar=np.full(self.pca_dimensionality, initial_lambda**2, dtype=float),
                l_bar=eigenvalue.copy(),
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

    def fit(self, data: np.ndarray, iterations: int) -> "NeuralGasPCA":
        if not self.units:
            self.initialize(data)
        data = _as_2d(data)
        for _ in range(int(iterations)):
            self.fit_single(data[self.rng.integers(0, len(data))])
        return self

    def fit_single(self, x: np.ndarray) -> "NeuralGasPCA":
        if not self.units:
            self.initialize(np.asarray(x, dtype=float)[None, :])
        x = np.asarray(x, dtype=float).reshape(-1)
        n_features = x.size

        rankings: list[tuple[int, float]] = []
        total_epsilon = 0.0
        for index, unit in enumerate(self.units):
            unit.x_c = x - unit.center
            unit.y = unit.weight.T @ unit.x_c
            rankings.append((index, self._potential(unit, n_features)))
            total_epsilon += unit.lr_state.epsilon

        self.rho = (
            (float(self.params.rho_init) - float(self.params.rho_final))
            * total_epsilon
            / self.number_units
            + float(self.params.rho_final)
        )
        rankings.sort(key=lambda item: item[1])
        self.ranking = [[float(index + 1), float(potential)] for index, potential in rankings]
        h = np.exp((1.0 - np.arange(1, self.number_units + 1)) / max(self.rho, 1e-12))

        for rank, (index, _) in enumerate(rankings):
            self.units[index].activity += float(self.params.mu) * float(h[rank])
            if h[rank] < float(self.params.softhard):
                break
            self._adapt_unit(self.units[index], h=float(h[rank]), n_features=n_features)

        activity_sum = sum(unit.activity for unit in self.units)
        if activity_sum > 0.0:
            for unit in self.units:
                unit.activity *= self.number_units / activity_sum

        self.iteration += 1
        return self

    def centers(self) -> np.ndarray:
        return np.vstack([unit.center for unit in self.units])

    def labels(self, data: np.ndarray) -> np.ndarray:
        data = _as_2d(data)
        return np.array([min(self._rank_for(x), key=lambda item: item[1])[0] for x in data], dtype=int)

    def _rank_for(self, x: np.ndarray) -> list[tuple[int, float]]:
        result: list[tuple[int, float]] = []
        n_features = x.size
        for index, unit in enumerate(self.units):
            unit.x_c = x - unit.center
            unit.y = unit.weight.T @ unit.x_c
            result.append((index, self._potential(unit, n_features)))
        return result

    def _potential(self, unit: LocalPCAUnit, n_features: int) -> float:
        eigenvalue = np.maximum(unit.eigenvalue, np.finfo(float).eps)
        basis = float(unit.y @ (unit.y / eigenvalue))
        lambda_rest = 0.0
        if unit.m < n_features:
            lambda_rest = unit.sigma_sqr / (n_features - unit.m)
            lambda_rest = max(lambda_rest, np.finfo(float).eps)
            basis += (float(unit.x_c @ unit.x_c) - float(unit.y @ unit.y)) / lambda_rest

        adaptive = float(self.params.rmax) ** (1.0 - min(unit.activity, 1.0))
        potential = self.params.potential_function
        if potential == "N":
            control = 1.0
        elif potential == "AR":
            control = adaptive ** -2.0 * float(np.prod(eigenvalue)) ** (1.0 / n_features)
            if unit.m < n_features:
                control *= lambda_rest ** ((n_features - unit.m) / n_features)
        elif potential == "AV":
            control = (adaptive ** -2.0 * float(np.prod(eigenvalue))) ** (1.0 / n_features)
            if unit.m < n_features:
                control *= lambda_rest ** ((n_features - unit.m) / n_features)
        elif potential == "VRV":
            control = adaptive ** (-2.0 / n_features)
        elif potential == "VRR":
            control = adaptive ** -2.0
        elif potential == "H":
            if unit.m < n_features:
                control = (float(np.prod(eigenvalue)) * lambda_rest ** (n_features - unit.m)) ** (1.0 / n_features)
            else:
                control = np.sqrt(float(np.prod(eigenvalue))) ** (2.0 / n_features)
        else:
            raise ValueError(f"unknown potential function {potential!r}")
        return float(basis * control)

    def _adapt_unit(self, unit: LocalPCAUnit, h: float, n_features: int) -> None:
        a_lowpass = float(self.params.mu) * h

        lr_recursion = Recursion(
            unit.lr_state,
            _learning_rate_field(),
            step_size=a_lowpass,
            name="epsilon_control",
            iteration=self.iteration,
        )
        unit.lr_state = lr_recursion.advance(
            {"y": unit.y.copy(), "eigenvalue": unit.eigenvalue.copy()},
            RecursionContext(iteration=self.iteration, rng=self.rng),
        )
        denominator = unit.lr_state.eta_bar + unit.lr_state.l_bar
        gamma = unit.lr_state.gamma_bar / np.maximum(denominator, np.finfo(float).eps)
        unit.lr_state.epsilon = float(np.mean(np.abs(gamma)))
        unit.alpha = unit.lr_state.epsilon * h

        center_field = VectorField(lambda center, x_c, ctx: x_c, name="center_tracking")
        center_recursion = Recursion(unit.center, center_field, step_size=unit.alpha, name="unit_center")
        unit.center = center_recursion.advance(unit.x_c, RecursionContext(iteration=self.iteration, rng=self.rng))

        self._eforrlsa(unit)

        if n_features != unit.m:
            residual = float(unit.x_c @ unit.x_c - unit.y @ unit.y)
            unit.sigma_sqr += unit.alpha * (residual - unit.sigma_sqr)

    def _eforrlsa(self, unit: LocalPCAUnit) -> None:
        old_weight = unit.weight.copy()
        l2 = np.zeros(unit.m, dtype=float)
        p = np.zeros(unit.m, dtype=float)
        q = np.zeros((unit.m, unit.m), dtype=float)
        r = np.zeros((unit.m, unit.m), dtype=float)
        t = 0.0
        d = float(unit.x_c @ unit.x_c)

        for i in range(unit.m):
            alpha_l = (1.0 - unit.alpha) * unit.eigenvalue[i]
            beta_y = unit.alpha * unit.y[i]
            if i > 0:
                t += p[i - 1] * p[i - 1]
                d = max(0.0, d - unit.y[i - 1] * unit.y[i - 1])

            s = (alpha_l + unit.alpha * d) * unit.y[i]
            l2[i] = alpha_l * alpha_l + beta_y * (alpha_l * unit.y[i] + s)
            n2 = max(float(l2[i] - s * s * t), np.finfo(float).eps)
            n = np.sqrt(n2)
            p[i] = (beta_y - s * t) / n

            column = p[i] * unit.x_c
            for j in range(i + 1):
                if j < i:
                    r[i, j] = r[i - 1, j] + p[i - 1] * q[i - 1, j]
                    q[i, j] = -(beta_y * unit.y[j] + s * r[i, j]) / n
                else:
                    q[i, j] = alpha_l / n
                column = column + q[i, j] * old_weight[:, j]
            unit.weight[:, i] = column

        unit.eigenvalue = np.sqrt(np.maximum(l2, 0.0))


def _as_2d(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")
    return data
