"""Potential functions for local PCA unit competition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np

PotentialName = Literal["AR", "N", "H", "AV", "VRV", "VRR"]


@dataclass(frozen=True)
class LocalPCAGeometry:
    """Geometry needed to rank one local PCA unit for one observation."""

    x_c: np.ndarray
    y: np.ndarray
    eigenvalue: np.ndarray
    sigma_sqr: float
    activity: float
    pca_dimensionality: int
    data_dimensionality: int


class PotentialFunction(Protocol):
    """Callable ranking criterion for local PCA unit competition."""

    name: str

    def __call__(self, geometry: LocalPCAGeometry) -> float:
        ...


@dataclass(frozen=True)
class NamedPotential:
    """Potential functions evaluated in Migenda, Moeller, and Schenck."""

    name: PotentialName = "AR"
    rmax: float = 200.0

    def __call__(self, geometry: LocalPCAGeometry) -> float:
        basis, lambda_rest, eigenvalue = _mahalanobis_basis(geometry)
        adaptive = self.rmax ** (1.0 - min(geometry.activity, 1.0))

        if self.name == "N":
            control = 1.0
        elif self.name == "AR":
            control = adaptive ** -2.0 * float(np.prod(eigenvalue)) ** (1.0 / geometry.data_dimensionality)
            if geometry.pca_dimensionality < geometry.data_dimensionality:
                control *= lambda_rest ** (
                    (geometry.data_dimensionality - geometry.pca_dimensionality)
                    / geometry.data_dimensionality
                )
        elif self.name == "AV":
            control = (adaptive ** -2.0 * float(np.prod(eigenvalue))) ** (
                1.0 / geometry.data_dimensionality
            )
            if geometry.pca_dimensionality < geometry.data_dimensionality:
                control *= lambda_rest ** (
                    (geometry.data_dimensionality - geometry.pca_dimensionality)
                    / geometry.data_dimensionality
                )
        elif self.name == "VRV":
            control = adaptive ** (-2.0 / geometry.data_dimensionality)
        elif self.name == "VRR":
            control = adaptive ** -2.0
        elif self.name == "H":
            volume = float(np.prod(eigenvalue))
            if geometry.pca_dimensionality < geometry.data_dimensionality:
                volume *= lambda_rest ** (geometry.data_dimensionality - geometry.pca_dimensionality)
            control = volume ** (1.0 / geometry.data_dimensionality)
        else:
            raise ValueError(f"unknown potential function {self.name!r}")

        return float(basis * control)


PotentialSpec = PotentialName | PotentialFunction | Callable[[LocalPCAGeometry], float]


def as_potential(spec: PotentialSpec, rmax: float = 200.0) -> PotentialFunction:
    if isinstance(spec, str):
        return NamedPotential(spec, rmax=rmax)
    if hasattr(spec, "name"):
        return spec  # type: ignore[return-value]
    return _CallablePotential(spec)


def _mahalanobis_basis(geometry: LocalPCAGeometry) -> tuple[float, float, np.ndarray]:
    eigenvalue = np.maximum(geometry.eigenvalue, np.finfo(float).eps)
    basis = float(geometry.y @ (geometry.y / eigenvalue))

    lambda_rest = 0.0
    if geometry.pca_dimensionality < geometry.data_dimensionality:
        lambda_rest = geometry.sigma_sqr / (geometry.data_dimensionality - geometry.pca_dimensionality)
        lambda_rest = max(lambda_rest, np.finfo(float).eps)
        residual = float(geometry.x_c @ geometry.x_c) - float(geometry.y @ geometry.y)
        basis += residual / lambda_rest

    return basis, lambda_rest, eigenvalue


@dataclass(frozen=True)
class _CallablePotential:
    fn: Callable[[LocalPCAGeometry], float]
    name: str = "custom"

    def __call__(self, geometry: LocalPCAGeometry) -> float:
        return float(self.fn(geometry))
