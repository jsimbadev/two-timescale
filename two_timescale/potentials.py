r"""Potential functions for local PCA unit competition.

Each potential maps the geometry of one local PCA unit and one observation to a
scalar ranking score.  The shared basis term is the geometric Mahalanobis
distance with reconstruction residual,

.. math::

    d_{MG}(\xi)
      = y^\top \Lambda^{-1}y
        + \check{\lambda}^{-1}\xi^\top(I - WW^\top)\xi,

where the second term is present only when the retained PCA dimensionality is
smaller than the observation dimensionality.

References:
    Migenda02/2_RGHQJ3PP:
        Migenda, N., Moeller, R., & Schenck, W. (2024). Adaptive local
        Principal Component Analysis improves the clustering of
        high-dimensional data. *Pattern Recognition*, 146, 110030. Equations
        (23), (24), (32), (33), (37), and (38). Zotero item key ``RGHQJ3PP``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import numpy as np

PotentialName = Literal["AR", "N", "H", "AV", "VRV", "VRR"]


@dataclass(frozen=True)
class LocalPCAGeometry:
    """Geometry needed to rank one local PCA unit for one observation.

    Args:
        x_c: Centered observation ``xi = x - c``.
        y: Projected observation ``W.T @ xi``.
        eigenvalue: Principal eigenvalues in the retained PCA subspace.
        sigma_sqr: Residual variance in unrepresented directions.
        activity: Unit activity used by adaptive potential functions.
        pca_dimensionality: Retained local PCA dimensionality ``m``.
        data_dimensionality: Observation dimensionality ``n``.
    """

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
    r"""Named NGPCA potential function.

    The package uses short names for the potential families described in the
    paper:

    ``N``
        Unregularized geometric Mahalanobis basis ``d_MG``.
    ``H``
        Volume-normalized potential ``d_H``.
    ``AV`` and ``AR``
        Activity-controlled absolute volume and absolute radius variants.
    ``VRV`` and ``VRR``
        Activity-controlled relative volume and relative radius variants.

    The activity support follows the radius-control mechanism

    .. math::

        r(a) = r_{max}^{1 - min(a, 1)}.

    Args:
        name: Potential name.
        rmax: Maximum support factor for low-activity units.
    """

    name: PotentialName = "AR"
    rmax: float = 200.0

    def __call__(self, geometry: LocalPCAGeometry) -> float:
        """Evaluate the named potential for one unit-observation geometry."""
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
    """Normalize a potential specification to a callable potential object.

    Args:
        spec: Named potential, object with a ``name`` attribute and
            ``__call__``, or arbitrary callable.
        rmax: Maximum support factor used when ``spec`` is a named potential.

    Returns:
        Callable potential object.
    """
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
