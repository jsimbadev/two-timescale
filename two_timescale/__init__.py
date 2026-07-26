"""Composable stochastic approximation tools for multiple timescales."""

from .core import (
    Control,
    EMARecursion,
    FreeParameter,
    LearningRate,
    Recursion,
    RecursionContext,
    ScheduledLearningRate,
    VectorField,
)
from .gmm import GMMMoments, OnlineGaussianMixture
from .ngpca import (
    NGPCAParameters,
    NeuralGasPCA,
)
from .potentials import LocalPCAGeometry, NamedPotential, PotentialFunction, PotentialName, as_potential
from .recursions import (
    CVTRecursion,
    EFORRLSAObservation,
    EFORRLSARecursion,
    EFORRLSAState,
    LocalPCARecursions,
    LearningRateRecursions,
    ResidualVarianceRecursion,
)

__all__ = [
    "Control",
    "CVTRecursion",
    "EMARecursion",
    "EFORRLSAObservation",
    "EFORRLSARecursion",
    "EFORRLSAState",
    "FreeParameter",
    "GMMMoments",
    "LearningRate",
    "LearningRateRecursions",
    "LocalPCARecursions",
    "LocalPCAGeometry",
    "NGPCAParameters",
    "NamedPotential",
    "NeuralGasPCA",
    "OnlineGaussianMixture",
    "PotentialFunction",
    "PotentialName",
    "Recursion",
    "RecursionContext",
    "ResidualVarianceRecursion",
    "ScheduledLearningRate",
    "VectorField",
    "as_potential",
]
