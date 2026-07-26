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
    CVTRecursion,
    EFORRLSAObservation,
    EFORRLSARecursion,
    EFORRLSAState,
    LearningRateRecursions,
    NGPCAParameters,
    NeuralGasPCA,
    ResidualVarianceRecursion,
)
from .potentials import LocalPCAGeometry, NamedPotential, PotentialFunction, PotentialName, as_potential

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
