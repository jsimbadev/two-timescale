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
    EFORRLSAObservation,
    EFORRLSARecursion,
    EFORRLSAState,
    LearningRateRecursions,
    NGPCAParameters,
    NeuralGasPCA,
)
from .potentials import LocalPCAGeometry, NamedPotential, PotentialFunction, PotentialName, as_potential

__all__ = [
    "Control",
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
    "ScheduledLearningRate",
    "VectorField",
    "as_potential",
]
