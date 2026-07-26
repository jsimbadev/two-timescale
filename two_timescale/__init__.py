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
    DEFAULT_NGPCA_INITIAL_GAIN,
    NGPCAUnitRecursions,
    NGPCAParameters,
    NeuralGasPCA,
)
from .potentials import LocalPCAGeometry, NamedPotential, PotentialFunction, PotentialName, as_potential
from .recursions import (
    AdaptiveGainRecursions,
    EFORRLSAObservation,
    EFORRLSARecursion,
    EFORRLSAState,
    ScalarTrackingRecursion,
    StateTrackingRecursion,
)

__all__ = [
    "AdaptiveGainRecursions",
    "Control",
    "DEFAULT_NGPCA_INITIAL_GAIN",
    "EMARecursion",
    "EFORRLSAObservation",
    "EFORRLSARecursion",
    "EFORRLSAState",
    "FreeParameter",
    "GMMMoments",
    "LearningRate",
    "LocalPCAGeometry",
    "NGPCAUnitRecursions",
    "NGPCAParameters",
    "NamedPotential",
    "NeuralGasPCA",
    "OnlineGaussianMixture",
    "PotentialFunction",
    "PotentialName",
    "Recursion",
    "RecursionContext",
    "ScalarTrackingRecursion",
    "ScheduledLearningRate",
    "StateTrackingRecursion",
    "VectorField",
    "as_potential",
]
