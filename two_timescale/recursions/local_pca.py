"""Composition of recursions used by one local PCA unit."""

from __future__ import annotations

from dataclasses import dataclass

from .cvt import CVTRecursion
from .eforrlsa import EFORRLSARecursion
from .learning_rate import LearningRateRecursions
from .residual_variance import ResidualVarianceRecursion


@dataclass
class LocalPCARecursions:
    learning_rate: LearningRateRecursions
    center: CVTRecursion
    shape: EFORRLSARecursion
    residual_variance: ResidualVarianceRecursion
