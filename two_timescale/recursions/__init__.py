"""Named stochastic recursions used by two-timescale algorithms."""

from .adaptive_gain import AdaptiveGainObservation, AdaptiveGainRecursions
from .adam import Adam, AdamMoments, AdamPreconditioner, AdamStepSize
from .eforrlsa import EFORRLSAObservation, EFORRLSARecursion, EFORRLSAState
from .gmm_moments import (
    GMMMomentRecursion,
    GMMMoments,
    StepSize,
    gmm_moment_residual,
    gmm_moment_target,
    moments_to_parameters,
    responsibilities,
)
from .scalar_tracking import ScalarTrackingRecursion
from .smoothed_statistics import SmoothedStatisticControl, SmoothedStatisticTransform
from .state_tracking import StateTrackingRecursion

__all__ = [
    "AdaptiveGainRecursions",
    "AdaptiveGainObservation",
    "Adam",
    "AdamMoments",
    "AdamPreconditioner",
    "AdamStepSize",
    "EFORRLSAObservation",
    "EFORRLSARecursion",
    "EFORRLSAState",
    "GMMMomentRecursion",
    "GMMMoments",
    "ScalarTrackingRecursion",
    "SmoothedStatisticControl",
    "SmoothedStatisticTransform",
    "StateTrackingRecursion",
    "StepSize",
    "gmm_moment_residual",
    "gmm_moment_target",
    "moments_to_parameters",
    "responsibilities",
]
