r"""Transforms and scalar gains built from smoothed diagnostic statistics.

The classes in this module name a recurring pattern in adaptive stochastic
approximation: compute a diagnostic statistic from the current state and
observation, smooth it with an EMA recursion, then map the smoothed statistic
to either a scalar step size or a vector-field transform.

In symbols, the shared inner recursion is

.. math::

    \bar{s}_{t+1} = \bar{s}_t + b_t(d(z_t, x_t) - \bar{s}_t),

where ``d`` is the diagnostic and ``b_t`` is the statistic smoothing step size.
The result is then passed to either a gain map ``g`` or a transform map ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeAlias, TypeVar

from ..core import Control, EMARecursion, RecursionContext

State = TypeVar("State")
Observation = TypeVar("Observation")
Statistic = TypeVar("Statistic")
Transform = TypeVar("Transform")

Diagnostic: TypeAlias = Callable[[State, Observation, RecursionContext | None], Statistic]
GainMap: TypeAlias = Callable[[Statistic, State, Observation, RecursionContext | None], float]
StatisticMap: TypeAlias = Callable[[Statistic, State, Observation, RecursionContext | None], Transform]
StepSize: TypeAlias = float | Control | Callable[[RecursionContext | None], float]


@dataclass
class SmoothedStatisticControl(Generic[State, Observation, Statistic]):
    """Scalar step-size source derived from an EMA-smoothed statistic.

    This object is useful when an outer recursion should be scaled by a scalar
    that depends on recent diagnostic behavior.  The diagnostic state is owned
    here, not by the outer recursion, which keeps the adaptive loop explicit.

    Args:
        statistic_average: EMA recursion holding the smoothed diagnostic.
        diagnostic: Function mapping ``(state, observation, context)`` to the
            instantaneous statistic.
        gain_map: Function mapping the smoothed statistic to a scalar gain.
        gain: Initial scalar gain before the first update.
        smoothing: EMA smoothing coefficient for the diagnostic.
        lower: Lower bound applied by :meth:`value`.
        upper: Optional upper bound applied by :meth:`value`.
    """

    statistic_average: EMARecursion[Statistic]
    diagnostic: Diagnostic
    gain_map: GainMap
    gain: float
    smoothing: float = 0.03
    lower: float = 0.0
    upper: float | None = None

    @classmethod
    def new(
        cls,
        initial_statistic: Statistic,
        diagnostic: Diagnostic,
        gain_map: GainMap,
        initial_gain: float = 0.03,
        smoothing: float = 0.03,
        lower: float = 0.0,
        upper: float | None = None,
        name: str = "smoothed_statistic",
    ) -> "SmoothedStatisticControl[State, Observation, Statistic]":
        """Construct a smoothed-statistic scalar step-size source.

        Args:
            initial_statistic: Initial value of the statistic EMA.
            diagnostic: Instantaneous diagnostic statistic.
            gain_map: Map from smoothed statistic to scalar gain.
            initial_gain: Initial gain returned before updates.
            smoothing: EMA smoothing coefficient.
            lower: Lower bound for the returned gain.
            upper: Optional upper bound for the returned gain.
            name: Name assigned to the statistic EMA recursion.

        Returns:
            Configured scalar step-size source.
        """
        return cls(
            statistic_average=EMARecursion(initial_statistic, name=name),
            diagnostic=diagnostic,
            gain_map=gain_map,
            gain=float(initial_gain),
            smoothing=float(smoothing),
            lower=float(lower),
            upper=None if upper is None else float(upper),
        )

    def value(self, context: RecursionContext | None = None) -> float:
        """Return the current clipped scalar gain."""
        value = max(float(self.gain), self.lower)
        if self.upper is not None:
            value = min(value, self.upper)
        return value

    def update(
        self,
        state: State,
        observation: Observation,
        context: RecursionContext | None = None,
    ) -> float:
        """Update the diagnostic average and return the new scalar gain.

        Args:
            state: State of the recursion being diagnosed.
            observation: Observation driving the diagnosed recursion.
            context: Optional runtime context.

        Returns:
            Clipped scalar gain derived from the updated statistic.
        """
        statistic = self.diagnostic(state, observation, context)
        self.statistic_average.advance(
            statistic,
            step_size=self.smoothing,
            context=context,
        )
        self.gain = float(self.gain_map(self.statistic_average.state, state, observation, context))
        return self.value(context)


@dataclass
class SmoothedStatisticTransform(Generic[State, Observation, Statistic, Transform]):
    """Transform derived from an EMA-smoothed diagnostic statistic.

    Unlike :class:`SmoothedStatisticControl`, this class does not require the
    mapped value to be a scalar.  Adam uses it to smooth gradient moments before
    building a coordinate-wise preconditioner; NGPCA uses it to name the three
    smoothed variance-matching statistics that determine a local learning rate.

    Args:
        statistic_average: EMA recursion holding the smoothed diagnostic.
        diagnostic: Function mapping ``(state, observation, context)`` to the
            instantaneous statistic.
        statistic_map: Function mapping the smoothed statistic to the returned
            transform.
    """

    statistic_average: EMARecursion[Statistic]
    diagnostic: Diagnostic
    statistic_map: StatisticMap

    @classmethod
    def new(
        cls,
        initial_statistic: Statistic,
        diagnostic: Diagnostic,
        statistic_map: StatisticMap,
        step_size: StepSize = 0.03,
        name: str = "smoothed_statistic",
    ) -> "SmoothedStatisticTransform[State, Observation, Statistic, Transform]":
        """Construct a smoothed-statistic transform.

        Args:
            initial_statistic: Initial value of the statistic EMA.
            diagnostic: Instantaneous diagnostic statistic.
            statistic_map: Map from smoothed statistic to the returned value.
            step_size: Default EMA smoothing coefficient or schedule.
            name: Name assigned to the statistic EMA recursion.

        Returns:
            Configured statistic transform.
        """
        return cls(
            statistic_average=EMARecursion(initial_statistic, step_size=step_size, name=name),
            diagnostic=diagnostic,
            statistic_map=statistic_map,
        )

    def update(
        self,
        state: State,
        observation: Observation,
        step_size: StepSize | None = None,
        context: RecursionContext | None = None,
    ) -> Transform:
        """Update the statistic average and map it to a transform.

        Args:
            state: State of the recursion being diagnosed.
            observation: Observation driving the diagnosed recursion.
            step_size: Optional smoothing coefficient for this update only.
            context: Optional runtime context.

        Returns:
            Value returned by ``statistic_map`` from the updated statistic.
        """
        statistic = self.diagnostic(state, observation, context)
        self.statistic_average.advance(statistic, step_size=step_size, context=context)
        return self.statistic_map(self.statistic_average.state, state, observation, context)
