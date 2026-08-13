"""Controls built from smoothed diagnostic statistics."""

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
    """Scalar control derived from an EMA-smoothed diagnostic statistic."""

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
    """Transform derived from an EMA-smoothed diagnostic statistic."""

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
        return cls(
            statistic_average=EMARecursion(initial_statistic, step_size=step_size, name=name),
            diagnostic=diagnostic,
            statistic_map=statistic_map,
        )

    def update(
        self,
        state: State,
        observation: Observation,
        context: RecursionContext | None = None,
    ) -> Transform:
        statistic = self.diagnostic(state, observation, context)
        self.statistic_average.advance(statistic, context=context)
        return self.statistic_map(self.statistic_average.state, state, observation, context)
