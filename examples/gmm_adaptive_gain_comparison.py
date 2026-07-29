"""Compare fixed-step and adaptive-control Gaussian-mixture recursions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from two_timescale import (
    AdaptiveGMMGain,
    GaussianMixtureRecursion,
    LearningRate,
    RecursionContext,
    SmoothedCovarianceControl,
    SmoothedMomentResidualControl,
)
from two_timescale.recursions import GMMMoments


@dataclass(frozen=True)
class ConvergencePoint:
    epoch_index: int
    iteration: int
    fixed_nll: float
    responsibility_nll: float
    responsibility_gain: float
    covariance_nll: float
    covariance_gain: float
    residual_nll: float
    residual_gain: float
    fixed_moment_difference: float
    responsibility_moment_difference: float
    covariance_moment_difference: float
    residual_moment_difference: float


def make_mixture(seed: int = 12, n_per_component: int = 220) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=(-2.4, -0.2), scale=(0.45, 0.7), size=(n_per_component, 2))
    right = rng.normal(loc=(2.1, 0.8), scale=(0.65, 0.35), size=(n_per_component, 2))
    top = rng.normal(loc=(0.0, 2.8), scale=(0.5, 0.5), size=(n_per_component, 2))
    return np.vstack([left, right, top])


def run_comparison(
    epochs: int = 12,
    fixed_step: float = 0.03,
    initial_gain: float = 0.03,
    seed: int = 12,
) -> tuple[
    list[ConvergencePoint],
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    np.ndarray,
]:
    data = make_mixture(seed=seed)
    rng = np.random.default_rng(seed)
    train, validation = _split_train_validation(data, rng)

    fixed = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        step_size=LearningRate(fixed_step),
        rng=np.random.default_rng(seed + 1),
    )
    responsibility_control = AdaptiveGMMGain.new(
        n_components=3,
        initial_gain=initial_gain,
    )
    responsibility = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        step_size=responsibility_control,
        rng=np.random.default_rng(seed + 1),
    )
    covariance_control = SmoothedCovarianceControl.new(
        n_components=3,
        n_features=2,
        initial_gain=initial_gain,
    )
    covariance = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        step_size=covariance_control,
        rng=np.random.default_rng(seed + 1),
    )
    residual_control = SmoothedMomentResidualControl.new(
        n_components=3,
        n_features=2,
        initial_gain=initial_gain,
    )
    residual = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        step_size=residual_control,
        rng=np.random.default_rng(seed + 1),
    )

    trace = [
        _trace_point(
            epoch_index=0,
            fixed=fixed,
            responsibility=responsibility,
            responsibility_control=responsibility_control,
            covariance=covariance,
            covariance_control=covariance_control,
            residual=residual,
            residual_control=residual_control,
            validation=validation,
        )
    ]

    for epoch_index in range(1, epochs + 1):
        fixed_start = _copy_moments(fixed.recursions.moments.state)
        responsibility_start = _copy_moments(responsibility.recursions.moments.state)
        covariance_start = _copy_moments(covariance.recursions.moments.state)
        residual_start = _copy_moments(residual.recursions.moments.state)

        for observation in train[rng.permutation(len(train))]:
            fixed.advance(observation)

            responsibility_context = RecursionContext(iteration=responsibility.recursions.moments.iteration, rng=rng)
            responsibility_control.update(responsibility.recursions.moments, observation, responsibility_context)
            responsibility.advance(observation)

            covariance_context = RecursionContext(iteration=covariance.recursions.moments.iteration, rng=rng)
            covariance_control.update(covariance.recursions.moments, observation, covariance_context)
            covariance.advance(observation)

            residual_context = RecursionContext(iteration=residual.recursions.moments.iteration, rng=rng)
            residual_control.update(residual.recursions.moments, observation, residual_context)
            residual.advance(observation)

        trace.append(
            _trace_point(
                epoch_index=epoch_index,
                fixed=fixed,
                responsibility=responsibility,
                responsibility_control=responsibility_control,
                covariance=covariance,
                covariance_control=covariance_control,
                residual=residual,
                residual_control=residual_control,
                validation=validation,
                fixed_start=fixed_start,
                responsibility_start=responsibility_start,
                covariance_start=covariance_start,
                residual_start=residual_start,
            )
        )

    return trace, fixed, responsibility, covariance, residual, validation


def negative_log_likelihood(model: GaussianMixtureRecursion, data: np.ndarray) -> float:
    weights, means, covariances = model.parameters()
    log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))
    log_likelihood = 0.0

    for observation in data:
        component_logp = np.array([
            log_weight + _log_gaussian(observation, mean, covariance)
            for log_weight, mean, covariance in zip(log_weights, means, covariances)
        ])
        log_likelihood += _logsumexp(component_logp)

    return float(-log_likelihood / len(data))


def plot(trace: list[ConvergencePoint], ax: Any | None = None) -> tuple[Any, Any]:
    plt = _load_pyplot()
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    iterations = np.array([point.iteration for point in trace])
    ax.plot(iterations, [point.fixed_nll for point in trace], marker="o", label="fixed step")
    ax.plot(iterations, [point.responsibility_nll for point in trace], marker="o", label="responsibility gain")
    ax.plot(iterations, [point.covariance_nll for point in trace], marker="o", label="covariance gain")
    ax.plot(iterations, [point.residual_nll for point in trace], marker="o", label="moment residual gain")
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation negative log likelihood")
    ax.legend()

    gain_axis = ax.twinx()
    gain_axis.plot(iterations, [point.responsibility_gain for point in trace], color="black", linestyle="--")
    gain_axis.plot(iterations, [point.covariance_gain for point in trace], color="gray", linestyle=":")
    gain_axis.plot(iterations, [point.residual_gain for point in trace], color="tab:red", linestyle="-.")
    gain_axis.set_ylabel("control value")
    return figure, ax


def plot_differences(trace: list[ConvergencePoint], ax: Any | None = None) -> tuple[Any, Any]:
    plt = _load_pyplot()
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    iterations = np.array([point.iteration for point in trace])
    floor = np.finfo(float).tiny
    ax.semilogy(iterations, np.maximum([point.fixed_moment_difference for point in trace], floor), marker="o")
    ax.semilogy(
        iterations,
        np.maximum([point.responsibility_moment_difference for point in trace], floor),
        marker="o",
    )
    ax.semilogy(iterations, np.maximum([point.covariance_moment_difference for point in trace], floor), marker="o")
    ax.semilogy(iterations, np.maximum([point.residual_moment_difference for point in trace], floor), marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative moment displacement")
    ax.legend(["fixed step", "responsibility gain", "covariance gain", "moment residual gain"])
    return figure, ax


def print_trace(trace: list[ConvergencePoint]) -> None:
    print("epoch iteration fixed_nll resp_nll resp_gain cov_nll cov_gain resid_nll resid_gain")
    for point in trace:
        print(
            f"{point.epoch_index:>5d} "
            f"{point.iteration:>9d} "
            f"{point.fixed_nll:>9.6f} "
            f"{point.responsibility_nll:>8.6f} "
            f"{point.responsibility_gain:>9.6f} "
            f"{point.covariance_nll:>7.6f} "
            f"{point.covariance_gain:>8.6f} "
            f"{point.residual_nll:>9.6f} "
            f"{point.residual_gain:>10.6f}"
        )


def _trace_point(
    epoch_index: int,
    fixed: GaussianMixtureRecursion,
    responsibility: GaussianMixtureRecursion,
    responsibility_control: AdaptiveGMMGain,
    covariance: GaussianMixtureRecursion,
    covariance_control: SmoothedCovarianceControl,
    residual: GaussianMixtureRecursion,
    residual_control: SmoothedMomentResidualControl,
    validation: np.ndarray,
    fixed_start: GMMMoments | None = None,
    responsibility_start: GMMMoments | None = None,
    covariance_start: GMMMoments | None = None,
    residual_start: GMMMoments | None = None,
) -> ConvergencePoint:
    return ConvergencePoint(
        epoch_index=epoch_index,
        iteration=fixed.recursions.moments.iteration,
        fixed_nll=negative_log_likelihood(fixed, validation),
        responsibility_nll=negative_log_likelihood(responsibility, validation),
        responsibility_gain=responsibility_control.value(),
        covariance_nll=negative_log_likelihood(covariance, validation),
        covariance_gain=covariance_control.value(),
        residual_nll=negative_log_likelihood(residual, validation),
        residual_gain=residual_control.value(),
        fixed_moment_difference=_optional_moment_difference(fixed_start, fixed.recursions.moments.state),
        responsibility_moment_difference=_optional_moment_difference(
            responsibility_start,
            responsibility.recursions.moments.state,
        ),
        covariance_moment_difference=_optional_moment_difference(covariance_start, covariance.recursions.moments.state),
        residual_moment_difference=_optional_moment_difference(residual_start, residual.recursions.moments.state),
    )


def _split_train_validation(data: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(data))
    split = int(round(0.8 * len(data)))
    return data[order[:split]], data[order[split:]]


def _copy_moments(moments: GMMMoments) -> GMMMoments:
    return GMMMoments(
        mass=np.array(moments.mass, copy=True),
        first=np.array(moments.first, copy=True),
        second=np.array(moments.second, copy=True),
    )


def _optional_moment_difference(before: GMMMoments | None, after: GMMMoments) -> float:
    if before is None:
        return 0.0
    before_vector = _flatten_moments(before)
    after_vector = _flatten_moments(after)
    scale = max(float(np.linalg.norm(before_vector)), np.finfo(float).eps)
    return float(np.linalg.norm(after_vector - before_vector) / scale)


def _flatten_moments(moments: GMMMoments) -> np.ndarray:
    return np.concatenate([
        moments.mass.ravel(),
        moments.first.ravel(),
        moments.second.ravel(),
    ])


def _log_gaussian(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = covariance + 1e-6 * np.eye(x.size)
        sign, logdet = np.linalg.slogdet(covariance)
    diff = x - mean
    solved = np.linalg.solve(covariance, diff)
    return float(-0.5 * (x.size * np.log(2.0 * np.pi) + logdet + diff @ solved))


def _logsumexp(values: np.ndarray) -> float:
    anchor = float(np.max(values))
    return anchor + float(np.log(np.sum(np.exp(values - anchor))))


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("Plotting examples require matplotlib. Install it to use --plot or --save.") from exc
    return plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12, help="number of full sweeps over the training split")
    parser.add_argument("--fixed-step", type=float, default=0.03, help="fixed recursion step size")
    parser.add_argument("--initial-gain", type=float, default=0.03, help="initial adaptive control value")
    parser.add_argument("--seed", type=int, default=12, help="random seed")
    parser.add_argument("--plot", action="store_true", help="show a Matplotlib validation-score plot")
    parser.add_argument("--plot-differences", action="store_true", help="show a Matplotlib moment-difference plot")
    parser.add_argument("--save", type=Path, help="write the validation-score plot to this path")
    parser.add_argument("--save-differences", type=Path, help="write the moment-difference plot to this path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result, _, _, _, _, _ = run_comparison(
        epochs=args.epochs,
        fixed_step=args.fixed_step,
        initial_gain=args.initial_gain,
        seed=args.seed,
    )
    print_trace(result)

    if args.plot or args.save:
        fig, _ = plot(result)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, bbox_inches="tight", dpi=150)
        if args.plot:
            _load_pyplot().show()

    if args.plot_differences or args.save_differences:
        fig, _ = plot_differences(result)
        if args.save_differences:
            args.save_differences.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_differences, bbox_inches="tight", dpi=150)
        if args.plot_differences:
            _load_pyplot().show()
