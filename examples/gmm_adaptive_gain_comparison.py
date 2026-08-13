"""Compare fixed-step and adaptive-step Gaussian-mixture recursions.

The script is a diagnostic harness for the package's composability claims.  It
keeps the GMM recursion fixed and varies only the step-size source: a constant
learning rate, a polynomial-decay deterministic schedule, and several
smoothed-statistic adaptive gains.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from two_timescale import (
    AdaptiveGMMGain,
    GaussianMixtureRecursion,
    LearningRate,
    RecursionContext,
    ScheduledLearningRate,
    SmoothedCovarianceControl,
    SmoothedMomentResidualControl,
)
from two_timescale.recursions import GMMMoments

Scenario = Literal["easy", "overlap", "drift"]
TRUE_COLOR = "black"
FIXED_COLOR = "tab:blue"
RM_COLOR = "tab:orange"
RESPONSIBILITY_COLOR = "tab:green"
COVARIANCE_COLOR = "tab:red"
RESIDUAL_COLOR = "tab:purple"


@dataclass(frozen=True)
class ConvergencePoint:
    """One checkpoint in the GMM comparison trace.

    Attributes:
        epoch_index: Epoch number of the checkpoint.
        iteration: GMM recursion iteration at the checkpoint.
        true_nll: Validation negative log likelihood under the generating
            mixture.
        fixed_nll: Validation score for the fixed-step recursion.
        rm_nll: Validation score for the deterministic schedule recursion.
        rm_step: Current deterministic schedule value.
        responsibility_nll: Validation score for responsibility-gain control.
        responsibility_gain: Current responsibility-gain value.
        covariance_nll: Validation score for covariance-mismatch control.
        covariance_gain: Current covariance-mismatch gain.
        residual_nll: Validation score for moment-residual control.
        residual_gain: Current moment-residual gain.
        fixed_moment_difference: Relative moment displacement during the epoch.
        rm_moment_difference: Relative moment displacement during the epoch.
        responsibility_moment_difference: Relative moment displacement during
            the epoch.
        covariance_moment_difference: Relative moment displacement during the
            epoch.
        residual_moment_difference: Relative moment displacement during the
            epoch.
    """

    epoch_index: int
    iteration: int
    true_nll: float
    fixed_nll: float
    rm_nll: float
    rm_step: float
    responsibility_nll: float
    responsibility_gain: float
    covariance_nll: float
    covariance_gain: float
    residual_nll: float
    residual_gain: float
    fixed_moment_difference: float
    rm_moment_difference: float
    responsibility_moment_difference: float
    covariance_moment_difference: float
    residual_moment_difference: float


@dataclass(frozen=True)
class MixtureSpec:
    """Parameters of a synthetic Gaussian mixture.

    Args:
        weights: Mixture weights.
        means: Component means.
        covariances: Component covariance matrices.
    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray


def make_mixture(seed: int = 12, n_per_component: int = 220) -> np.ndarray:
    """Generate the separated synthetic comparison mixture."""
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=(-2.4, -0.2), scale=(0.45, 0.7), size=(n_per_component, 2))
    right = rng.normal(loc=(2.1, 0.8), scale=(0.65, 0.35), size=(n_per_component, 2))
    top = rng.normal(loc=(0.0, 2.8), scale=(0.5, 0.5), size=(n_per_component, 2))
    return np.vstack([left, right, top])


def make_overlap_mixture(seed: int = 12, n_samples: int = 660) -> np.ndarray:
    """Generate the overlapping and imbalanced comparison mixture."""
    return _sample_mixture(_overlap_spec(), n_samples, np.random.default_rng(seed))


def make_drift_mixture(seed: int = 12, epoch_index: int = 0, n_samples: int = 528) -> np.ndarray:
    """Generate one epoch of the drifting comparison mixture."""
    return _sample_mixture(_drift_spec(epoch_index), n_samples, np.random.default_rng(seed))


def run_comparison(
    epochs: int = 12,
    fixed_step: float = 0.03,
    rm_initial_step: float | None = None,
    rm_decay: float = 0.001,
    rm_power: float = 0.6,
    initial_gain: float = 0.03,
    seed: int = 12,
    scenario: Scenario = "easy",
) -> tuple[
    list[ConvergencePoint],
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    GaussianMixtureRecursion,
    np.ndarray,
]:
    """Run the fixed-step, scheduled, and adaptive-gain GMM comparison.

    Args:
        epochs: Number of training epochs.
        fixed_step: Constant step size for the fixed baseline.
        rm_initial_step: Optional initial value for the deterministic schedule.
        rm_decay: Decay multiplier for the deterministic schedule.
        rm_power: Polynomial power for the deterministic schedule.
        initial_gain: Initial value for adaptive gains.
        seed: Random seed controlling data and model initialization.
        scenario: Synthetic data scenario.

    Returns:
        Trace, fitted models for each step-size source, and final validation
        data.
    """
    rng = np.random.default_rng(seed)
    train, validation = _initial_data(scenario, seed, rng)

    fixed = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        step_size=LearningRate(fixed_step),
        rng=np.random.default_rng(seed + 1),
    )
    rm_schedule = ScheduledLearningRate(
        LearningRate(fixed_step if rm_initial_step is None else rm_initial_step),
        decay=rm_decay,
        power=rm_power,
    )
    robbins_monro = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        step_size=rm_schedule,
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
            robbins_monro=robbins_monro,
            rm_schedule=rm_schedule,
            responsibility=responsibility,
            responsibility_control=responsibility_control,
            covariance=covariance,
            covariance_control=covariance_control,
            residual=residual,
            residual_control=residual_control,
            validation=validation,
            scenario=scenario,
        )
    ]

    for epoch_index in range(1, epochs + 1):
        if scenario == "drift":
            train = make_drift_mixture(seed=seed + epoch_index, epoch_index=epoch_index)
            validation = make_drift_mixture(seed=seed + 10_000 + epoch_index, epoch_index=epoch_index, n_samples=132)

        fixed_start = _copy_moments(fixed.recursions.moments.state)
        rm_start = _copy_moments(robbins_monro.recursions.moments.state)
        responsibility_start = _copy_moments(responsibility.recursions.moments.state)
        covariance_start = _copy_moments(covariance.recursions.moments.state)
        residual_start = _copy_moments(residual.recursions.moments.state)

        for observation in train[rng.permutation(len(train))]:
            fixed.advance(observation)
            robbins_monro.advance(observation)

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
                robbins_monro=robbins_monro,
                rm_schedule=rm_schedule,
                responsibility=responsibility,
                responsibility_control=responsibility_control,
                covariance=covariance,
                covariance_control=covariance_control,
                residual=residual,
                residual_control=residual_control,
                validation=validation,
                scenario=scenario,
                fixed_start=fixed_start,
                rm_start=rm_start,
                responsibility_start=responsibility_start,
                covariance_start=covariance_start,
                residual_start=residual_start,
            )
        )

    return trace, fixed, robbins_monro, responsibility, covariance, residual, validation


def negative_log_likelihood(model: GaussianMixtureRecursion, data: np.ndarray) -> float:
    """Compute mean negative log likelihood under a fitted GMM recursion."""
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


def true_negative_log_likelihood(scenario: Scenario, data: np.ndarray, epoch_index: int = 0) -> float:
    """Compute mean negative log likelihood under the generating mixture."""
    spec = _scenario_spec(scenario, epoch_index)
    log_weights = np.log(np.maximum(spec.weights, np.finfo(float).tiny))
    log_likelihood = 0.0

    for observation in data:
        component_logp = np.array([
            log_weight + _log_gaussian(observation, mean, covariance)
            for log_weight, mean, covariance in zip(log_weights, spec.means, spec.covariances)
        ])
        log_likelihood += _logsumexp(component_logp)

    return float(-log_likelihood / len(data))


def plot(trace: list[ConvergencePoint], ax: Any | None = None) -> tuple[Any, Any]:
    """Plot validation scores and step-size values over the trace."""
    plt = _load_pyplot()
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    iterations = np.array([point.iteration for point in trace])
    ax.plot(iterations, [point.true_nll for point in trace], color=TRUE_COLOR, linestyle="-", label="true mixture")
    ax.plot(iterations, [point.fixed_nll for point in trace], color=FIXED_COLOR, marker="o", label="fixed step")
    ax.plot(iterations, [point.rm_nll for point in trace], color=RM_COLOR, marker="o", label="RM schedule")
    ax.plot(
        iterations,
        [point.responsibility_nll for point in trace],
        color=RESPONSIBILITY_COLOR,
        marker="o",
        label="responsibility gain",
    )
    ax.plot(
        iterations,
        [point.covariance_nll for point in trace],
        color=COVARIANCE_COLOR,
        marker="o",
        label="covariance gain",
    )
    ax.plot(
        iterations,
        [point.residual_nll for point in trace],
        color=RESIDUAL_COLOR,
        marker="o",
        label="moment residual gain",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation negative log likelihood")
    ax.legend()

    gain_axis = ax.twinx()
    gain_axis.plot(iterations, [point.rm_step for point in trace], color=RM_COLOR, linestyle="--")
    gain_axis.plot(
        iterations,
        [point.responsibility_gain for point in trace],
        color=RESPONSIBILITY_COLOR,
        linestyle="--",
    )
    gain_axis.plot(iterations, [point.covariance_gain for point in trace], color=COVARIANCE_COLOR, linestyle=":")
    gain_axis.plot(iterations, [point.residual_gain for point in trace], color=RESIDUAL_COLOR, linestyle="-.")
    gain_axis.set_ylabel("control value")
    return figure, ax


def plot_differences(trace: list[ConvergencePoint], ax: Any | None = None) -> tuple[Any, Any]:
    """Plot relative moment displacement for each comparison model."""
    plt = _load_pyplot()
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    iterations = np.array([point.iteration for point in trace])
    ax.semilogy(iterations, _positive_values([point.fixed_moment_difference for point in trace]), marker="o")
    ax.semilogy(iterations, _positive_values([point.rm_moment_difference for point in trace]), marker="o")
    ax.semilogy(
        iterations,
        _positive_values([point.responsibility_moment_difference for point in trace]),
        marker="o",
    )
    ax.semilogy(iterations, _positive_values([point.covariance_moment_difference for point in trace]), marker="o")
    ax.semilogy(iterations, _positive_values([point.residual_moment_difference for point in trace]), marker="o")
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative moment displacement")
    ax.legend(["fixed step", "RM schedule", "responsibility gain", "covariance gain", "moment residual gain"])
    return figure, ax


def print_trace(trace: list[ConvergencePoint]) -> None:
    """Print the comparison trace as a fixed-width table."""
    print("epoch iteration true_nll fixed_nll rm_nll rm_step resp_nll resp_gain cov_nll cov_gain resid_nll resid_gain")
    for point in trace:
        print(
            f"{point.epoch_index:>5d} "
            f"{point.iteration:>9d} "
            f"{point.true_nll:>8.6f} "
            f"{point.fixed_nll:>9.6f} "
            f"{point.rm_nll:>6.6f} "
            f"{point.rm_step:>7.6f} "
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
    robbins_monro: GaussianMixtureRecursion,
    rm_schedule: ScheduledLearningRate,
    responsibility: GaussianMixtureRecursion,
    responsibility_control: AdaptiveGMMGain,
    covariance: GaussianMixtureRecursion,
    covariance_control: SmoothedCovarianceControl,
    residual: GaussianMixtureRecursion,
    residual_control: SmoothedMomentResidualControl,
    validation: np.ndarray,
    scenario: Scenario,
    fixed_start: GMMMoments | None = None,
    rm_start: GMMMoments | None = None,
    responsibility_start: GMMMoments | None = None,
    covariance_start: GMMMoments | None = None,
    residual_start: GMMMoments | None = None,
) -> ConvergencePoint:
    return ConvergencePoint(
        epoch_index=epoch_index,
        iteration=fixed.recursions.moments.iteration,
        true_nll=true_negative_log_likelihood(scenario, validation, epoch_index),
        fixed_nll=negative_log_likelihood(fixed, validation),
        rm_nll=negative_log_likelihood(robbins_monro, validation),
        rm_step=rm_schedule.value(RecursionContext(iteration=robbins_monro.recursions.moments.iteration)),
        responsibility_nll=negative_log_likelihood(responsibility, validation),
        responsibility_gain=responsibility_control.value(),
        covariance_nll=negative_log_likelihood(covariance, validation),
        covariance_gain=covariance_control.value(),
        residual_nll=negative_log_likelihood(residual, validation),
        residual_gain=residual_control.value(),
        fixed_moment_difference=_optional_moment_difference(fixed_start, fixed.recursions.moments.state),
        rm_moment_difference=_optional_moment_difference(rm_start, robbins_monro.recursions.moments.state),
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


def _initial_data(scenario: Scenario, seed: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if scenario == "easy":
        return _split_train_validation(make_mixture(seed=seed), rng)
    if scenario == "overlap":
        return _split_train_validation(make_overlap_mixture(seed=seed), rng)
    if scenario == "drift":
        train = make_drift_mixture(seed=seed, epoch_index=0)
        validation = make_drift_mixture(seed=seed + 10_000, epoch_index=0, n_samples=132)
        return train, validation
    raise ValueError(f"unknown scenario: {scenario}")


def _sample_mixture(spec: MixtureSpec, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    components = rng.choice(len(spec.weights), size=n_samples, p=spec.weights)
    data = np.empty((n_samples, spec.means.shape[1]))
    for component in range(len(spec.weights)):
        count = int(np.sum(components == component))
        if count:
            data[components == component] = rng.multivariate_normal(
                spec.means[component],
                spec.covariances[component],
                size=count,
            )
    return data


def _scenario_spec(scenario: Scenario, epoch_index: int = 0) -> MixtureSpec:
    if scenario == "easy":
        return _easy_spec()
    if scenario == "overlap":
        return _overlap_spec()
    if scenario == "drift":
        return _drift_spec(epoch_index)
    raise ValueError(f"unknown scenario: {scenario}")


def _easy_spec() -> MixtureSpec:
    return MixtureSpec(
        weights=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        means=np.array([[-2.4, -0.2], [2.1, 0.8], [0.0, 2.8]]),
        covariances=np.array([
            [[0.45**2, 0.0], [0.0, 0.7**2]],
            [[0.65**2, 0.0], [0.0, 0.35**2]],
            [[0.5**2, 0.0], [0.0, 0.5**2]],
        ]),
    )


def _overlap_spec() -> MixtureSpec:
    return MixtureSpec(
        weights=np.array([0.55, 0.30, 0.15]),
        means=np.array([[-0.9, -0.2], [0.8, 0.3], [0.0, 1.35]]),
        covariances=np.array([
            [[0.95, 0.45], [0.45, 0.80]],
            [[0.75, -0.35], [-0.35, 0.65]],
            [[0.50, 0.15], [0.15, 0.85]],
        ]),
    )


def _drift_spec(epoch_index: int) -> MixtureSpec:
    phase = 0.45 * epoch_index
    means = np.array([
        [-1.4 + 0.45 * np.sin(phase), -0.5 + 0.25 * np.cos(phase)],
        [1.3 + 0.35 * np.cos(phase), 0.2 + 0.35 * np.sin(phase)],
        [-0.1 + 0.25 * np.sin(0.5 * phase), 1.6 + 0.45 * np.cos(0.7 * phase)],
    ])
    weights = np.array([
        0.42 + 0.12 * np.sin(0.7 * phase),
        0.36 + 0.10 * np.cos(0.5 * phase),
        0.22,
    ])
    weights = weights / weights.sum()
    return MixtureSpec(
        weights=weights,
        means=means,
        covariances=np.array([
            [[0.75, 0.20], [0.20, 0.55]],
            [[0.65, -0.18], [-0.18, 0.45]],
            [[0.45, 0.08], [0.08, 0.60]],
        ]),
    )


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


def _positive_values(values: list[float]) -> np.ndarray:
    positive = np.array(values, dtype=float)
    positive[positive <= 0.0] = np.nan
    return positive


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
    parser.add_argument(
        "--scenario",
        choices=["easy", "overlap", "drift"],
        default="easy",
        help="data stream to use: separated blobs, overlapping imbalanced blobs, or epochwise drift",
    )
    parser.add_argument("--epochs", type=int, default=12, help="number of full sweeps over the training split")
    parser.add_argument("--fixed-step", type=float, default=0.03, help="fixed recursion step size")
    parser.add_argument(
        "--rm-initial-step",
        type=float,
        help="initial Robbins-Monro step size; defaults to --fixed-step",
    )
    parser.add_argument("--rm-decay", type=float, default=0.001, help="Robbins-Monro schedule decay")
    parser.add_argument("--rm-power", type=float, default=0.6, help="Robbins-Monro schedule power")
    parser.add_argument("--initial-gain", type=float, default=0.03, help="initial adaptive control value")
    parser.add_argument("--seed", type=int, default=12, help="random seed")
    parser.add_argument("--plot", action="store_true", help="show a Matplotlib validation-score plot")
    parser.add_argument("--plot-differences", action="store_true", help="show a Matplotlib moment-difference plot")
    parser.add_argument("--save", type=Path, help="write the validation-score plot to this path")
    parser.add_argument("--save-differences", type=Path, help="write the moment-difference plot to this path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result, _, _, _, _, _, _ = run_comparison(
        epochs=args.epochs,
        fixed_step=args.fixed_step,
        rm_initial_step=args.rm_initial_step,
        rm_decay=args.rm_decay,
        rm_power=args.rm_power,
        initial_gain=args.initial_gain,
        seed=args.seed,
        scenario=args.scenario,
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
