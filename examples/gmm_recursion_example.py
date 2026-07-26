"""Gaussian-mixture recursion example with the same recursion algebra."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from two_timescale import GaussianMixtureRecursion, LearningRate


def make_mixture(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(loc=(-2.5, 0.0), scale=(0.45, 0.8), size=(300, 2))
    right = rng.normal(loc=(2.0, 1.0), scale=(0.7, 0.35), size=(300, 2))
    top = rng.normal(loc=(0.0, 3.0), scale=(0.5, 0.5), size=(240, 2))
    return np.vstack([left, right, top])


def run() -> tuple[GaussianMixtureRecursion, np.ndarray]:
    rng = np.random.default_rng(10)
    data = make_mixture(seed=10)
    model = GaussianMixtureRecursion(
        n_components=3,
        n_features=2,
        learning_rate=LearningRate(0.03),
        rng=rng,
    )
    stream = data[[rng.integers(0, len(data)) for _ in range(4_000)]]
    model.advance(stream)
    return model, data


def plot(model: GaussianMixtureRecursion, data: np.ndarray, ax: Any | None = None) -> tuple[Any, Any]:
    """Plot observations and covariance ellipses for the GMM recursion."""

    plt = _load_pyplot()
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    labels = model.assignments(data)
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=8, marker="o", alpha=0.35, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    weights, means, covariances = model.parameters()
    for weight, mean, covariance in zip(weights, means, covariances):
        _draw_covariance_ellipse(ax, mean, covariance, label=f"{weight:.2f}")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return figure, ax


def _draw_covariance_ellipse(ax: Any, mean: np.ndarray, covariance: np.ndarray, label: str) -> None:
    values, vectors = np.linalg.eigh(0.5 * (covariance[:2, :2] + covariance[:2, :2].T))
    values = np.maximum(values, 0.0)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = mean[:2, None] + vectors @ (2.0 * np.sqrt(values)[:, None] * circle)
    ax.plot(ellipse[0], ellipse[1], color="black", linewidth=1.0)
    ax.scatter([mean[0]], [mean[1]], c="black", s=16, marker="x")
    ax.annotate(label, mean[:2], xytext=(4, 4), textcoords="offset points", fontsize=8)


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("Plotting examples require matplotlib. Install it to use --plot or --save.") from exc
    return plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="show a Matplotlib plot of the recursion state")
    parser.add_argument("--save", type=Path, help="write the plot to this path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fitted, observations = run()
    weights, means, covariances = fitted.parameters()
    print("weights")
    print(np.round(weights, 3))
    print("means")
    print(np.round(means, 3))
    if args.plot or args.save:
        fig, _ = plot(fitted, observations)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, bbox_inches="tight", dpi=150)
        if args.plot:
            _load_pyplot().show()
