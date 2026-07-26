"""Neural Gas PCA example using DistGen samples and the local MATLAB defaults."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from two_timescale import FreeParameter, NGPCAParameters, NeuralGasPCA

DEFAULT_SAMPLE_NAME = "banana_samples"
DEFAULT_NUM_UNITS = 4
DEFAULT_ITERATIONS = 20_000
DEFAULT_PCA_DIMENSIONALITY = 2


class MatlabTwister:
    """MATLAB rng(0,'twister') compatible rand stream."""

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int = 0) -> None:
        # MATLAB's rng(0,'twister') exposes a standard MT19937 state seeded
        # as mt19937ar's default 5489 state.
        self._state = [0] * self._N
        self._index = self._N + 1
        self.seed(seed)

    def seed(self, seed: int) -> None:
        mt_seed = 5489 if seed == 0 else int(seed)
        self._state[0] = mt_seed & 0xFFFFFFFF
        for index in range(1, self._N):
            previous = self._state[index - 1]
            self._state[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF
        self._index = self._N

    def random(self, size: int | tuple[int, ...] | None = None) -> float | np.ndarray:
        if size is None:
            return self._rand_float()
        if isinstance(size, int):
            return np.array([self._rand_float() for _ in range(size)], dtype=float)
        values = np.array([self._rand_float() for _ in range(int(np.prod(size)))], dtype=float)
        return values.reshape(size, order="F")

    def integers(self, low: int, high: int | None = None) -> int:
        if high is None:
            high = low
            low = 0
        return low + int(np.floor((high - low) * self._rand_float()))

    def _rand_float(self) -> float:
        first = self._uint32() >> 5
        second = self._uint32() >> 6
        return (first * 67108864.0 + second) / 9007199254740992.0

    def _uint32(self) -> int:
        if self._index >= self._N:
            self._twist()
        value = self._state[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    def _twist(self) -> None:
        mag01 = [0, self._MATRIX_A]
        for kk in range(self._N - self._M):
            value = (self._state[kk] & self._UPPER_MASK) | (self._state[kk + 1] & self._LOWER_MASK)
            self._state[kk] = self._state[kk + self._M] ^ (value >> 1) ^ mag01[value & 1]
        for kk in range(self._N - self._M, self._N - 1):
            value = (self._state[kk] & self._UPPER_MASK) | (self._state[kk + 1] & self._LOWER_MASK)
            self._state[kk] = self._state[kk + (self._M - self._N)] ^ (value >> 1) ^ mag01[value & 1]
        value = (self._state[self._N - 1] & self._UPPER_MASK) | (self._state[0] & self._LOWER_MASK)
        self._state[self._N - 1] = self._state[self._M - 1] ^ (value >> 1) ^ mag01[value & 1]
        self._index = 0


def default_distgen_csv_path(sample_name: str = DEFAULT_SAMPLE_NAME) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root.parent / "geometry_experiments" / "GeomDiagnostics.jl" / "outputs" / f"{sample_name}.csv"


def load_distgen_data(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found at: {path}")
    data = np.loadtxt(path, delimiter=",")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"NGPCA requires at least 2 columns. Found {data.shape[1]}.")
    return data


def run(
    data_path: Path | None = None,
    sample_name: str | None = None,
    num_units: int | None = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> tuple[NeuralGasPCA, np.ndarray]:
    rng = MatlabTwister(0)
    sample_name = sample_name or os.getenv("SAMPLE_NAME") or DEFAULT_SAMPLE_NAME
    data = load_distgen_data(data_path or default_distgen_csv_path(sample_name))
    num_units = num_units or _env_int("NGPCA_NUM_UNITS", DEFAULT_NUM_UNITS)
    params = NGPCAParameters(
        softhard=FreeParameter(1.0, name="softhard", lower=0.0, upper=1.0),
    )
    model = NeuralGasPCA(
        number_units=num_units,
        pca_dimensionality=DEFAULT_PCA_DIMENSIONALITY,
        params=params,
        rng=rng,
        weight_initialization="uniform",
    )
    model.initialize(data).fit(data, iterations=iterations)
    return model, data


def plot(model: NeuralGasPCA, data: np.ndarray, ax: Any | None = None) -> tuple[Any, Any]:
    """Plot data and local PCA units like the reference MATLAB NGPCA examples."""

    plt = _load_pyplot()
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must have at least two columns to plot")

    ax.scatter(data[:, 0], data[:, 1], s=5, marker="o", alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for unit in model.units:
        _draw_ngpca_unit(ax, unit)

    ax.set_title("NGPCA on DistGen samples")
    # MATLAB's examples call axis manual before draw(), so unit ellipses do not
    # expand the view beyond the data cloud.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return figure, ax


def to_matlab_json_dict(model: NeuralGasPCA) -> dict[str, Any]:
    """Serialize a fitted model with the same field names as ngpca_to_json.m."""

    params = model.params
    n_features = 0 if not model.units else int(model.units[0].center.size)

    return {
        "potentialFunction": params.potential_function,
        "softhard": float(params.softhard),
        "learningRate": float(params.learning_rate),
        "activity": float(params.activity),
        "rho_init": float(params.rho_init),
        "rho_final": float(params.rho_final),
        "mu": float(params.mu),
        "rmax": float(params.rmax),
        "lambda": float(params.initial_eigenvalue),
        "numberUnits": model.number_units,
        "PCADimensionality": model.pca_dimensionality,
        "iterations": model.iteration,
        "rho": model.rho,
        "units": [_unit_to_matlab_json_dict(unit) for unit in model.units],
        "r": model.ranking,
        "CI": [],
        "DU": [],
        "NMI": [],
        "initialized": int(bool(model.units)),
        "dataDimensionality": n_features,
    }


def save_matlab_json(model: NeuralGasPCA, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_matlab_json_dict(model), separators=(",", ":")))


def _unit_to_matlab_json_dict(unit: Any) -> dict[str, Any]:
    return {
        "m": unit.m,
        "center": unit.center.tolist(),
        "weight": unit.weight.tolist(),
        "eigenvalue": unit.eigenvalue.tolist(),
        "sigma_sqr": unit.sigma_sqr,
        "x_c": unit.x_c.tolist(),
        "x_c_bar": np.ones(unit.center.size).tolist(),
        "y": unit.y.tolist(),
        "activity": unit.activity,
        "eta_bar": unit.lr_state.eta_bar.tolist(),
        "l_bar": unit.lr_state.l_bar.tolist(),
        "gamma_bar": unit.lr_state.gamma_bar.tolist(),
        "alpha": unit.alpha,
        "epsilon": unit.lr_state.epsilon,
        "H": 0,
    }


def _draw_ngpca_unit(ax: Any, unit: Any) -> None:
    projected_weight = unit.weight[:2, :]
    covariance = (projected_weight * unit.eigenvalue[None, :]) @ projected_weight.T
    if unit.weight.shape[0] != unit.m:
        rest = unit.sigma_sqr / (unit.weight.shape[0] - unit.m)
        covariance = covariance + rest * (np.eye(2) - projected_weight @ projected_weight.T)

    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    values = np.maximum(values, 0.0)
    _plot_ellipse(ax, vectors, np.sqrt(values), unit.center[:2])

    axis_indices = np.argsort(unit.eigenvalue)[-2:][::-1]
    original_axes = unit.weight[:2, axis_indices]
    original_scale = np.sqrt(np.maximum(unit.eigenvalue[axis_indices], 0.0))
    position = unit.center[:2]
    for direction, scale in zip(original_axes.T, original_scale):
        endpoint = position + scale * direction
        ax.plot([position[0], endpoint[0]], [position[1], endpoint[1]], color="black", linewidth=1.0)


def _plot_ellipse(ax: Any, vectors: np.ndarray, radii: np.ndarray, position: np.ndarray) -> None:
    theta = np.deg2rad(np.arange(0.0, 360.0 + 0.5, 0.5))
    axis = radii[0] * vectors[:, 0]
    angle = np.arctan2(vectors[:, 0] @ np.array([0.0, 1.0]), vectors[:, 0] @ np.array([1.0, 0.0]))
    x = np.linalg.norm(axis) * np.cos(theta)
    y = np.linalg.norm(radii[1] * vectors[:, 1]) * np.sin(theta)
    nx = position[0] + np.cos(angle) * x - np.sin(angle) * y
    ny = position[1] + np.sin(angle) * x + np.cos(angle) * y
    ax.plot(nx, ny, color="black", linewidth=1.0)


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("Plotting examples require matplotlib. Install it to use --plot or --save.") from exc
    return plt


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(round(float(raw)))
    except ValueError:
        return default
    return value if value >= 1 else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-name", default=os.getenv("SAMPLE_NAME", DEFAULT_SAMPLE_NAME))
    parser.add_argument("--data-path", type=Path, help="read DistGen samples from this CSV path")
    parser.add_argument("--num-units", type=int, default=_env_int("NGPCA_NUM_UNITS", DEFAULT_NUM_UNITS))
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--plot", action="store_true", help="show a Matplotlib plot of the fitted model")
    parser.add_argument("--save", type=Path, help="write the plot to this path")
    parser.add_argument("--save-json", type=Path, help="write a MATLAB-schema JSON serialization to this path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fitted, observations = run(
        data_path=args.data_path,
        sample_name=args.sample_name,
        num_units=args.num_units,
        iterations=args.iterations,
    )
    print("centers")
    print(np.round(fitted.centers(), 3))
    if args.save_json:
        save_matlab_json(fitted, args.save_json)
    if args.plot or args.save or _env_bool("NGPCA_PLOT"):
        fig, _ = plot(fitted, observations)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, bbox_inches="tight", dpi=150)
        if args.plot:
            _load_pyplot().show()
