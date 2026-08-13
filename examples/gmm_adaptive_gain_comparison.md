# Adaptive Step-Size Sources for GMM Recursions

The comparison script exercises one fixed-step online GMM, one
polynomial-decay scheduled online GMM, and three adaptive step-size sources.
The GMM recursion itself remains the online moment recursion

```math
\theta_{n+1}
=
\theta_n
+
\alpha_n
\left(
T(x_n;\theta_n) - \theta_n
\right),
```

where `theta = (mass, first, second)` and
`T(x; theta) = (r, r x, r x x^T)`.

The step-size sources live outside the GMM vector field. Each one observes the
current GMM moment recursion and the current observation, updates a smoothed
diagnostic statistic, then exposes a scalar `value(...)` used as the next GMM
step size.

## Step-Size Sources

- `AdaptiveGMMGain`: reuses `AdaptiveGainRecursions` on the responsibility
  mismatch `r_n - w_n`.
- `SmoothedCovarianceControl`: smooths componentwise local covariance
  observations `(x_n - mu_k)(x_n - mu_k)^T` and maps their relative mismatch
  with `Sigma_k` to a gain.
- `SmoothedMomentResidualControl`: smooths the full GMM moment residual
  `T(x_n; theta_n) - theta_n` and maps its relative norm to a gain.

This keeps the model recursion task-agnostic while making the adaptive
diagnostic explicit and swappable.

The polynomial-decay baseline uses

```math
\alpha_n =
\frac{\alpha_0}{(1 + d n)^p},
```

with `--rm-initial-step`, `--rm-decay`, and `--rm-power`. If
`--rm-initial-step` is omitted, it starts from `--fixed-step`.

## Scenarios

Run the separated baseline:

```bash
uv run python -m examples.gmm_adaptive_gain_comparison --scenario easy
```

Run a harder stationary mixture with overlap, correlated covariances, and
unequal weights:

```bash
uv run python -m examples.gmm_adaptive_gain_comparison --scenario overlap
```

Run an epochwise drifting stream:

```bash
uv run python -m examples.gmm_adaptive_gain_comparison --scenario drift
```

The printed `true_nll` column is the negative log likelihood of the generating
mixture on the validation data for that epoch. In the drift scenario the
validation distribution changes each epoch, so this reference value moves too.
