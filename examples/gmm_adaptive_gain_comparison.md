# Adaptive Controls for GMM Recursions

The comparison script exercises one fixed-step online GMM and three adaptive
controls. The GMM recursion itself remains the stochastic-EM moment recursion

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

The controls live outside the GMM vector field. Each control observes the
current GMM moment recursion and the current observation, updates a smoothed
diagnostic statistic, then exposes a scalar `value(...)` used as the next GMM
step size.

## Controls

- `AdaptiveGMMGain`: reuses `AdaptiveGainRecursions` on the responsibility
  mismatch `r_n - w_n`.
- `SmoothedCovarianceControl`: smooths componentwise local covariance
  observations `(x_n - mu_k)(x_n - mu_k)^T` and maps their relative mismatch
  with `Sigma_k` to a gain.
- `SmoothedMomentResidualControl`: smooths the full stochastic-EM residual
  `T(x_n; theta_n) - theta_n` and maps its relative norm to a gain.

This keeps the model recursion task-agnostic while making the adaptive-control
diagnostic explicit and swappable.
