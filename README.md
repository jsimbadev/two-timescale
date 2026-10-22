# two-timescale

Composable Python abstractions for online stochastic recursions.

The library is built around the stochastic-approximation recursion

```math
z_{t+1} = z_t + a_t F(z_t, X_{t+1}).
```

In code, `z_t` is owned by a `Recursion`, `F` is a `VectorField`, and `a_t` is
a `step_size`. The step size may be a float, a schedule, or another recursion.
That gives a small vocabulary for writing recursive systems: outer algorithms
own their state, auxiliary recursions own their own state, and the wiring
between them is explicit.

## Minimal Example

```python
import numpy as np

from two_timescale import LearningRate, Recursion, VectorField

mean_field = VectorField(
    lambda state, observation, context: observation - state,
    name="mean_field",
)

mean = Recursion(
    state=np.zeros(2),
    field=mean_field,
    step_size=LearningRate(0.05),
)

for x in observations:
    mean.advance(x)
```

The same language is used by the package-level algorithms:

- `AdamPreconditioner` owns Adam's first moment, second moment, and bias
  recursions, then preconditions an outer vector field.
- `NeuralGasPCA` expresses local adaptive gains as smoothed statistics and uses
  separate recursions for centers, PCA shape, and residual variance.
- `GaussianMixtureRecursion` tracks online mixture sufficient statistics.

## Examples

Run the NGPCA example:

```bash
python -m examples.ngpca_example
```

Plot or save the NGPCA result:

```bash
python -m examples.ngpca_example --plot
python -m examples.ngpca_example --save outputs/ngpca.png
python -m examples.ngpca_example --save-json outputs/unit_banana_samples_python.json
```

Run the online Gaussian mixture recursion example:

```bash
python -m examples.gmm_recursion_example
python -m examples.gmm_recursion_example --plot
```

Plotting uses Matplotlib:

```bash
uv sync --extra examples
```

## Documentation

Build the Sphinx documentation locally:

```bash
uv sync --extra docs
uv run --extra docs sphinx-build -b html docs docs/_build/html
```

The docs include notation, concepts, reusable patterns, algorithm
decompositions with equations and citations, how-to examples, developer notes,
and an API reference.

## Tests

```bash
uv run python -m unittest
```
