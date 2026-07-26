# two-timescale

Composable Python abstractions for multiple-timescale stochastic recursions.
The first concrete model is Neural Gas PCA, based on the update structure in
Migenda, Moeller, and Schenck, "Adaptive local Principal Component Analysis
improves the clustering of high-dimensional data" (Pattern Recognition, 2024).

The core abstraction is:

```python
z[n + 1] = z[n] + a[n] * F(z[n], x[n])
```

where `F` is a `VectorField`, `a[n]` is a typed control such as a
`LearningRate`, and the state can itself belong to another `Recursion`.
Vector fields support algebraic scaling and addition, so an adaptive control
recursion can multiply another vector field.

```python
from two_timescale import LearningRate, Recursion, VectorField

control = Recursion(0.5, VectorField(lambda state, obs, ctx: 0.1), step_size=1.0)
field = VectorField(lambda state, obs, ctx: obs - state) * control
tracked = Recursion(0.0, field, step_size=LearningRate(1.0))

tracked.advance(2.0)  # 1.0
control.advance(None) # control state is now 0.6
tracked.advance(2.0)  # 1.6
```

`EMARecursion` is the same recursion algebra specialized to exponential moving
averages:

```python
from two_timescale import EMARecursion

average = EMARecursion(0.0, step_size=0.1)
average.advance(5.0)
```

NGPCA exposes the paper's online update as `advance`. Pass a single observation
or a two-dimensional stream; the caller owns any random sampling policy. Inside
each local PCA unit, the adaptive learning-rate filters are `EMARecursion`
objects. The update step composes those filters with generic stochastic
approximation objects: `StateTrackingRecursion` for the center,
`EFORRLSARecursion` for the eigensystem, and `ScalarTrackingRecursion` for the
unrepresented variance.

```python
from two_timescale import NGPCAParameters, NamedPotential, NeuralGasPCA

model = NeuralGasPCA(
    4,
    params=NGPCAParameters(potential_function=NamedPotential("AR", rmax=200.0)),
)
model.initialize(data)
model.advance(data[:100])
```

## Examples

Run the NGPCA example with the same DistGen sample source and NGPCA
configuration as `../geometry_experiments/NGPCA-Clustering/Example_distgen.m`.
By default it reads
`../geometry_experiments/GeomDiagnostics.jl/outputs/banana_samples.csv`, uses
hard clustering, 4 units, PCA dimensionality 2, 20,000 iterations, and a
MATLAB-compatible `rng(0,"twister")` random stream.

```bash
python -m examples.ngpca_example
```

The MATLAB environment overrides are also supported:

```bash
SAMPLE_NAME=banana_samples NGPCA_NUM_UNITS=4 NGPCA_PLOT=1 python -m examples.ngpca_example
```

Plot or save the NGPCA result with MATLAB-style unit ellipses and PCA axes:

```bash
python -m examples.ngpca_example --plot
python -m examples.ngpca_example --save outputs/ngpca.png
python -m examples.ngpca_example --save-json outputs/unit_banana_samples_python.json
```

Run the online Gaussian mixture recursion example:

```bash
python -m examples.gmm_recursion_example
```

Its fitted components can be plotted the same way:

```bash
python -m examples.gmm_recursion_example --plot
python -m examples.gmm_recursion_example --save outputs/gmm.png
```

Plotting uses Matplotlib. With uv, install the example extra first:

```bash
uv sync --extra examples
```

Run smoke tests:

```bash
uv run python -m unittest
```
