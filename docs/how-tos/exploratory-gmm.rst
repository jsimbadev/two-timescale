Run An Online Gaussian Mixture Recursion
========================================

``GaussianMixtureRecursion`` tracks sufficient statistics for mixture weights,
means, and covariances.

.. code-block:: python

   import numpy as np

   from two_timescale import GaussianMixtureRecursion, LearningRate

   model = GaussianMixtureRecursion(
       n_components=3,
       n_features=2,
       step_size=LearningRate(0.03),
       rng=np.random.default_rng(42),
   )

   model.advance(data)

   weights = model.weights
   means = model.means
   labels = model.assignments(data)

The GMM controls in this package are exploratory step-size sources.  They use
the same smoothed-statistic vocabulary as the rest of the library, but should
be treated as diagnostics rather than a sourced reference algorithm.
