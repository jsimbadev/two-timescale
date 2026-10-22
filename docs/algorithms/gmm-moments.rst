GMM Moment Recursions
=====================

Mathematical Form
-----------------

The Gaussian-mixture wrapper tracks sufficient statistics by stochastic
approximation.  For responsibilities ``r_k(x_t)`` and component index ``k``,
the target one-sample moments are

.. math::

   M_{0,k}(x_t) = r_k(x_t),
   \qquad
   M_{1,k}(x_t) = r_k(x_t)x_t,

.. math::

   M_{2,k}(x_t) = r_k(x_t)x_t x_t^\top.

The running moments update as

.. math::

   M_{t+1} = M_t + a_t(\hat{M}(x_t; M_t) - M_t).

This is the same stochastic-recursion pattern described in the core concepts
:cite:p:`kushner2003stochastic`.

Library Decomposition
---------------------

.. list-table::
   :header-rows: 1

   * - Quantity
     - Library object
   * - masses, first moments, second moments
     - ``GMMMoments``
   * - online moment update
     - ``GMMMomentRecursion``
   * - user-facing model
     - ``GaussianMixtureRecursion``
   * - exploratory adaptive gains
     - ``SmoothedStatisticControl`` subclasses in ``gmm_controls``

Minimal Example
---------------

.. code-block:: python

   import numpy as np

   from two_timescale import GaussianMixtureRecursion

   model = GaussianMixtureRecursion(
       n_components=3,
       n_features=2,
       rng=np.random.default_rng(42),
   )

   model.advance(data)

   weights, means, covariances = model.parameters()

The GMM controls are documented as exploratory diagnostics.  They share the
library vocabulary, but the docs should not present them as a sourced algorithm
unless a specific reference is added.
