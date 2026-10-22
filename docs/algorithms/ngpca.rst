Neural Gas PCA
==============

Mathematical Form
-----------------

The implemented model follows the online local-PCA structure of
:cite:t:`migenda2024adaptive`.  Each unit has a center ``c_j``, a local basis
``W_j``, principal eigenvalues ``\lambda_j``, and a residual variance estimate.
For an observation ``x_t``,

.. math::

   \xi_j = x_t - c_j,
   \qquad
   y_j = W_j^\top \xi_j.

Units are ranked by a potential function and updated through a neighborhood
weight.  The implementation uses the package potential object to compute the
ranking and applies the resulting weight to the local adaptive gain.

Library Decomposition
---------------------

.. list-table::
   :header-rows: 1

   * - Model quantity
     - Library object
   * - center ``c_j``
     - ``StateTrackingRecursion``
   * - basis and eigenvalues
     - ``EFORRLSARecursion``
   * - residual variance
     - ``ScalarTrackingRecursion``
   * - adaptive local gain
     - ``AdaptiveGainRecursions``
   * - potential/ranking
     - ``Potential`` from ``NamedPotential`` or a callable

The adaptive local gain is the important abstraction match with the rest of the
library.  It is expressed as smoothed statistics rather than as opaque mutable
fields on the outer model.

Minimal Example
---------------

.. code-block:: python

   from two_timescale import NGPCAParameters, NamedPotential, NeuralGasPCA

   model = NeuralGasPCA(
       number_units=4,
       pca_dimensionality=2,
       params=NGPCAParameters(
           potential_function=NamedPotential("AR", rmax=200.0),
       ),
   )

   model.initialize(data)
   model.advance(data)

   centers = model.centers()
   labels = model.labels(data)

Structural Similarity To Adam
-----------------------------

Adam and NGPCA are structurally similar in this package.  Both have an outer
recursion that moves the main state and inner recursions that estimate adaptive
quantities.  In Adam, the inner quantities are moment estimates that produce a
coordinate-wise preconditioner.  In NGPCA, the inner quantities are local
variance diagnostics that produce a scalar gain for center and shape updates.
