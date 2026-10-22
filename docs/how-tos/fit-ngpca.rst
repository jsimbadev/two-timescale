Fit Neural Gas PCA
==================

``NeuralGasPCA`` consumes one observation or a two-dimensional stream.  Each
local unit owns recursions for its center, local PCA shape, residual variance,
and adaptive gain.

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

The algorithm page explains how these recursions correspond to the local PCA
updates described by :cite:p:`migenda2024adaptive`.
