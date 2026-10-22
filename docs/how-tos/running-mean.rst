Track A Running Mean
====================

Use a residual vector field and a fixed step size:

.. math::

   z_{t+1} = z_t + a(X_{t+1} - z_t).

.. code-block:: python

   import numpy as np

   from two_timescale import Recursion, VectorField

   mean_field = VectorField(lambda state, observation, context: observation - state)
   mean = Recursion(state=np.zeros(2), field=mean_field, step_size=0.05)

   for x in observations:
       mean.advance(x)

The estimate is available as ``mean.state``.
