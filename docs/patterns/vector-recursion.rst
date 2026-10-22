Vector Recursion
================

Vector states use the same interface as scalar states.  For a running vector
mean,

.. math::

   z_{t+1} = z_t + a_t(X_{t+1} - z_t).

.. code-block:: python

   import numpy as np

   from two_timescale import Recursion, VectorField

   residual = VectorField(
       lambda state, observation, context: observation - state,
       name="residual",
   )

   mean = Recursion(state=np.zeros(3), field=residual, step_size=0.05)

   for x in observations:
       mean.advance(x)

The vector field is the only object that needs to know the shape of the
increment.
