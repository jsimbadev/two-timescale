Recursive Step Size
===================

A recursive step-size source separates "how the gain is estimated" from "how
the outer state moves".  Let

.. math::

   g_{t+1} = g_t + b_t(r_t - g_t)

track a scalar diagnostic ``r_t``.  The outer recursion can then use ``g_t`` as
its current step size.

.. code-block:: python

   import numpy as np

   from two_timescale import EMARecursion, Recursion, VectorField

   gain = EMARecursion(0.1, step_size=0.01, name="gain")
   residual = VectorField(lambda state, observation, context: observation - state)

   recursion = Recursion(
       state=np.zeros(2),
       field=residual,
       step_size=gain.control(),
   )

   for x in observations:
       error = np.linalg.norm(x - recursion.state)
       gain.advance(error / (1.0 + error))
       recursion.advance(x)

The update order is part of the model: this example updates the gain before the
outer state consumes it.
