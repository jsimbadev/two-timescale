Quickstart
==========

The smallest useful recursion tracks a running mean.  Its vector field is the
current residual,

.. math::

   F(z_t, X_{t+1}) = X_{t+1} - z_t,

so the recursion

.. math::

   z_{t+1} = z_t + a_t (X_{t+1} - z_t)

is an exponential moving average when ``a_t`` is fixed.

.. code-block:: python

   import numpy as np

   from two_timescale import LearningRate, Recursion, VectorField

   mean_field = VectorField(
       lambda state, observation, context: observation - state,
       name="mean_residual",
   )

   mean = Recursion(
       state=np.zeros(2),
       field=mean_field,
       step_size=LearningRate(0.05),
       name="running_mean",
   )

   for x in observations:
       mean.advance(x)

The step size can also come from another recursion.  Here a scalar EMA tracks a
bounded residual diagnostic and supplies the next outer step size.

.. math::

   b_{t+1} = b_t + \eta \left(
       \frac{\|X_{t+1} - z_t\|}{1 + \|X_{t+1} - z_t\|} - b_t
   \right).

.. code-block:: python

   from two_timescale import EMARecursion

   gain = EMARecursion(0.1, step_size=0.01, name="residual_gain")

   adaptive_mean = Recursion(
       state=np.zeros(2),
       field=mean_field,
       step_size=gain.control(),
       name="adaptive_mean",
   )

   for x in observations:
       residual = np.linalg.norm(x - adaptive_mean.state)
       gain.advance(residual / (1.0 + residual))
       adaptive_mean.advance(x)
