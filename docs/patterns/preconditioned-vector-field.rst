Preconditioned Vector Field
===========================

A preconditioner turns one vector field into another:

.. math::

   \tilde{F}_t(z_t, X_{t+1}) = P_t(F(z_t, X_{t+1})).

The outer recursion still owns the step size.

.. code-block:: python

   import numpy as np

   from two_timescale import AdamPreconditioner, Recursion, VectorField

   field = VectorField(
       lambda state, observation, context: observation - state,
       name="residual",
   )

   preconditioner = AdamPreconditioner.new(shape=(2,))

   recursion = Recursion(
       state=np.zeros(2),
       field=preconditioner.precondition(field),
       step_size=0.001,
   )

   recursion.advance(np.ones(2))

This is the same mechanism used by the high-level ``Adam`` wrapper, but the
preconditioned field can be attached to any compatible recursion.
