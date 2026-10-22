Vector Fields
=============

A vector field is the update direction ``F``.  For a running mean,

.. math::

   F(z_t, X_{t+1}) = X_{t+1} - z_t.

In the library, vector fields are callable objects with algebraic composition.

.. code-block:: python

   from two_timescale import VectorField

   residual = VectorField(
       lambda state, observation, context: observation - state,
       name="residual",
   )

Fields can be scaled by step-size sources or added to other fields.  This is
how the package keeps algorithmic wiring explicit: a higher-level method can
construct a field, wrap it with a preconditioner, and pass the result to a
recursion.

.. math::

   \tilde{F}_t(z_t, X_{t+1}) = P_t(F(z_t, X_{t+1})).

Here ``P_t`` is stateful, but it is still separate from the outer recursion's
step size.
