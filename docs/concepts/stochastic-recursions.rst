Stochastic Recursions
=====================

Stochastic approximation studies recursive estimates of the form

.. math::

   z_{t+1} = z_t + a_t F(z_t, X_{t+1}),

where the observation sequence supplies noisy information about a target vector
field :cite:p:`kushner2003stochastic`.  The implementation keeps this form
literal:

.. code-block:: python

   from two_timescale import Recursion, VectorField

   field = VectorField(lambda state, observation, context: observation - state)
   recursion = Recursion(state=0.0, field=field, step_size=0.05)

   recursion.advance(2.0)

The same class is used for model states and auxiliary states.  That is the
main structural decision in the package: recursive algorithms are built by
wiring recursions together instead of introducing a separate abstraction for
each algorithmic role.
