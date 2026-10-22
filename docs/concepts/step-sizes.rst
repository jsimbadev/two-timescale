Step Sizes
==========

The outer step size is the scalar ``a_t`` in

.. math::

   z_{t+1} = z_t + a_t F(z_t, X_{t+1}).

In code, ``step_size`` can be a number, a schedule, a scalar object exposing
``value(context)``, or a scalar recursion state.

.. code-block:: python

   from two_timescale import LearningRate, ScheduledLearningRate

   fixed = LearningRate(0.05)
   scheduled = ScheduledLearningRate(LearningRate(0.2), decay=0.01, power=0.5)

Recursive step-size sources are useful when ``a_t`` depends on recent
diagnostics.  The source owns its state; the outer recursion only asks for the
current scalar value.

.. code-block:: python

   from two_timescale import EMARecursion, Recursion, VectorField

   gain = EMARecursion(0.1, step_size=0.01, name="gain")
   field = VectorField(lambda state, observation, context: observation - state)
   recursion = Recursion(state=0.0, field=field, step_size=gain.control())

   gain.advance(0.2)
   recursion.advance(1.0)

This separation matters for composability.  The step-size source can be tested,
replaced, or reused without changing the vector field it scales.
