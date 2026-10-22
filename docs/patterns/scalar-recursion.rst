Scalar Recursion
================

A scalar recursion is useful for gains, diagnostics, and running losses.  The
same stochastic-approximation form applies:

.. math::

   u_{t+1} = u_t + a_t(f(u_t, X_{t+1}) - u_t).

An exponential moving average is available directly:

.. code-block:: python

   from two_timescale import EMARecursion

   loss_average = EMARecursion(0.0, step_size=0.02, name="loss_average")

   for loss in losses:
       loss_average.advance(loss)

If the scalar should drive another recursion, expose it through
``loss_average.control()``.
