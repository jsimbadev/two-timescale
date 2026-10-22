Glossary
========

.. glossary::

   recursion
      Stateful object that advances an estimate by applying a vector field and
      a step size.

   vector field
      Callable update direction ``F(z_t, X_{t+1})``.

   step size
      Scalar multiplier ``a_t`` applied to an outer vector-field update.

   step-size source
      Fixed scalar, schedule, callable, or scalar recursion that supplies a
      step size at update time.

   diagnostic
      Instantaneous statistic computed from a state, observation, and optional
      recursion context.

   smoothed statistic
      Diagnostic average tracked by an inner recursion.

   gain
      Scalar value derived from a parameter or statistic and used to scale an
      update.

   transform
      Value derived from a smoothed statistic.  A transform may be scalar,
      vector-valued, matrix-valued, or structured.

   preconditioner
      Stateful transform ``P_t`` that maps a vector-field value into a new
      vector-field value before the outer step size is applied.

   timescale
      Relative update speed of a recursion in a coupled system.
