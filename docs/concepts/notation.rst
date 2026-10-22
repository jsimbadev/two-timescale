Notation
========

The documentation uses one notation across examples and algorithms.

.. list-table::
   :header-rows: 1

   * - Symbol
     - Meaning
     - Library object
   * - ``z_t``
     - State owned by the outer recursion.
     - ``Recursion.state``
   * - ``X_{t+1}``
     - Observation or mini-batch driving the next update.
     - ``observation`` argument
   * - ``F``
     - Vector field evaluated at the current state and observation.
     - ``VectorField``
   * - ``a_t``
     - Step size multiplying the outer update.
     - ``step_size``
   * - ``d``
     - Diagnostic statistic computed from a state and observation.
     - ``diagnostic`` callable
   * - ``\bar{s}_t``
     - Smoothed diagnostic statistic.
     - ``EMARecursion.state``
   * - ``P_t``
     - Stateful transform applied to a vector field.
     - preconditioner

The standard outer recursion is

.. math::

   z_{t+1} = z_t + a_t F(z_t, X_{t+1}).

An inner recursion uses the same form, but its state is auxiliary.  For
example, an adaptive step-size source may track a diagnostic average
``\bar{s}_t`` while the outer recursion tracks ``z_t``.
