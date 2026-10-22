Adam
====

Mathematical Form
-----------------

Adam estimates first and second raw moments of a gradient-like vector ``g_t``
:cite:p:`kingma2017adam`:

.. math::

   m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,

.. math::

   v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2.

The initialized moments are bias corrected as

.. math::

   \hat{m}_t = \frac{m_t}{1-\beta_1^t},
   \qquad
   \hat{v}_t = \frac{v_t}{1-\beta_2^t}.

The Adam update direction is

.. math::

   P_t(g_t) = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.

Library Decomposition
---------------------

.. list-table::
   :header-rows: 1

   * - Mathematical object
     - Library object
   * - ``m_t``
     - first ``SmoothedStatisticTransform``
   * - ``v_t``
     - second ``SmoothedStatisticTransform``
   * - ``1-\beta_1^t`` and ``1-\beta_2^t``
     - bias ``EMARecursion`` objects
   * - ``P_t``
     - ``AdamPreconditioner``
   * - ``\alpha_t``
     - outer ``step_size``

The high-level ``Adam`` wrapper is equivalent to an outer recursion using an
``AdamPreconditioner``.  The preconditioner owns the moment state; the outer
recursion owns ``\alpha_t``.

Minimal Example
---------------

.. code-block:: python

   import numpy as np

   from two_timescale import Adam

   adam = Adam.new(state=np.zeros(2), step_size=0.001)

   for gradient in gradients:
       adam.update(-gradient)

Use the lower-level preconditioner when the Adam direction should wrap a
different vector field.

.. code-block:: python

   from two_timescale import AdamPreconditioner, Recursion, VectorField

   field = VectorField(lambda state, observation, context: observation - state)
   preconditioner = AdamPreconditioner.new(shape=(2,))

   recursion = Recursion(
       state=np.zeros(2),
       field=preconditioner.precondition(field),
       step_size=0.001,
   )
