Preconditioners
===============

A preconditioner is a stateful transform applied to a vector-field value before
the outer recursion applies its own step size:

.. math::

   z_{t+1} = z_t + a_t P_t(F(z_t, X_{t+1})).

Adam is written this way in the library:

.. code-block:: text

   gradient-like field -> AdamPreconditioner -> outer step size

The preconditioner owns the first moment, second moment, and bias-correction
recursions.  It returns the Adam direction

.. math::

   P_t(g_t) = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.

The outer recursion owns the ``alpha`` step size from Adam
:cite:p:`kingma2017adam`.

.. code-block:: python

   from two_timescale import AdamPreconditioner, LearningRate, Recursion, VectorField

   gradient_field = VectorField(lambda state, observation, context: observation)
   preconditioner = AdamPreconditioner.new(shape=(2,))

   adam_field = preconditioner.precondition(gradient_field)
   state = Recursion(
       state=np.zeros(2),
       field=adam_field,
       step_size=LearningRate(0.001),
   )

This separation is useful outside optimization.  Any vector field with the
same shape can be preconditioned by the same pattern.
