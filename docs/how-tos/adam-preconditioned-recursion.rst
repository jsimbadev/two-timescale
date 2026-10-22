Use An Adam-Preconditioned Recursion
====================================

Adam can be used as a complete wrapper when the observations are already update
directions:

.. code-block:: python

   import numpy as np

   from two_timescale import Adam

   adam = Adam.new(state=np.zeros(2), step_size=0.001)

   for gradient in gradients:
       adam.update(-gradient)

For a generic vector field, use only the preconditioner:

.. math::

   z_{t+1}
   =
   z_t + \alpha_t
   \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.

.. code-block:: python

   from two_timescale import AdamPreconditioner, Recursion, VectorField

   field = VectorField(lambda state, observation, context: observation - state)
   preconditioner = AdamPreconditioner.new(shape=(2,))

   recursion = Recursion(
       state=np.zeros(2),
       field=preconditioner.precondition(field),
       step_size=0.001,
   )

   for x in observations:
       recursion.advance(x)
