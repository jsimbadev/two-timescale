Smoothed Diagnostic
===================

``SmoothedStatisticControl`` names the common pattern

.. math::

   \bar{s}_{t+1}
   =
   \bar{s}_t + b_t(d(z_t, X_{t+1}) - \bar{s}_t),
   \qquad
   a_{t+1} = h(\bar{s}_{t+1}).

The diagnostic ``d`` and map ``h`` are supplied by the caller.

.. code-block:: python

   from two_timescale import SmoothedStatisticControl

   gain = SmoothedStatisticControl.new(
       initial_statistic=0.0,
       diagnostic=lambda state, observation, context: abs(observation - state),
       gain_map=lambda statistic, state, observation, context: statistic / (1.0 + statistic),
       initial_gain=0.05,
       smoothing=0.02,
       upper=0.5,
   )

   gain.update(state=0.0, observation=1.0)

Use ``SmoothedStatisticTransform`` when the mapped quantity is not a scalar
step size.
