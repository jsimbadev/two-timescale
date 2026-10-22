Smoothed Statistics
===================

Adaptive algorithms often compute a diagnostic, smooth it, and map it to a
gain or transform:

.. math::

   \bar{s}_{t+1}
   =
   \bar{s}_t + b_t\left(d(z_t, X_{t+1}) - \bar{s}_t\right).

The diagnostic ``d`` is an instantaneous quantity.  The state ``\bar{s}_t`` is
the slow quantity the algorithm actually uses.

``SmoothedStatisticControl`` maps the smoothed statistic to a scalar.  Use it
when the statistic should become a step-size source.

.. code-block:: python

   from two_timescale import SmoothedStatisticControl

   gain = SmoothedStatisticControl.new(
       initial_statistic=0.0,
       diagnostic=lambda state, observation, context: abs(observation - state),
       gain_map=lambda statistic, state, observation, context: statistic / (1.0 + statistic),
       initial_gain=0.1,
       smoothing=0.02,
       upper=0.5,
   )

``SmoothedStatisticTransform`` maps the smoothed statistic to any value, not
only a scalar.  Adam uses this for first and second moments
:cite:p:`kingma2017adam`; NGPCA uses it for the variance-matching statistics
that define a local gain :cite:p:`migenda2024adaptive`.
