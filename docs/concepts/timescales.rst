Timescales
==========

A recursive system can have several coupled update speeds.  The package does
not require a special "two-timescale" class.  Instead, each stateful quantity is
itself a recursion or recursion-like object.

For an outer state ``z_t`` and an inner statistic ``\bar{s}_t``, a typical
slow-fast system has the form

.. math::

   \bar{s}_{t+1}
   =
   \bar{s}_t + b_t(d(z_t, X_{t+1}) - \bar{s}_t),

.. math::

   z_{t+1}
   =
   z_t + a_t G(z_t, X_{t+1}, \bar{s}_{t+1}).

The inner update may be faster or slower depending on the relative sizes of
``a_t`` and ``b_t``.  In code, that relationship is expressed by composition:
the inner object updates first, then exposes a scalar gain or transform to the
outer recursion.

Adam fits this language because the first and second moments evolve inside an
``AdamPreconditioner`` while the parameter recursion owns the outer step size.
NGPCA fits the same language because local variance diagnostics evolve inside
adaptive-gain recursions while center and shape recursions consume the resulting
gain.
