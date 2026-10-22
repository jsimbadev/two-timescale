Adding Algorithms
=================

New algorithms should be decomposed into existing abstractions before adding a
new top-level class.

Use this checklist:

1. Identify the outer state ``z_t``.
2. Identify the vector field ``F`` driving that state.
3. Identify the outer step size ``a_t``.
4. Identify auxiliary recursions, diagnostics, and smoothed statistics.
5. Decide whether any auxiliary state is a scalar step-size source or a
   vector-field preconditioner.

Only add a new abstraction when the existing vocabulary cannot name a repeated
structure.  Prefer smaller recursion bundles when they make the mathematical
decomposition clearer.

Algorithm documentation should include a table mapping paper quantities to
library objects.  If the implementation intentionally differs from a paper, the
page should say where the package is using a general stochastic-recursion
pattern rather than reproducing a named algorithm exactly.
