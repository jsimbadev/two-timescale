Design Principles
=================

The package treats online algorithms as systems of named recursions.  A method
is clearer when each moving quantity owns its state and exposes how it is used
by the next level of composition.

The design has four rules.

First, a ``Recursion`` owns state.  It advances that state and increments its
own iteration counter.

Second, a ``VectorField`` computes the direction of motion.  Fields can be
added, scaled, or wrapped by preconditioners.

Third, a ``step_size`` is any scalar source that can be evaluated at update
time.  It may be a fixed number, a schedule, or the state of another recursion.

Fourth, adaptive quantities should be named.  Adam moments, Neural Gas PCA
variance diagnostics, and exploratory GMM gains are represented as smoothed
statistics or recursion bundles instead of hidden temporary values.

This vocabulary makes algorithms declarative: the code states which recursions
exist, which statistics they smooth, and which vector field they control.
