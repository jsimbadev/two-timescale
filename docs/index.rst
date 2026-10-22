two-timescale
=============

``two-timescale`` is a Python library for writing online algorithms as
composable stochastic recursions.  The central object is

.. math::

   z_{t+1} = z_t + a_t F(z_t, X_{t+1}),

where the state ``z_t``, vector field ``F``, and step size ``a_t`` are named
objects.  The same vocabulary describes fixed learning rates, recursive
step-size sources, smoothed diagnostics, Adam-style preconditioners, and the
slow-fast structure of Neural Gas PCA.

The documentation is written as a bridge between equations and code.  Concept
pages define the mathematical language, pattern pages show reusable
compositions, algorithm pages decompose larger methods into library objects,
and the API reference documents the public surface.

.. toctree::
   :maxdepth: 2

   quickstart
   concepts/index
   patterns/index
   algorithms/index
   how-tos/index
   development/index
   api/index
   glossary
   references

Installation
------------

Install the package in editable mode from the repository:

.. code-block:: bash

   uv sync

Install documentation dependencies when building the docs:

.. code-block:: bash

   uv sync --extra docs
   uv run sphinx-build -b html docs docs/_build/html
