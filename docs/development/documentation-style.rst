Documentation Style
===================

Documentation should make the mathematical object visible before introducing
the implementation.

Use this order for concept and algorithm pages:

1. Define the equation or recurrence.
2. State the role of each symbol.
3. Map mathematical quantities to library objects.
4. Show the smallest useful code example.
5. Cite the source when the equation or algorithm is taken from literature.

Docstrings should use Google style and include equations when the implementation
would otherwise be opaque.  Public docs and docstrings should cite papers by
standard bibliographic labels or rendered citations.  Local reference-manager
identifiers should not appear in rendered documentation.

Avoid adding prose that makes claims beyond the cited sources.  When a feature
is exploratory, say so directly.
