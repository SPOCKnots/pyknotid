Pyknot2
=======

Python (and optional Cython) modules for detecting and measuring
knotting and linking. pyknot2 can analyse space-curves, i.e. sets of
points in three-dimensions, or can parse standard topological
representations of knot diagrams.

Installation
------------

pyknot2 supports both Python 2 and Python 3, you can install it with::

  $ pip install pyknot2

This will automatically install any Python dependencies that are not
already present.

Requirements
~~~~~~~~~~~~

If installing pyknot2 without pip, the following dependencies are required:

- cython (not essential, but strongly recommended)
- numpy
- peewee
- networkx
- planarity

Most of these are not hard requirements, but some functionality will
not be available if they are not present.


Example usage
-------------

.. code:: python

    In [1]: import pyknot2.spacecurves as sp

    In [2]: import pyknot2.make as mk

    In [3]: k = sp.Knot(mk.knot.three_twist(100)*10)

    In [4]: k.plot()

    In [5]: k.alexander_polynomial(-1)
    Finding crossings
    i = 0 / 97
    7 crossings found

    Simplifying: initially 14 crossings
    -> 10 crossings after 2 runs
    Out[5]: 6.9999999999999991

    In [6]: import sympy as sym

    In [7]: t = sym.var('t')

    In [8]: k.alexander_polynomial(t)
    Simplifying: initially 10 crossings
    -> 10 crossings after 1 runs
    Out[8]: 2/t - 3/t**2 + 2/t**3

    In [9]: k.octree_simplify(5)
    Run 0 of 5, 100 points remain
    Run 1 of 5, 98 points remain
    Run 2 of 5, 104 points remain
    Run 3 of 5, 92 points remain
    Run 4 of 5, 77 points remain

    Reduced to 77 points

    In [10]: k.plot()
