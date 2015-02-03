Pyknot
======

Python (and optional Cython) modules for dealing with knotting and linking.

Pyknot is a work in progress, but in principle should support python2
and python3 (if it doesn't, that's a bug), and provide classes to help
with knots and links as space curves, detecting crossings, calculating
and analysing knot invariants etc.


Example usage
-------------

::

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
