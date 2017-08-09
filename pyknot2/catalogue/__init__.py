'''
Knot catalogue
==============

pyknot2 provides lookup access for different knot types from a
pre-build database of knot invariants.

Lookup by name
--------------

Use :func:`pyknot2.catalogue.identify.get_knot`::

    from pyknot2.catalogue.identify import get_knot
    trefoil = get_knot('3_1')
    figure_eight = get_knot('4_1')

Lookup by invariants
--------------------
Use :func:`pyknot2.catalogue.identify.from_invariants`::

    from pyknot2.catalogue.identify import from_invariants

    from_invariants(determinant=5, max_crossings=9)
    # returns [<Knot 4_1>, <Knot 5_1>]

    import sympy as sym
    t = sym.var('t')
    from_invariants(alexander=1-t+t**2, max_crossings=9)
    # returns [<Knot 3_1>]

For a full list of lookup parameters, see :func:`~pyknot2.catalogue.identify.from_invariants`.

'''

from .identify import from_invariants, get_knot, first_from_invariants
