'''pyknotid provides knot lookup by name or invariant values, using a
prebuilt database.

The knot database includes information about all knots with up to 15
crossings, with topological invariants following those indexed by the
`Knot Atlas <http://katlas.org/wiki/Main_Page>`__ and the `KnotInfo
Table of Knot Invariants <http://www.indiana.edu/~knotinfo/>`__, or
calculated by pyknotid using the Dowker-Thistlethwaite codes of the
knots.

Downloading the database
------------------------

The database must normally be downloaded separately, and is currently
approximately 230MB in size.

If you do not download the database, most of pyknotid will work
fine. Only the explicit knot identification by database lookup, or
direct database queries, are not available.

To download the knot database::

    from pyknotid.catalogue.getdb import download_database
    download_database()

After this has completed (it may take a few seconds), the database
functions should all work immediately.

For other database management functions, see :doc:`getdb`.

Lookup by name
--------------

Use :func:`pyknotid.catalogue.identify.get_knot`::

    from pyknotid.catalogue.identify import get_knot
    trefoil = get_knot('3_1')
    figure_eight = get_knot('4_1')

Lookup by invariants
--------------------
Use :func:`pyknotid.catalogue.identify.from_invariants`::

    from pyknotid.catalogue.identify import from_invariants

    from_invariants(determinant=5, max_crossings=9)
    # returns [<Knot 4_1>, <Knot 5_1>]

    import sympy as sym
    t = sym.var('t')
    from_invariants(alexander=1-t+t**2, max_crossings=9)
    # returns [<Knot 3_1>]

For a full list of lookup parameters, see :func:`~pyknotid.catalogue.identify.from_invariants`.

Exploring properties of knots
-----------------------------

You can view more properties of any knot returned by the database::

  from pyknotid.catalogue import get_knot, from_invariants

  k = get_knot('5_2')
  k.pretty_print()  # prints some information from the database:
                    #  Identifier: 5_2
                    #  Min crossings: 5
                    #  Fibered: False
                    #  Gauss code: -1, 5, -2, 1, -3, 4, -5, 2, -4, 3
                    #  Planar diagram: X_1425 X_3849 X_5,10,6,1 X_9,6,10,7 X_7283
                    #  DT code: 4 8 10 2 6
                    #  Determinant: 7
                    #  Signature: -2
                    #  Alexander: 2*t**2 - 3*t + 2
                    #  Jones: 1/q - 1/q**2 + 2/q**3 - 1/q**4 + q**(-5) - 1/q**6
                    #  HOMFLY: -a**6 + a**4*z**2 + a**4 + a**2*z**2 + a**2
                    #  Hyperbolic volume: 2.82812
                    #  Vassiliev order 2: 2
                    #  Vassiliev order 3: -3
                    #  Symmetry: reversible

Properties of the knot can also be accessed directly::

  k.determinant  # 7

For a full list of attributes available, see
:class:`pyknotid.catalogue.database.Knot`.

'''

from pyknotid.catalogue.identify import from_invariants, get_knot, first_from_invariants
from pyknotid.catalogue.getdb import download_database

db_version = 1
