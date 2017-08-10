Overview
========

pyknot2 is a Python module for doing calculations on knots or links,
whether specified as space-curves, or via standard topological
notations.


Space curve analysis
--------------------

pyknot2 can perform many calculations on space curves specified as a
set of points, as well as plotting the curves in three dimensions or
in projection. See the :doc:`space curve documentation
<spacecurves/index>` for more information.

Example::

  from pyknot2.make import trefoil
  from pyknot2.spacecurves import Knot

  k = Knot(trefoil())

  k.determinant()  # 3
  k.gauss_code()  # 1+a,2-a,3+a,1-a,2+a,3-a
  k.identify()  # [<Knot 3_1>]


Topological representations
---------------------------

pyknot2 can accept input using several standard topological notations,
using these to calculate topological invariants, or even to
reconstruct a 3D space curve. See the :doc:`representation
documentation <representations/index>` for more information.

Example::

  from pyknot2.representations import GaussCode, Representation

  gc = GaussCode('1+a,2-a,3+a,1-a,2+a,3-a')
  gc.simplify() # does nothing here, as no Reidemeister moves can be
                # performed to immediately simplify the curve

  # Representation is a generic topological representation providing
  # more methods
  rep = Representation(gc)
  rep.determinant()  # 3
  rep.space_curve()  # <Knot with 34 points>, a space curve with the
                     # given Gauss code on projection

Knot catalogue
--------------

pyknot2 can look up knot types in a prebuilt database, according to
the knot name (e.g. ``3_1`` for the trefoil knot, ``4_1`` for the
figure-eight knot etc.), or the values of its knot invariants. See the :doc:`knot catalogue documentation` for more information.

Example::

  from pyknot2.catalogue import get_knot, from_invariants

  k = get_knot('5_2')
  k.vassiliev_2  # 2
  k.determinant()  # 3

  k = get_knot('7_3').space_curve()  # <Knot with 83 points>, a space curve
                                     # that forms a 7_3 knot.

  from_invariants(determinant=7, max_crossings=11) # [<Knot 5_2>,
                                                   #  <Knot 7_1>,
                                                   #  <Knot 9_42>,
                                                   #  <Knot K11n57>,
                                                   #  <Knot K11n96>,
                                                   #  <Knot K11n111>]
  
