'''Topological representations
===========================

Knots and links can be encoded in many different ways, generally by
enumerating their self-intersections in projection along some
axis. We provide here

This module contains classes and functions for representing
knots in knot diagrams, mainly the
:class:`pyknot2.representations.gausscode.GaussCode` and
:class:`pyknot2.representations.planardiagram.PlanarDiagram`.

These provide convenient methods to convert between different
representations, and to simplify via Reidemeister moves.

Creating representations
########################

Knot representations can be calculated from space curves, or created directly by inputting standard notations.

From space curves
-----------------

pyknot2's space curve classes can all return topological representations. For instance::

  from pyknot2.spacecurves import Knot
  from pyknot2.make import trefoil
  k = Knot(trefoil())
  
You can extract a :class:`~pyknot2.representations.gausscode.GaussCode` object::

  k.gauss_code()  # 1+a,2-a,3+a,1-a,2+a,3-a

or a :class:`~pyknot2.representations.planardiagram.PlanarDiagram`::

  k.planar_diagram()  # PD with 3: X_{2,5,3,6} X_{4,1,5,2} X_{6,3,1,4}

or a Gauss diagram::

  k.gauss_diagram()  # plots the diagram in a new window using matplotlib

or a generic :class:`~pyknot2.representations.representation.Representation`::

  k.representation()  # 1+a,2-a,3+a,1-a,2+a,3-a (but provides more methods than a GaussCode)

By direct input
---------------

Gauss codes
~~~~~~~~~~~

A Gauss code is a list of crossings in a projection of a curve,
labelled by numbers, and in each case indicating whether the curve
passes over (+) or under (-) itself. Each crossing also has a local
orientation, represented here by 'c' for clockwise, or 'a' for
anticlockwise.

With these rules, you can enter Gauss codes as comma-separated lists::

  from pyknot2.representations import GaussCode
  gc = GaussCode('1+c,2-c,3+c,1-c,2+c,3-c')

If you do not know the crossing orientations (c/a), pyknot2 can
calculate them automatically::

  gc = GaussCode.calculating_orientations('1+,2-,3+,1-,2+,3-')

If you do this with a chiral not, the chirality is selected
arbitrarily.


Calculating invariants
----------------------

You can calculate many invariants using the functions of
:doc:`../invariants`.

pyknot2 also provides a more convenient interface using the
:class:`~pyknot2.representations.representation.Representation`
class. Internally this wraps a Gauss code::

  from pyknot2.representations import Representation
  rep = Representation('1-c,2+c,3-a,4+a,2-c,1+c,4-a,3+a')

You can then calculate many quantities via methods of this object::

  rep.vassiliev_degree_2()  # 1
  rep.vassiliev_degree_3()  # -1
  rep.identify()  # [<Knot 3_1>]

For a full list of available functions, see
:class:`~pyknot2.representations.representation.Representation`.

'''

from pyknot2.representations.gausscode import GaussCode
from pyknot2.representations.planardiagram import PlanarDiagram
from pyknot2.representations.dtnotation import DTNotation
from pyknot2.representations.representation import Representation
__all__ = ['GaussCode', 'PlanarDiagram', 'DTNotation', 'Representation', ]

