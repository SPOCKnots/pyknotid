'''
Knots and links can be encoded in many different ways, generally by
enumerating their self-intersections in projection along some
axis. We provide here

This module contains classes and functions for representing
knots in knot diagrams, mainly the
:class:`pyknotid.representations.gausscode.GaussCode` and
:class:`pyknotid.representations.planardiagram.PlanarDiagram`.

These provide convenient methods to convert between different
representations, and to simplify via Reidemeister moves.

Creating representations
########################

Knot representations can be calculated from space curves, or created directly by inputting standard notations.

From space curves
-----------------

pyknotid's space curve classes can all return topological representations. For instance::

  from pyknotid.spacecurves import Knot
  from pyknotid.make import trefoil
  k = Knot(trefoil())
  
You can extract a :class:`~pyknotid.representations.gausscode.GaussCode` object::

  k.gauss_code()  # 1+a,2-a,3+a,1-a,2+a,3-a

or a :class:`~pyknotid.representations.planardiagram.PlanarDiagram`::

  k.planar_diagram()  # PD with 3: X_{2,5,3,6} X_{4,1,5,2} X_{6,3,1,4}

or a Gauss diagram::

  k.gauss_diagram()  # plots the diagram in a new window using matplotlib

.. image:: example_gauss_diagram_k10_93.png
   :alt: An example Gauss diagram for the knot 10_93
   :align: center
   :scale: 50%

or a generic :class:`~pyknotid.representations.representation.Representation`::

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

  from pyknotid.representations import GaussCode
  gc = GaussCode('1+c,2-c,3+c,1-c,2+c,3-c')

If you do not know the crossing orientations (c/a), pyknotid can
calculate them automatically::

  gc = GaussCode.calculating_orientations('1+,2-,3+,1-,2+,3-')

If you do this with a chiral not, the chirality is selected
arbitrarily.


Calculating invariants
######################

You can calculate many invariants using the functions of
:doc:`../invariants`.

pyknotid also provides a more convenient interface using the
:class:`~pyknotid.representations.representation.Representation`
class. Internally this wraps a Gauss code::

  from pyknotid.representations import Representation
  rep = Representation('1-c,2+c,3-a,4+a,2-c,1+c,4-a,3+a')

You can then calculate many quantities via methods of this object::

  rep.vassiliev_degree_2()  # 1
  rep.vassiliev_degree_3()  # -1
  rep.identify()  # [<Knot 4_1>]

For a full list of available functions, see
:class:`~pyknotid.representations.representation.Representation`.

'''

from pyknotid.representations.gausscode import GaussCode
from pyknotid.representations.planardiagram import PlanarDiagram
from pyknotid.representations.dtnotation import DTNotation
from pyknotid.representations.representation import Representation
__all__ = ['GaussCode', 'PlanarDiagram', 'DTNotation', 'Representation', ]

