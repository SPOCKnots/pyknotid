'''.. image:: random_walk_length_30.png
   :scale: 50%
   :alt: A closed random walks with 30 steps
   :align: center

This module contains classes and functions for working with knots and
links as three-dimensional space curves, or calling functions
elsewhere in pyknotid to perform topological analysis. Functionality
includes manipulating knots/links via translation, rotation and
scaling, plotting diagrams, finding crossings and identifying knots.

Different knot classes
----------------------

pyknotid includes the following classes for topological calculation:

- :doc:`spacecurve`: Provides functions for calculations on a single
  curve, including plotting, some geometrical properties and finding
  crossings in projection.

- :doc:`knot`: Provides functions for topological calculations on a
  single curve, such as the Alexander polynomial or Vassiliev
  invariants.

- :doc:`openknot`: Provides functions for topological calculations on
  an open curve that does not form a closed loop. Open curves are
  topologically trivial from a mathematical perspective, but can be
  analysed in terms of the topology of different closures.

- :doc:`link`: Provides the same interface to collections of multiple
  curves, and can calculate linking invariants.

- :doc:`periodiccell`: Provides some convenience functions for
  managing collections of curves in periodic boundaries.

Creating space curves
---------------------

The space curve classes are specified via N by 3 arrays of points in
three dimensions, representing a piecewise linear curve.

For instance, the following code produces and plots a
:class:`~pyknotid.spacecurves.knot.Knot` from a set of manually
specified points::

  import numpy as np
  from pyknotid.spacecurves import Knot

  points = np.array([[9.0, 0.0, 0.0],
                     [0.781, 4.43, 2.6],
                     [-4.23, 1.54, -2.6],
                     [-4.5, -7.79, -7.35e-16],
                     [3.45, -2.89, 2.6],
                     [3.45, 2.89, -2.6],
                     [-4.5, 7.79, 0.0],
                     [-4.23, -1.54, 2.6],
                     [0.781, -4.43, -2.6]])

  k = Knot(points)
  k.plot()

.. image:: trefoil_few_points.png
   :align: center
   :alt: A trefoil knot specified by vertex points
   :scale: 50%

The :doc:`pyknotid.make module<../make/index>` provides functions for
creating many types of example knots, such as torus knots or some
specific knot types::

  import numpy as np
  from pyknotid.make import torus_knot

   k = torus_knot(7, 4)
   k.plot()


.. image:: p7_q4__torus_knot.png
   :align: center
   :scale: 50%
   :alt: A p=7 and q=4 knot produced by the above code

'''

from pyknotid.spacecurves.spacecurve import SpaceCurve
from pyknotid.spacecurves.knot import Knot
from pyknotid.spacecurves.link import Link
from pyknotid.spacecurves.openknot import OpenKnot
from pyknotid.spacecurves.periodiccell import Cell

__all__ = ('SpaceCurve', 'Knot', 'Link', 'OpenKnot', 'Cell', )

