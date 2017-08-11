'''
.. image:: random_walk_length_30.png
   :scale: 50%
   :alt: A closed random walks with 30 steps
   :align: center

This module contains classes and functions for working with knots and
links as three-dimensional space curves, or calling functions
elsewhere in pyknot2 to perform topological analysis. Functionality
includes manipulating knots/links via translation, rotation and
scaling, plotting diagrams, finding crossings and identifying knots.

Different knot classes
----------------------

pyknot2 includes the following classes for topological calculation:

- :doc:`spacecurve`: Provides functions for calculations on a single
  curve, including plotting, some geometrical properties and finding
  crossings in projection.

- :doc:`knot`: Provides functions for topological calculations on a
  single curve, such as the Alexander polynomial or Vassiliev
  invariants.

- :doc:`link`: Provides the same interface to collections of multiple
  curves, and can calculate linking invariants.

- :doc:`periodiccell`: Provides some convenience functions for
  managing collections of curves in periodic boundaries.

Creating space curves
---------------------

The space curve classes are specified via N by 3 arrays of points in
three dimensions, representing a piecewise linear curve.

For instance, the following code produces and plots a
:class:`~pyknot2.spacecurves.knot.Knot` from a set of manually
specified points::

  import numpy as np
  from pyknot2.spacecurves import Knot

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

The following code could produce points representing other torus knots::

  import numpy as np
  from pyknot2.spacecurves import Knot

  def torus_knot(p=2, q=3, num_points=100):
      # NOTE: p and q must be coprime!
      points = np.zeros((num_points, 3))
      ts = np.linspace(0, 2*np.pi, num_points)
      rs = np.cos(q * ts) + 2.
      points[:, 0] = rs * np.cos(p * ts)
      points[:, 1] = rs * np.sin(p * ts)
      points[:, 2] = -1 * np.sin(q * ts)

      return points * 5
   
   k = Knot(torus_knot(4, 7))
   k.plot()


.. image:: p7_q4__torus_knot.png
   :align: center
   :scale: 50%
   :alt: A p=7 and q=4 knot produced by the above code

'''

from pyknot2.spacecurves.spacecurve import SpaceCurve
from pyknot2.spacecurves.knot import Knot
from pyknot2.spacecurves.link import Link
from pyknot2.spacecurves.openknot import OpenKnot
from pyknot2.spacecurves.periodiccell import Cell

__all__ = ('SpaceCurve', 'Knot', 'Link', 'OpenKnot', 'Cell', )

