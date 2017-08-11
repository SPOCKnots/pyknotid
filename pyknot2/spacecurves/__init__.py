'''
This module contains classes and functions for working with knots and
links as three-dimensional space curves, or calling functions
elsewhere in pyknot2 to perform topological analysis. Functionality
includes manipulating knots/links via translation, rotation and
scaling, plotting diagrams, finding crossings and identifying knots.

..
   .. image:: trefoil.png
      :scale: 50%
      :alt: A trefoil knot as seen in a pyknot2 visualisation
      :align: center

.. image:: random_walk_length_30.png
   :scale: 50%
   :alt: A closed random walks with 30 steps
   :align: center
'''

from pyknot2.spacecurves.spacecurve import SpaceCurve
from pyknot2.spacecurves.knot import Knot
from pyknot2.spacecurves.link import Link
from pyknot2.spacecurves.openknot import OpenKnot
from pyknot2.spacecurves.periodiccell import Cell

__all__ = ('SpaceCurve', 'Knot', 'Link', 'OpenKnot', 'Cell', )

