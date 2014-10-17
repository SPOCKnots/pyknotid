'''
Space curves
============

This module contains classes and functions for working with
(knots and links as) three-dimensional space curves. Many of
these tools call functions elsewhere in pyknot2, bringing them
all together in a convenient interface.

This includes manipulating knots/links via translation, rotation
and scaling, plotting diagrams, finding crossings and identifying
knots.

'''

from .knot import Knot
from .link import Link
__all__ = ('Knot', 'Link', )

