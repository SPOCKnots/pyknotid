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

'''

from pyknot2.representations.gausscode import GaussCode
from pyknot2.representations.planardiagram import PlanarDiagram
from pyknot2.representations.dtnotation import DTNotation
from pyknot2.representations.representation import Representation
__all__ = ['GaussCode', 'PlanarDiagram', 'DTNotation', 'Representation', ]

