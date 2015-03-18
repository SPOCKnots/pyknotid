'''
Representation
==============

An abstract representation of a Knot, providing methods for
the calculation of topological invariants.
'''

from pyknot2.representations.gausscode import GaussCode
import numpy as n


class Representation(GaussCode):
    '''
    An abstract representation of a knot or link. Internally
    this is just a Gauss code, but it exposes extra topological methods
    and may in future be refactored to work differently.
    '''

    def gauss_code(self):
        return GaussCode(self)
    
    def planar_diagram(self):
        from pyknot2.representations.planardiagram import PlanarDiagram
        return PlanarDiagram(self)

    def alexander_polynomial(self, variable=-1, quadrant='lr',
                             mode='python'):
        '''
        Returns the Alexander polynomial at the given point,
        as calculated by :func:`pyknot2.invariants.alexander`.

        See :func:`pyknot2.invariants.alexander` for the meanings
        of the named arguments.
        '''
        from ..invariants import alexander
        self.simplify()
        return alexander(self, variable=variable, quadrant=quadrant,
                         simplify=False, mode=mode)

    def alexander_at_root(self, root, round=True):
        '''
        Returns the Alexander polynomial at the given root of unity,
        i.e. evaluated at exp(2 pi I / root).

        The result returned is the absolute value.

        Parameters
        ----------
        root : int
            The root of unity to use, i.e. evaluating at exp(2 pi I / root).
            If this is iterable, this method returns a list of the results
            at every value of that iterable.
        round : bool
            If True and n in (1, 2, 3, 4), the result will be rounded
            to the nearest integer for convenience, and returned as an
            integer type.
        **kwargs :
            These are passed directly to :meth:`alexander_polynomial`.
        '''
        if hasattr(root, '__contains__'):
            return [self.alexander_at_root(r) for r in root]
        variable = n.exp(2 * n.pi * 1.j / root)
        value = self.alexander_polynomial(variable)
        value = n.abs(value)
        if round and root in (1, 2, 3, 4):
            value = int(n.round(value))
        return value

    def vassiliev_degree_2(self, simplify=True):
        '''
        Returns the Vassiliev invariant of degree 2 for the Knot.

        Parameters
        ==========
        simplify : bool
            If True, simplifies the Gauss code of self before
            calculating the invariant. Defaults to True, but
            will work fine if you set it to False (and might even
            be faster).
        **kwargs :
            These are passed directly to :meth:`gauss_code`.
        '''
        from ..invariants import vassiliev_degree_2
        if simplify:
            self.simplify()
        return vassiliev_degree_2(self)
