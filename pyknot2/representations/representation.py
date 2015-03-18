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

    def vassiliev_degree_3(self, simplify=True, try_cython=True):
        '''Returns the Vassiliev invariant of degree 3 for the Knot.

        Parameters
        ==========
        simplify : bool
            If True, simplifies the Gauss code of self before
            calculating the invariant. Defaults to True, but
            will work fine if you set it to False (and might even
            be faster).
        try_cython : bool
            Whether to try and use an optimised cython version of the
            routine (takes about 1/3 of the time for complex
            representations).  Defaults to True, but the python
            fallback will be *slower* than setting it to False if the
            cython function is not available.
        **kwargs :
            These are passed directly to :meth:`gauss_code`.

        '''
        from ..invariants import vassiliev_degree_3
        if simplify:
            self.simplify(verbose=self.verbose)
        return vassiliev_degree_3(self, try_cython=try_cython)

    def hyperbolic_volume(self):
        '''
        Returns the hyperbolic volume at the given point, via
        :meth:`pyknot2.representations.PlanarDiagram.as_spherogram`.
        '''
        from ..invariants import hyperbolic_volume
        return hyperbolic_volume(self.planar_diagram())

    def exterior_manifold(self):
        '''
        The knot complement manifold of self as a SnapPy class
        giving access to all of SnapPy's tools.

        This method requires that Spherogram, and possibly SnapPy,
        are installed.
        '''
        return self.planar_diagram().as_spherogram().exterior()

    def identify(self, determinant=True, alexander=False, roots=(2, 3, 4),
                 min_crossings=True):
        '''
        Provides a simple interface to
        :func:`pyknot2.catalogue.identify.from_invariants`, by passing
        the given invariants. This does *not* support all invariants
        available, or more sophisticated identification methods,
        so don't be afraid to use the catalogue functions directly.

        Parameters
        ----------
        determinant : bool
            If True, uses the knot determinant in the identification.
            Defaults to True.
        alexander : bool
            If True-like, uses the full alexander polynomial in the
            identification. If the input is a dictionary of kwargs,
            these are passed straight to self.alexander_polynomial.
        roots : iterable
            A list of roots of unity at which to evaluate. Defaults
            to (2, 3, 4), the first of which is redundant with the
            determinant. Note that higher roots can be calculated, but
            aren't available in the database.
        min_crossings : bool
            If True, the output is restricted to knots with fewer crossings
            than the current projection of this one. Defaults to True. The
            only reason to turn this off is to see what other knots have
            the same invariants, it is never not useful for direct
            identification.
        '''
        if not roots:
            roots = []
        roots = set(roots)
        if determinant:
            roots.add(2)

        identify_kwargs = {}
        for root in roots:
            identify_kwargs[
                'alex_imag_{}'.format(root)] = self.alexander_at_root(root)

        if alexander:
            if not isinstance(alexander, dict):
                import sympy as sym
                alexander = {'variable': sym.var('t')}
            poly = self.alexander_polynomial(**alexander)
            identify_kwargs['alexander'] = poly

        from pyknot2.catalogue.identify import from_invariants
        from pyknot2.catalogue.database import Knot as DBKnot
        if min_crossings and len(self.gauss_code()) < 16:
            identify_kwargs['other'] = (
                DBKnot.min_crossings <= len(self.gauss_code()), )

        return from_invariants(**identify_kwargs)

    def self_linking(self):
        '''Returns the self linking number J(K) of the Gauss code, an
        invariant of virtual knots. See Kauffman 2004 for more information.

        Returns
        -------
        slink_counter : int
            The self linking number of the open curve
        '''
        from ..invariants import self_linking
        return self_linking(self)


    def is_virtual(self):
        '''
        Takes an open curve and checks (for the default projection) if its 
        Gauss code corresponds to a virtual knot or not. Returns a Boolean of 
        this information.

        Returns
        -------
        virtual : bool
            True if the Gauss code corresponds to a virtual knot. False
            otherwise.
        '''
        gauss_code = self._gauss_code[0][:, 0]
        l = len(gauss_code)
        total_crossings = l / 2
        crossing_counter = 1
        virtual = False
        
        while (not virtual and crossing_counter < total_crossings + 1): 
            occurences = n.where(gauss_code == crossing_counter)[0]
            first_occurence = occurences[0]
            second_occurence = occurences[1]
            crossing_difference = second_occurence - first_occurence        
                  
            if(crossing_difference % 2 == 0):
                virtual = True
                  
            crossing_counter += 1
                
        return virtual   
