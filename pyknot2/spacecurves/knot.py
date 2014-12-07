'''
Knot
====

Class for dealing with knots; a single space-curve, which may be
topologically trivial.
'''

import numpy as n
import sys
from scipy.interpolate import interp1d

from pyknot2.spacecurves.spacecurve import SpaceCurve

# We must be careful only to import modules that do not depend on this one, to
# prevent import loops
from pyknot2.io import to_json_file, from_json_file, from_csv

__all__ = ('Knot', )


class Knot(SpaceCurve):
    '''
    Class for holding the vertices of a single line, providing helper
    methods for convenient manipulation and analysis.

    A :class:`Knot` just represents a single space curve, it may be
    topologically trivial!

    This class deliberately combines methods to do many different kinds
    of measurements or manipulations. Some of these are externally
    available through other modules in pyknot2 - if so, this is usually
    indicated in the method docstrings.

    Parameters
    ----------
    points : array-like
        The 3d points (vertices) of a piecewise
        linear curve representation
    verbose : bool
        Indicates whether the Knot should print
        information during processing
    add_closure : bool
        If True, adds a final point to the knot near to the start point,
        so that it will appear visually to close when plotted.
    '''

    def copy(self):
        '''Returns another knot with the same points and verbosity
        as self. Other attributes (e.g. cached crossings) are not
        preserved.'''
        return Knot(self.points.copy(), verbose=self.verbose)

    def alexander_polynomial(self, variable=-1, quadrant='lr',
                             mode='python', **kwargs):
        '''
        Returns the Alexander polynomial at the given point,
        as calculated by :func:`pyknot2.invariants.alexander`.

        See :func:`pyknot2.invariants.alexander` for the meanings
        of the named arguments.
        '''
        from ..invariants import alexander
        gc = self.gauss_code(**kwargs)
        gc.simplify(verbose=self.verbose)
        return alexander(gc, variable=variable, quadrant=quadrant,
                         simplify=False, mode=mode)

    def alexander_at_root(self, root, round=True, **kwargs):
        '''
        Returns the Alexander polynomial at the given root of unity,
        i.e. evaluated at exp(2 pi I / root). 

        The result returned is the absolute value.

        Parameters
        ----------
        root : int
            The root of unity to use, i.e. evaluating at exp(2 pi I / root)
        round : bool
            If True and n in (1, 2, 3, 4), the result will be rounded
            to the nearest integer for convenience, and returned as an
            integer type.
        **kwargs :
            These are passed directly to :meth:`alexander_polynomial`.
        '''
        variable = n.exp(2 * n.pi * 1.j / root)
        value = self.alexander_polynomial(variable, **kwargs)
        value = n.abs(value)
        if round and root in (1, 2, 3, 4):
            value = int(n.round(value))
        return value

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

    def __str__(self):
        if self._crossings is not None:
            return '<Knot with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<Knot with {} points>'.format(len(self.points))

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
                alexander = {'variable': sym.var('t')}
            poly = self.alexander_polynomial(**alexander)
            identify_kwargs['alexander'] = poly

        from pyknot2.catalogue.identify import from_invariants
        from pyknot2.catalogue.database import Knot as DBKnot
        if min_crossings and len(self.gauss_code()) < 16:
            identify_kwargs['other'] = (
                DBKnot.min_crossings <= len(self.gauss_code()), )

        return from_invariants(**identify_kwargs)

    def slipknot_alexander(self, num_samples=0, **kwargs):
        '''
        Parameters
        ----------
        num_samples : int
            The number of indices to cut at. Defaults to 0, which
            means to sample at all indices.
        **kwargs :
            Keyword arguments, passed directly to
            :meth:`pyknot2.spacecurves.openknot.OpenKnot.alexander_fractions.
        '''
        points = self.points
        if num_samples == 0:
            num_samples = len(points)
        indices = n.linspace(0, len(points), num_samples).astype(n.int)

        from pyknot2.spacecurves.openknot import OpenKnot

        arr = n.ones((num_samples, num_samples))
        for index, points_index in enumerate(indices):
            self._vprint('\rindex = {} / {}'.format(index, len(indices)),
                         False)
            for other_index, other_points_index in enumerate(indices[(index+2):]):
                k = OpenKnot(points[points_index:other_points_index], verbose=False)
                if len(k.points) < 4:
                    alex = 1.
                else:
                    alex = k.alexander_fractions(**kwargs)[-1][0]
                arr[index, other_index] = alex

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(arr, interpolation='none')

        ax.plot(n.linspace(0, num_samples, 100) - 0.5,
                n.linspace(num_samples, 0, 100) - 0.5,
                color='black',
                linewidth=3)

        ax.set_xlim(-0.5, num_samples-0.5)
        ax.set_ylim(-0.4, num_samples-0.5)

        fig.show()

        return fig, ax
        
