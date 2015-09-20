'''
Knot
====

Class for dealing with knots; a single space-curve, which may be
topologically trivial.
'''

from __future__ import division

import numpy as n

from pyknot2.spacecurves.spacecurve import SpaceCurve

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

    @property
    def points(self):
        return super(Knot, self).points

    @points.setter
    def points(self, points):
        super(Knot, self.__class__).points.fset(self, points)
        self._cached_isolated = None

    def copy(self):
        '''Returns another knot with the same points and verbosity
        as self. Other attributes (e.g. cached crossings) are not
        preserved.'''
        return Knot(self.points.copy(), verbose=self.verbose)

    def plot(self, **kwargs):
        super(Knot, self).plot(closed=True, **kwargs)

    def reconstructed_space_curve(self):
        r = self.representation()
        k = Knot(r.space_curve())
        k.zero_centroid()
        return k

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
        gc.simplify()
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
        value = self.alexander_polynomial(variable, **kwargs)
        value = n.abs(value)
        if round and root in (1, 2, 3, 4):
            value = int(n.round(value))
        return value

    def determinant(self):
        '''
        Returns the determinant of the knot. This is the Alexander
        polynomial evaluated at -1.
        '''
        return self.alexander_at_root(2)

    def vassiliev_degree_2(self, simplify=True, **kwargs):
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
        gc = self.gauss_code(**kwargs)
        if simplify:
            gc.simplify()
        return vassiliev_degree_2(gc)

    def vassiliev_degree_3(self, simplify=True, try_cython=True, **kwargs):
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
        gc = self.gauss_code(**kwargs)
        if simplify:
            gc.simplify()
        return vassiliev_degree_3(gc, try_cython=try_cython)

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
                import sympy as sym
                alexander = {'variable': sym.var('t')}
            poly = self.alexander_polynomial(**alexander)
            identify_kwargs['alexander'] = poly

        if min_crossings and len(self.gauss_code()) < 16:
            identify_kwargs['max_crossings'] = len(self.gauss_code())

        from pyknot2.catalogue.identify import from_invariants
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
            for other_index, other_points_index in enumerate(
                    indices[(index+2):]):
                k = OpenKnot(points[points_index:other_points_index],
                             verbose=False)
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

    def isolate_knot(self):
        '''Return indices of self.points within which the knot (if any)
        appears to lie, according to a simple closure algorithm.

        This method is experimental and may not provide very good results.
        '''
        if self._cached_isolated is not None:
            return self._cached_isolated
        determinant = self.determinant()

        from pyknot2.spacecurves import OpenKnot

        k1 = OpenKnot(self.points, verbose=False)
        start, end = _isolate_open_knot(k1, determinant, 0, len(k1))[1:]

        if end - start < 0.6 * len(k1):
            self._cached_isolated = (start, end)
            return start, end

        roll_dist = int(0.25*len(self.points))
        k2 = OpenKnot(n.roll(self.points, roll_dist, axis=0), verbose=False)
        start, end = _isolate_open_knot(k2, determinant, 0, len(k2))[1:]
        print('se', start, end)
        start -= roll_dist
        start %= len(self)
        end -= roll_dist
        end %= len(self)
        print('now', start, end)
        self._cached_isolated = (start, end)
        return start, end

    def plot_isolated(self, **kwargs):
        '''
        Plots the curve in red, except for the isolated local knot which
        is coloured blue. The local knot is found with self.isolate_knot,
        which may not be reliable or have good resolution.

        Parameters
        ==========
        **kwargs :
            kwargs are passed directly to :meth:`Knot.plot`.
        '''
        start, end = self.isolate_knot()
        if end < start:
            end, start = start, end
        mus = n.zeros(len(self.points))
        mus[start:end+1] = 0.4
        if end - start > 0.6*len(self) or end == start:
            mus = 0.4 - mus
        self.plot(mus=mus, **kwargs)

    def slip_triangle(self, func):

        from pyknot2.representations import Representation
        r = self.representation()

        length = len(r)

        cs = self.raw_crossings()

        results = {}

        invs = {}
        
        for i in range(length + 1):
            for j in range(length + 1):
                if i + j >= length:
                    continue

                new_r = Representation(r)

                points = self.points.copy()

                new_cs = cs.copy()

                new_start = 0
                new_end = -1
                for _ in range(i):
                    new_start = new_cs[0, 0] + 0.5*(new_cs[1, 0] - new_cs[0, 0])
                    new_cs = new_cs[new_cs[:, 1] != new_cs[0, 0]]
                    new_cs = new_cs[new_cs[:, 0] != new_cs[0, 0]]
                    new_r._remove_crossing(new_r._gauss_code[0][0, 0])
                for _ in range(j):
                    new_end = new_cs[-2, 0] + 0.5*(new_cs[-1, 0] - new_cs[-2, 0])
                    new_cs = new_cs[new_cs[:, 1] != new_cs[-1, 0]]
                    new_cs = new_cs[new_cs[:, 0] != new_cs[-1, 0]]
                    new_r._remove_crossing(new_r._gauss_code[0][-1, 0])

                new_points = points[int(n.ceil(new_start)):int(new_end)]
                new_start_remainder = new_start % 1
                new_end_remainder = new_end % 1

                new_points = n.vstack((points[int(new_start)] + new_start_remainder * (points[int(new_start) + 1] - points[int(new_start)]),
                                       new_points,
                                       points[int(new_end)] + new_end_remainder * (points[int(new_end) + 1] - points[int(new_end)])))
                # results[(i, j)] = points[int(new_start):int(new_end)]
                results[(i, j)] = new_points
                invs[(i, j)] = func(new_r)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        points = self.points
        mins = n.min(self.points, axis=0)
        maxs = n.max(self.points, axis=0)
        span = max((maxs - mins)[:2])
        size = span * 1.16

        xmin = mins[0] - 0.08*span
        xmax = maxs[0] + 0.08*span
        ymin = mins[1] - 0.08*span
        ymax = maxs[1] + 0.08*span

        fig = plt.figure()
        grid = GridSpec(length, length)
        for coords, points in results.items():
            print('coords are', coords)
            print length
            ax = plt.subplot(grid[length - 1 - coords[0], coords[1]])
            ax.plot(points[:, 0], points[:, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            inv = invs[coords]
            colour = 'red' if inv != 0 else 'green'
            ax.patch.set_facecolor(colour)
            ax.patch.set_alpha(0.1)

        fig.tight_layout()
        fig.show()
        return fig, ax
        

def _isolate_open_knot(k, det, start, end):
    from pyknot2.spacecurves import OpenKnot
    if len(k.points) < 10:  
        return (False, None, None)

    alexanders = k.alexander_fractions()
    main_det, main_frac = alexanders[-1]


    if main_det != det:
        return (False, None, None)

    half_len = len(k.points) // 2
    k1 = OpenKnot(k.points[:half_len], verbose=False)
    k1_knotted, k1_start, k1_end = _isolate_open_knot(
        k1, det, start, start + half_len - 1)
    if k1_knotted:
        return (True, k1_start, k1_end)

    k2 = OpenKnot(k.points[half_len:], verbose=False)
    k2_knotted, k2_start, k2_end = _isolate_open_knot(
        k2, det, start + half_len, end)
    if k2_knotted:
        return (True, k2_start, k2_end)

    quarter_len = len(k.points) // 4
    k3 = OpenKnot(k.points[quarter_len:(quarter_len + half_len)], verbose=False)
    k3_knotted, k3_start, k3_end = _isolate_open_knot(
        k3, det, start + quarter_len, start + quarter_len + half_len)
    if k3_knotted:
        return (True, k3_start, k3_end)

    k4 = OpenKnot(k.points[quarter_len:], verbose=False)
    k4_knotted, k4_start, k4_end = _isolate_open_knot(
        k4, det, start + quarter_len, end)
    if k4_knotted:
        return (True, k4_start, k4_end)

    k5 = OpenKnot(k.points[:-quarter_len], verbose=False)
    k5_knotted, k5_start, k5_end = _isolate_open_knot(
        k5, det, start, end - quarter_len)
    if k5_knotted:
        return (True, k5_start, k5_end)

    return (True, start, end)
    


