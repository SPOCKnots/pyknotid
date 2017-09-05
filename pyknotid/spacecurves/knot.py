'''
Knot
----

Class for dealing with curves as knots. :class:`Knot` provides many
methods for topological manipulation and calculations.

API documentation
~~~~~~~~~~~~~~~~~

'''

from __future__ import division

import numpy as n
import numpy as np

from pyknotid.spacecurves.spacecurve import SpaceCurve

__all__ = ('Knot', )


class Knot(SpaceCurve):
    '''
    Class for holding the vertices of a single line, providing helper
    methods for convenient manipulation and analysis.

    A :class:`Knot` just represents a single space curve, it may be
    topologically trivial!

    This class deliberately combines methods to do many different kinds
    of measurements or manipulations. Some of these are externally
    available through other modules in pyknotid - if so, this is usually
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

    @SpaceCurve.points.setter
    def points(self, points):
        super(Knot, self.__class__).points.fset(self, points)
        self._cached_isolated = None

    def tangents(self):
        return super(Knot, self).tangents(closed=True)

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
        as calculated by :func:`pyknotid.invariants.alexander`.

        See :func:`pyknotid.invariants.alexander` for the meanings
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
        :meth:`pyknotid.representations.PlanarDiagram.as_spherogram`.

        Returns
        -------
        volume : float
            A float representing the volume returned.
        accuracy : int
            The number of digits of precision. This is significant
            digits, e.g. 0.00021 with 1 digit precision = 2E-4.
        solution_type : str
            The solution type of the manifold. Normally one of:
            - 'contains degenerate tetrahedra' => may not be a valid result
            - 'all tetrahedra positively oriented' =>
              really probably hyperbolic
        '''
        from ..invariants import hyperbolic_volume
        m = self.exterior_manifold()
        v = m.volume()
        return (float(v), v.accuracy, m.solution_type())

    def exterior_manifold(self):
        '''
        The knot complement manifold of self as a SnapPy class
        giving access to all of SnapPy's tools.

        This method requires that Spherogram, and possibly SnapPy,
        are installed.
        '''
        link = self.planar_diagram().as_spherogram()
        link.simplify()  # necessary with snappy 2.5, which can't deal
                         # with extra Reidemeister moves sometimes
        return link.exterior()

    def __str__(self):
        if self._crossings is not None:
            return '<Knot with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<Knot with {} points>'.format(len(self.points))

    def identify(self, determinant=True, alexander=False, roots=(2, 3, 4),
                 min_crossings=True):
        '''
        Provides a simple interface to
        :func:`pyknotid.catalogue.identify.from_invariants`, by passing
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

        from pyknotid.catalogue.identify import from_invariants
        return from_invariants(**identify_kwargs)

    def planar_writhe_quantities(self, num_angles=100, **kwargs):
        '''Returns the second order writhes, and arnold 2St+J+ values, for a
        range of different projection directions.

        '''
        from pyknotid.spacecurves.rotation import (
            get_rotation_angles, rotate_to_top)

        angles = get_rotation_angles(num_angles)

        results = []

        print_dist = int(max(1, 3000. / len(self.points)))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))

            results.append((angs, k, k.planar_second_order_writhe(),
                            k.arnold_2St_2Jplus()))

            # ax.clear()
            # k.plot_projection(fig_ax=(fig, ax))
            # fig.set_size_inches((3, 3))
            # ax.set_title('Wr2 = {}, unknot Wr2 = {}'.format(results[-1][2],
            #                                                 results[-1][3]))
            # fig.tight_layout()
            # fig.savefig('planar_writhe_quantities-{:05d}.png'.format(i))

        return results


    def slipknot_alexander(self, num_samples=0, **kwargs):
        '''
        Parameters
        ----------
        num_samples : int
            The number of indices to cut at. Defaults to 0, which
            means to sample at all indices.
        **kwargs :
            Keyword arguments, passed directly to
            :meth:`pyknotid.spacecurves.openknot.OpenKnot.alexander_fractions.
        '''
        points = self.points
        if num_samples == 0:
            num_samples = len(points)
        indices = n.linspace(0, len(points), num_samples).astype(n.int)

        from pyknotid.spacecurves.openknot import OpenKnot

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

        from pyknotid.spacecurves import OpenKnot

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

        from pyknotid.representations import Representation
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

    def whitney_index(self):
        '''The degree of the Gauss map mapping a point on the curve to the
        direction of the positive tangent vector at this point.'''

        points = self.points
        tangents = self.tangents()

        directions = np.apply_along_axis(
            lambda arr: np.arctan2(arr[1], arr[0]), 1, tangents)

        directions[directions > np.pi] -= 2*np.pi
        directions[directions < -np.pi] += 2*np.pi

        jumps = np.roll(directions, -1) - directions


        index = np.sum(jumps > np.pi) - np.sum(jumps < -np.pi)

        return index

    def arnold_2St_2Jplus(self, **kwargs):
        from pyknotid.invariants import arnold_2St_2Jplus
        gc = self.gauss_code(**kwargs)
        # Do *not* simplify, as this is only a plane curve invariant
        return arnold_2St_2Jplus(gc)

    def arnold_2St_2Jminus(self, **kwargs):
        from pyknotid.invariants import arnold_2St_2Jminus
        gc = self.gauss_code(**kwargs)
        # Do *not* simplify, as this is only a plane curve invariant
        return arnold_2St_2Jminus(gc)

    def plot_secant_manifold(self, plot_extrema=True):
        import matplotlib.pyplot as plt
        # from colorsys import hls_to_rgb
        from hsluv import hsluv_to_rgb, hpluv_to_rgb
        from scipy.special import erfinv

        # import pyknotid.hsluvcolormap  # causes hsluv to be registered

        points = self.points

        colours = np.ones((len(points), len(points), 3))
        heights = np.zeros((len(points), len(points)))

        thetas = []
        phis = []

        for i1, point in enumerate(points):
            for i2, other_point in enumerate(points[i1+1:]):
                i2 += i1 + 1
                direction = other_point - point
                theta = np.arccos(direction[2] / mag(direction))
                phi = np.arctan2(direction[1], direction[0])

                # colours[i2, i1] = hls_to_rgb((phi + np.pi) / (2*np.pi), erfinv((theta / np.pi) * 2 - 1.) / 4.4 + 0.5, 1)
                colours[i2, i1] = hsluv_to_rgb((360 * (phi + np.pi) / (2*np.pi), 100, 65))# * (erfinv((theta / np.pi) * 2 - 1.) / 4.4 + 0.5)))

                colours[i1, i2] =  hsluv_to_rgb((360 * ((phi + 2*np.pi) % (2*np.pi)) / (2*np.pi), 100, 65))

                phis.append(phi)
                thetas.append(theta)

                heights[i2, i1] = theta
                heights[i1, i2] = np.pi - theta

        print(np.min(phis), np.max(phis), np.min(thetas), np.max(thetas))

        fig, ax = plt.subplots()

        im = ax.imshow(colours, origin='lower', zorder=-1)
        # ax.contour(heights, cmap='Greys', levels=np.linspace(-1, 1, 13))
        ax.contour(heights, cmap='Greys_r', levels=np.linspace(0, np.pi, 11),
                   zorder=0)

        crossings = self.raw_crossings()

        # Plot the lines between crossings
        unique_crossings = []
        crossings_done = set()
        for crossing in crossings:
            if crossing[0] in crossings_done:
                continue
            crossings_done.add(crossing[1])
            unique_crossings.append(crossing)

        for i, crossing in enumerate(unique_crossings):
            start1, end1, height1, sign1 = crossing
            for other_crossing in unique_crossings[i+1:]:
                start2, end2, height2, sign2 = other_crossing

                if not (start2 > start1 and end1 > start2 and end2 > end1):
                    continue

                colour = 'crimson' if sign1 * sign2 > 0 else 'lime'
                ax.plot([start1, start2], [end1, end2], color=colour, zorder=1)

        # Plot the crossing points
        for crossing in crossings:
            if crossing[1] > crossing[0]:
                colour = 'crimson' if crossing[3] > 0 else 'lime'
                edge_colour = 'white' if crossing[2] > 0 else 'black'
                ax.scatter([crossing[0]], [crossing[1]], color=colour, edgecolors=edge_colour,
                           s=30, zorder=2)

        # Plot extrema of the planar projection
        zs = points[:, 1]
        maxima_indices = np.argwhere((zs > np.roll(zs, 1)) & (zs > np.roll(zs, -1))).T[0]
        minima_indices = np.argwhere((zs < np.roll(zs, 1)) & (zs < np.roll(zs, -1))).T[0]
        print('maxima indices', maxima_indices)
        print('minima indices', minima_indices)
        for maximum in maxima_indices:
            ax.scatter([maximum], [maximum],  color='purple', edgecolors='pink', zorder=5)
        for minimum in minima_indices:
            ax.scatter([minimum], [minimum],  color='cyan', edgecolors='pink', zorder=5)

        xs = np.arange(len(points))
        ax.plot(xs, xs, linewidth=8, color='black', zorder=4)

        ax.set_xticks([])
        ax.set_yticks([])


        fig.tight_layout()
        
        # Plot the knot projection inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_ax = inset_axes(ax, width="45%", height="45%", loc=4)
        # inset_ax = fig.add_axes([0.6, 0.1, 0.4, 0.4])
        self.plot_projection(fig_ax=(fig, inset_ax))
        # inset_ax.set_axis_off()
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.patch.set_alpha(0.5)

        ax.set_xlim(0.5, len(points) - 0.5)
        ax.set_ylim(0.5, len(points) - 0.5)

        ax.set_ylabel('i2')
        ax.set_xlabel('i1')

        return fig, ax

    def plot_secant_crossings(self, radius=None, **kwargs):
        
        crossings = self.raw_crossings(**kwargs)

        if radius is None:
            radii = self.points - np.average(self.points, axis=0)
            radius = 2.5 * np.max(np.sqrt(np.sum(radii**2, axis=1)))

        unique_crossings = []
        crossings_done = set()
        for crossing in crossings:
            if crossing[0] in crossings_done:
                continue
            crossings_done.add(crossing[1])
            unique_crossings.append(crossing)

        points = self.points

        lines = []

        for i, crossing in enumerate(unique_crossings):
            start1, end1, height1, sign1 = crossing
            for other_crossing in unique_crossings[i+1:]:
                start2, end2, height2, sign2 = other_crossing

                if not (start2 > start1 and end1 > start2 and end2 > end1):
                    continue

                start_point1 = points[int(start1)]
                start_point1 += (start1 - int(start1)) * (
                    points[int(start1) + 1] - points[int(start1)])
                segment_point1s = points[int(start1) + 1:int(start2) + 1]
                end_point1 = points[int(start2)]
                end_point1 += (start2 - int(start2)) * (
                    points[int(start2) + 1] - points[int(start2)])

                segment1 = np.vstack(
                    [start_point1, segment_point1s, end_point1])

                #
                start_point2 = points[int(end1)]
                start_point2 += (end1 - int(end1)) * (
                    points[int(end1) + 1] - points[int(end1)])
                segment_point2s = points[int(end1) + 1:int(end2) + 1]
                end_point2 = points[int(end2)]
                end_point2 += (end2 - int(end2)) * (
                    points[int(end2) + 1] - points[int(end2)])

                segment2 = np.vstack(
                    [start_point2, segment_point2s, end_point2])

                directions = np.vstack([
                    start_point1 - segment2,
                    segment1 - end_point2])

                lines.append(directions)

        sphere_lines = []

        for i, line in enumerate(lines):
            radii = np.sqrt(np.sum(line**2, axis=1))
            thetas = np.arccos(line[:, 2] / radii)
            phis = np.arctan2(line[:, 1], line[:, 0])

            local_radius = radius - 0.02*i*radius*np.sin(2*np.sin(thetas))

            sphere_points = np.zeros(line.shape)
            sphere_points[:, 0] = radius * np.sin(thetas) * np.cos(phis)
            sphere_points[:, 1] = radius * np.sin(thetas) * np.sin(phis)
            sphere_points[:, 2] = radius * np.cos(thetas)

            # sphere_points += 2*(np.random.random(sphere_points.shape) - 0.5) * 0.005 * radius

            sphere_lines.append(sphere_points)

        self.plot(tube_radius=0.1, colour=(0.3, 0.3, 0.3, 1))

        # Plot the poles
        from vispy.geometry import create_sphere
        from vispy.scene import Mesh
        meshdata = create_sphere(radius=0.035 * radius)
        vertices = meshdata.get_vertices()
        faces = meshdata.get_faces()
        vertices[:, 2] += radius
        mesh1 = Mesh(vertices=vertices, faces=faces, vertex_colors=np.array([(0.3, 0.3, 0.3, 1) for v in vertices]))
        vertices = vertices.copy()
        vertices[:, 2] -= 2*radius
        mesh2 = Mesh(vertices=vertices, faces=faces, vertex_colors=np.array([(0.3, 0.3, 0.3, 1) for v in vertices]))

        import pyknotid.visualise as pvis
        pvis.vispy_canvas.view.add(mesh1)
        pvis.vispy_canvas.view.add(mesh2)

        from colorsys import hsv_to_rgb
        colours = [hsv_to_rgb(hue, 1, 0.8) for hue in np.linspace(0, 1, len(sphere_lines) + 1)][:-1]
        colours = np.array(colours)
        np.random.shuffle(colours)
        for line, colour in zip(sphere_lines, colours):
            from pyknotid.spacecurves.openknot import OpenKnot
            k = OpenKnot(line)
            k.plot(clf=False, tube_radius=0.15, colour=colour, zero_centroid=False)


def mag(v):
    return n.sqrt(v.dot(v))


def _isolate_open_knot(k, det, start, end):
    from pyknotid.spacecurves import OpenKnot
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
    


