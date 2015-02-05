'''
SpaceCurve
==========

A class for dealing with space curves (both open and closed),
with methods for measuring arclength, writhe etc. and plotting
the curve.
'''

import numpy as n
import sys
from scipy.interpolate import interp1d

try:
    from pyknot2.spacecurves import chelpers
except ImportError:
    from pyknot2.spacecurves import helpers as chelpers
from pyknot2.spacecurves import helpers as helpers
from pyknot2.spacecurves.geometry import arclength, radius_of_gyration
from pyknot2.spacecurves.smooth import smooth

# We must be careful only to import modules that do not depend on this one, to
# prevent import loops
from pyknot2.visualise import plot_line, plot_projection
from pyknot2.io import to_json_file, from_json_file, from_csv
from pyknot2.utils import (mag, get_rotation_matrix,
                           ensure_shape_tuple)


class SpaceCurve(object):
    '''
    Class for holding the vertices of a single line, providing helper
    methods for convenient manipulation and analysis.

    The methods of this class are largely geometrical (though this
    includes listing the crossings in projection and extracting a
    Gauss code etc.). For topological measurements, you should use
    a :class:`~pyknot2.spacecurves.knot.Knot`.

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
        Indicates whether the SpaceCurve should print
        information during processing
    add_closure : bool
        If True, adds a final point to the knot near to the start point,
        so that it will appear visually to close when plotted.
    zero_centroid : bool
        If True, shifts the coordinates of the points so that their centre
        of mass is at the origin.
    '''

    def __init__(self, points, verbose=True, add_closure=False,
                 zero_centroid=False):
        if isinstance(points, SpaceCurve):
            points = points.points.copy()
        self._points = n.zeros((0, 3))
        self._crossings = None  # Will store a list of crossings if
                                # self.crossings() has been called
        self.points = n.array(points).astype(n.float)
        self.verbose = verbose

        self._cached_writhe_and_crossing_numbers = None
        self._gauss_code = None

        self._recent_octree = None

        if add_closure:
            self._add_closure()
        if zero_centroid:
            self.zero_centroid()

    def copy(self):
        '''Returns another knot with the same points and verbosity
        as self. Other attributes (e.g. cached crossings) are not
        preserved.'''
        return SpaceCurve(self.points.copy(), verbose=self.verbose)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points
        # Changing the points may change the crossings, so
        # reset any cached crossing attributes.
        self._crossings = None
        self._cached_writhe_and_crossing_numbers = None
        self._gauss_code = None

    def _vprint(self, s, newline=True):
        '''Prints s, with optional newline. Intended for internal use
        in displaying progress.'''
        if self.verbose:
            sys.stdout.write(s)
            if newline:
                sys.stdout.write('\n')
            sys.stdout.flush()

    def _add_closure(self):
        closing_distance = mag(self.points[-1] - self.points[0])
        if closing_distance > 0.02:
            self.points = n.vstack((self.points, self.points[:1] -
                                    0.001*(self.points[0] - self.points[-1])))

    def _unwrap_periodicity(self, shape):
        '''
        Walks along the points of self, assuming that a periodic
        boundary on a
        lattice bounded by :param:`shape` has been crossed whenever one
        point is too far from the previous one. When this occurs,
        subtracts the lattice vector in this direction.

        Parameters
        ----------
        shape : array-like
            The x, y, z distances of the periodic boundary.
        '''

        dx, dy, dz = shape
        points = self.points
        for i in range(1, len(points)):
            prevLine = points[i-1]
            curLine = points[i]
            rest = points[i:]
            change = curLine - prevLine
            if -1.05*dx < change[0] < -0.95*dx:
                rest[:, 0] += dx
            if 1.05*dx > change[0] > 0.95*dx:
                rest[:, 0] -= dx
            if -1.05*dy < change[1] < -0.95*dy:
                rest[:, 1] += dy
            if 1.05*dy > change[1] > 0.95*dy:
                rest[:, 1] -= dy
            if -1.05*dz < change[2] < -0.95*dz:
                rest[:, 2] += dz
            if 1.05*dz > change[2] > 0.95*dz:
                rest[:, 2] -= dz

        self.points = points

    @classmethod
    def from_periodic_line(cls, line, shape, perturb=True, **kwargs):
        '''Returns a :class:`SpaceCurve` instance in which the line has been
        unwrapped through
        the periodic boundaries.

        Parameters
        ----------
        line : array-like
            The Nx3 vector of points in the line
        shape : array-like
            The x, y, z distances of the periodic boundary
        perturb : bool
            If True, translates and rotates the knot to avoid any lattice
            problems.
        '''
        shape = ensure_shape_tuple(shape)
        knot = cls(line, **kwargs)
        knot._unwrap_periodicity(shape)
        if perturb:
            knot.translate(n.array([0.00123, 0.00231, 0.00321]))
            knot.rotate((0.0002, 0.0001, 0.0001))
        return knot

    @classmethod
    def from_lattice_data(cls, line):
        '''Returns a :class:`SpaceCurve` instance in which the line has been
        slightly translated and rotated, in order to (practically) ensure
        no self intersections in closure or coincident points in
        projection.

        Parameters
        ----------
        line : array-like
            The list of points in the line. May be any type that SpaceCurve
            normally accepts.

        Returns
        -------
        :class:`SpaceCurve`
        '''
        knot = cls(line)
        knot.translate(n.array([0.00123, 0.00231, 0.00321]))
        knot.rotate(n.random.random(3) * 0.012)
        return knot

    def translate(self, vector):
        '''Translates all the points of self by the given vector.

        Parameters
        ----------
        vector : array-like
            The x, y, z translation distances
        '''
        self.points = self.points + n.array(vector)

    def zero_centroid(self):
        '''
        Translate such that the centroid (average position of vertices)
        is at (0, 0, 0).
        '''
        centroid = n.average(self.points, axis=0)
        self.translate(-1*centroid)

    def rotate(self, angles=None):
        '''
        Rotates all the points of self by the given angles in each axis.

        Parameters
        ----------
        angles : array-like
            The rotation angles about x, y and z. If None, random
            angles are used. Defaults to None.
        '''
        if angles is None:
            angles = n.random.random(3)
        phi, theta, psi = angles
        rot_mat = get_rotation_matrix(angles)
        self._apply_matrix(rot_mat)

    def _apply_matrix(self, mat):
        '''
        Applies the given matrix to all of self.points.
        '''
        self.points = n.apply_along_axis(mat.dot, 1, self.points)

    def raw_crossings(self, mode='use_max_jump', include_closure=True,
                      recalculate=False, try_cython=True):
        '''Returns the crossings in the diagram of the projection of the
        space curve into its z=0 plane.

        The crossings will be calculated the first time this function
        is called, then cached until an operation that would change
        the list (e.g. rotation, or changing ``self.points``).

        Multiple modes are available (see parameters) - you should be
        aware of this because different modes may be vastly slower or
        faster depending on the type of line.

        Parameters
        ----------
        mode : str, optional
            One of ``'count_every_jump'`` or ``'use_max_jump'`` or
            ``'naive'``. In the first case,
            walking along the line uses information about the length of
            every step. In the second, it guesses that all steps have the
            same length as the maximum step length. In the last, no
            optimisation is made and every crossing is checked.
            The optimal choice depends on the data but is usually
            ``'use_max_jump'``, which is the default.
        include_closure : bool, optional
            Whether to include crossings with the
            line joining the start and end points. Defaults to True.
        recalculate : bool, optional
            Whether to force a recalculation of the crossing positions.
            Defaults to False.
        try_cython : bool, optional
            Whether to force the use of the python (as opposed to cython)
            implementation of find_crossings. This will make no difference
            if the cython could not be loaded, in which case python is already
            used automatically. Defaults to True.

        Returns
        -------
        array-like
            The raw array of floats representing crossings, of the
            form [[line_index, other_index, +-1, +-1], ...], where the
            line_index and other_index are in arclength parameterised
            by integers for each vertex and linearly interpolated,
            and the +-1 represent over/under and clockwise/anticlockwise
            respectively.
        '''

        if not recalculate and self._crossings is not None:
            return self._crossings

        if try_cython:
            helpers_module = chelpers
        else:
            helpers_module = helpers

        self._vprint('Finding crossings')

        points = self.points
        segment_lengths = n.roll(points[:, :2], -1, axis=0) - points[:, :2]
        segment_lengths = n.sqrt(n.sum(segment_lengths * segment_lengths,
                                       axis=1))
        # if include_closure:
        #     max_segment_length = n.max(segment_lengths)
        # else:
        max_segment_length = n.max(segment_lengths[:-1])

        numtries = len(points) - 3

        crossings = []

        jump_mode = {'count_every_jump': 1, 'use_max_jump': 2,
                     'naive': 3}[mode]

        for i in range(len(points)-2):
            if self.verbose:
                if i % 100 == 0:
                    self._vprint('\ri = {} / {}'.format(i, numtries),
                                 False)
            v0 = points[i]
            dv = points[(i+1) % len(points)] - v0

            s = points[i+2:]
            vnum = i
            compnum = i+2

            crossings.extend(helpers_module.find_crossings(
                v0, dv, s, segment_lengths[compnum:],
                vnum, compnum,
                max_segment_length,
                jump_mode
                ))

        if include_closure:
            closure_segment_length = segment_lengths[-1]
            max_segment_length = n.max([closure_segment_length,
                                        max_segment_length])
            v0 = points[-1]
            dv = points[0] - v0
            s = points[1:-1]
            vnum = len(points) - 1
            compnum = 1
            crossings.extend(helpers_module.find_crossings(
                v0, dv, s, segment_lengths[compnum:],
                vnum, compnum,
                max_segment_length,
                jump_mode))

        self._vprint('\n{} crossings found\n'.format(len(crossings) / 2))
        crossings.sort(key=lambda s: s[0])
        crossings = n.array(crossings)
        self._crossings = crossings

        return crossings

    def planar_writhe(self, **kwargs):
        '''
        Returns the current planar writhe of the knot; the signed sum
        of crossings of the current projection.

        The 'true' writhe is the average of this quantity over all
        projection directions, and is available from the :meth:`writhe`
        method.

        Parameters
        ----------
        **kwargs :
            These are passed directly to :meth:`raw_crossings`.
        '''
        crossings = self.raw_crossings(**kwargs)
        return n.sum(crossings[:, 3]) / 2.

    def writhe(self, samples=10, recalculate=False, **kwargs):
        '''
        The (approximate) writhe of the space curve, obtained by averaging
        the planar writhe over the given number of directions.

        Parameters
        ----------
        samples : int
            The number of directions to average over. Defaults to 10.
        recalculate : bool
            Whether to recalculate the writhe even if a cached result
            is available. Defaults to False.
        **kwargs :
            These are passed directly to :meth:`raw_crossings`.
        '''
        crossing_number, writhe = self._writhe_and_crossing_numbers(
            samples, recalculate=recalculate, **kwargs)
        return writhe

    def average_crossing_number(self, samples=10, recalculate=False,
                                **kwargs):
        '''
        The (approximate) average crossing number of the space curve,
        obtained by averaging
        the planar writhe over the given number of directions.

        Parameters
        ----------
        samples : int
            The number of directions to average over.
        recalculate : bool
            Whether to recalculate the ACN.
        **kwargs :
            These are passed directly to :meth:`raw_crossings`.
        '''
        crossing_number, writhe = self._writhe_and_crossing_numbers(
            samples, recalculate=recalculate, **kwargs)
        return crossing_number

    def _writhe_and_crossing_numbers(self, samples=10, recalculate=False,
                                     **kwargs):
        '''
        Calculates and stores the writhe and average crossing number.
        Internal (not intended for external use).
        '''
        if (self._cached_writhe_and_crossing_numbers is not None and
            self._cached_writhe_and_crossing_numbers[0] == samples and
            not recalculate):
            return self._cached_writhe_and_crossing_numbers[1]

        from .complexity import writhe_and_crossing_number
        numbers = writhe_and_crossing_number(self.points, samples,
                                             verbose=self.verbose,
                                             **kwargs)
        self._cached_writhe_and_crossing_numbers = (samples, numbers)

        return numbers

    def gauss_code(self, recalculate=False, **kwargs):
        '''
        Returns a :class:`~pyknot2.representations.gausscode.GaussCode`
        instance representing the crossings of the knot.

        The GaussCode instance is cached internally. If you want to
        recalculate it (e.g. to get an unsimplified version if you
        have simplified it), you should pass `recalculate=True`.

        This method passes kwargs directly to :meth:`raw_crossings`,
        see the documentation of that function for all options.
        '''

        from ..representations.gausscode import GaussCode
        if not recalculate and self._gauss_code is not None:
            return self._gauss_code
        crossings = self.raw_crossings(recalculate=recalculate, **kwargs)
        gc = GaussCode(crossings)
        self._gauss_code = gc
        return gc

    def planar_diagram(self, **kwargs):
        '''
        Returns a
        :class:`~pyknot2.representations.planardiagram.PlanarDiagram`
        instance representing the crossings of the knot.

        This method passes kwargs directly to :meth:`raw_crossings`,
        see the documentation of that function for all options.
        '''
        from ..representations.planardiagram import PlanarDiagram
        crossings = self.raw_crossings(**kwargs)
        return PlanarDiagram(crossings)

    def gauss_diagram(self, simplify=False, **kwargs):
        '''
        Returns a
        :class:`~pyknot2.representations.gaussdiagram.GaussDiagram`
        instance representing the crossings of the knot.

        This method passes kwargs directly to :meth:`raw_crossings`,
        see the documentation of that function for all options.
        '''
        from ..representations.gaussdiagram import GaussDiagram
        gc = self.gauss_code(**kwargs)
        if simplify:
            gc.simplify()
        return GaussDiagram(gc)

    def plot(self, mode='auto', clf=True, **kwargs):
        '''
        Plots the line. See :func:`pyknot2.visualise.plot_line` for
        full documentation.
        '''
        plot_line(self.points, mode=mode, clf=clf, **kwargs)

    def plot_projection(self, with_crossings=True, mark_start=False,
                        fig_ax=None, show=True):
        points = self.points
        crossings = None
        plot_crossings = []
        if with_crossings:
            crossings = self.raw_crossings()
            plot_crossings = []
            for crossing in crossings:
                x, y, over, orientation = crossing
                xint = int(x)
                r = points[xint]
                dr = points[(xint+1) % len(points)] - r
                plot_crossings.append(r + (x-xint) * dr)
        fig, ax = plot_projection(points,
                                  crossings=n.array(plot_crossings),
                                  mark_start=mark_start,
                                  fig_ax=fig_ax,
                                  show=show)
        return fig, ax

    def __str__(self):
        if self._crossings is not None:
            return '<SpaceCurve with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<SpaceCurve with {} points>'.format(len(self.points))

    def __repr__(self):
        return str(self)

    def to_json(self, filen):
        '''
        Writes the knot points to the given filename, in a json format
        that can be read later by :meth:`SpaceCurve.from_json`. Uses
        :func:`pyknot2.io.to_json_file` internally.
        '''
        to_json_file(self.points, filen)

    @classmethod
    def from_json(cls, filen):
        '''
        Loads knot points from the given filename, assuming json format,
        and returns a :class:`SpaceCurve` with those points.
        '''

        points = from_json_file(filen)
        return cls(points)

    @classmethod
    def from_csv(cls, filen, **kwargs):
        '''
        Loads knot points from the given csv file, and returns a
        :class:`SpaceCurve` with those points.
        
        Arguments are passed straight to :func:`pyknot.io.from_csv`.
        '''
        return cls(from_csv(filen, **kwargs))

    def octree_simplify(self, runs=1, plot=False, rotate=True,
                        obey_knotting=True, **kwargs):
        '''
        Simplifies the curve via the octree reduction of
        :module:`pyknot2.simplify.octree`.

        Parameters
        ----------
        runs : int
            The number of times to run the octree simplification.
            Defaults to 1.
        plot : bool
            Whether to plot the curve after each run. Defaults to False.
        rotate : bool
            Whether to rotate the space curve before each run. Defaults
            to True as this can make things much faster.
        obey_knotting : bool
            Whether to not let the line pass through itself. Defaults to
            True as this is always what you want for a closed curve.
        **kwargs
            Any remaining kwargs are passed to the
            :class:`pyknot2.simplify.octree.OctreeCell`
            constructor.

        '''
        from pyknot2.simplify.octree import OctreeCell
        for i in range(runs):
            if len(self.points) > 30:
                self._vprint('\rRun {} of {}, {} points remain'.format(
                    i, runs, len(self.points)))

            if rotate:
                rot_mat = get_rotation_matrix(n.random.random(3))
                self._apply_matrix(rot_mat)

            oc = OctreeCell.from_single_line(self.points, **kwargs)
            oc.simplify(obey_knotting)
            self._recent_octree = oc
            self.points = oc.get_single_line()

            if rotate:
                self._apply_matrix(rot_mat.T)

            if plot:
                self.plot()

        self._vprint('\nReduced to {} points'.format(len(self.points)))

    def arclength(self, include_closure=True):
        '''
        Returns the arclength of the line, the sum of lengths of each
        piecewise linear segment.

        Parameters
        ----------
        include_closure : bool
            Whether to include the distance between the final and
            first points. Defaults to True.
        '''
        return arclength(self.points, include_closure)

    def radius_of_gyration(self):
        '''
        Returns the radius of gyration of the points of self,
        assuming each has equal weight and ignoring the connecting
        lines.
        '''
        return radius_of_gyration(self.points)

    def __len__(self):
        return len(self.points)

    def reparameterised(self, mode='arclength', num_points=None,
                        interpolation='linear'):
        '''
        Returns a new :class:`SpaceCurve` where new points have been selected
        by interpolating the current ones.

        .. warning:: This doesn't do what you expect! The new segments
                     will probably not all be separated by the right amount
                     in terms of the new parameterisation.

        Parameters
        ----------
        mode : str
            The function to reparameterise by. Defaults to 'arclength',
            which is currently the only option.
        num_points : int
            The number of points in the new parameterisation. Defaults
            to None, which means the same as the current number.
        interpolation : str
            The type of interpolation to use, passed directly to the
            ``kind`` option of ``scipy.interpolate.interp1d``. Defaults
            to 'linear', and other options have not been tested.
        '''
        indices = self._new_indices_by_arclength(num_points)

        interp_xs = interp1d(range(len(self.points)+1),
                             n.hstack((self.points[:, 0],
                                      self.points[:, 0][:1])))
        interp_ys = interp1d(range(len(self.points)+1),
                             n.hstack((self.points[:, 1],
                                      self.points[:, 1][:1])))
        interp_zs = interp1d(range(len(self.points)+1),
                             n.hstack((self.points[:, 2],
                                      self.points[:, 2][:1])))

        new_points = n.zeros((len(indices), 3), dtype=n.float)
        new_points[:, 0] = interp_xs(indices)
        new_points[:, 1] = interp_ys(indices)
        new_points[:, 2] = interp_zs(indices)

        return SpaceCurve(new_points)

    def _new_indices_by_arclength(self, number, step=None, gap=0):
        if number is None:
            number = len(self.points)
        total_arclength = self.arclength()
        if step is None:
            arclengths = n.linspace(0, total_arclength - gap, number+1)[:-1]
        else:
            arclengths = n.arange(0, total_arclength - gap, step)

        arclengths[0] += 0.000001

        points = self.points
        segment_arclengths = self.segment_arclengths()
        cumulative_arclength = n.hstack([[0.], n.cumsum(segment_arclengths)])
        total_arclength = self.arclength()

        indices = []

        for arclength in arclengths:
            first_greater_index = n.argmax(cumulative_arclength > arclength)
            last_lower_index = (first_greater_index - 1) % len(points)
            arclength_below = cumulative_arclength[last_lower_index]
            step_arclength = segment_arclengths[last_lower_index]
            step_fraction = (arclength - arclength_below) / step_arclength
            indices.append(last_lower_index + step_fraction)

        return indices

    def segment_arclengths(self):
        '''
        Returns an array of arclengths of every step in the line
        defined by self.points.
        '''
        return n.apply_along_axis(
            mag, 1, n.roll(self.points, -1, axis=0) - self.points)

    def smooth(self, repeats=1, periodic=True, window_len=10,
               window='hanning'):
        '''
        Smooths each of the x, y and z components of self.points by
        convolving with a window of the given type and size.

        Parameters
        ----------
        repeats : int
            Number of times to run the smoothing algorithm. Defaults to 1.
        periodic : bool
            If True, the convolution window wraps around the curve.
            Defaults to True.
        window_len : int
            Width of the convolution window. Defaults to 10.
            Passed to :func:`pyknot2.spacecurves.smooth.smooth`.
        window : string
            The type of convolution window. Defaults to 'hanning'.
            Passed to :func:`pyknot2.spacecurves.smooth.smooth`.
        '''
        points = self.points
        if periodic:
            points = n.vstack((points[-(window_len + 1):],
                               points,
                               points[:(window_len + 1)]))
        for i in range(repeats):
            points[:, 0] = smooth(points[:, 0], window_len, window)
            points[:, 1] = smooth(points[:, 1], window_len, window)
            points[:, 2] = smooth(points[:, 2], window_len, window)
        self.points = points[(window_len + 1):-(window_len + 1)]
