'''
Knot
====

Class for dealing with knots; a single space-curve, which may be
topologically trivial.
'''

import numpy as n
import sys
from scipy.interpolate import interp1d

import chelpers
from .geometry import arclength, radius_of_gyration

from ..visualise import plot_line, plot_projection
from ..io import to_json_file, from_json_file
from ..utils import vprint, mag, get_rotation_matrix

__all__ = ('Knot', 'Link')


class Knot(object):
    '''
    Class for holding the vertices of a single line, providing helper
    methods for convenient manipulation and analysis.

    A :class:`Knot` just represents a single space curve, it may be
    topologically trivial!

    This class deliberately combines methods to do many different kinds
    of measurements or manipulations. Some of these are externally
    available through other modules in pyknot2 - if so, this is usually
    indicated in the method docstrings.

    :param array-like points: the 3d points (vertices) of a piecewise
                              linear curve representation
    :param bool verbose: indicates whether the Knot should print
                         information during processing
    '''

    def __init__(self, points, verbose=True):
        if isinstance(points, Knot):
            points = points.points.copy()
        self._points = n.zeros((0, 3))
        self._crossings = None  # Will store a list of crossings if
                                # self.crossings() has been called
        self.points = n.array(points).astype(n.float)
        self.verbose = verbose

        self._cached_writhe_and_crossing_numbers = None
        self._gauss_code = None

        self._recent_octree = None 

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

    def _unwrap_periodicity(self, shape):
        '''Walks along the points of self, assuming that a periodic boundary on a

        lattice bounded by :param:`shape` has been crossed whenever one
        point is too far from the previous one. When this occurs,
        subtracts the lattice vector in this direction.

        :param array-like shape: The x, y, z distances of the periodic boundary.
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
    def from_periodic_line(cls, line, shape, perturb=True):
        '''Returns a :class:`Knot` instance in which the line has been
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
        knot = cls(line)
        knot._unwrap_periodicity(shape)
        if perturb:
            knot.translate(n.array([0.00123, 0.00231, 0.00321]))
            knot.rotate()
        return knot

    @classmethod
    def from_lattice_data(cls, line):
        '''Returns a :class:`Knot` instance in which the line has been
        slightly translated and rotated, in order to (practically) ensure
        no self intersections in closure or coincident points in
        projection.

        :rtype: :class:`Knot`
        '''
        knot = cls(line)
        knot.translate(n.array([0.00123, 0.00231, 0.00321]))
        knot.rotate(n.random.random(3) * 0.012)
        return knot

    def translate(self, vector):
        '''Translates all the points of self by the given vector.

        :param array-like vector: The x, y, z translation distances
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
        sin = n.sin
        cos = n.cos
        rot_mat = get_rotation_matrix(angles)
        self._apply_matrix(rot_mat)

    def _apply_matrix(self, mat):
        '''
        Applies the given matrix to all of self.points.
        '''
        self.points = n.apply_along_axis(mat.dot, 1, self.points)

    def raw_crossings(self, mode='use_max_jump', include_closure=True,
                      recalculate=False):
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
            One of ``'count_every_jump'`` and ``'use_max_jump'``. In the former
            case,
            walking along the line uses information about the length of
            every step. In the latter, it guesses that all steps have the
            same length as the maximum step length. The optimal choice
            depends on the data, but is usually ``'use_max_jump'``, which
            is the default.
        include_closure : bool, optional
            Whether to include crossings with the
            line joining the start and end points. Defaults to True.
        recalculate : bool, optional
            Whether to force a recalculation of the crossing positions.
            Defaults to False.

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

        self._vprint('Finding crossings')
        
        points = self.points
        segment_lengths = n.roll(points[:, :2], -1, axis=0) - points[:, :2]
        segment_lengths = n.sqrt(n.sum(segment_lengths * segment_lengths,
                                       axis=1))
        if include_closure:
            max_segment_length = n.max(segment_lengths)
        else:
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

            crossings.extend(chelpers.find_crossings(
                v0, dv, s, segment_lengths[compnum:],
                vnum, compnum,
                max_segment_length,
                jump_mode
                ))

        if include_closure:
            v0 = points[-1]
            dv = points[0] - v0
            s = points[1:-1]
            vnum = len(points) - 1
            compnum = 1
            crossings.extend(chelpers.find_crossings(
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
            Whether to recalculate the writhe.
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

    def gauss_code(self, **kwargs):
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
        if self._gauss_code is not None:
            return self._gauss_code
        crossings = self.raw_crossings(**kwargs)
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

    def plot(self, mode='mayavi', clf=True, **kwargs):
        '''
        Plots the line. See :func:`pyknot2.visualise.plot_line` for
        full documentation.
        '''
        plot_line(self.points, mode=mode, clf=clf, **kwargs)

    def plot_projection(self, with_crossings=True, mark_start=False,
                        fig_ax=None):
        points = self.points
        crossings = None
        plot_crossings = []
        if with_crossings:
            crossings = self.raw_crossings()
            plot_crossings = []
            for crossing in crossings:
                x, y, over, orientation = crossing
                xint = int(x)
                yint = int(y)
                r = points[xint]
                dr = points[(xint+1) % len(points)] - r
                plot_crossings.append(r + (x-xint) * dr)
        fig, ax = plot_projection(points,
                                  crossings=n.array(plot_crossings),
                                  mark_start=mark_start,
                                  fig_ax=fig_ax)
        return fig, ax

    def __str__(self):
        if self._crossings is not None:
            return '<Knot with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<Knot with {} points>'.format(len(self.points))

    def __repr__(self):
        return str(self)

    def to_json(self, filen):
        '''
        Writes the knot points to the given filename, in a json format
        that can be read later by :meth:`Knot.from_json`. Uses
        :func:`pyknot2.io.to_json_file` internally.
        '''
        to_json_file(self.points, filen)

    @classmethod
    def from_json(cls, filen):
        '''
        Loads knot points from the given filename, assuming json format,
        and returns a :class:`Knot` with those points.
        ''' 

        points = from_json_file(filen)
        return cls(points)

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
        Returns a new :class:`Knot` where new points have been selected
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

        return Knot(new_points)

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
            last_lower_point = points[last_lower_index]
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



class Link(object):
    '''
    Class for holding the vertices of multiple lines, providing helper
    methods for convenient manipulation and analysis.

    The data is stored
    internally as multiple :class:`Knots`.

    Parameters
    ----------
    lines : list of nx3 array-like or Knots
        List with the points of each line.
    verbose : bool
        Whether to print information during processing. Defaults
        to True.
    '''

    def __init__(self, lines, verbose=True):
        self._lines = []
        self.verbose = verbose

        lines = [Knot(line) for line in lines]
        self.lines = lines

        self._crossings = None

        self._recent_octree = None

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        self._lines = lines

    def _reset_cache(self):
        self._crossings = None

    @classmethod
    def from_periodic_lines(cls, lines, shape, perturb=True):
        '''Returns a :class:`Link` instance in which the lines have
        been unwrapped through the periodic boundaries.

        Parameters
        ----------
        line : list
            A list of the Nx3 vectors of points in the lines
        shape : array-like
            The x, y, z distances of the periodic boundary
        perturb : bool
            If True, translates and rotates the knot to avoid any lattice
            problems.
        '''
        lines = [Knot.from_periodic_line(line, shape, False)
                 for line in lines]
        link = cls(lines)
        if perturb:
            link.translate(n.array([0.00123, 0.00231, 0.00321]))
            link.rotate()
        return link

    def raw_crossings(self, mode='use_max_jump', only_with_other_lines=True,
                      include_closures=True,
                      recalculate=False):
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
            One of ``'count_every_jump'`` and ``'use_max_jump'``. In the former
            case,
            walking along the line uses information about the length of
            every step. In the latter, it guesses that all steps have the
            same length as the maximum step length. The optimal choice
            depends on the data, but is usually ``'use_max_jump'``, which
            is the default.
        only_with_other_lines : bool
            If True, ignores self-crossings (i.e. the knot type of the loops)
            and returns only a list of crossings between the loops. Defaults
            to True
        include_closures : bool, optional
            Whether to include crossings with the
            lines joining their start and end points. Defaults to True.
        recalculate : bool, optional
            Whether to force a recalculation of the crossing positions.
            Defaults to False.

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

        lines = self.lines

        # Get length of each line
        line_lengths = [0.]
        line_lengths.extend([line.arclength() for line in lines])
        cumulative_lengths = n.cumsum(line_lengths)

        if only_with_other_lines:
            crossings = [[] for _ in lines]
        else:
            self._vprint('Calculating self-crossings for all {} '
                         'component lines'.format(len(lines)))
            crossings = [k.raw_crossings(
                mode=mode, include_closure=include_closures,
                recalculate=recalculate) for k in lines]
            for index, cum_length in enumerate(cumulative_lengths):
                crossings[index][:, :2] += cum_length
                crossings[index] = crossings[index].tolist()

        jump_mode = {'count_every_jump': 1, 'use_max_jump': 2,
                     'naive': 3}[mode]

        segment_lengths = [
            n.roll(line.points[:, :2], -1, axis=0) - line.points[:, :2] for
            line in lines]
        segment_lengths = [
            n.sqrt(n.sum(lengths * lengths, axis=1)) for
            lengths in segment_lengths]

        if include_closures:
            max_segment_length = n.max(n.hstack(segment_lengths))
        else:
            max_segment_length = n.max(n.hstack([
                lengths[:-1] for lengths in segment_lengths]))

        for line_index, line in enumerate(lines):
            for other_index, other_line in enumerate(lines[line_index+1:]):
                self._vprint(
                    '\rComparing line {} with {}'.format(line_index,
                                                         other_index))

                other_index += line_index + 1
                
                points = line.points
                comparison_points = other_line.points
                if include_closures:
                    comparison_points = n.vstack((comparison_points,
                                                  comparison_points[:1]))

                other_seg_lengths = segment_lengths[other_index]
                # other_seg_lengths is already corrected to include
                # closures if necessary
                
                first_line_range = range(len(points))
                if not include_closures:
                    first_line_range = first_line_range[:-1]

                for i in first_line_range:
                    if i % 100 == 0:
                        self._vprint(
                            '\ri = {} / {}'.format(
                                i, len(comparison_points)), False)
                    v0 = points[i]
                    dv = points[(i + 1) % len(points)] - v0

                    vnum = i
                    compnum = 0  # start at beginning of other line
                    
                    new_crossings = chelpers.find_crossings(
                        v0, dv, comparison_points, other_seg_lengths,
                        vnum, compnum,
                        max_segment_length,
                        jump_mode)

                    if not len(new_crossings):
                        continue
                    first_crossings = n.array(new_crossings[::2])
                    first_crossings[:, 0] += cumulative_lengths[line_index]
                    first_crossings[:, 1] += cumulative_lengths[other_index]
                    sec_crossings = n.array(new_crossings[1::2])
                    sec_crossings[:, 0] += cumulative_lengths[other_index]
                    sec_crossings[:, 1] += cumulative_lengths[line_index]

                    crossings[line_index].extend(first_crossings.tolist())
                    crossings[other_index].extend(sec_crossings.tolist())

        self._vprint('\n{} crossings found\n'.format(
            [len(cs) / 2 for cs in crossings]))
        [cs.sort(key=lambda s: s[0]) for cs in crossings]
        crossings = [n.array(cs) for cs in crossings]
        self._crossings = crossings

        return crossings
                    


        

        return crossings
        
        
    def translate(self, vector):
        '''Translate all points in all lines of self.

        Parameters
        ----------
        vector : array-like
            The x, y, z translation distances
        '''
        for line in self.lines:
            line.translate(vector)
        
    def rotate(self, angles=None):
        '''
        Rotates all the points of each line of self by the given angle
        in each axis.

        Parameters
        ----------
        angles : array-like
            Rotation angles about x, y and z axes. If None, random angles
            are used. Defaults to None.
        '''
        if angles is None:
            angles = n.random.random(3)
        for line in self.lines:
            line.rotate(angles)
        self._reset_cache()

    def plot(self, mode='mayavi', clf=True, **kwargs):
        '''
        Plots all the lines. See :func:`pyknot2.visualise.plot_line` for
        full documentation.
        '''
        lines = self.lines
        lines[0].plot(mode=mode, clf=clf, **kwargs)
        for line in lines[1:]:
            line.plot(mode=mode, clf=False, **kwargs)

    def octree_simplify(self, runs=1, plot=False, rotate=True,
                        obey_knotting=False, **kwargs):
        '''
        Simplifies the curves via the octree reduction of
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
            False - knotting of individual components will be ignored!
            This is *much* faster than the alternative.

        kwargs are passed to the :class:`pyknot2.simplify.octree.OctreeCell`
        constructor.
        '''
        from ..simplify.octree import OctreeCell, remove_nearby_points
        for line in self.lines:
            line.points = remove_nearby_points(line.points)
        for i in range(runs):
            if n.sum([len(knot.points) for knot in self.lines]) > 30:
                vprint('\rRun {} of {}, {} points remain'.format(
                    i, runs, len(self)), False, self.verbose)

            if rotate:
                rot_mat = get_rotation_matrix(n.random.random(3))
                for line in self.lines:
                    line._apply_matrix(rot_mat)

            oc = OctreeCell.from_lines([line.points for line in self.lines],
                                       **kwargs)
            oc.simplify(obey_knotting)
            self._recent_octree = oc
            self.lines = [Knot(line) for line in oc.get_lines()]

            if rotate:
                for line in self.lines:
                    line._apply_matrix(rot_mat.T)

            if plot:
                self.plot()

        vprint('\nReduced to {} points'.format(len(self)))

    def __len__(self):
        return n.sum(map(len, self.lines))

    def arclength(self, include_closures=True):
        '''
        Returns the sum of arclengths of the lines.

        Parameters
        ----------
        include_closures : bool
            Whether to include the distance between the final and
            first points of each line. Defaults to True.
        '''
        return n.sum(k.arclength(include_closures) for k in self.lines)

    def _vprint(self, s, newline=True):
        '''Prints s, with optional newline. Intended for internal use
        in displaying progress.'''
        if self.verbose:
            vprint(s, newline)
        

        

