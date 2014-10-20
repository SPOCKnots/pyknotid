'''
Link
====

Class for dealing with links; multiple space-curves, which may not
necessarily actually be linked with one another.
'''

import numpy as n

import chelpers

from .knot import Knot
from ..visualise import plot_line, plot_projection
from ..io import to_json_file, from_json_file
from ..utils import vprint, mag, get_rotation_matrix


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

        if isinstance(lines, Link):
            self.lines = [Knot(line) for line in lines.lines]
        else:
            lines = [Knot(line) for line in lines]
            self.lines = lines

        self._crossings = None
        self._gauss_code = {}

        self._recent_octree = None

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        self._lines = lines
        self._reset_cache()

    def _reset_cache(self):
        self._crossings = None
        self._gauss_code = {}

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

        if (not recalculate and self._crossings is not None and
            self._crossings[0] == only_with_other_lines):
            return self._crossings[1]

        lines = self.lines

        # Get length of each line
        line_lengths = [0.]
        line_lengths.extend([line.arclength() for line in lines])
        cumulative_lengths = n.cumsum(line_lengths)[:-1]

        if only_with_other_lines:
            crossings = [[] for _ in lines]
        else:
            self._vprint('Calculating self-crossings for all {} '
                         'component lines'.format(len(lines)))
            crossings = [k.raw_crossings(
                mode=mode, include_closure=include_closures,
                recalculate=recalculate) for k in lines]
            for index, cum_length in enumerate(cumulative_lengths):
                if len(crossings[index]):
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
                    if i % 1000 == 0:
                        self._vprint(
                            '\ri = {} / {}'.format(
                                i, len(first_line_range)), False)
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
        self._crossings = (only_with_other_lines, crossings)

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

    def plot_projection(self, with_crossings=True, mark_start=False,
                        include_self_crossings=False):
        all_points = [line.points for line in self.lines]
        lengths = [k.arclength() for k in self.lines]
        cum_lengths = n.hstack([[0], n.cumsum(lengths)])

        crossings = None
        plot_crossings = []
        if with_crossings:
            all_crossings = self.raw_crossings(
                only_with_other_lines=(not include_self_crossings))
            plot_crossings = []
            for index, crossings in enumerate(all_crossings):
                remove_length = cum_lengths[index]
                points = all_points[index]
                for crossing in crossings:
                    x, y, over, orientation = crossing

                    # Work out which line the crossing is on
                    # next_x_start = n.argmax(cum_lengths > x)
                    # x_line = next_x_start - 1
                    # x -= cum_lengths[x_line]
                    x -= remove_length

                    xint = int(x)
                    r = points[xint]
                    dr = points[(xint+1) % len(points)] - r
                    plot_crossings.append(r + (x-xint) * dr)
        fig, ax = plot_projection(all_points[0],
                                  crossings=n.array(plot_crossings),
                                  mark_start=mark_start)
        for line in all_points[1:]:
            plot_projection(line,
                            crossings=n.array(plot_crossings),
                            mark_start=mark_start,
                            fig_ax=(fig, ax))

        ax.autoscale(tight=False)
        return fig, ax

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
        include_self_linking = not kwargs.get('only_with_other_lines', True)
        if ('recalculate' not in kwargs and
            include_self_linking in self._gauss_code):
            return self._gauss_code[include_self_linking]
        crossings = self.raw_crossings(**kwargs)
        gc = GaussCode(crossings)
        self._gauss_code[include_self_linking] = gc
        return gc

    def linking_number(self):
        '''
        Returns the linking number of the lines in the Link, the
        sum of signed crossings between them, ignoring crossings of
        a line with itself.
        '''
        crossings = self.raw_crossings(only_with_other_lines=True)
        number = 0
        for line in crossings:
            if len(line):
                number += n.sum(line[:, 3])
        return int(n.abs(number / 2))
