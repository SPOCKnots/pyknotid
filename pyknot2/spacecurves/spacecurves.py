'''
Classes for dealing with knots (a single line, which may be
topologically trivial) and links (multiple Knot classes).
'''

import numpy as n
import sys

import chelpers

from ..visualise import plot_line, plot_projection


class Knot(object):
    '''
    Class for holding the vertices of a single line, providing helper
    methods for convenient manipulation and analysis.

    A Knot is just a single space curve, it may be topologically
    trivial!

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

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points
        self._crossings = None

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
    def from_periodic_line(cls, line, shape):
        '''Returns a :class:`Knot` instance in which the line has been
        unwrapped through
        the periodic boundaries.

        :param array-like line: The Nx3 vector of points in the line
        :param array-like shape: The x, y, z distances of the periodic boundary
        '''
        knot = cls(line)
        knot._unwrap_periodicity(shape)
        return knot

    @classmethod
    def from_lattice_data(cls, line):
        '''returns a :class:`Knot` instance in which the line has been
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
        rotmat = n.array([
                [cos(theta)*cos(psi),
                 -1*cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi),
                 sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi)],
                [cos(theta)*sin(psi),
                 cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi),
                 -1*sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi)],
                [-1*sin(theta), sin(phi)*cos(theta),
                 cos(phi)*cos(theta)]])
        self._apply_matrix(rotmat)

    def _apply_matrix(self, mat):
        '''
        Applies the given matrix to all of self.points.
        '''
        self.points = n.apply_along_axis(mat.dot, 1, self.points)

    def crossings(self, mode='count_every_jump', include_closure=True,
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
        mode : str
            One of 'count_every_jump' and 'use_max_jump'. In the former
            case,
            walking along the line uses information about the length of
            every step. In the latter, it guesses that all steps have the
            same length as the maximum step length. The optimal choice
            depends on the data.
        include_closure : bool
            Whether to include crossings with the
            line joining the start and end points. Defaults to True.
        recalculate : bool
            Whether to force a recalculation of the crossing positions.
            Defaults to False.

        :rtype: :class:`pyknot.representations.gausscode.GaussCode`
                ^^ TODO
        '''

        if not recalculate and self._crossings is not None:
            return self._crossings

        self._vprint('Finding crossings')
        
        points = self.points
        segment_lengths = n.roll(points, -1, axis=0) - points
        segment_lengths = n.sqrt(n.sum(segment_lengths * segment_lengths,
                                       axis=1))
        max_segment_length = n.max(segment_lengths)
        numtries = len(points) - 3
        
        crossings = []

        jump_mode = {'count_every_jump': 1, 'use_max_jump': 2}[mode]
        
        for i in range(len(points)-3):
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
                max_segment_length))

        self._vprint('\n{} crossings found\n'.format(len(crossings)))
        crossings.sort(key=lambda s: s[0])
        crossings = n.array(crossings)
        self._crossings = crossings

        return crossings

    def plot(self, mode='mayavi', clf=True, **kwargs):
        '''
        Plots the line. See :func:`pyknot2.visualise.plot_line` for
        full documentation.
        '''
        plot_line(self.points, mode=mode, clf=clf, **kwargs)

    def plot_projection(self, with_crossings=True):
        points = self.points
        crossings = None
        plot_crossings = []
        if with_crossings:
            crossings = self.crossings()
            plot_crossings = []
            for crossing in crossings:
                x, y, over, orientation = crossing
                xint = int(x)
                yint = int(y)
                r = points[xint]
                dr = points[(xint+1) % len(points)] - r
                plot_crossings.append(r + (x-xint) * dr)
        plot_projection(points, crossings=n.array(plot_crossings))

    def __str__(self):
        if self._crossings is not None:
            return '<Knot with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<Knot with {} points>'.format(len(self.points))

    def __repr__(self):
        return str(self)


class Link(object):
    '''
    Class for holding the vertices of multiple lines, providing helper
    methods for convenient manipulation and analysis.

    The data is stored
    internally as multiple Knots.

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

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        self._lines = lines
        
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

    def plot(self, mode='mayavi', clf=True, **kwargs):
        '''
        Plots all the lines. See :func:`pyknot2.visualise.plot_line` for
        full documentation.
        '''
        lines = self.lines
        lines[0].plot(mode=mode, clf=clf, **kwargs)
        for line in lines[1:]:
            line.plot(mode=mode, clf=False, **kwargs)
        

def lineprint(x):
    sys.stdout.write('\r' + x)
    sys.stdout.flush()
    return 1
