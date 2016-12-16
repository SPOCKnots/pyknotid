'''
Tools for working with a periodic cell of spacecurves.
'''
from pyknot2.utils import ensure_shape_tuple
from pyknot2.spacecurves import Knot, OpenKnot
import numpy as n
import numpy as np

class Cell(object):
    '''Class for holding the vertices of some number of lines with
    periodic boundary conditions.

    Parameters
    ==========
    lines : list
        Must be a list of Knots or ndarrays of vertices.
    shape : tuble or int
        The shape of the cell, in whatever units the lines use.
    periodic : bool
        Whether the cell is periodic. If True, lines are marked as
        'nth' or 'loop' in self.line_types. Defaults to True.
    '''

    def __init__(self, lines, shape, periodic=True, cram=False, downsample=None):
        self.shape = ensure_shape_tuple(shape)
        lines = [l for l in lines if len(l) > 1]
        if downsample is not None:
            lines = [l[::downsample] for l in lines]
        lines = list(map(_interpret_line, lines))
        if cram:
            lines = [_cram_into_cell(l, self.shape) for l in lines]
        self.lines = [_cut_line_at_jumps(l, self.shape) for l in lines]
        self.periodic = periodic

        self.line_types = None
        if periodic:
            self.line_types = [_test_periodicity(l, self.shape) for l in self.lines]

    @classmethod
    def from_qwer(cls, qwer, shape, **kwargs):
        '''
        Returns an instance of Cell having parsed the output line format
        of Sandy's simulations.
        '''
        q, w, e, r = qwer
        if len(w) > 0 and isinstance(w[0], tuple):
            w = [l[2] for l in w]
        if len(e) > 0 and isinstance(e[0], tuple):
            e = [l[2] for l in e]
        output = cls(q+w+e, shape, **kwargs)
        output.qwer = (q, w, e, r)
        return output

    def append(self, line, cram=False):
        line = _interpret_line(line)
        if cram:
            line = _cram_into_cell(line)
        self.lines.append(line)

    def plot(self, boundary=True, clf=True, tube_radius=1.0,
             length_colours=None,
             **kwargs):
        from pyknot2.visualise import plot_cell
        boundary = self.shape if boundary else None

        if length_colours is not None:
            if 'colours' in kwargs:
                raise ValueError(
                    'colours and length_colours cannot both be set')

            q, w, e, r = self.qwer
            from pyknot2.spacecurves.openknot import OpenKnot

            lengths = []
            for line in q + w + e:
                k = OpenKnot.from_periodic_line(line, self.shape)
                length = k.arclength()
                lengths.append(length)
            total_length = np.sum(lengths)
            lengths = [l / total_length for l in lengths]

            colours = []
            for length in lengths:
                for bound, colour in length_colours[:-1]:
                    if length < bound:
                        colours.append(colour)
                        break
                else:
                    colours.append(length_colours[-1])
            kwargs['colours'] = colours

        plot_cell(self.lines, boundary=boundary, clf=clf,
                  tube_radius=tube_radius, **kwargs)

    def smooth(self, repeats=1, window_len=10):
        from pyknot2.spacecurves import Knot
        new_lines = []
        for i, line in enumerate(self.lines):
            new_segments = []
            for segment in line:
                if len(segment) > window_len:
                    k = Knot(segment)
                    k.smooth(repeats, periodic=False, window_len=window_len)
                    new_segments.append(k.points)
                else:
                    new_segments.append(segment)
            new_lines.append(new_segments)
        self.lines = new_lines

    def to_povray(self, filen, spline='cubic_spline'):
        from pyknot2.visualise import cell_to_povray
        cell_to_povray(filen, self.lines, self.shape)

    def get_lengths(self):
        from pyknot2.spacecurves.openknot import OpenKnot
        lengths = []
        for line in self.lines:
            points = n.vstack(line)
            k = OpenKnot.from_periodic_line(points, self.shape,
                                            perturb=False)
            lengths.append(k.arclength())
        return lengths


                


        
def _interpret_line(line):
    if isinstance(line, Knot):
        return line.points
    elif isinstance(line, n.ndarray):
        return line

    return ValueError('Lines must be Knots or ndarrays.')

def _test_periodicity(line, shape):
    closing_vector = (line[-1][-1] - line[0][0]) / n.array(shape)
    if n.any(closing_vector > 0.2):
        return 'nth'
    return 'loop'

def _cut_line_at_jumps(line, shape):
    x, y, z = shape
    shape = n.array(shape)
    if x < 0 or y < 0 or z < 0:
        return line
    line = line.copy()
    i = 0
    out = []
    while i < (len(line)-1):
        cur = line[i]
        nex = line[i+1]
        if n.any(n.abs(nex-cur) > 0.9*shape):
            first_half = line[:(i+1)]
            second_half = line[(i+1):]
            out.append(first_half)
            line = second_half
            i = 0
        else:
            i += 1
    out.append(line)
    return out

def _cram_into_cell(line, shape):
    '''Imposes the shape as periodic boundary conditions.'''
    shape = n.array(shape)
    dx, dy, dz = shape

    points = n.array(line).copy()
    for i in range(1, len(points)):
        prev = points[i-1]
        cur = points[i]
        rest = points[i:]
        if cur[0] < 0:
            rest[:, 0] += dx
        if cur[0] > dx:
            rest[:, 0] -= dx
        if cur[1] < 0:
            rest[:, 1] += dy
        if cur[1] > dy:
            rest[:, 1] -= dy
        if cur[2] < 0:
            rest[:, 2] += dz
        if cur[2] > dz:
            rest[:, 2] -= dz

    return points
