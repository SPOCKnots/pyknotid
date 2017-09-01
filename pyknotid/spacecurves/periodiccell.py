'''
PeriodicCell
------------

Tools for working with a periodic cell of spacecurves.

API documentation
~~~~~~~~~~~~~~~~~
'''
from pyknotid.utils import ensure_shape_tuple
from pyknotid.spacecurves import Knot, OpenKnot
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
        '''Returns an instance of Cell from a quartet of differently
        classified lines in periodic boundaries.

        Parameters
        ----------
        qwer : tuple
            Should be a 4-tuple of lists q, w, e, r. q is closed
            loops, w is lines with non-trivial homology, e is lines that
            terminate on the boundaries of the cell, r is any remaining
            (unclassified) lines.
        shape : int or tuple
            The size of the cell along each axis. If a single number
            is passed, all axes are assumed to be the same length.

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

    def plot(self, boundary=True, clf=True, tube_radius=0.5,
             length_colours=None,
             **kwargs):
        from pyknotid.visualise import plot_cell
        boundary = self.shape if boundary else None

        if length_colours is not None:
            if 'colours' in kwargs:
                raise ValueError(
                    'colours and length_colours cannot both be set')

            q, w, e, r = self.qwer
            from pyknotid.spacecurves.openknot import OpenKnot

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
        '''Smooth each line in the curve, equivalent to
        :meth:`~pyknotid.spacecurves.spacecurve.SpaceCurve.smooth`.

        '''
        from pyknotid.spacecurves import Knot
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
        from pyknotid.visualise import cell_to_povray
        cell_to_povray(filen, self.lines, self.shape)

    def get_lengths(self):
        from pyknotid.spacecurves.openknot import OpenKnot
        lengths = []
        for line in self.lines:
            points = n.vstack(line)
            k = OpenKnot.from_periodic_line(points, self.shape,
                                            perturb=False)
            lengths.append(k.arclength())
        return lengths

    def simplify(self, num=1, cut_selection='uniform', **kwargs):
        from pyknotid.simplify import octree

        for i in range(num):
            print('i = {} / {}'.format(i, num))
            ot = octree.OctreeCell.from_cell_lines(self.lines, self.shape,
                                                cut_selection=cut_selection,
                                                **kwargs)
            ot.simplify()

            self.lines = [_cut_line_at_jumps(l, self.shape) for l in
                        ot.get_lines()]

    def linking_matrix(self):
        '''Get the linking numbers of each line in the cell with every
        other.
        '''
        from pyknotid.spacecurves.openknot import OpenKnot
        from pyknotid.spacecurves.link import Link
        from collections import defaultdict

        lines = self.lines
        
        types = ['loop' if _is_closed_loop(
            OpenKnot.from_periodic_line(np.vstack(l), self.shape,
                                        perturb=False).points)
                 else 'line' for l in lines]

        linkings = np.zeros((len(lines), len(lines)))

        linkings = {}

        for i1, details in enumerate(zip(lines, types)):
            line1, line_type1 = details
            print('i1 is', i1, len(lines))
            for i2, other_details in enumerate(zip(lines[i1:], types[i1:])):
                i2 += i1
                print('i2 is', i2)
                line2, line_type2 = other_details

                if line_type1 == 'loop' and line_type2 == 'loop':

                    linking = get_linking_between_loops(
                        line1, line2, self.shape, same_loop=(i1==i2))
                    
                    print('linking is', linking)

                    if linking:
                        linkings[(i1, i2)] = linking

                elif line_type1 == 'loop' and line_type2 == 'line':
                    linking = get_linking_between_loop_and_line(
                        line1, line2, self.shape)

                # Same as previous condition, just the opposite order
                elif line_type1 == 'line' and line_type2 == 'loop':
                    linking = get_linking_between_loop_and_line(
                        line2, line1, self.shape)

                elif line_type1 == 'line' and line_type2 == 'line':
                    pass

        return linkings
        

    # def efficient_linking_matrix(self):
    #     '''Get the linking numbers of each line in the cell with every
    #     other.
    #     '''
    #     # Should this be the periodic linking number or not?

    #     linkings = np.zeros((len(self.lines),
    #                          len(self.lines)))

    #     import chelpers

    #     all_crossings = []

    #     for i1, line in enumerate(self.lines):
    #         segments_1 = line

    #         cum_lengths_1 = np.cumsum(map(len, segments_1))
    #         print('i1 is', i1, len(self.lines))

    #         for i2, other_line in enumerate(self.lines[i1+1:]):
    #             i2 += 1
    #             segments_2 = line
    #             cum_lengths_2 = np.cumsum(map(len, segments_2))

    #             steps_2 = [
    #                 np.roll(segment[:, :2], -1, axis=0) - segment[:, :2] for
    #                 segment in segments_1]
    #             step_lengths_2 = []
    #             for steps in steps_2:
    #                 step_lengths_2.append(np.array(
    #                     [np.sqrt(np.sum(row**2))
    #                      for row in steps]))
    #             max_step_lengths_2 = [np.max(ls) for ls in
    #                                       step_lengths_2]

    #             current_linking = 0
    #             current_offset = (0, 0, 0)

    #             crossings = []

    #             for segment_1, current_index in zip(segments_1, cum_lengths_1):
    #                 for i, point in enumerate(segment_1[:-1]):
    #                     next_point = segment_1[i+1]
    #                     crossing_index = current_index + i
    #                     dv = next_point - point
    #                     for crossing_index_2, segment_2, lengths_2, max_step_length in zip(
    #                             cum_lengths_2, segments_2,
    #                             step_lengths_2,
    #                             max_step_lengths_2):

    #                         new_crossings = chelpers.find_crossings(
    #                             point, dv,
    #                             segment_2,
    #                             lengths_2,
    #                             crossing_index,
    #                             0,
    #                             max_step_length
    #                             )
    #                         if new_crossings:
    #                             new_crossings = np.array(new_crossings)
    #                             new_crossings[::2, 1] += crossing_index_2
    #                             new_crossings[1::2, 0] += crossing_index_2
    #                             # new_crossings[:, 1] += crossing_index_2
    #                             crossings.append(new_crossings)

    #             if crossings:
    #                 all_crossings.append((i1, i2, np.vstack(crossings)))

    #     linkings = []

    #     return all_crossings
                    
        
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

def _is_closed_loop(line, cutoff=5.):
    closing_distance = np.sqrt(np.sum((line[-1] - line[0])**2))
    return closing_distance < cutoff


from pyknotid.spacecurves.link import Link

class BoundingBox(object):
    def __init__(self, l):
        if hasattr(l, 'points'):
            # crudely get the points if l is a
            # Knot or SpaceCurve
            l = l.points

        self.mins = np.min(l, axis=0)
        self.maxs = np.max(l, axis=0)

    def intersects(self, b):
        b1 = self
        b2 = b

        b1xmax, b1ymax, b1zmax = b1.maxs
        b1xmin, b1ymin, b1zmin = b1.mins

        b2xmax, b2ymax, b2zmax = b2.maxs
        b2xmin, b2ymin, b2zmin = b2.mins

        return (b1xmax > b2xmin and
                b1xmin < b2xmax and
                b1ymax > b2ymin and
                b1ymin < b2ymax and
                b1zmax > b2zmin and
                b1zmin < b2zmax)

    def translations_to(self, b, shape):
        '''
        Returns the translations of b1 that could make it overlap b2.
        '''
        b1 = self
        b2 = b

        size1 = b1.maxs - b1.mins
        size2 = b2.maxs - b2.mins

        if not isinstance(shape, (int, float)):
            assert len(set(shape)) == 1
            shape = float(shape[0])

        # should these have +1 and -1?
        steps_mins = np.floor((b2.mins - b1.maxs) / shape).astype(np.int) + 1
        steps_maxs = np.floor((b2.maxs - b1.mins) / shape).astype(np.int)

        return steps_mins, steps_maxs

        distance_from_b2_to_b1 = b2.mins - b1.mins
        num_steps_from_b2_to_b1 = distance_from_b2_to_b1 / shape


def get_linking_between_loops(line1, line2, shape,
                              same_loop=False,
                              periodic=True):
    if periodic:
        l = Link.from_periodic_lines((np.vstack(line1),
                                      np.vstack(line2)),
                                     shape,
                                     perturb=False)
    else:
        l = Link((line1, line2))
    b1 = BoundingBox(l.lines[0].points)
    b2 = BoundingBox(l.lines[1].points)

    # Get the range of x, y and z translations under
    # which the bounding boxes collide

    translations = b2.translations_to(b1, shape)

    print('translations are', translations)

    linkings = {}

    for dx in range(translations[0][0], translations[1][0] + 1):
        for dy in range(translations[0][1], translations[1][1] + 1):
            for dz in range(translations[0][2], translations[1][2] + 1):
                if same_loop:
                    if dx == dy == dz == 0:
                        continue
                print('dx dy dz', dx, dy, dz, translations[0], translations[1])
                points1 = l.lines[0].points.copy()
                points2 = l.lines[1].points.copy()

                points2 += np.array([dx, dy, dz]) * np.array(shape)

                b2 = BoundingBox(points2)

                if not b1.intersects(b2):
                    continue

                l_new = Link((points1, points2), verbose=False)
                l_new.rotate()

                linking_number = l_new.linking_number()
                if linking_number:
                    linkings[(dx, dy, dz)] = linking_number

    return linkings

def get_linking_between_loop_and_line(loop, line, shape,
                                      periodic=True):

    loop= np.vstack(loop)
    line = np.vstack(line)
    loop = Knot.from_periodic_line(loop, shape)
    line = OpenKnot.from_periodic_line(line, shape)

    line_closure = (line.points[-1] - line.points[0]) / shape[0]
    line_closure = np.round(line_closure).astype(np.int)

    print('closure is', line_closure)

    b1 = BoundingBox(line.points)
    b2 = BoundingBox(loop.points)

    # Get the range of x, y and z translations under
    # which the bounding boxes collide

    translations = b2.translations_to(b1, shape)

    print('translations are', translations)

    linkings = {}

    from pyknotid.spacecurves.link import Link
    from pyknotid.spacecurves.rotation import rotate_vector_to_top
    from pyknotid.utils import get_rotation_matrix

    random_matrix = get_rotation_matrix(np.random.random(size=3) * 2 * np.pi)

    # loop._apply_matrix(rotation_matrix)
    # line._apply_matrix(rotation_matrix)

    loop._apply_matrix(random_matrix)
    line._apply_matrix(random_matrix)

    to_top_matrix = rotate_vector_to_top(line.points[-1] - line.points[0])

    loop._apply_matrix(to_top_matrix)
    line._apply_matrix(to_top_matrix)

    # Switch x and z axes
    # loop.points[:, 0], loop.points[:, 2] = loop.points[:, 2], loop.points[:, 0]
    # line.points[:, 0], line.points[:, 2] = line.points[:, 2], line.points[:, 0]

    axis_rotation = get_rotation_matrix((np.pi / 2., 0, 0))

    line._apply_matrix(axis_rotation)
    loop._apply_matrix(axis_rotation)

    for dx in range(translations[0][0], translations[1][0] + 1):
        for dy in range(translations[0][1], translations[1][1] + 1):
            for dz in range(translations[0][2], translations[1][2] + 1):

                print(dx, dy, dz)

                translation = random_matrix.dot(np.array((dx, dy, dz)) * np.array(shape))
                translation = to_top_matrix.dot(translation)
                translation = axis_rotation.dot(translation)

                print('translation is', translation)

                points1 = line.points.copy()
                points2 = loop.points.copy()
                points1[-1] = points1[0]

                points2 += translation

                b1 = BoundingBox(points1)
                b2 = BoundingBox(points2)


                if not b1.intersects(b2):
                    continue

                l = Link((points1, points2), verbose=False)

                linking_number = l.linking_number(include_closures=False)

                # if linking_number:
                #     import ipdb
                #     ipdb.set_trace()


                if linking_number:
                    linkings[(dx, dy, dz)] = linking_number
                
    return linkings
                

                # We can calculate the linking number as a link in the current plane, as the crossings will be made up later

    # 0.1) Choose a random rotation

    # 1) Rotate the line to have homology vector in z=0
    # 2) For every translation where the loop overlaps the line at all...
    #    ...but factoring out translations that are multiples of the line's homology vector!
    # 3) Extend the line as far as necessary
    # 4) Calculate the linking

    # For every position where the 


loop1 = np.array([[0., 0, 0],
                  [5, 0, 0],
                  [5, 5, 0],
                  [0, 5, 0],
                  [0, 0.1, 0]])

loop2 = np.array([[2.5, 2.5, -3],
                  [2.5, 2.5, 3],
                  [7, 8, 3],
                  [7, 8, -3],
                  [2.7, 2.65, -3]])

loop3 = loop2 + np.array([30., 0, 0])
