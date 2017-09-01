'''
Octree space curve simplification
=================================

Module for simplifying lines with an octree decomposition.
'''

import numpy as n
try:
    from coctree import (angle_exceeds as cangle_exceeds,
                         line_to_segments as cline_to_segments)
except ImportError:
    cangle_exceeds = None
    cline_to_segments = None


class OctreeError(Exception):
    pass


class OctreeCell(object):
    '''Stores line segments, and is capable of splitting and simplifying
    them.

    - lines must be a list of LineSegments. If your data is in another form,
      consider using the from_line or from_cell_lines class methods.
    - shape is the size of the cell
    - depth is the current recursion depth
    - state is a dictionary that stores information from the recursion tree
    - min_cell is the minimum cell side length
    - max_depth is the maximum recursion depth beyond which simplification is
      not attempted
    - top_level should be True if this OctreeCell is the first created, in which
      case a LineHandle is instantiated for each segment so that it can be
      reconstructed after simplification.
    - handles is a list of Handles to use, if it doesn't exist and top_level
      is True then one is created per segment. This *will* fail with some
      input, so use the class methods for building the cell to automatically
      pass the right handles.
    - cut_selection is a string saying how to select the cut planes in self.
      It may be 'regular' (8 equally sized octants), 'uniform' (values
      selected from a uniform distribution in each axis), or 'com' (values
      selected to pass through the line centre of mass of the cell).
    - is_knotted_func is an optional function to test for knotting.

    The Handle system is used to ensure correct behaviour (with simple code!)
    even for non-continuous segments etc.
    '''
    def __init__(self, lines, shape, depth=0, top_level=True,
                 state=None, min_cell=1., max_depth=100,
                 handles=None, cut_selection='regular',
                 is_knotted_func=None):

        self.depth = depth
        self.shape = shape
        self.max_depth = max_depth
        self.min_cell = min_cell

        self.cut_selection = cut_selection
        if cut_selection == 'regular':
            self.cut_planes = self.get_equidistant_planes()
        elif cut_selection == 'uniform':
            self.cut_planes = self.get_uniform_random_planes()
        elif cut_selection == 'com':
            raise OctreeError('com cut selection not yet supported')
        else:
            raise OctreeError('invalid choice of cut_selection')

        if state is None:
            state = {}
        self.state = state

        self.is_knotted_func = is_knotted_func

        if 'boundaries' not in state:
            state['boundaries'] = []
        state['boundaries'].append(self.boundary_lines())

        self.original_segments = lines
        self.segments = lines

        if handles is None:
            handles = []
        self.handles = handles
        if top_level and not self.handles:
            # Make a Handle to keep track of each segment even when
            # it is replaced with multiple shorter segments
            self.handles = [Handle(segment, segment.prev) for
                            segment in self.segments]
            for handle in self.handles:
                handle.next.prev = handle
                handle.prev.next = handle

    @classmethod
    def from_single_line(cls, line, shape=None, **kwargs):
        '''Returns an OctreeCell created from the given line, optionally
        with the given boundingbox shape. If the shape is not provided,
        it is calculated from the line extents.

        Other kwargs are passed directly to the OctreeCell.
        '''
        if shape is None:
            shape = [n.min(line[:, 0]), n.max(line[:, 0]),
                     n.min(line[:, 1]), n.max(line[:, 1]),
                     n.min(line[:, 2]), n.max(line[:, 2])]
        lineseg = LineSegment(line, identifier=1)
        lineseg.next = lineseg
        lineseg.prev = lineseg

        return cls([lineseg], shape, **kwargs)


    @classmethod
    def from_lines(cls, lines, shape=None, **kwargs):
        '''Returns an OctreeCell created from the given lines, optionally
        with the given boundingbox shape. If the shape is not provided,
        it is calculated from the maximum line extents.

        Other kwargs are passed directly to the OctreeCell.
        '''
        if shape is None:
            extremes = [[n.min(line[:, 0]), n.max(line[:, 0]),
                         n.min(line[:, 1]), n.max(line[:, 1]),
                         n.min(line[:, 2]), n.max(line[:, 2])] for
                         line in lines]
            extremes = n.array(extremes)
            shape = [n.min(extremes[:, 0]), n.max(extremes[:, 1]),
                     n.min(extremes[:, 2]), n.max(extremes[:, 3]),
                     n.min(extremes[:, 4]), n.max(extremes[:, 5])]
        segments = []
        for identifier, line in enumerate(lines):
            lineseg = LineSegment(line, identifier=identifier)
            lineseg.next = lineseg
            lineseg.prev = lineseg
            segments.append(lineseg)

        # import ipdb
        # ipdb.set_trace()

        return cls(segments, shape, **kwargs)
            


    @classmethod
    def from_cell_lines(cls, lines, shape, **kwargs):
        '''Returns an OctreeCell created from the given list of lines,
        which are assumed to be periodic.

        The shape may be all 6 values (xmin, xmax, ymin, ymax, zmin, zmax)
        or just xmax, ymax, zmax with the assumption that the minima are
        all zero.
        '''

        if len(shape) == 3:
            shape = (0, shape[0], 0, shape[1], 0, shape[2])
        short_shape = shape[1::2]
        
        if isinstance(lines[0], list):
            line_segs = lines
        else:
            line_segs = [
                split_cell_line(line, [s - 10. for s in short_shape])
                for line in lines]
        all_joined_segments = []
        handles = []
        for identifier, segments in enumerate(line_segs):
            segments = filter(lambda j: len(j) > 1, segments)
            seg_classes = [LineSegment(seg, identifier=identifier)
                           for seg in segments]
            handle = Handle(seg_classes[0])
            for i, seg_class in enumerate(seg_classes):
                nex = seg_classes[(i+1) % len(seg_classes)]
                # The CellJumpSegments isn't in self.segments, so it
                # won't be simplified
                jump_segment = CellJumpSegment(  
                    n.vstack((seg_class.points[-1], nex.points[0])))
                seg_class.next = jump_segment
                nex.prev = jump_segment
                jump_segment.prev = seg_class
                jump_segment.next = nex
            handle.prev = handle.next.prev
            handle.next.prev = handle
            handle.prev.next = handle
            handles.append(handle)

            all_joined_segments.extend(seg_classes)

        return cls(all_joined_segments, shape, handles=handles, **kwargs)

    def simplify(self, obey_knotting=True):
        '''
        Performs a suitable action depending on the segments in the cell:
        - if 1 segment, checks the total angle and reduces to a straight
          line if it is less than 3.8pi (4pi is needed for a knot)

        - otherwise, does the octant simplification if the cell min size
          and recursion depth have both not been hit

        The `obey_knotting` parameter may be False to ignore self-entangling
        of lines, i.e. a single line may just pass through itself so no
        angle check is necessary.
        '''

        segments = self.segments
        do_octants = False
        shape = self.shape
        size = self.get_cell_size()

        # Do something reasonable if there are two segments that include
        # the line start point
        if len(segments) == 2:
            s1, s2 = segments
            if s1.next == s2.prev:
                if isinstance(s1.next, Handle):
                    handle = s1.next
                    points = n.vstack((s1.points, s2.points))
                    new_segment = LineSegment(points,
                                              identifier=s1.identifier)
                    handle.prev = s1.prev
                    handle.prev.next = handle
                    handle.next = new_segment
                    new_segment.prev = handle
                    new_segment.next = s2.next
                    new_segment.next.prev = new_segment
                    self.segments = segments = [new_segment]
            elif s2.next == s1.prev:
                if isinstance(s2.next, Handle):
                    handle = s2.next
                    points = n.vstack((s2.points, s1.points))
                    new_segment = LineSegment(points,
                                              identifier=s2.identifier)
                    handle.prev = s2.prev
                    handle.prev.next = handle
                    handle.next = new_segment
                    new_segment.prev = handle
                    new_segment.next = s1.next
                    new_segment.next.prev = new_segment
                    self.segments = segments = [new_segment]

        if (len(segments) == 0 or self.depth >= self.max_depth or
            any([span < self.min_cell for span in size])):
            # If any of these are True, no simplification is possible
            return

        if not obey_knotting:
            identifiers = [segment.identifier for segment in segments]
            if None not in identifiers and len(set(identifiers)) == 1:
                for segment in segments:
                    segment.replace_with_straight_line()
                return


        elif len(segments) == 1:
            segment = segments[0]
            if not obey_knotting:
                segment.replace_with_straight_line()
                return
            if not angle_exceeds_func(segment.points, 2.*n.pi, False):
                segment.replace_with_straight_line()
                return
            if (self.is_knotted_func is not None and
                not self.is_knotted_func(segment.points)):
                segment.replace_with_straight_line()
                return

        # If we get here, the cell is too complex to simplify topologically,
        # but we are allowed to split into octants
                
        self._simplify_via_octants(obey_knotting)
        
    def _simplify_via_octants(self, obey_knotting=True):
        '''Splits each segment of self into sub-segments in each octant
        of self, and resolves them with a new OctreeCell for each octant.'''

        segments = self.segments
        cuts = self.cut_planes


        # Get new segments to be sorted into the new octants
        new_segments = []
        for seg in segments:
            new_segs = seg.cut_at(cuts)
            for each_new_seg in new_segs:
                new_segments.append(each_new_seg)

        # Place the segments in appropriate octants
        octant_contents = find_octants_of_segments(new_segments,
                                                   self.cut_planes)

        # Make and run a new OctreeCell for each octant
        fur, flr, fll, ful, bur, blr, bll, bul = octant_contents
        xmin, xmax, ymin, ymax, zmin, zmax = self.shape
        cut_x, cut_y, cut_z = self.cut_planes

        shapes = ((cut_x, xmax, cut_y, ymax, cut_z, zmax),
                  (cut_x, xmax, ymin, cut_y, cut_z, zmax),
                  (xmin, cut_x, ymin, cut_y, cut_z, zmax),
                  (xmin, cut_x, cut_y, ymax, cut_z, zmax),
                  (cut_x, xmax, cut_y, ymax, zmin, cut_z),
                  (cut_x, xmax, ymin, cut_y, zmin, cut_z),
                  (xmin, cut_x, ymin, cut_y, zmin, cut_z),
                  (xmin, cut_x, cut_y, ymax, zmin, cut_z))

        for i in range(8):
            octant_segments = octant_contents[i]
            if not len(octant_segments):
                continue
            shape = shapes[i]
            cxmin, cxmax, cymin, cymax, czmin, czmax = shape
            cell_size = (cxmax-cxmin) + (cymax-cymin) + (czmax-czmin)

            q = OctreeCell(octant_segments, shape, self.depth+1,
                           top_level=False,
                           state=self.state, min_cell=self.min_cell,
                           max_depth=self.max_depth,
                           cut_selection=self.cut_selection)
            q.simplify(obey_knotting)

    
    def get_equidistant_planes(self):
        '''Returns the x, y and z values halfway along each axis of the
        cell, according to self.shape.'''
        xmin, xmax, ymin, ymax, zmin, zmax = self.shape
        cut_x = (xmax + xmin) / 2.
        cut_y = (ymax + ymin) / 2.
        cut_z = (zmax + zmin) / 2.
        return (cut_x, cut_y, cut_z)

    def get_uniform_random_planes(self):

        '''Returns x, y and z values each uniformly randomly distributed
        through self.shape.'''
        xmin, xmax, ymin, ymax, zmin, zmax = self.shape
        return (n.array([xmin, ymin, zmin]) +
                (0.1 + 0.8*n.random.random(3)) *
                n.array(self.get_cell_size()))

    def get_cell_size(self):
        return (self.shape[1] - self.shape[0],
                self.shape[3] - self.shape[2],
                self.shape[5] - self.shape[4])
        
    def boundary_lines(self):
        '''Get a line tracing the boundary of the cell. These lines
        are passed recursively up the OctreeCell tree so that the top level one
        keeps a list of all of them - useful for plotting the tree state
        later.'''
                 
        xmin, xmax, ymin, ymax, zmin, zmax = self.shape
        return n.array([[xmin, ymin, zmin],
                        [xmax, ymin, zmin],
                        [xmax, ymax, zmin],
                        [xmin, ymax, zmin],
                        [xmin, ymin, zmin],
                        [xmin, ymin, zmax],
                        [xmax, ymin, zmax],
                        [xmax, ymin, zmin],
                        [xmax, ymin, zmax],
                        [xmax, ymax, zmax],
                        [xmax, ymax, zmin],
                        [xmax, ymax, zmax],
                        [xmin, ymax, zmax],
                        [xmin, ymax, zmin],
                        [xmin, ymax, zmax],
                        [xmin, ymin, zmax]])

    def get_lines(self, remove_unnecessary_jumps=True):
        '''Return all the lines of self by joining any segments back
        together again.'''
        handles = self.handles
        lines = [handle.reconstruct_line(remove_unnecessary_jumps)
                 for handle in handles]
        return lines

    def get_single_line(self, remove_unnecessary_jumps=True):
        '''Return the first line in the set of lines reconstructed from
        segments in self. This is useful for OctreeCells that only had
        a single line to begin with, e.g. simplifying a knot.
        '''
        lines = self.get_lines(remove_unnecessary_jumps)
        if len(lines) == 0:
            raise OctreeError('Tried to get single line but no lines found.')
        return lines[0]


        
class Handle(object):
    '''Simply stores a next and previous LineSegment, with a method to
    reconstruct the line by walking along the segments.

    The Handle is like a LineSegment with no point data - all it does is
    sit in the line keeping track of its next and prev segments (which
    may be modified by the cut_at method of those LineSegments). Since
    it will never change itself, it can be used later to rejoin all these
    LineSegments which may have been created arbitrarily deep in the
    decomposition, returning the same line that corresponds to its input!
    '''

    def __init__(self, next=None, prev=None, identifier=None):
        self.next = next  
        self.prev = prev  
        self.identifier = identifier

    def reconstruct_line(self, remove_unnecessary_jumps=True):
        '''Returns the line stored by the handle, by walking along it until
        returning to the Handle or reaching a segment with no next entry.

        If ``remove_unnecessary_jumps`` is True, will remove jumps connected
        by a trivial segment along the periodic cell wall.
        '''
        comp_segs = [self.next]
        current = self.next
        while current.next is not self and current.next is not None:
            current = current.next
            comp_segs.append(current)

        if remove_unnecessary_jumps:
            invalid_indices = []
            for i, segment in enumerate(comp_segs):
                if i not in invalid_indices:
                    segment2 = comp_segs[(i+1) % len(comp_segs)]
                    segment3 = comp_segs[(i+1) % len(comp_segs)]
                    if (isinstance(segment, CellJumpSegment) and
                        isinstance(segment3, CellJumpSegment) and
                        len(segment2.points) == 2):
                        invalid_indices.append(i)

            comp_segs = [seg for i, seg in enumerate(comp_segs) if
                         i not in invalid_indices]

        return resample(n.vstack([seg.points[:-1] for
                                  seg in comp_segs] +
                                 [comp_segs[0].points[:1] +
                                  0.00001]))

    def get_line_components(self):
        '''Returns a list of all the LineSegments in the line with
        this handle.'''
        component_segments = [self.next]
        current = self.next
        while current.next is not self and current.next is not None:
            current = current.next
            component_segments.append(current)
        return component_segments

        

class LineSegment(object):
    '''Stores a section of line, and can split at boundaries.'''

    def __init__(self, points, next=None, prev=None, identifier=None):
        self.points = points
        self.next = next  # the next LineSegment
        self.prev = prev  # the previous LineSegment

        self.identifier = identifier  # identifies all segments of the
                                      # same line


    def cell(self, cut_planes):
        '''Returns a string representing where the LineSegment is,
        in terms of which octant of the space divided by cut_planes
        x, y, z the LineSegment appears in.

        This assumes the LineSegment is only within one octant (except
        perhaps points on the boundary).
        '''

        cut_x, cut_y, cut_z = cut_planes
        p = self.points[0] + 0.51123*(self.points[1] - self.points[0])
        x, y, z = p
        if z > cut_z:
            if x > cut_x and y > cut_y:
                return 'fur'
            elif x > cut_x and y <= cut_y:
                return 'flr'
            elif x <= cut_x and y <= cut_y:
                return 'fll'
            elif x <= cut_x and y > cut_y:
                return 'ful'
            else:
                raise OctreeError('Segment not in any region?')
        elif z <= cut_z:
            if x > cut_x and y >= cut_y:
                return 'bur'
            elif x > cut_x and y <= cut_y:
                return 'blr'
            elif x <= cut_x and y <= cut_y:
                return 'bll'
            elif x <= cut_x and y > cut_y:
                return 'bul'
            else:
                raise OctreeError('Segment not in any region?')
        else:
            raise OctreeError('Segment not in any region?')

    def cut_at(self, cuts):
        '''Returns a list of new LineSegments, resulting from cutting self at
        the given cut planes.
        '''
        points = self.points.copy()
        crude_segs = line_to_segments(points, cuts, join_ends=False)
        segs = crude_segs_to_segs(crude_segs, self.identifier)

        # crude_segs_to_segs assumes the segments form a loop
        # here we instead connect the first to self.prev, and
        # the last to self.next
        segs[0].prev = self.prev
        self.prev.next = segs[0]
        segs[-1].next = self.next
        self.next.prev = segs[-1]

        return segs

    def replace_with_straight_line(self):
        '''
        Replaces self.points with a straight line (i.e. just the first
        and last vertices).

        This is only called when only one segment lies in a quadcell,
        therefore components can be cut out safely without worrying
        about any other parts of the full curve.'''

        if self.next is not self.prev:
            self.points = n.vstack((self.points[0], self.points[-1]))
        else:  # if the line is a loop, just cut out most points
            self.points = self.points[::int(len(self.points)/3.)]


class CellJumpSegment(LineSegment):
    '''A special LineSegment that denotes a jump through periodic boundary
    conditions. This has no special behaviour of its own, but is
    subclassed in case this is useful later.
    '''


def line_to_segments(line, cuts=None, join_ends=True):
    '''Takes a line (set of points), a list of cut planes in
    x, y, z, and a parameter to decide whether the line
    joining the first and last point should also be cut.

    Returns a list of shorter lines resulting from cutting at
    all these cut planes.'''

    line = line.copy()

    if cuts is None:
        xmin = n.min(line[:,0]) - 1
        xmax = n.max(line[:,0]) + 1
        ymin = n.min(line[:,1]) - 1
        ymax = n.max(line[:,1]) + 1
        zmin = n.min(line[:,2]) - 1
        zmax = n.max(line[:,2]) + 1

        cut_x = (xmax + xmin) / 2.
        cut_y = (ymax + ymin) / 2.
        cut_z = (zmin + zmax) / 2.
    else:
        cut_x, cut_y, cut_z = cuts

    # Cut the line wherever it passes through a quad cell boundary
    segments = []
    cut_i = 0
    for i in range(len(line)-1):
        cur = line[i]
        nex = line[i+1]

        dv = nex - cur
        dx, dy, dz = dv

        cross_cut_x = n.sign(cur[0] - cut_x) != n.sign(nex[0] - cut_x)
        cross_cut_y = n.sign(cur[1] - cut_y) != n.sign(nex[1] - cut_y)
        cross_cut_z = n.sign(cur[2] - cut_z) != n.sign(nex[2] - cut_z)

        if cross_cut_x and cross_cut_y and cross_cut_z:
            x_cut_pos = -1 * (cur[0]-cut_x)/dx
            y_cut_pos = -1 * (cur[1]-cut_y)/dy
            z_cut_pos = -1 * (cur[2]-cut_z)/dz
            order = n.sort((x_cut_pos, y_cut_pos, z_cut_pos))
            # assert 0 < x_cut_pos < 1 and 0 < y_cut_pos < 1 and 0 < z_cut_pos < 1
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            join_point_3 = cur + order[2]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            third_seg = n.vstack((join_point_2, join_point_3))
            line[i] = join_point_3
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
            segments.append(third_seg)
        elif cross_cut_x and cross_cut_y:
            x_cut_pos = -1 * (cur[0]-cut_x)/dx
            y_cut_pos = -1 * (cur[1]-cut_y)/dy
            order = n.sort((x_cut_pos, y_cut_pos))
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            line[i] = join_point_2
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
        elif cross_cut_x and cross_cut_z:
            x_cut_pos = -1 * (cur[0]-cut_x)/dx
            z_cut_pos = -1 * (cur[2]-cut_z)/dz
            order = n.sort((x_cut_pos, z_cut_pos))
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            line[i] = join_point_2
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
        elif cross_cut_y and cross_cut_z:
            y_cut_pos = -1 * (cur[1]-cut_y)/dy
            z_cut_pos = -1 * (cur[2]-cut_z)/dz
            order = n.sort((y_cut_pos, z_cut_pos))
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            line[i] = join_point_2
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
        elif cross_cut_x:
            cut_pos = -1 * (cur[0]-cut_x)/dx
            assert 0. <= cut_pos <= 1.
            join_point = cur + cut_pos*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point))
            line[i] = join_point
            cut_i = i
            segments.append(first_seg)
        elif cross_cut_y:
            cut_pos = -1 * (cur[1]-cut_y)/dy
            assert 0. <= cut_pos <= 1.
            join_point = cur + cut_pos*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point))
            line[i] = join_point
            cut_i = i
            segments.append(first_seg)
        elif cross_cut_z:
            cut_pos = -1 * (cur[2]-cut_z)/dz
            assert 0. <= cut_pos <= 1.
            join_point = cur + cut_pos*dv

            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point))
            line[i] = join_point
            # second_seg = n.vstack((join_point, line[(i+1):]))

            cut_i = i
            segments.append(first_seg)

    final_seg = line[cut_i:]
    if cut_i > 0:
        if join_ends:
            first_seg = segments.pop(0)
            segments.append(n.vstack((final_seg, first_seg)))
        else:
            segments.append(final_seg)
    else:
        segments.append(final_seg)

    return segments


def crude_segs_to_segs(csegs, identifier=None):
    '''Takes a list of line segments and makes them into LineSegment
    instances, under the assumption that the last is joined to the
    first. The segments are labelled by the passed identifier.
    '''
    segs = []
    for s in csegs:
        s = LineSegment(s, identifier=identifier)
        segs.append(s)
    for i in range(len(segs)):
        cur = segs[i]
        prev = segs[(i-1) % len(segs)]
        next = segs[(i+1) % len(segs)]
        cur.prev = prev
        cur.next = next
    return segs


def angle_exceeds(ps, val=2*n.pi, include_closure=True):
    '''Returns True if the sum of angles along ps exceeds
    val, else False.

    If include_closure, includes the angles with the line closing
    the end and start points.
    '''
    angle = 0.
    nex = ps[0]
    nex2 = ps[1]
    dv2 = nex2-nex
    dv2 /= mag(dv2)
    checks = range(len(ps)) if include_closure else range(len(ps)-2)
    for i in checks:
        cur = nex
        nex = nex2
        nex2 = ps[(i+2) % len(ps)]
        dv = dv2
        dv2 = nex2-nex
        dv2 /= mag(dv2)
        increment = angle_between(dv, dv2)
        if n.isnan(increment):
            return True
        angle += increment
        if angle > val:
            return True
    assert not n.isnan(angle)
    return False

def angle_between(v1, v2):
    '''Returns angle between v1 and v2, assuming they are normalised to 1.'''
    # clip becaus v1.dot(v2) may exceed 1 due to floating point
    return n.arccos(n.clip(n.abs(v1.dot(v2)), 0., 1.))


def mag(v):
    return n.sqrt(v.dot(v))

def find_octants_of_segments(segments, cuts):
    '''For each LineSegment in the passed segments, returns an
    identifier for the octant it should be placed in'''
    fur = []  # front upper right etc.
    flr = []
    fll = []
    ful = []
    bur = []
    blr = []
    bll = []
    bul = []
    for seg in segments:
        cell = seg.cell(cuts)
        if cell == 'fur':
            fur.append(seg)
        elif cell == 'flr':
            flr.append(seg)
        elif cell == 'fll':
            fll.append(seg)
        elif cell == 'ful':
            ful.append(seg)
        elif cell == 'bur':
            bur.append(seg)
        elif cell == 'blr':
            blr.append(seg)
        elif cell == 'bll':
            bll.append(seg)
        elif cell == 'bul':
            bul.append(seg)
        else:
            raise OctreeError('Segment not in recognised cell.')
    return (fur, flr, fll, ful, bur, blr, bll, bul)

def resample(points):
    '''Takes a set of points, and cuts out the middle ones if they form
    part of a straight line, or if they are identical to the previous
    point.'''
    if len(points) < 2:
        return points

    changing = True
    while changing:
        changing = False
        length = len(points)
        points = points.copy()
        keep = n.ones(len(points), dtype=bool)
        cur = points[0]
        nex = points[1]
        dv2 = nex-cur
        dv2 /= mag(dv2)

        for i in range(1, len(points) - 1):
            prev = cur
            cur = nex
            nex = points[i+1]
            if n.all(n.less(n.abs(cur-nex), 0.0001)):
                keep[i] = False
            else:
                dv1 = cur - prev
                dv1 /= mag(dv1)
                dv2 = nex - cur
                dv2 /= mag(dv2)
                if n.abs(dv1.dot(dv2) - 1) < 0.0001:
                    keep[i] = False
        points = points[keep]
        new_length = len(points)
        if new_length != length:
            changing = True
    return points[keep]

def remove_nearby_points(points):
    '''Takes a set of points, and removes those that are no distance from
    the previous one.'''
    points = points.copy()
    keep = n.ones(len(points), dtype=n.bool)

    comparator = points[0]
    for i, point in enumerate(points):
        nex = points[(i+1) % len(points)]
        if n.all(n.abs((nex - point)) < 0.00001):
            keep[i] = False
    keep[-1] = True

    return points[keep]
                 

def split_cell_line(line, shape=(10, 10, 10.)):
    '''Takes a cell lines, and cuts at the periodic boundaries
    (in this represenation, this is where it jumps)'''
    shape = n.array(shape)
    line = line.copy()
    i = 0
    out = []
    while i < (len(line)-1):
        cur = line[i]
        nex = line[i+1]
        if n.any(n.abs(nex-cur) > shape):
            firsthalf = line[:(i+1)]
            secondhalf = line[(i+1):]
            out.append(firsthalf)
            line = secondhalf
            i = 0
        else:
            i += 1
    out.append(line)
    return out

# Setup functions that *may* depend on cython
angle_exceeds_func = angle_exceeds
# angle_exceeds_func = (cangle_exceeds if cangle_exceeds is not None
#                       else angle_exceeds)
line_to_segments_func = line_to_segments
# line_to_segments_func = (cline_to_segments if cline_to_segments is not None
#                       else line_to_segments)

def determinant_indicates_knot(points):
    if len(points) > 500:
        return True
    from pyknotid.spacecurves import Knot
    k = Knot(points)
    return (k.determinant() > 1)
        
