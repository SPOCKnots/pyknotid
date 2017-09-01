'''
PlanarDiagram
=============

Classes for working with planar diagram notation of knot diagrams.

See individual class documentation for more details.

API documentation
~~~~~~~~~~~~~~~~~
'''

from __future__ import print_function

import numpy as n


class PlanarDiagram(list):
    '''A class for containing and manipulating planar diagrams.

    Just provides convenient display and conversion methods for now.
    In the future, will support simplification.

    Shorthand input may be of the form ``X_1,4,2,5 X_3,6,4,1 X_5,2,6,3``.
    This is (should be?) the same as returned by repr.

    Parameters
    ----------
    crossings : array-like or string or GaussCode
        The list of crossings in the diagram, which will be converted
        to an internal planar diagram representation. Currently these are
        mostly converted via a GaussCode instance, so in addition to the
        shorthand any array-like supported by
        :class:`~pyknotid.representations.gausscode.GaussCode` may be used.
    '''

    def __init__(self, crossings=''):
        from pyknotid.representations import gausscode
        if isinstance(crossings, str):
            self.extend(shorthand_to_crossings(crossings))
        elif isinstance(crossings, gausscode.GaussCode):
            self.extend(gausscode_to_crossings(crossings))
        else:
            self.extend(gausscode_to_crossings(
                gausscode.GaussCode(crossings)))

    def __str__(self):
        lenstr = 'PD with {0}: '.format(len(self))
        return lenstr + ' '.join([str(crossing) for crossing in self])

    def __repr__(self):
        return self.__str__()

    def as_mathematica(self):
        '''
        Returns a mathematica code representation of self, usable in the
        mathematica knot tools.
        '''
        s = 'PD['
        s = s + ', '.join(crossing.as_mathematica() for crossing in self)
        return s + ']'

    def as_spherogram(self):
        '''
        Get a planar diagram class from the Spherogram module, which
        can be used to access SnapPy's manifold tools.

        This method requires that spherogram and SnapPy are installed.
        '''
        from snappy import Link
        from spherogram import Crossing, Link
        scs = [Crossing() for crossing in self]

        indices = {}
        for i in range(len(self)):
            c = self[i]
            for j in range(len(c)):
                number = c[j]
                if number in indices:
                    otheri, otherj = indices.pop(number)
                    scs[i][j] = scs[otheri][otherj]
                else:
                    indices[number] = (i, j)
        return Link(scs)

    def as_networkx_extended(self):
        '''(internal use only) Returns a networkx Graph along with extra
        information about the crossings.
        '''
        # print('pd is', self)
        import networkx as nx
        edges = []
        cache = {}
        heights = {}
        edge_directions = []
        from collections import defaultdict
        intermediate_edges_by_node = defaultdict(lambda : [None, None, None, None])
        intermediate_node_index = len(self)
        for node_index, crossing in enumerate(self):
            for crossing_index, arc_number in enumerate(crossing):
                if arc_number in cache:
                    other_node_index, other_height, other_crossing_index = cache[arc_number]
                    # edges.append([other_node_index, node_index])
                    edges.append([other_node_index, intermediate_node_index])
                    edges.append([node_index, intermediate_node_index])

                    height = index_height(crossing_index)

                    if crossing.is_outgoing(arc_number):
                        # heights[node_index, other_node_index, arc_number] = (height, other_height)
                        heights[node_index, intermediate_node_index, arc_number] = (height, 0.)
                        heights[intermediate_node_index, other_node_index, arc_number] = (0., other_height)

                        # edge_directions.append((node_index, other_node_index, arc_number))
                        edge_directions.append((node_index, intermediate_node_index, arc_number))
                        edge_directions.append((intermediate_node_index, other_node_index, arc_number))
                    else:
                        # heights[other_node_index, node_index, arc_number] = (other_height, height)
                        heights[other_node_index, intermediate_node_index, arc_number] = (other_height, 0.)
                        heights[intermediate_node_index, node_index, arc_number] = (0., height)

                        # edge_directions.append((other_node_index, node_index, arc_number))
                        edge_directions.append((other_node_index, intermediate_node_index, arc_number))
                        edge_directions.append((intermediate_node_index, node_index, arc_number))

                    intermediate_edges_by_node[node_index][crossing_index] = intermediate_node_index
                    intermediate_edges_by_node[other_node_index][other_crossing_index] = intermediate_node_index

                    intermediate_node_index += 1 
                    
                else:
                    cache[arc_number] = (node_index, index_height(crossing_index), crossing_index)

        for node, intermediate_nodes in intermediate_edges_by_node.items():
            for i, intermediate in enumerate(intermediate_nodes):
                next_intermediate = intermediate_nodes[(i+1) % 4]
                edges.append([intermediate, next_intermediate])

        # print('intermediates are', intermediate_edges_by_node)
        g = nx.Graph()
        g.add_nodes_from(list(range(len(self)))*3)
        g.add_edges_from(edges)

        # print('possible heights:')
        # for key, value in sorted(heights.items()):
        #     print(key, value)


        seen = set()
        duplicates = list()
        for edge in edges:
            edge = tuple(edge)
            if edge in seen:
                duplicates.append(sorted(edge))
            seen.add(edge)

        return g, duplicates, heights, sorted(edge_directions)[0]

    def as_networkx(self):
        '''Get a networkx graph representing the planar diagram, where
        each node is a crossing and each edge is an arc. This is a
        non-directed non-multi graph; where two arcs join the same crossing,
        they are represented as a single edge, but information about
        duplicates is returned alongside the graph.

        Returns
        -------
        g : Graph
            The networkx graph
        duplicates : list
            A list of tuples representing nodes joined by multiple edges.
        heights : dict
            A dictionary of (start, end, arc_number) graph edges,
            containing the start and end height of each edge.
        first_edge : tuple
            The first edge in the graph, including (start, end, arc_number).
        '''
        # print('pd is', self)
        import networkx as nx
        edges = []
        cache = {}
        heights = {}
        edge_directions = []
        for node_index, crossing in enumerate(self):
            for crossing_index, arc_number in enumerate(crossing):
                if arc_number in cache:
                    other_node_index, other_height = cache[arc_number]
                    edges.append([other_node_index, node_index])

                    height = index_height(crossing_index)

                    if crossing.is_outgoing(arc_number):
                        # print('node {} {}, crossing {} is outgoing to {}'.format(
                        #     node_index, crossing, arc_number, other_node_index))
                        heights[node_index, other_node_index, arc_number] = (height, other_height)
                        edge_directions.append((node_index, other_node_index, arc_number))
                    else:
                        # print('node {} {}, crossing {} is incoming from {}'.format(
                            # node_index, crossing, arc_number, other_node_index))
                        heights[other_node_index, node_index, arc_number] = (other_height, height)
                        edge_directions.append((other_node_index, node_index, arc_number))
                    
                else:
                    cache[arc_number] = node_index, index_height(crossing_index)

        g = nx.Graph()
        g.add_nodes_from(range(len(self)))
        g.add_edges_from(edges)

        # print('possible heights:')
        # for key, value in sorted(heights.items()):
        #     print(key, value)


        seen = set()
        duplicates = list()
        for edge in edges:
            edge = tuple(edge)
            if edge in seen:
                duplicates.append(sorted(edge))
            seen.add(edge)

        return g, duplicates, heights, sorted(edge_directions)[0]
            
def index_height(index):
    '''Returns the height based on the index of the crossing in an entry
    of a planar diagram; the 0th and 2nd indices are under crossings,
    and the 1st and 3rd are over crossings.

    '''
    if index in (0, 2):
        return -1.
    return 1.

class Crossing(list):
    '''
    A single crossing in a planar diagram. Each :class:`PlanarDiagram`
    is a list of these.

    Parameters
    ----------
    a : int or None
        The first entry in the list of lines meeting at this Crossing.
    b : int or None
        The second entry in the list of lines meeting at this Crossing.
    c : int or None
        The third entry in the list of lines meeting at this Crossing.
    d : int or None
        The fourth entry in the list of lines meeting at this Crossing.
    '''

    def __init__(self, a=None, b=None, c=None, d=None):
        super(Crossing, self).__init__()
        self.extend([a, b, c, d])

    def valid(self):
        '''
        True if all intersecting lines are not None.
        '''
        if all([entry is not None for entry in self]):
            return True
        return False

    def components(self):
        '''
        Returns a de-duplicated list of lines intersecting at this Crossing.

        :rtype: list
        '''
        return list(set(self))

    def __str__(self):
        return 'X_{{{0},{1},{2},{3}}}'.format(
            self[0], self[1], self[2], self[3])

    def __repr__(self):
        return self.__str__()

    def as_mathematica(self):
        '''
        Get a string of mathematica code that can represent the Crossing
        in mathematica's knot library.

        The mathematica code won't be valid if any lines of self are None.

        :rtype: str
        '''
        return 'X[{}, {}, {}, {}]'.format(
            self[0], self[1], self[2], self[3])

    def __hash__(self):
        return tuple(self).__hash__()

    def update_line_number(self, old, new):
        '''
        Replaces all instances of the given line number in self.

        Parameters
        ----------
        old : int
            The old line number
        new : int
            The number to replace it with
        '''
        for i in range(4):
            if self[i] == old:
                self[i] = new

    def is_incoming(self, index):
        if index not in self:
            raise ValueError('arc index doesn\'t intersect crossing')
        other_index = self[(self.index(index) + 2) % 4]
        if other_index == 1 and index != 2:  # => index is the final arc
            return True
        if index == 1 and other_index != 2:  # => other_index is the final arc
            return False
        if other_index > index:
            return True
        return False

    def is_outgoing(self, index):
        return not self.is_incoming(index)

def shorthand_to_crossings(s):
    '''
    Takes a planar diagram shorthand string, and returns a list of
    :class:`Crossing`s.
    '''
    crossings = []
    cs = s.split(' ')
    for entry in cs:
        entry = entry.split('_')
        if entry[0] == 'X':
            a, b, c, d = [int(j) for j in entry[1].split(',')]
            crossings.append(Crossing(a, b, c, d))
        elif entry[0] == 'P':
            a, b = [int(j) for j in entry[1].split(',')]
            crossings.append(Point(a, b))
    return crossings


def gausscode_to_crossings(gc):
    cl = gc._gauss_code
    crossings = []
    incomplete_crossings = {}
    line_lengths = [len(line) for line in cl]
    total_lines = sum(line_lengths)
    line_indices = [1] + list(n.cumsum(line_lengths)[:-1] + 1)

    curline = 1
    for i, line in enumerate(cl):
        curline = line_indices[i]
        for index, over, clockwise in line:
            if index in incomplete_crossings:
                crossing = incomplete_crossings.pop(index)
            else:
                crossing = Crossing()

            inline = curline
            curline += 1
            if curline >= (line_indices[i] + line_lengths[i]):
                curline = line_indices[i]
            outline = curline

            if over == -1:
                crossing[0] = inline
                crossing[2] = outline
                crossings.append(crossing)
            else:
                if clockwise == 1:
                    crossing[3] = inline
                    crossing[1] = outline
                else:
                    crossing[1] = inline
                    crossing[3] = outline

            if not crossing.valid():
                incomplete_crossings[index] = crossing

    return crossings
