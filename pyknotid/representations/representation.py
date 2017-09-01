'''
Representation
==============

An abstract representation of a Knot, providing methods for
the calculation of topological invariants.

API documentation
~~~~~~~~~~~~~~~~~
'''

from __future__ import print_function, division
from pyknotid.representations.gausscode import GaussCode
from collections import defaultdict
import numpy as n


class Representation(GaussCode):
    '''
    An abstract representation of a knot or link. Internally
    this is just a Gauss code, but it exposes extra topological methods
    and may in future be refactored to work differently.
    '''

    @classmethod
    def calculating_orientations(cls, code, **kwargs):
        gc = super(Representation, cls).calculating_orientations(code)
        return Representation(gc, **kwargs)

    def gauss_code(self):
        return GaussCode(self)
    
    def planar_diagram(self):
        from pyknotid.representations.planardiagram import PlanarDiagram
        return PlanarDiagram(self)

    def alexander_polynomial(self, variable=-1, quadrant='lr',
                             mode='python', force_no_simplify=False):
        '''
        Returns the Alexander polynomial at the given point,
        as calculated by :func:`pyknotid.invariants.alexander`.

        See :func:`pyknotid.invariants.alexander` for the meanings
        of the named arguments.
        '''
        from ..invariants import alexander
        if not force_no_simplify:
            self.simplify()
        return alexander(self, variable=variable, quadrant=quadrant,
                         simplify=False, mode=mode)

    def jones_polynomial(self, variable=-1, simplify=True):
        if simplify:
            self.simplify()
        from pyknotid.invariants import jones_polynomial
        p = self.planar_diagram()
        return jones_polynomial(p)

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

    def vassiliev_degree_2(self, simplify=True):
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
        if simplify:
            self.simplify()
        return vassiliev_degree_2(self)

    def vassiliev_degree_3(self, simplify=True, try_cython=True):
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
        if simplify:
            self.simplify()
        return vassiliev_degree_3(self, try_cython=try_cython)

    def virtual_vassiliev_degree_3(self):
        '''Returns the virtual Vassiliev invariant of degree 3 for the
        representation.

        '''
        from ..invariants import virtual_vassiliev_degree_3
        return virtual_vassiliev_degree_3(self)

    def hyperbolic_volume(self):
        '''
        Returns the hyperbolic volume at the given point, via
        :meth:`pyknotid.representations.PlanarDiagram.as_spherogram`.
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

    def identify(self, determinant=True, vassiliev_2=True,
                 vassiliev_3=None, alexander=False, roots=(2, 3, 4),
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
        vassiliev_2 : bool
            If True, uses the Vassiliev invariant of order 2. Defaults to True.
        vassiliev_3 : bool
            If True, uses the Vassiliev invariant of order 3. Defaults to None,
            which means the invariant is used only if the representation has
            less than 30 crossings.
        '''
        if not roots:
            roots = []
        roots = set(roots)
        if determinant:
            roots.add(2)

        if len(self) < 30 and vassiliev_3 is None:
            vassiliev_3 = True

        identify_kwargs = {}
        for root in roots:
            identify_kwargs[
                'alex_imag_{}'.format(root)] = self.alexander_at_root(root)

        if vassiliev_2:
            identify_kwargs['v2'] = self.vassiliev_degree_2()
        if vassiliev_3:
            identify_kwargs['v3'] = self.vassiliev_degree_3()

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

    def is_virtual(self):
        '''
        Takes an open curve and checks (for the default projection) if its 
        Gauss code corresponds to a virtual knot or not. Returns a Boolean of 
        this information.

        Returns
        -------
        virtual : bool
            True if the Gauss code corresponds to a virtual knot. False
            otherwise.
        '''
        if len(self._gauss_code) == 0:
            return False
        if len(self._gauss_code[0]) == 0:
            return False
        gauss_code = self._gauss_code[0][:, 0]
        l = len(gauss_code)
        total_crossings = l / 2
        crossing_counter = 1
        virtual = False
        
        for crossing_number in self.crossing_numbers:
            occurences = n.where(gauss_code == crossing_number)[0]
            first_occurence = occurences[0]
            second_occurence = occurences[1]
            crossing_difference = second_occurence - first_occurence        
                  
            if crossing_difference % 2 == 0:
                return True
        return False

    def self_linking(self):
        '''Returns the self linking number J(K) of the Gauss code, an
        invariant of virtual knots. See Kauffman 2004 for more information.

        Returns
        -------
        slink_counter : int
            The self linking number of the open curve
        '''
        from ..invariants import self_linking
        return self_linking(self)

    def writhe(self):
        writhe = 0

        return int(n.round(n.sum([n.sum(l[:, -1]) for l in self._gauss_code])/2.))

    def slip_triangle(self, func):

        code = self._gauss_code[0]

        length = len(self)
        array = n.ones((len(self) + 1, len(self) + 1)) * -0
        
        for i in range(length + 1):
            for j in range(length + 1):
                if i + j > length:
                    continue
                new_gc = Representation(self)

                for _ in range(i):
                    new_gc._remove_crossing(new_gc._gauss_code[0][0, 0])
                for _ in range(j):
                    new_gc._remove_crossing(new_gc._gauss_code[0][-1, 0])

                invariant = func(new_gc)

                array[-1*(i + 1), j] = invariant

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(array, interpolation='none', cmap='jet')

        ticks = range(length + 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks])

        ax.plot([0, length+1], [0, length+1], color='black', linewidth=2)
        ax.set_xlim(-0.5, length+0.5)
        ax.set_ylim(length+0.5, -0.5)

        
        fig.show()

        return array, fig, ax

    def _construct_planar_graph(self):
        pd = self.planar_diagram()
        g, duplicates, heights, first_edge = pd.as_networkx_extended()

        import planarity

        pg = planarity.PGraph(g)
        pg.embed_drawplanar()
        g = planarity.networkx_graph(pg)


        node_labels = {}
        xs = []
        ys = []

        nodes_by_height = {}
        node_xs_by_y = {}
        node_xs_ys = {}
        node_lefts_rights = {}

        for node, data in g.nodes(data=True):
            y = data['pos']
            xb = data['start']
            xe = data['end']
            x = int((xe + xb) / 2.)

            node_labels[node] = (x, y)
            xs.extend([xb, xe])
            ys.append(y)

            nodes_by_height[data['pos']] = node
            node_xs_by_y[data['pos']] = x
            node_xs_ys[node] = (x, y)
            node_lefts_rights[node] = (xb, xe)

        lines = []

        rightmost_x = n.max(xs)
        leftmost_x = n.min(xs)
        x_span = rightmost_x - leftmost_x
        safe_yshift = 0.5 / x_span

        extra_x_shifts = []
        
        for n1, n2, data in g.edges(data=True):
            x = data['pos']
            yb = data['start']
            ye = data['end']

            start_node = nodes_by_height[yb]
            end_node = nodes_by_height[ye]
            if start_node >= len(self) and end_node >= len(self):
                continue

            start_left, start_right = node_lefts_rights[start_node]
            end_left, end_right = node_lefts_rights[end_node]

            start_frac = n.abs((x - start_left) / (start_right - start_left) - 0.5)
            start_frac = 0.5 - start_frac
            if True:  # ye < ys:  # This always evaluated to True - a bug?
                start_frac *= -1
            start_shift = start_frac

            end_frac = n.abs((x - end_left) / (end_right - end_left) - 0.5)
            end_frac = 0.5 - end_frac
            if False:  # ye > ys:  # This always evaluated to False - a bug?
                end_frac *= -1
            end_shift = end_frac

            start_node_x = node_xs_by_y[yb]
            start_node_y = yb
            
            end_node_x = node_xs_by_y[ye]
            end_node_y = ye

            line = n.array([[start_node_x, start_node_y],
                            [x, start_node_y - start_shift],
                            [x, end_node_y - end_shift],
                            [end_node_x, end_node_y]])
            if x == start_node_x:
                line = line[1:]
                line[0, 1] = start_node_y
            if x == end_node_x:
                line = line[:-1]
                line[-1, 1] = end_node_y
            lines.append(line)

            if sorted((n1, n2)) in duplicates:
                line = line.copy()
                n1x, n1y = node_xs_ys[n1]
                n2x, n2y = node_xs_ys[n2]

                lx, hx = sorted([n1x, n2x])
                ly, hy = sorted([n1y, n2y])
                if len(line) == 4:
                    join_1 = n.array([line[2, 0], line[2, 1], 0]) - n.array([line[0, 0], line[0, 1], 0])
                    normal_1 = n.cross(join_1, [0, 0, 1])[:2]
                    normal_1 /= n.linalg.norm(normal_1)

                    join_2 = n.array([line[3, 0], line[3, 1], 0]) - n.array([line[1, 0], line[1, 1], 0])
                    normal_2 = n.cross(join_2, [0, 0, 1])[:2]
                    normal_2 /= n.linalg.norm(normal_2)

                    extra_x_shifts.append(line[1][0] + 0.005 * normal_1[0])

                    line[1] += 0.01*normal_1
                    line[2] += 0.01*normal_2

                elif len(line) == 3:
                    join_1 = n.array([line[2, 0], line[2, 1], 0]) - n.array([line[0, 0], line[0, 1], 0])
                    normal_1 = n.cross(join_1, [0, 0, 1])[:2]
                    normal_1 /= n.linalg.norm(normal_1)

                    extra_x_shifts.append(line[1][0] + 0.005 * normal_1[0])

                    line[1] += 0.01*normal_1
                elif len(line) == 2:
                    join_1 = n.array([line[1, 0], line[1, 1], 0]) - n.array([line[0, 0], line[0, 1], 0])
                    normal_1 = n.cross(join_1, [0, 0, 1])[:2]
                    normal_1 /= n.linalg.norm(normal_1)

                    line = n.vstack([line[0], line[0] + 0.5*(line[1] - line[0]) + 0.01*normal_1, line[1]])

                    extra_x_shifts.append(line[1][0] - 0.005 * normal_1[0])
                
                    
                lines.append(line)

        extra_x_shifts = sorted(extra_x_shifts)[::-1]

        return g, lines, node_labels, nodes_by_height, (leftmost_x, rightmost_x), first_edge, heights, extra_x_shifts

    def draw_planar_graph(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection

        g, lines, node_labels, nodes_by_height, xlims, first_edge, heights, extra_x_shifts = self._construct_planar_graph()
        leftmost_x, rightmost_x = xlims

        patches = []
        for node, data in g.nodes(data=True):
            y = data['pos']
            xb = data['start']
            xe = data['end']
            x = int((xe + xb) / 2.)
            patches.append(Circle((x, y), 0.25))
        
        plt.ion()

        fig, ax = plt.subplots()
        p = PatchCollection(patches, facecolor='none')
        ax.add_collection(p)
        # plt.axis('equal')

        for node, (x, y) in node_labels.items():
            plt.text(x, y, node,
                     horizontalalignment='center',
                     verticalalignment='center')

        for line in lines:
            ax.plot(line[:, 0], line[:, 1], linewidth=1)
        
        ax.set_xlim(leftmost_x - 1, rightmost_x + 1)
        fig.show()
        
    def space_curve(self, **kwargs):
        from pyknotid.spacecurves import Knot
        self.simplify()

        if len(self) == 0:
            thetas = n.linspace(0, 2*n.pi, 10)
            xs = n.sin(thetas) * 3
            ys = n.cos(thetas) * 3
            zs = n.zeros(10)
            return Knot(n.vstack((xs, ys, zs)).T)
        
        # self.draw_planar_graph()
        g, lines, node_labels, nodes_by_height, xlims, first_edge, heights, extra_x_shifts = self._construct_planar_graph()
        leftmost_x, rightmost_x = xlims

        cg = CrossingGraph()
        for line in lines:
            start_node = nodes_by_height[n.int(n.round(line[0, 1]))]
            end_node = nodes_by_height[n.int(n.round(line[-1, 1]))]
            cl = CrossingLine(start_node, end_node, line)
            cg[start_node].append(cl)
            cg[end_node].append(cl.reversed())

        cg.assert_four_valency()

        cg.align_nodes()
        first_node = 0
        next_node = 1

        points = cg.retrieve_space_curve(
            first_edge[0], first_edge[1], first_edge[2], heights)

        for shift in extra_x_shifts:
            for i in range(len(points)):
                cur_x = points[i, 0]
                if cur_x > shift:
                    points[i, 0] += 1.

        ys = sorted(set(points[:, 1].tolist()))
        for i, y in enumerate(ys):
            points[:, 1][points[:, 1] > y + i + 0.0000001] += 1.

        points[:, 2] *= 1.5

        k = Knot(points*5, verbose=self.verbose)
        k.zero_centroid()
        k.rotate((0.05, 0.03, 0.02))
        return k


class CrossingLine(object):
    def __init__(self, start, end, points):
        self.start = start
        self.end = end
        self.points = points

    def reversed(self):
        return CrossingLine(self.end, self.start, self.points[::-1])

    def __str__(self):
        return '<CrossingLine joining {} with {}>'.format(
            self.start, self.end)

    def __repr__(self):
        return '<CrossingLine joining {} with {}>'.format(
            self.start, self.end)

class CrossingGraph(defaultdict):
    def __init__(self):
        super(CrossingGraph, self).__init__(list)

    def number_of_crossings(self):
        return int(len(self) / 3)
        
    def assert_four_valency(self):
        return True
        for key, value in self.items():
            if len(value) != 4:
                raise ValueError('CrossingGraph is not 4-valent')

    def align_nodes(self):
        '''Orders the lines of each node to be in order, clockwise, depending
        on their incoming angle.
        '''
        for key, value in self.items():
            self[key] = sorted(
                value, key=lambda l: n.arctan2(l.points[1, 1] - l.points[0, 1],
                                               l.points[1, 0] - l.points[0, 0]))

    def retrieve_space_curve(self, first, next, initial_arc_number, heights):
        first_node_lines = self[first]
        for line in first_node_lines:
            if line.end == next:
                break
        else:
            raise ValueError('Node {} is not connected to node {}'.format(first, next))

        possible_start_lines = [l for l in first_node_lines if (l.end == next)]
        if len(possible_start_lines) not in (1, 2):
            raise ValueError('Invalid number of start lines: {}'.format(
                len(possible_start_lines)))

        if len(possible_start_lines) == 1:
            return self._retrieve_space_curve(possible_start_lines[0],
                                              initial_arc_number,
                                              heights)
        else:

            print('possible starts:')
            for line in possible_start_lines:
                print(line)
            
            try:
                return self._retrieve_space_curve(possible_start_lines[0],
                                                  initial_arc_number,
                                                  heights)
            except KeyError as err:
                return self._retrieve_space_curve(possible_start_lines[1],
                                                  initial_arc_number,
                                                  heights)

    def _retrieve_space_curve(self, line, initial_arc_number, heights):
        current_line = line
        segments = []
        h = 1.
        arc_number = initial_arc_number
        for _ in range(4*self.number_of_crossings()):
        # for _ in range(len(self)*2):
            current_points = current_line.points.copy()
            ps = n.zeros((len(current_points), 3))
            ps[:, :-1] = current_points

            height = heights[(current_line.start, current_line.end, arc_number)]
            ps[0, -1] = height[0] # height
            # ps[0, -1] = h; h *= -1.

            segments.append(ps[:-1])

            next_lines = self[current_line.end]
            incoming_angle = n.arctan2(current_points[-2, 1] - current_points[-1, 1],
                                       current_points[-2, 0] - current_points[-1, 0])

            other_incoming_angles = [
                n.arctan2(l.points[1, 1] - l.points[0, 1],
                          l.points[1, 0] - l.points[0, 0]) for l in next_lines]

            angle_distances = [
                angle_distance(angle, incoming_angle)
                for angle in other_incoming_angles]

            incoming_index = n.argmin(angle_distances)
            if len(angle_distances) == 4:
                outgoing_index = (incoming_index + 2) % 4 
            elif len(angle_distances) == 2:
                outgoing_index = (incoming_index + 1) % 2
            else:
                raise ValueError('Encountered node with neither 2 nor 4 arcs, should be impossible')
            current_line = next_lines[outgoing_index]

            # This is the old arc number calculation before adding intermediates
            # arc_number = (arc_number % (len(self) * 2)) + 1

            if len(angle_distances) == 4:
                arc_number = (arc_number % (2*self.number_of_crossings())) + 1

        return n.vstack(segments)

def angle_distance(a1, a2):
    dist = n.abs(a2 - a1)
    if dist > n.pi:
        dist = 2*n.pi - dist
    return dist

