from __future__ import print_function

import numpy as n

from pyknot2.utils import ensure_shape_tuple, vprint
from periodiccell import _cram_into_cell, _cut_line_at_jumps, _interpret_line

try:
    from pyknot2.spacecurves import chelpers
except ImportError:
    from pyknot2.spacecurves import helpers as chelpers

class PeriodicKnot(object):
    def __init__(self, points, shape, period=None):
        # points will automatically be folded
        points = _interpret_line(points)

        self.shape = n.array(ensure_shape_tuple(shape))
        self.points = _cut_line_at_jumps(points, self.shape)

        if period is None:
            period = (n.round((self.unfolded_points()[-1] - self.points[0][0]) /
                              self.shape)) 
        self.period = period
    
    @classmethod
    def folding(cls, points, shape, origin=(0, 0, 0.), **kwargs):
        '''Return an instance of PeriodicKnot resulting from folding the
        points within the given shape.'''
        origin = n.array(origin)
        shape = ensure_shape_tuple(shape)

        points -= origin
        points[:, 0] %= shape[0]
        points[:, 1] %= shape[1]
        points[:, 2] %= shape[2]
        return cls(points, shape, **kwargs)

    def plot(self, boundary=True, clf=True, tube_radius=1.0,
             **kwargs):
        from pyknot2.visualise import plot_cell
        boundary = self.shape if boundary else None
        plot_cell([self.points], boundary=boundary, clf=clf,
                  tube_radius=tube_radius, **kwargs)

    def unfolded_points(self):
        points = [self.points[0]]
        current_shift = n.array([0., 0., 0.])
        for i, segment in enumerate(self.points[1:], 1):
            segment = segment - current_shift * self.shape
            diff = segment[0] - points[-1][-1]
            diff = n.round(diff / self.shape)
            current_shift += diff 
            points.append(segment - diff * self.shape)
        return n.vstack(points)

    def unfolded_points_with_translations(self, num=3, mat=None):
        points = []
        unfolded = self.unfolded_points()
        for i in range(-num, num+1):
            points.append(unfolded + self.period_vector() * i)
        points = n.vstack(points)
        if mat is not None:
             points = n.apply_along_axis(mat.dot, 1, points)
        return n.vstack(points)


    def translate(self, distance):
        distance = n.array(ensure_shape_tuple(distance))
        points = self.unfolded_points() + distance
        points = _fold(points, self.shape)
        self.points = _cut_line_at_jumps(points, self.shape)
        self._fix_endpoints()
 
    def _fix_endpoints(self):
        if len(self.points) < 2:
            return
        closing_vector = self.points[-1][-1] - self.points[0][0]
        if n.all(closing_vector < 0.9*self.shape):
            first = self.points[0]
            self.points = self.points[1:]
            last = self.points.pop()
            self.points.append(n.vstack((last, first)))
            
    def period_vector(self):
        return self.period * self.shape

    def raw_crossings_by_unfolding(self, num_translations=0):
        from pyknot2.spacecurves.rotation import rotate_to_top
        points = self.unfolded_points_with_translations(
            num_translations, rotate_to_top(1.1, 0.))

        num_points = len(points)
        core_num = len(self.unfolded_points())
        core_index = num_translations * core_num

        segment_lengths = n.roll(points[:, :2], -1, axis=0) - points[:, :2]
        segment_lengths = n.sqrt(n.sum(segment_lengths**2, axis=1))
        max_segment_length = n.max(segment_lengths[:-1])
        jump_mode = 2

        core_points = points[core_index:(core_index + core_num + 1)]
        core_segment_lengths = segment_lengths[core_index:(core_index + core_num + 1)]

        first_block = points[:(core_index + 1)]
        first_block_lengths = segment_lengths[:(core_index + 1)]

        second_block = points[(core_index + core_num + 1):]
        second_block_lengths = segment_lengths[(core_index + core_num + 1):]
        
        crossings = []
        for i in range(len(core_points) - 1):
            v0 = core_points[i]
            dv = core_points[i+1] - v0

            vnum = core_index + i

            compnum = i
            crossings.extend(chelpers.find_crossings(
                v0, dv, first_block, first_block_lengths,
                vnum, compnum,
                max_segment_length,
                jump_mode))

            compnum = i + (core_index + core_num + 1)
            crossings.extend(chelpers.find_crossings(
                v0, dv, second_block, second_block_lengths,
                vnum, compnum,
                max_segment_length,
                jump_mode))

            compnum = core_index + i + 2
            crossings.extend(chelpers.find_crossings(
                v0, dv, core_points[i+2:], core_segment_lengths[i+2:],
                vnum, compnum,
                max_segment_length,
                jump_mode))

        crossings.sort(key=lambda j: j[0])
        return n.array(crossings), core_num, core_index

    def plot_projection_by_unfolding(self, num_translations=0):
        from pyknot2.visualise import plot_line, plot_projection
        from pyknot2.spacecurves.rotation import rotate_to_top
        points = self.unfolded_points_with_translations(
            num_translations, rotate_to_top(1.1, 0))
        crossings = self.raw_crossings_by_unfolding(num_translations)[0]
        plot_crossings = []
        for crossing in crossings:
            x, y, over, orientation = crossing
            xint = int(x)
            r = points[xint]
            dr = points[(xint+1) % len(points)] - r
            plot_crossings.append(r + (x-xint) * dr)
        fig, ax = plot_projection(points,
                                  crossings=n.array(plot_crossings),
                                  mark_start=True,
                                  show=True)
        return fig, ax

    def gauss_code_by_unfolding(self, num_translations=0, mat=None):
        '''Returns the Gauss code alongside a list of  that are
        crossings of the non-translated curve with the translated one.'''
        crossings, core_num, core_index = self.raw_crossings_by_unfolding(
            num_translations=num_translations)
        from pyknot2.representations import GaussCode
        gc = GaussCode(crossings)
        if num_translations == 0:
            return gc, gc.crossing_numbers

        admissible_indices = set()
        for line_index, gauss_index in zip(crossings[:, 0], gc._gauss_code[0][:, 0]):
            if core_index < line_index < core_index + core_num + 1:
                admissible_indices.add(gauss_index)
        return gc, list(admissible_indices)
        

    def raw_crossings(self, num_translations=0, mat=None):
        from pyknot2.spacecurves import OpenKnot
        points = self.unfolded_points()
        translated_points = points + num_translations * self.period_vector()

        if mat is not None:
            points = n.apply_along_axis(mat.dot, 1, points)
            translated_points = n.apply_along_axis(mat.dot, 1, translated_points)

        # 1) get crossings of self with self
        self_crossings = OpenKnot(points, verbose=False).raw_crossings().tolist()
        if num_translations == 0:
            return n.array(self_crossings)
        
        # # 2) get crossings of other with other
        # other_crossings = OpenKnot(translated_points).raw_crossings().tolist()
        # if len(other_crossings):
        #     other_crossings[:, :2] += len(points) 

        # 3) get crossings of self with other
        inter_crossings = []
        segment_lengths = (n.roll(translated_points[:, :2], -1, axis=0) -
                           translated_points[:, :2])
        segment_lengths = n.sqrt(n.sum(segment_lengths * segment_lengths,
                                       axis=1))
        max_segment_length = n.max(segment_lengths[:-1])
        jump_mode = 2
        for i in range(len(points) - 1):
            v0 = points[i]
            dv = points[(i+1) % len(points)] - v0

            s = translated_points
            vnum = i
            compnum = len(points)

            inter_crossings.extend(chelpers.find_crossings(
                v0, dv, s, segment_lengths[compnum:],
                vnum, compnum,
                max_segment_length,
                jump_mode
                ))
        # if len(inter_crossings):
        #     inter_crossings = n.array(inter_crossings)
        #     inter_crossings[::2, 1] += len(points)
        #     inter_crossings[1::2, 0] += len(points)
        #     inter_crossings = inter_crossings.tolist()

        all_crossings = self_crossings + inter_crossings # + other_crossings
        all_crossings.sort(key=lambda j: j[0])

        return n.array(all_crossings)

    def gauss_code(self, num_translations=0, mat=None):
        '''Returns the Gauss code alongside a list of  that are
        crossings of the non-translated curve with the translated one.'''
        crossings = self.raw_crossings(num_translations=num_translations, mat=mat)
        from pyknot2.representations import GaussCode
        gc = GaussCode(crossings)
        if num_translations == 0:
            return gc, gc.crossing_numbers

        admissible_indices = set()
        for line_index, gauss_index in zip(crossings[:, 0], gc._gauss_code[0][:, 0]):
            if line_index > len(self.unfolded_points()):
                admissible_indices.add(gauss_index)
        return gc, list(admissible_indices)
        

    def plot_projection(self, num_translations=0, mat=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        points = self.unfolded_points()
        translated_points = points + num_translations * self.period_vector()
        if mat is not None:
            points = n.apply_along_axis(mat.dot, 1, points)
            translated_points = n.apply_along_axis(mat.dot, 1, translated_points)
        ax.plot(points[:, 0], points[:, 1])
        ax.plot(translated_points[:, 0], translated_points[:, 1])
        fig.show()
        return fig, ax
        
    def periodic_vassiliev_degree_2(self, num_translations=3,
                                    number_of_samples=100):
        from pyknot2.spacecurves.rotation import get_rotation_angles, rotate_to_top            
        angles = get_rotation_angles(number_of_samples)
        v2s = []
        for translations in range(-1*num_translations, num_translations + 1):
            vprint('Checking translation {}'.format(translations), False)
            v2s_cur = []
            for theta, phi in angles:
                matrix = rotate_to_top(theta + 0.13, phi + 0.43)
                gc, numbers = self.gauss_code(num_translations=translations,
                                              mat=matrix)
                v2s_cur.append(periodic_vassiliev_degree_2(gc, numbers))
            print()
            print('translation {:03}, v2s_cur {}'.format(translations, v2s_cur))
            v2s.append(n.average(v2s_cur))
        vprint()

        return n.sum(v2s)

    def periodic_vassiliev_degree_2_by_unfolding(self, num_translations=3):
        gc = self.gauss_code_by_unfolding(3)[0]
        from pyknot2.invariants import vassiliev_degree_2
        return vassiliev_degree_2(gc)
        

def periodic_vassiliev_degree_2(representation, relevant_crossing_numbers=[]):
    # Hacky periodic version of the vassiliev function in
    # pyknot2.invariants
    from pyknot2.invariants import _crossing_arrows_and_signs

    relevant_crossing_numbers = set(relevant_crossing_numbers)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')

    gc = gc[0]
    arrows, signs = _crossing_arrows_and_signs(
        gc, representation.crossing_numbers)
    
    crossing_numbers = list(representation.crossing_numbers)
    representations_sum = 0
    for index, i1 in enumerate(crossing_numbers):
        arrow1 = arrows[i1]
        a1s, a1e = arrow1
        for i2 in crossing_numbers[index+1:]:
            if i1 not in relevant_crossing_numbers and i2 not in relevant_crossing_numbers:
                continue
            arrow2 = arrows[i2]
            a2s, a2e = arrow2

            if a2s > a1s and a2e < a1s and a1e > a2s:
                representations_sum += signs[i1] * signs[i2]
            elif a1s > a2s and a1e < a2s and a2e > a1s:
                representations_sum += signs[i1] * signs[i2]

    return representations_sum

def _fold(points, shape):
    '''Imposes the shape as periodic boundary conditions.'''
    shape = n.array(shape)
    dx, dy, dz = shape

    points = n.vstack([[points[0] + 0.001], points])
    rest = n.array(points).copy()
    new_points = []
    i = 0
    while i < len(rest) - 1:
        cur = rest[i]
        nex = rest[i+1]
        print('i is', i)
        print(cur, nex)
        if nex[0] < 0:
            new_points.append(rest[:i+1])
            boundary_point = cur + cur[0] / (cur[0] - nex[0]) * (nex - cur)
            new_points.append([boundary_point])
            rest = n.vstack([[boundary_point], rest[i+1:]])
            rest[:, 0] += dx
            i = 0
        elif nex[0] > dx:
            new_points.append(rest[:i+1])
            boundary_point = cur + (dx - cur[0]) / (nex[0] - cur[0]) * (nex - cur)
            new_points.append([boundary_point])
            rest = n.vstack([[boundary_point], rest[i+1:]])
            rest[:, 0] -= dx
            i = 0
        elif nex[1] < 0:
            new_points.append(rest[:i+1])
            boundary_point = cur + cur[1] / (cur[1] - nex[1]) * (nex - cur)
            new_points.append([boundary_point])
            rest = n.vstack([[boundary_point], rest[i+1:]])
            rest[:, 1] += dy
            i = 0
        elif nex[1] > dy:
            new_points.append(rest[:i+1])
            boundary_point = cur + (dx - cur[1]) / (nex[1] - cur[1]) * (nex - cur)
            new_points.append([boundary_point])
            rest = n.vstack([[boundary_point], rest[i+1:]])
            rest[:, 1] -= dy
            i = 0
        elif nex[2] < 0:
            new_points.append(rest[:i+1])
            boundary_point = cur + cur[2] / (cur[2] - nex[2]) * (nex - cur)
            new_points.append([boundary_point])
            rest = n.vstack([[boundary_point], rest[i+1:]])
            rest[:, 2] += dz
            i = 0
        elif nex[2] > dz:
            new_points.append(rest[:i+1])
            boundary_point = cur + (dx - cur[2]) / (nex[2] - cur[2]) * (nex - cur)
            new_points.append([boundary_point])
            rest = n.vstack([[boundary_point], rest[i+1:]])
            rest[:, 2] -= dz
            i = 0
        else:
            i += 1
    new_points.append(rest)

    print('new points', new_points)

    return n.vstack(new_points)[1:]
    
