from __future__ import print_function

import numpy as n

from pyknot2.utils import ensure_shape_tuple, vprint
from periodiccell import _cram_into_cell, _cut_line_at_jumps, _interpret_line

from collections import defaultdict

try:
    from pyknot2.spacecurves import chelpers
except ImportError:
    from pyknot2.spacecurves import helpers as chelpers

ROTATION_MAGIC_NUMBERS = (1., 0.)

class PeriodicKnot(object):
    def __init__(self, points, period_vector=None):
        self.points = points
        self._period_vector = n.array(period_vector)

        from pyknot2.spacecurves.rotation import rotate_to_top
        self._apply_matrix(rotate_to_top(*ROTATION_MAGIC_NUMBERS))

        if self._period_vector is not None:
            self._period_vector = rotate_to_top(*ROTATION_MAGIC_NUMBERS).dot(self._period_vector)

    @property
    def period_vector(self):
        if self._period_vector is not None:
            return self._period_vector
        return self.points[-1] - self.points[0]

    def roll(self, num):
        if n.abs(num) > 1:
            raise Exception('roll does not work properly for rolls bigger than 1 yet')
        pv = self.period_vector
        ps = self.points

        if num > 0:
            ps = n.vstack((ps[1:], [ps[1] + pv]))
        elif num < 0:
            ps = n.vstack(([ps[-2] - pv], ps[:-1]))
        self.points = ps

    def plot_with_translation(self, with_translation=0, **kwargs):
        from pyknot2.spacecurves import OpenKnot, Link
        if with_translation == 0:
            OpenKnot(self.points).plot(**kwargs)
        else:
            Link([self.points,
                  self.translated_points(with_translation)]).plot(
                      colours=['red', 'blue'])

    def plot(self, num_translations=3, colour_minimal_unfoldings=True,
             **kwargs):
        from pyknot2.spacecurves import OpenKnot, Link
        if not colour_minimal_unfoldings:
            points = self.points_with_translations(num_translations)
            k = OpenKnot(points)
            k.plot(**kwargs)
            return k
        lines = []
        for i in range(-num_translations, num_translations+1):
            lines.append(self.translated_points(i))
        l = Link(lines)
        l.plot(**kwargs)
        return l

    def translated_points(self, num):
        return self.points + num * self.period_vector

    def plot_projection_with(self, num_translations=0, mat=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        points = self.points
        translated_points = self.translated_points(num_translations)
        if mat is not None:
            points = n.apply_along_axis(mat.dot, 1, points)
            translated_points = n.apply_along_axis(
                mat.dot, 1, translated_points)
        ax.plot(points[:, 0], points[:, 1], color='purple', linewidth=1.5)
        ax.plot(translated_points[:, 0],
                translated_points[:, 1], color='green')
        fig.show()
        return fig, ax

    def points_with_translations(self, num_translations=3):
        if num_translations == 0:
            return self.points
        new_points = []
        for i in range(-num_translations, num_translations):
            new_points.append(self.translated_points(i)[:-1])
        new_points.append(self.translated_points(num_translations))
        points = n.vstack(new_points)
        return points

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

    def plot_projection(self, num_translations=3):
        from pyknot2.visualise import plot_line, plot_projection
        points = self.points_with_translations(num_translations)
        crossings, core_num, core_index = self.raw_crossings(num_translations)
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
        core_points = points[num_translations * (len(self.points) - 1):
                             num_translations * (len(self.points) - 1) + len(self.points)]
        # ax.plot(points[:, 0], points[:, 1], 'o', color='green')
        ax.plot(core_points[:, 0], core_points[:, 1], color='purple', linewidth=1.5)
        return fig, ax

    def raw_crossings(self, num_translations=3):
        from pyknot2.spacecurves.rotation import rotate_to_top

        points = self.points_with_translations(num_translations)

        num_points = len(points)
        core_num = len(self.points) - 1
        core_index = num_translations * core_num 

        segment_lengths = n.roll(points[:, :2], -1, axis=0) - points[:, :2]
        segment_lengths = n.sqrt(n.sum(segment_lengths**2, axis=1))
        max_segment_length = n.max(segment_lengths[:-1])
        jump_mode = 2

        core_points = points[(core_index):(core_index + core_num + 1)]
        core_segment_lengths = segment_lengths[core_index:(core_index + core_num + 1)]

        first_block = points[:(core_index + 1)]
        first_block_lengths = segment_lengths[:(core_index + 1)]

        second_block = points[(core_index + core_num):]
        second_block_lengths = segment_lengths[(core_index + core_num):]
        
        crossings = []
        for i in range(len(core_points) - 1):
            v0 = core_points[i]
            dv = core_points[i+1] - v0

            vnum = core_index + i

            compnum = 0
            crossings.extend(chelpers.find_crossings(
                v0, dv, first_block, first_block_lengths,
                vnum, compnum,
                max_segment_length,
                jump_mode))

            compnum = core_index + core_num
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

    def gauss_code(self, num_translations=3):
        crossings, core_num, core_index = self.raw_crossings(num_translations)

        equivalent_crossing_indices = get_equivalent_crossing_indices(
            crossings, len(self.points) - 1)

        from pyknot2.representations import GaussCode
        gc = GaussCode(crossings)
        if num_translations == 0:
            return gc, defaultdict(set)

        equivalent_crossing_numbers = get_equivalent_crossing_numbers(
            equivalent_crossing_indices, gc._gauss_code[0])

        return gc, equivalent_crossing_numbers
    

def get_equivalent_crossing_indices(crossings, span):
    equivalencies = defaultdict(set)
    for i, row1 in enumerate(crossings):
        for j, row2 in enumerate(crossings[i+1:], i+1):
            if n.abs((row1[0] - row2[0])) % span < 0.000001:
                equivalencies[i].add(j)
                for val in equivalencies[i]:
                    equivalencies[val].add(j)
                    equivalencies[j].add(val)
                equivalencies[j].add(i)
    for key, value in equivalencies.items():
        try:
            value.remove(key)
        except KeyError:
            pass
    return equivalencies


def get_equivalent_crossing_numbers(indices, gauss_code):
    equivalent_crossing_numbers = defaultdict(set)
    for key, values in indices.items():
        number = gauss_code[key][0]
        equivalent_crossing_numbers[number] = set([int(gauss_code[value][0]) for value in values])
        equivalent_crossing_numbers[number].add(number)
    return equivalent_crossing_numbers


class CellKnot(object):
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

    def raw_crossings_by_unfolding(self, num_translations=0, shift=0):
        from pyknot2.spacecurves.rotation import rotate_to_top
        points = self.unfolded_points_with_translations(
            num_translations, rotate_to_top(*ROTATION_MAGIC_NUMBERS))

        num_points = len(points)
        core_num = len(self.unfolded_points())
        core_index = num_translations * core_num + shift

        segment_lengths = n.roll(points[:, :2], -1, axis=0) - points[:, :2]
        segment_lengths = n.sqrt(n.sum(segment_lengths**2, axis=1))
        max_segment_length = n.max(segment_lengths[:-1])
        jump_mode = 2

        core_points = points[(core_index):(core_index + core_num + 1)]
        core_segment_lengths = segment_lengths[(core_index + shift):(core_index + shift + core_num + 1)]

        first_block = points[:(core_index + 1)]
        first_block_lengths = segment_lengths[:(core_index + 1)]

        second_block = points[(core_index + core_num):]
        second_block_lengths = segment_lengths[(core_index + core_num):]
        
        crossings = []
        for i in range(len(core_points) - 1):
            v0 = core_points[i]
            dv = core_points[i+1] - v0

            vnum = core_index + i

            compnum = 0
            crossings.extend(chelpers.find_crossings(
                v0, dv, first_block, first_block_lengths,
                vnum, compnum,
                max_segment_length,
                jump_mode))

            compnum = core_index + core_num
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

    def plot_projection_by_unfolding(self, num_translations=0, shift=0):
        from pyknot2.visualise import plot_line, plot_projection
        from pyknot2.spacecurves.rotation import rotate_to_top
        points = self.unfolded_points_with_translations(
            num_translations, rotate_to_top(*ROTATION_MAGIC_NUMBERS))
        crossings = self.raw_crossings_by_unfolding(num_translations, shift=shift)[0]
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
        core_points = points[num_translations * len(self.unfolded_points()) + shift:
                             num_translations * len(self.unfolded_points()) + shift + len(self.unfolded_points())]
        # ax.plot(points[:, 0], points[:, 1], 'o', color='green')
        ax.plot(core_points[:, 0], core_points[:, 1], color='purple', linewidth=1.5)
        return fig, ax

    def gauss_code_by_unfolding(self, num_translations=0, mat=None, shift=None):
        '''Returns the Gauss code alongside a list of  that are
        crossings of the non-translated curve with the translated one.'''
        crossings, core_num, core_index = self.raw_crossings_by_unfolding(
            num_translations=num_translations, shift=shift)

        equivalent_crossing_indices = defaultdict(set)
        length = len(self.points)
        for i, row1 in enumerate(crossings):
            for j, row2 in enumerate(crossings[i+1:], i+1):
                # print(i, j, n.abs((row1[0] - row2[0]) % 40))
                if n.abs((row1[0] - row2[0]) % len(self.unfolded_points())) < 0.000001:
                    equivalent_crossing_indices[i].add(j)
                    for val in equivalent_crossing_indices[i]:
                        equivalent_crossing_indices[val].add(j)
                        equivalent_crossing_indices[j].add(val)
                    equivalent_crossing_indices[j].add(i)
        for key, value in equivalent_crossing_indices.items():
            try:
                value.remove(key)
            except KeyError:
                pass
        
        from pyknot2.representations import GaussCode
        gc = GaussCode(crossings)
        if num_translations == 0:
            return gc, defaultdict(set)

        equivalent_crossing_numbers = defaultdict(set)
        code = gc._gauss_code[0]
        for key, values in equivalent_crossing_indices.items():
            number = code[key][0]
            equivalent_crossing_numbers[number] = set([int(code[value][0]) for value in values])
            equivalent_crossing_numbers[number].add(number)

        translation_numbers = {}
        if shift is None:
            shift = 0
        centre_start = num_translations * len(self.unfolded_points()) + shift
        for i, row in enumerate(crossings):
            crossing_number = code[i][0]
            if crossing_number not in translation_numbers:
                position1 = row[0]
                translation1 = int(n.floor((position1 - centre_start) / (len(self.unfolded_points() + 0))))

                position2 = row[1]
                translation2 = int(n.floor((position2 - centre_start) / (len(self.unfolded_points() + 0))))

                translation_numbers[crossing_number] = (translation1, translation2)
                
        return gc, equivalent_crossing_numbers, translation_numbers
        

    def raw_crossings(self, num_translations=0, mat=None, shift=0):
        from pyknot2.spacecurves import OpenKnot
        points = self.unfolded_points()
        points = n.vstack((points[shift:], points[:(shift+1)] + self.period_vector()))
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
            if num_translations > 0:
                vnum = i
                compnum = len(points)
            else:
                vnum = len(points) + i
                compnum = 0

            inter_crossings.extend(chelpers.find_crossings(
                v0, dv, s, segment_lengths,
                vnum, compnum,
                max_segment_length,
                jump_mode
                ))
        # if len(inter_crossings):
        #     inter_crossings = n.array(inter_crossings)
        #     inter_crossings[::2, 1] += len(points)
        #     inter_crossings[1::2, 0] += len(points)
        #     inter_crossings = inter_crossings.tolist()

        if num_translations < 0:
            self_crossings = n.array(self_crossings)
            self_crossings[:, :2] += len(points)
            self_crossings = self_crossings.tolist()

        all_crossings = self_crossings + inter_crossings # + other_crossings
        all_crossings.sort(key=lambda j: j[0])

        return n.array(all_crossings)

    def gauss_code(self, num_translations=0, mat=None, shift=0):
        '''Returns the Gauss code alongside a list of  that are
        crossings of the non-translated curve with the translated one.'''
        crossings = self.raw_crossings(num_translations=num_translations, mat=mat, shift=shift)
        from pyknot2.representations import GaussCode
        gc = GaussCode(crossings)
        if num_translations == 0:
            return gc, gc.crossing_numbers

        admissible_indices = set()
        for line_index, gauss_index in zip(crossings[:, 0], gc._gauss_code[0][:, 0]):
            if num_translations > 0:
                if line_index > len(self.unfolded_points()):
                    admissible_indices.add(gauss_index)
            elif num_translations < 0:
                if line_index < len(self.unfolded_points()):
                    admissible_indices.add(gauss_index)
        return gc, list(admissible_indices)
        

    def plot_projection(self, num_translations=0, mat=None, shift=0):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        points = self.unfolded_points()
        points = n.vstack((points[shift:], points[:(shift+1)] + self.period_vector()))
        translated_points = points + num_translations * self.period_vector()
        if mat is not None:
            points = n.apply_along_axis(mat.dot, 1, points)
            translated_points = n.apply_along_axis(mat.dot, 1, translated_points)
        ax.plot(points[:, 0], points[:, 1])
        ax.plot(translated_points[:, 0], translated_points[:, 1])
        fig.show()
        return fig, ax

    def plot(self, num_translations=0, mat=None, shift=0):
        import matplotlib.pyplot as plt
        points = self.unfolded_points()
        points = n.vstack((points[shift:], points[:(shift+1)] + self.period_vector()))
        translated_points = points + num_translations * self.period_vector()
        if mat is not None:
            points = n.apply_along_axis(mat.dot, 1, points)
            translated_points = n.apply_along_axis(mat.dot, 1, translated_points)
        from pyknot2.spacecurves import Link
        Link([points, translated_points]).plot()
        
    def periodic_vassiliev_degree_2(self, num_translations=3,
                                    number_of_samples=100,
                                    shift=0):
        from pyknot2.spacecurves.rotation import get_rotation_angles, rotate_to_top            
        angles = get_rotation_angles(number_of_samples)
        v2s = []
        v2_arrs = []
        for translations in range(-1*num_translations, num_translations + 1):
            vprint('Checking translation {}'.format(translations), False)
            v2s_cur = []
            for theta, phi in angles:
                matrix = rotate_to_top(theta + 0.13, phi + 0.43)
                gc, numbers = self.gauss_code(num_translations=translations,
                                              mat=matrix, shift=shift)
                v2s_cur.append(periodic_vassiliev_degree_2(gc, numbers))
                # print(theta + 0.13, phi + 0.43, gc, numbers)
            print()
            print('translation {:03}, v2s_cur {}'.format(translations, v2s_cur))
            v2s.append(n.average(v2s_cur))
            v2_arrs.append(v2s_cur)
        vprint()
        print('totals are', n.sum(v2_arrs, axis=0))
        return n.sum(v2s)

    def periodic_vassiliev_degree_2_by_unfolding(self, num_translations=3, shift=0):
        gc, equivalencies, translation_indices = self.gauss_code_by_unfolding(num_translations, shift=shift)
        from pyknot2.invariants import vassiliev_degree_2
        return periodic_vassiliev_degree_2_without_double_count(gc, equivalencies, translation_indices)
        
    def periodic_vassiliev_degree_3_by_unfolding(self, num_translations=3, shift=0):
        gc = self.gauss_code_by_unfolding(num_translations, shift=shift)[0]
        from pyknot2.invariants import vassiliev_degree_3
        return vassiliev_degree_3(gc)

def periodic_vassiliev_degree_2_without_double_count(representation, equivalent_crossing_numbers={},
                                                     translation_indices={}):
    # Hacky periodic version of the vassiliev function in
    # pyknot2.invariants
    from pyknot2.invariants import _crossing_arrows_and_signs

    # print('gc is', representation)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')

    gc = gc[0]
    arrows, signs = _crossing_arrows_and_signs(
        gc, representation.crossing_numbers)

    crossings_already_done = set()
    
    crossing_numbers = list(representation.crossing_numbers)
    representations_sum = 0
    for index, i1 in enumerate(crossing_numbers):
        arrow1 = arrows[i1]
        a1s, a1e = arrow1
        for i2 in crossing_numbers[index+1:]:
            # if (i1 not in relevant_crossing_numbers and
            #     i2 not in relevant_crossing_numbers):
            #     continue
            if tuple(sorted([i1, i2])) in crossings_already_done:
                continue
            arrow2 = arrows[i2]
            a2s, a2e = arrow2

            if a2s > a1s and a2e < a1s and a1e > a2s:

                ti1 = translation_indices[i1]
                ti2 = translation_indices[i2]
                
                representations_sum += signs[i1] * signs[i2]

                print('vass with', i1, i2, signs[i1] * signs[i2], 'indices',
                      translation_indices[i1], translation_indices[i2])
                for i1other in equivalent_crossing_numbers[i1].union({i1}):
                    for i2other in equivalent_crossing_numbers[i2].union({i2}):
                        crossings_already_done.add(tuple(sorted([i1other, i2other])))
            elif a1s > a2s and a1e < a2s and a2e > a1s:
                representations_sum += signs[i1] * signs[i2]

                print('vass with', i1, i2, '...', signs[i1] * signs[i2], 'indices',
                      translation_indices[i1], translation_indices[i2])
                for i1other in equivalent_crossing_numbers[i1].union({i1}):
                    for i2other in equivalent_crossing_numbers[i2].union({i2}):
                        crossings_already_done.add(tuple(sorted([i1other, i2other])))

    return representations_sum

def periodic_vassiliev_degree_3_without_double_count(representation,
                                                     equivalent_crossing_numbers):
    ## See Polyak and Viro
    from pyknot2.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')

    gc = gc[0]
    arrows, signs = _crossing_arrows_and_signs(
        gc, representation.crossing_numbers)
    
    crossings_already_done = set()

    crossing_numbers = list(representation.crossing_numbers)
    used_sets = set()
    representations_sum_1 = 0
    representations_sum_2 = 0
    for index, i1 in enumerate(crossing_numbers):
        if index % 10 == 0:
            vprint('\rCurrently comparing index {}'.format(index), False)
        arrow1 = arrows[i1]
        a1s, a1e = arrow1
        a1e = (a1e - a1s) % len(gc)
        for i2 in crossing_numbers:
            arrow2 = arrows[i2]
            a2s, a2e = arrow2
            a2s = (a2s - a1s) % len(gc)
            a2e = (a2e - a1s) % len(gc)
            for i3 in crossing_numbers:
                arrow3 = arrows[i3]
                a3s, a3e = arrow3
                a3s = (a3s - a1s) % len(gc)
                a3e = (a3e - a1s) % len(gc)

                ordered_indices = tuple(sorted((i1, i2, i3)))
                if ordered_indices in used_sets:
                    continue

                if (a2s < a1e and a3e < a1e and a3e > a2s and
                    a3s > a1e and a2e > a3s):
                    representations_sum_1 += (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                if (a2e < a1e and a3s < a1e and a3s > a2e and
                    a2s > a1e and a3e > a2s):
                    representations_sum_2 += (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)

    print 
    
    return int(round(representations_sum_1 / 2.)) + representations_sum_2

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
            if (i1 not in relevant_crossing_numbers and
                i2 not in relevant_crossing_numbers):
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
    
