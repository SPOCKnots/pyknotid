'''
Gauss code
==========

Classes for working with Gauss codes representing planar projections
of curves.

See class documentation for more details.
'''

from __future__ import print_function
import numpy as n
import re
import sys


class GaussCode(object):
    '''
    Class for containing and manipulating Gauss codes.

    This implements *only* extended Gauss code that includes the sign
    of each crossing (clockwise or anticlockwise).

    The length of a Gauss code (e.g. ``len(GaussCode())``) is the
    number of crossings in it.

    Parameters
    ----------
    crossings : array-like or string or PlanarDiagram or GaussCode
        A raw_crossings array from a
        :class:`~pyknot2.spacecurves.spacecurves.Knot`
        or :class:`~pyknot2.spacecurves.spacecurves.Link`, or a string
        representation of the form (e.g.)
        ``1+c,2-c,3+c,1-c,2+c,3-c``, with commas between entries,
        and with multiple link components separated by spaces and/or
        newlines. If a PlanarDiagram or GaussCode is passed, the code
        is duplicated.
    verbose : bool
        Whether to print information during calculations. Defaults to
        True.
    '''

    def __init__(self, crossings='', verbose=True):
        from pyknot2.representations import planardiagram
        if isinstance(crossings, str):
            self._init_from_string(crossings)
        elif isinstance(crossings, GaussCode):
            self._gauss_code = [row.copy() for row in crossings._gauss_code]
        elif isinstance(crossings, planardiagram.PlanarDiagram):
            raise NotImplementedError(
                'planar diagram -> gauss code not implemented')
        else:
            if isinstance(crossings, n.ndarray):
                crossings = [crossings]
            self._init_from_raw_crossings_array(crossings)

        self.crossing_numbers = _get_crossing_numbers(self._gauss_code)
        self.verbose = verbose

    def __len__(self):
        return int(sum([len(c) for c in self._gauss_code]) / 2)

    def _init_from_raw_crossings_array(self, crossings):
        '''
        Takes a list of crossings in (optionally multiple) lines,
        and converts to internal Gauss code representation.
        '''
        assigned_indices = {}
        gauss_code = []
        current_index = 1
        for line in crossings:
            line_gauss_code = []
            for ident, other_ident, over, clockwise in line:
                if ident not in assigned_indices:
                    assigned_indices[other_ident] = current_index
                    index = current_index
                    current_index += 1
                else:
                    index = assigned_indices.pop(ident)
                line_gauss_code.append([index, int(over),
                                        int(clockwise)])
            gauss_code.append(n.array(line_gauss_code))

        self._gauss_code = gauss_code

    def _init_from_string(self, crossings):
        '''
        Converts the string into internal Gauss code representation.
        '''
        regex = re.compile('[ \n]*')
        lines = regex.split(crossings)

        gauss_code = []
        over_under = {'+': 1, '-': -1}
        signs = {'c': 1, 'a': -1}

        for line in lines:
            line_gauss_code = []
            line_crossings = line.split(',')
            for line_crossing in line_crossings:
                line_gauss_code.append([int(line_crossing[:-2]),
                                        over_under[line_crossing[-2]],
                                        signs[line_crossing[-1]]])
            gauss_code.append(n.array(line_gauss_code))

        self._gauss_code = gauss_code

    def __repr__(self):
        out_strs = []

        gauss_code = self._gauss_code
        for line in gauss_code:
            if len(line) == 0:
                out_strs.append('----')
                out_strs.append(',')
            else:
                for entry in line:
                    out_strs.append(str(entry[0]))
                    over = (entry[1] > 0)
                    if over:
                        out_strs.append('+')
                    else:
                        out_strs.append('-')
                    clockwise = (entry[2] > 0)
                    if clockwise:
                        out_strs.append('c')
                    else:
                        out_strs.append('a')
                    out_strs.append(',')
            out_strs = out_strs[:-1]
            out_strs.append('\n')
        out_strs = out_strs[:-1]
        return ''.join(out_strs)

    def __str__(self):
        return repr(self)

    def _do_reidemeister_moves(self, one=True, two=True, one_extended=True):
        '''
        Performs the given Reidemeister moves a single time, iterating
        over all the crossings of self.
        '''

        code = self._gauss_code
        crossing_numbers = self.crossing_numbers

        rm2_store = {}

        # These lists keep track of which crossings have been removed, to
        # avoid modifying the arrays every time an RM is performed
        keeps = [n.ones(l.shape[0], dtype=bool) for l in code]

        # First do RM1 and RM2
        for line_index, line in enumerate(code):
            keep = keeps[line_index]

            for row_index, row in enumerate(line):
                next_index = (row_index + 1) % len(line)
                next_row = line[next_index]

                if (one and (row[0] == next_row[0]) and keep[row_index] and
                    keep[next_index]):
                    number = row[0]
                    crossing_numbers.remove(number)
                    keep[row_index] = False
                    keep[next_index] = False

                if (two and keep[row_index] and keep[next_index] and
                    (row[1] == next_row[1])):  # both over or under
                    numbers = tuple(sorted([row[0], next_row[0]]))
                    if numbers not in rm2_store:
                        rm2_store[numbers] = (line_index, row_index,
                                              next_index)
                    else:
                        other_indices = rm2_store.pop(numbers)
                        crossing_numbers.remove(row[0])
                        crossing_numbers.remove(next_row[0])
                        keep[row_index] = False
                        keep[next_index] = False
                        keeps[other_indices[0]][other_indices[1]] = False
                        keeps[other_indices[0]][other_indices[2]] = False

        # Next do RM1_extended separately (could be more efficient?)
        if one_extended:
            # Do extended RM1 as a separate step
            print
            code = [(line[keep] if len(line) > 0 else line) for (line, keep) in zip(code, keeps)]
            keeps = [n.ones(l.shape[0], dtype=bool) for l in code]

            crossing_indices = {}
            for line_index, line in enumerate(code):
                for row_index, row in enumerate(line):
                    identifier, over, crossings = row
                    if identifier not in crossing_indices:
                        crossing_indices[identifier] = []
                    crossing_indices[identifier].append((line_index,
                                                         row_index))
            for number in list(crossing_numbers):
                if number not in crossing_numbers:
                    continue  # The crossing has already been removed
                locations = crossing_indices[number]

                if locations[0][0] != locations[1][0]:  # not on same line
                    continue
                line_index = locations[0][0]
                first_index = locations[0][1]
                second_index = locations[1][1]
                first_index, second_index = sorted(
                    [first_index, second_index])

                # First, check crossings in the middle of the list
                in_between = code[line_index][first_index+1:second_index]
                in_between_keeps = keeps[line_index][first_index+1:second_index]
                if len(in_between) > 0:
                    in_between = in_between[in_between_keeps]

                if n.abs(n.sum(in_between[:, 1])) == len(in_between):
                    # all crossings over or under
                    keeps[line_index][first_index] = False
                    keeps[line_index][second_index] = False
                    for entry in in_between:
                        identifier = entry[0]
                        if identifier in crossing_numbers:
                            crossing_numbers.remove(identifier)
                        indices = crossing_indices[identifier]
                        keeps[indices[0][0]][indices[0][1]] = False
                        keeps[indices[1][0]][indices[1][1]] = False
                    crossing_numbers.remove(number)

                # Second, wrap around the list if a big rm1 wasn't performed
                if number not in crossing_numbers:
                    continue
                in_between = n.vstack((code[line_index][second_index+1:],
                                       code[line_index][:first_index]))
                in_between_keeps = n.hstack((keeps[line_index][second_index+1:],
                                             keeps[line_index][:first_index]))
                if len(in_between) > 0:
                    in_between = in_between[in_between_keeps]

                if n.abs(n.sum(in_between[:, 1])) == len(in_between):
                    keeps[line_index][first_index] = False
                    keeps[line_index][second_index] = False
                    for entry in in_between:
                        identifier = entry[0]
                        if identifier in crossing_numbers:
                            crossing_numbers.remove(identifier)
                        indices = crossing_indices[identifier]
                        keeps[indices[0][0]][indices[0][1]] = False
                        keeps[indices[1][0]][indices[1][1]] = False
                    crossing_numbers.remove(number)


        # Get rid of all crossings that have been removed by RMs
        self._gauss_code = [(line[keep] if len(line) > 0 else line) for (line, keep) in zip(code, keeps)]
        self.crossing_numbers = crossing_numbers

    def simplify(self, one=True, two=True, one_extended=True):
        '''
        Simplifies the GaussCode, performing the given Reidemeister moves
        everywhere possible, as many times as possible, until the
        GaussCode is no longer changing.

        This modifies the GaussCode - (non-topological) information may
        be lost!

        Parameters
        ----------
        one : bool
            Whether to use Reidemeister 1, defaults to True.
        two : bool
            Whether to use Reidemeister 2, defaults to True.
        one_extended : bool
            Whether to use extended Reidemeister 1, which removes crossings
            connected by arcs which include only over or only under crossings
            (and which must thus be topologically irrelevant). Defaults
            to True.
        '''

        if self.verbose:
            print('Simplifying: initially {} crossings'.format(
                n.sum([len(line) for line in self._gauss_code])))

        number_of_runs = 0
        while True:
            original_gc = self._gauss_code
            original_len = n.sum([len(line) for line in original_gc])
            self._do_reidemeister_moves(one, two)
            new_gc = self._gauss_code
            new_len = n.sum([len(line) for line in new_gc])
            number_of_runs += 1
            if self.verbose:
                sys.stdout.write('\r-> {} crossings after {} runs'.format(
                    n.sum([len(line) for line in new_gc]), number_of_runs))
                sys.stdout.flush()
            if new_len == original_len:
                break

        if self.verbose:
            print()


def _get_crossing_numbers(gc):
    '''
    Given GaussCode internal data, returns a list of all
    the crossing numbers within
    '''
    crossing_vals = set()
    for line in gc:
        for entry in line:
            crossing_vals.add(entry[0])
    return crossing_vals

