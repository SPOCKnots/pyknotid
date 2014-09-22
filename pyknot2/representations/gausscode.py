'''
Classes for working with Gauss codes representing planar projections
of curves.

See class documentation for more details.
'''

from __future__ import print_function
import numpy as n
import re
import sys
import planardiagram

class GaussCode(object):
    '''
    Class for containing and manipulating Gauss codes.

    Just provides convenient display and conversion methods for now.
    In the future, will support simplification.

    This implements *only* extended Gauss code that includes the sign
    of each crossing (clockwise or anticlockwise).

    Parameters
    ----------
    crossings : array-like or string or PlanarDiagram
        One of:
        - a raw_crossings array from a :class:`~pyknot2.spacecurves.Knot`
          or :class:`~pyknot2.spacecurves.Link`.
        - a string representation of the form (e.g.)
          ``1+c,2-c,3+c,1-c,2+c,3-c``, with commas between entries,
          and with multiple link components separated by spaces and/or
          newlines.
    '''

    def __init__(self, crossings=''):

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
                                        over_under(line_crossing[-2]),
                                        signs(line_crossing[-1])])
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

    def _do_reidemeister_moves(self, one=True, two=True, two_extended=False):
        '''
        Performs the given Reidemeister moves a single time, iterating
        over all the crossings of self.
        '''
        if two_extended:
            raise NotImplementedError('can\'t do extended moves yet')

        code = self._gauss_code
        crossing_numbers = self.crossing_numbers

        rm2_store = {}

        # These lists keep track of which crossings have been removed, to
        # avoid modifying the arrays every time an RM is performed
        keeps = [n.ones(l.shape[0], dtype=bool) for l in code]

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
                        rm2_store[numbers] = (line_index, row_index, next_index)
                    else:
                        other_indices = rm2_store.pop(numbers)
                        crossing_numbers.remove(row[0])
                        crossing_numbers.remove(next_row[0])
                        keep[row_index] = False
                        keep[next_index] = False
                        keeps[other_indices[0]][other_indices[1]] = False
                        keeps[other_indices[0]][other_indices[2]] = False

        # Get rid of all crossings that have been removed by RMs
        self._gauss_code = [line[keep] for (line, keep) in zip(code, keeps)]
        self.crossing_numbers = crossing_numbers
                        
        
    def simplify(self, one=True, two=True, verbose=True):
        '''
        Simplifies the GaussCode, performing the given Reidemeister moves
        everywhere possible, as many times as possible, until the
        GaussCode is no longer changing.

        This modifies the GaussCode - (non-topological) information may
        be lost!

        Parameters
        ----------
        one : bool
            Whether to use Reidemeister 1
        two : bool
            Whether to use Reidemeister 2
        '''

        if verbose:
            print('Simplifying: initially {} crossings'.format(
                n.sum(map(len, self._gauss_code))))

        number_of_runs = 0
        while True:
            original_gc = self._gauss_code
            original_len = n.sum(map(len, original_gc))
            self._do_reidemeister_moves(one, two)
            new_gc = self._gauss_code
            new_len = n.sum(map(len, new_gc))
            number_of_runs += 1
            if verbose:
                sys.stdout.write('\r-> {} crossings after {} runs'.format(
                    n.sum(map(len, new_gc)), number_of_runs))
                sys.stdout.flush()
            if new_len == original_len:
                break

        if verbose:
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
    return list(crossing_vals)
    
