'''
Classes for working with Gauss codes representing planar projections
of curves.

See class documentation for more details.
'''

import numpy as n
import re
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
        elif isinstance(crossings, planardiagram.PlanarDiagram):
            raise NotImplementedError(
                'planar diagram -> gauss code not implemented')
        else:
            if isinstance(crossings, n.ndarray):
                crossings = [crossings]
            self._init_from_raw_crossings_array(crossings)

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
        
