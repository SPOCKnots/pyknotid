'''
DTNotation
==========

Classes for working with DT notation representing planar projections
of curves.

API documentation
~~~~~~~~~~~~~~~~~
'''

import numpy as n
import re
import sys

if sys.version_info.major == 2:
    string_types = basestring
else:
    string_types = str

class DTNotation(object):
    '''
    Class for containing and manipulation DT notation.

    Parameters
    ----------
    code : str or array-like
        The DT code. Must be either a string of entries separated
        by spaces, or an array.
    '''

    def __init__(self, code):
        if isinstance(code, string_types):
            self._init_from_string(code)
        elif isinstance(code, n.ndarray):
            code = [code]
            self._dt = code
        elif isinstance(code, list):
            self._dt = code
        else:
            raise ValueError('Could not parse code as a valid type')

    def _init_from_string(self, code):
        '''
        Converts the string into internal DT notation representation.
        '''

        regex = re.compile('[\n]*')
        lines = regex.split(code)

        dt = []

        for line in lines:
            numbers = line.split(' ')
            dt.append(n.array([int(number) for number in numbers], dtype=n.int))

        self._dt = dt

    def gauss_code_string(self):
        '''Returns a string containing a Gauss code, in the format accepted
        by :class:`~pyknotid.representations.gausscode.GaussCode`.

        To get a :class:`~pyknotid.representations.gausscode.GaussCode`
        object, you can pass this string when initialising it, or use
        :meth:`DTNotation.representation`.

        '''
        if len(self._dt) > 1:
            raise ValueError('DTNotation -> GaussCode does not yet '
                             'work with links')

        dt = self._dt[0]
        arr = n.zeros((len(dt) * 2, 2), dtype=n.int)


        for index, even in enumerate(dt, 0):
            odd = 2*index
            sign = n.sign(even)
            even = n.abs(even) - 1

            arr[odd, 0] = index + 1
            arr[odd, 1] = sign

            arr[even, 0] = index + 1
            arr[even, 1] = -1 * sign

        str_entries = ['{}{}'.format(index, '+' if sign > 0 else '-')
                       for index, sign in arr]

        return ','.join(str_entries)

    def representation(self, **kwargs):
        '''Returns a
        :class:`~pyknotid.representations.representation.Representation`
        representing the same DT code. The crossing orientations (and
        therefore resulting chirality) are chosen arbitrarily.
        '''
        from pyknotid.representations import Representation
        return Representation.calculating_orientations(
            self.gauss_code_string(), **kwargs)

    # def space_curve(self):
    #     gc = 

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if len(self._dt) == 0:
            return '----'
        strings = [' '.join(map(str, l)) for l in self._dt]
        return '\n'.join(strings)
