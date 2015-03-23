'''
Dowker-Thistlethwaite notation
==============================

Classes for working with DT notation representing planar projections
of curves.
'''

import numpy as n
import re

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
        if isinstance(code, str):
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

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if len(self._dt) == 0:
            return '----'
        strings = [' '.join(map(str, l)) for l in self._dt]
        return '\n'.join(strings)
