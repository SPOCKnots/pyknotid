'''
Planar diagrams
===============

Classes for working with planar diagram notation of knot diagrams.

See individual class documentation for more details.
'''


import numpy as n
import sys



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
        :class:`~pyknot2.representations.gausscode.GaussCode` may be used.
    '''

    def __init__(self, crossings=''):
        from pyknot2.representations import gausscode
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

    def __init__(self,a=None, b=None, c=None, d=None):
        super(Crossing, self).__init__()
        self.extend([a,b,c,d])

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
            crossings.append(Crossing(a,b,c,d))
        elif entry[0] == 'P':
            a, b = [int(j) for j in entry[1].split(',')]
            crossings.append(Point(a,b))
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
