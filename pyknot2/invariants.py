'''
Invariants
==========

Functions for retrieving invariants of knots and links.

Functions whose name ends with ``_mathematica`` try to create an
external Mathematica process to calculate the answer. They may hang
or have other problems if Mathematica isn't available in your
``$PATH``.

.. warning:: This module may be broken into multiple components at
             some point.
'''

def alexander(representation, variable=-1, quadrant='lr', simplify=True):
    '''
    Calculates the Alexander polynomial of the given knot. The
    representation *must* have just one knot component, or the calculation
    will fail or potentially give bad results.

    The result is returned with whatever numerical precision the
    algorithm produces, it is not rounded.

    The given representation *must* be simplified (RM1 performed if
    possible) for this to work, otherwise the matric has overlapping
    elements. This is so important that this function automatically
    calls :meth:`pyknot2.representations.gausscode.GaussCode.simplify`,
    you must disable this manually if you don't want to do it.

    Parameters
    ----------
    representation : Anything convertible to a
                     :class:`~pyknot2.representations.gausscode.GaussCode`
        A pyknot2 representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`).
    variable : float or sympy variable
        The value to caltulate the Alexander polynomial at. Defaults to -1,
        but may be switched to the sympy variable ``t`` in the future.
        Supports numeric types (will be treated as floats) or sympy
        expressions.
    quadrant : str
        Determines what principal minor of the Alexander matrix should be
        used in the calculation; all choices *should* give the same answer.
        Must be 'lr', 'ur', 'ul' or 'll' for lower-right, upper-right,
        upper-left or lower-left respectively.
    simplify : bool
        Whether to call the GaussCode simplify method, defaults to True.
    '''

    from .representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    if simplify:
        representation.simplify(one=True, two=True, one_extended=True)

    code_list = representation._gauss_code
    if len(code_list) == 0:
        return 1
    elif len(code_list) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')
    
    crossings = code_list[0]

    if len(crossings) == 0:
        return 1
    if quadrant not in ['lr','ur','ul','ll']:
        raise Exception('invalid quadrant')

    if isinstance(variable, (int, float, complex)):
        return _alexander_numpy(crossings, variable, quadrant)
    else:
        return _alexander_sympy(crossings, variable, quadrant)


def _alexander_numpy(crossings, variable=-1.0, quadrant='lr'):
    '''
    Numpy implementation of the Alexander polynomial (evaluated
    at a float), assuming the input has been sanitised by
    :func:`alexander`.
    '''
    import numpy as n
    num_crossings = len(crossings)/2
    dtype = n.complex if isinstance(variable, n.complex) else n.float
    matrix = n.zeros((num_crossings, num_crossings), dtype=dtype)
    line_num = 0
    crossing_num_counter = 0
    crossing_dict = {}
    crossing_exists = False
    
    over_clock = 1. - 1. / variable
    under_clock_before = -1.
    under_clock_after = 1. / variable
    over_aclock = 1 - variable
    under_aclock_before = -1.
    under_aclock_after = variable

    for i in range(len(crossings)):
        identifier, over, clockwise = crossings[i]
        if identifier in crossing_dict:
            crossing_num = crossing_dict.pop(identifier)
            crossing_exists = True
        if not crossing_exists:
            crossing_num = crossing_num_counter
            crossing_num_counter += 1
            crossing_dict[identifier] = crossing_num
        crossing_exists = False

        if over > 0.99999:
            mat_elt = over_clock if clockwise > 0.999 else over_aclock
            matrix[crossing_num, line_num % num_crossings] = mat_elt
        else:
            new_mat_elt = (under_clock_after if clockwise > 0.999 else
                           under_aclock_after)
            matrix[crossing_num, line_num % num_crossings] = -1
            line_num += 1
            matrix[crossing_num, line_num % num_crossings] = new_mat_elt
    if quadrant == 'lr':
        poly_val = n.linalg.det(matrix[1:, 1:])
    elif quadrant == 'ur':
        poly_val = n.linalg.det(matrix[:-1, 1:])
    elif quadrant == 'ul':
        poly_val = n.linalg.det(matrix[:-1:, :-1])
    elif quadrant == 'll':
        poly_val = n.linalg.det(matrix[1:, :-1])
    if not isinstance(poly_val, n.complex):
        poly_val = n.abs(poly_val)
    return poly_val

def _alexander_sympy(crossings, variable=None, quadrant='lr'):
    '''
    Sympy implementation of the Alexander polynomial, evaluated
    with the variable replaced by some sympy expression.
    '''
    import sympy as sym
    if variable is None:
        variable = sym.var('t')
    num_crossings = len(crossings)/2
    matrix = sym.zeros((num_crossings, num_crossings))
    line_num = 0
    crossing_num_counter = 0
    crossing_dict = {}
    crossing_exists = False
    
    over_clock = 1 - 1 / variable
    under_clock_before = -1
    under_clock_after = 1 / variable
    over_aclock = 1 - variable
    under_aclock_before = -1
    under_aclock_after = variable

    for i in range(len(crossings)):
        identifier, over, clockwise = crossings[i]
        if identifier in crossing_dict:
            crossing_num = crossing_dict.pop(identifier)
            crossing_exists = True
        if not crossing_exists:
            crossing_num = crossing_num_counter
            crossing_num_counter += 1
            crossing_dict[identifier] = crossing_num
        crossing_exists = False

        if over > 0.99999:
            mat_elt = over_clock if clockwise > 0.999 else over_aclock
            matrix[crossing_num, line_num % num_crossings] = mat_elt
        else:
            new_mat_elt = (under_clock_after if clockwise > 0.999 else
                           under_aclock_after)
            matrix[crossing_num, line_num % num_crossings] = -1
            line_num += 1
            matrix[crossing_num, line_num % num_crossings] = new_mat_elt
    if quadrant == 'lr':
        poly_val = matrix[1:, 1:].det()
    elif quadrant == 'ur':
        poly_val = matrix[:-1, 1:].det()
    elif quadrant == 'ul':
        poly_val = matrix[:-1:, :-1].det()
    elif quadrant == 'll':
        poly_val = matrix[1:, :-1].det()
    return poly_val

