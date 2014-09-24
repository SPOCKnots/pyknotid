'''
Invariants
==========

Functions for retrieving invariants of knots and links.

Functions whose name ends with ``_mathematica`` try to create an
external Mathematica process to calculate the answer. They may hang
or have other problems if Mathematica isn't available in your
``$PATH``, so be careful using them.

.. warning:: This module may be broken into multiple components at
             some point.
'''
from __future__ import print_function
import subprocess
import re


def alexander(representation, variable=-1, quadrant='lr', simplify=True):
    '''
    Calculates the Alexander polynomial of the given knot. The
    representation *must* have just one knot component, or the calculation
    will fail or potentially give bad results.

    The result is returned with whatever numerical precision the
    algorithm produces, it is not rounded.

    The given representation *must* be simplified (RM1 performed if
    possible) for this to work, otherwise the matrix has overlapping
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
    variable : float or complex or sympy variable
        The value to caltulate the Alexander polynomial at. Defaults to -1,
        but may be switched to the sympy variable ``t`` in the future.
        Supports int/float/complex types (fast, works for thousands of
        crossings) or sympy
        expressions (much slower, works mostly only for <100 crossings).
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
    if quadrant not in ['lr', 'ur', 'ul', 'll']:
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
    under_clock_after = 1. / variable
    over_aclock = 1 - variable
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
    # This is almost the same as the numpy implementation...they should
    # probably be merged.
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
    under_clock_after = 1 / variable
    over_aclock = 1 - variable
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


def alexander_mathematica(representation, quadrant='ul', verbose=False,
                          via_file=True):
    '''
    Returns the Alexander polynomial of the given representation, by
    creating a Mathematica process and running its knot routines.
    The Mathematica installation must include the KnotTheory package.

    The function only supports evaluating at the variable ``t``.

    Parameters
    ----------
    representation : Anything convertible to a
                     :class:`~pyknot2.representations.gausscode.GaussCode`
        A pyknot2 representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`).
    quadrant : str
        Determines what principal minor of the Alexander matrix should be
        used in the calculation; all choices *should* give the same answer.
        Must be 'lr', 'ur', 'ul' or 'll' for lower-right, upper-right,
    verbose : bool
        Whether to print information about the procedure. Defaults to False.
    via_file : bool
        If True, calls Mathematica via a written file ``mathematicascript.m``,
        otherwise calls Mathematica directly with ``runMath``. The latter
        had a nasty bug in at least one recent Mathematica version, so the
        default is to True.
    '''
    from .representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    code = representation._gauss_code

    if len(code) == 0:
        return 1
    elif len(code) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')
    mat_mat = _mathematica_matrix(code[0], quadrant=quadrant, verbose=verbose)
    if not via_file:
        det = subprocess.check_output(['runMath', 'Det[' + mat_mat + ']'])[:-1]
    else:
        _write_mathematica_script('mathematicascript.m', 'Print[Det[' +
                                  mat_mat + ']]')
        det = subprocess.check_output(['bash', 'mathematicascript.m'])
    t = sym.var('t')
    return eval(det.replace('^', '**'))


def jones_mathematica(representation):
    '''
    Returns the Jones polynomial of the given representation, by
    creating a Mathematica process and running its knot routines.
    The Mathematica installation must include the KnotTheory package.

    The function only supports evaluating at the variable ``q``.

    Parameters
    ----------
    representation : A PlanarDiagram, or anything convertible to a
                     :class:`~pyknot2.representations.gausscode.GaussCode`
        A pyknot2 representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`), or a PlanarDiagram.
    '''
    from .representations.gausscode import GaussCode
    from .representations.planardiagram import PlanarDiagram
    if not isinstance(representation, (GaussCode, PlanarDiagram)):
        representation = GaussCode(representation)
    if isinstance(representation, GaussCode):
        representation = PlanarDiagram(representation)

    mathematica_code = representation.as_mathematica()
    _write_mathematica_script('jonesscript.m',
                              '<< KnotTheory`\nPrint[Jones[' +
                              mathematica_code + '][q]]')
    result = subprocess.check_output(['bash', 'jonesscript.m']).split('\n')[-2]
    q = sym.var('q')
    result = result.replace('[', '(')
    result = result.replace(']', ')')
    result = result.replace('Sqrt', 'sym.sqrt')
    r = re.compile('([0-9]+)')
    result = r.sub(r'sym.Integer(\1)', result)
    return eval(result.replace('^', '**'))


def _write_mathematica_script(filen, text):
    '''
    Write the given text (mathematica code) to the given filename. It will
    be wrapped in some MathKernel calling stuff first.
    '''
    with open(filen, 'w') as fileh:
        fileh.write('MathKernel -noprompt -run "commandLine={${1+\"$1\"}}; '
                    '$(sed \'1,/^exit/d\' $0) ; Exit[]"\nexit $?\n')
        fileh.write(text)
        fileh.close()


def _mathematica_matrix(cs, quadrant='lr', verbose=False):
    '''
    Turns the given crossings into a string of Mathematica code
    representing the Alexander matrix.

    This functions is for internal use only.
    '''
    if len(cs) == 0:
        return ''
    mathmat_entries = []

    line_num = 0
    num_crossings = len(cs)/2
    crossing_num_counter = 0
    crossing_dict = {}
    crossing_exists = False
    written_indices = []
    for i, crossing in enumerate(cs):
        if verbose and (i+1) % 100 == 0:
            sys.stdout.write('\ri = {0} / {1}'.format(i, len(cs)))
        print('crossing is', crossing)
        identifier, upper, direc = crossing
        for entry in crossing_dict:
            if entry[0] == identifier:
                crossing_num = crossing_dict[entry]
                crossing_entry = entry
                crossing_exists = True
        if not crossing_exists:
            crossing_num = crossing_num_counter
            crossing_num_counter += 1
            crossing_dict[tuple(crossing)] = crossing_num
        else:
            crossing_dict.pop(crossing_entry)
        crossing_exists = False

        if upper > 0.99999:
            if direc > 0.99999:
                matrix_element = '1-1/t'
            else:
                matrix_element = '1-t'
            mathmat_entries.append((crossing_num,
                                    line_num % num_crossings,
                                    matrix_element))
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))
        else:
            if direc > 0.99999:
                new_matrix_element = '1/t'
            else:
                new_matrix_element = 't'
            mathmat_entries.append((crossing_num,
                                    line_num % num_crossings, '-1'))
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))
            line_num += 1
            mathmat_entries.append((crossing_num,
                                    line_num % num_crossings,
                                    new_matrix_element))
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))

    if quadrant == 'lr':
        mathmat_entries = filter(
            lambda j: j[0] != 0 and j[1] != 0, mathmat_entries)
        mathmat_entries = map(
            lambda j: (j[0]-1, j[1]-1, j[2]), mathmat_entries)
    if quadrant == 'ur':
        mathmat_entries = filter(
            lambda j: j[0] != num_crossings-1 and j[1] != 0, mathmat_entries)
        mathmat_entries = map(
            lambda j: (j[0], j[1]-1, j[2]), mathmat_entries)
    if quadrant == 'ul':
        mathmat_entries = filter(
            lambda j: j[0] != num_crossings-1 and j[1] != num_crossings-1,
            mathmat_entries)
    if quadrant == 'll':
        mathmat_entries = filter(
            lambda j: j[0] != 0 and j[1] != num_crossings-1,
            mathmat_entries)
        mathmat_entries = map(lambda j: (j[0]-1, j[1], j[2]), mathmat_entries)

    if verbose:
        print
    outstr = 'Sparse_array[{ '
    for entry in mathmat_entries:
        outstr += '{%d,%d}->%s, ' % (entry[0]+1, entry[1]+1, entry[2])
    outstr = outstr[:-2]
    outstr += ' }]'
    return outstr
