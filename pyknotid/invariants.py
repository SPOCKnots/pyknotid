'''Invariants
==========

Functions for retrieving invariants of knots and links.

Many of these functions can be called in a more convenient way via
methods of the :doc:`space curve classes <spacecurves/index>`
(e.g. :class:`~pyknotid.spacecurves.knot.Knot`) or the
:class:`~pyknotid.representations.representation.Representation` class.


Mathematica
-----------

Functions whose name ends with ``_mathematica`` try to create an
external Mathematica process to calculate the answer. They may hang
or have other problems if Mathematica isn't available in your
``$PATH``, so be careful using them.

.. warning:: This module may be broken into multiple components at
             some point.

API documentation
-----------------

'''
from __future__ import print_function
import subprocess
import re
import sympy as sym
import numpy as n

from pyknotid.utils import vprint

def alexander(representation, variable=-1, quadrant='lr', simplify=True,
              mode='python'):
    '''Calculates the Alexander polynomial of the given knot. The
    representation *must* have just one knot component, or the
    calculation will fail or potentially give bad results.

    The result is returned with whatever numerical precision the
    algorithm produces, it is not rounded.

    The given representation *must* be simplified (RM1 performed if
    possible) for this to work, otherwise the matrix has overlapping
    elements. This is so important that this function automatically
    calls
    :meth:`pyknotid.representations.gausscode.GaussCode.simplify`, you
    must disable this manually if you don't want to do it.

    .. note:: If 'maxima' or 'mathematica' is chosen as the mode, the
              variable will automatically be set to ``t``.

    .. note:: If the mode is 'cypari', the quadrant argument will be
              ignored and the upper-left quadrant always used.

    Parameters
    ----------
    representation : Anything convertible to a
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
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
    mode : string
        One of 'python', 'maxima', 'cypari' or 'mathematica'. 
        denotes what
        tools to use; if python, the calculation is performed with
        numpy or sympy as appropriate. If maxima or mathematica, that
        program is called by the function - this will only work if the
        external tool is installed and available. Defaults to python.

    '''

    from pyknotid.representations.gausscode import GaussCode
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

    mode = mode.lower()
    if mode == 'maxima':
        return alexander_maxima(representation, quadrant,
                                verbose=False,
                                simplify=False)
    elif mode == 'cypari':
        return alexander_cypari(representation, quadrant,
                                verbose=False,
                                simplify=False)
    elif mode == 'mathematica':
        return alexander_mathematica(representation, quadrant,
                                     verbose=False)
                                     

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
    num_crossings = int(len(crossings)/2)
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
    num_crossings = int(len(crossings)/2)
    matrix = sym.zeros(num_crossings)
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


def alexander_maxima(representation, quadrant='ul', verbose=False,
                     simplify=True):
    '''
    Returns the Alexander polynomial of the given representation, by
    calculating the matrix determinant in maxima.

    The function only supports evaluating at the variable ``t``.

    Parameters
    ----------
    representation : Anything convertible to a
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`).
    quadrant : str
        Determines what principal minor of the Alexander matrix should be
        used in the calculation; all choices *should* give the same answer.
        Must be 'lr', 'ur', 'ul' or 'll' for lower-right, upper-right,
    verbose : bool
        Whether to print information about the procedure. Defaults to False.
    simplify : bool
        If True, tries to simplify the representation before calculating
        the polynomial. Defaults to True.
    '''

    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    if simplify:
        representation.simplify(one=True, two=True, one_extended=True)

    if len(representation._gauss_code) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')

    code = representation._gauss_code[0]

    if len(code) == 0:
        return 1

    mat_mat = _maxima_matrix(code, quadrant=quadrant, verbose=verbose)

    code = ''.join([#'sparse: true;\n',
                    'ratmx: true;\n',
                    'display2d: false;\n',
                    mat_mat,
                    'expand(determinant(m2));'])

    with open('maxima_batch.maxima', 'w') as fileh:
        fileh.write(code)

    result = subprocess.check_output(
        ['maxima', '-b', 'maxima_batch.maxima'])
    print('Maxima output is:\n', result)
    result = result.decode('utf-8').split('\n')[-3][6:]

    t = sym.var('t')

    return eval(result.replace('^', '**'))


def alexander_cypari(representation, quadrant='ul', verbose=False,
                     simplify=True):
    '''
    Returns the Alexander polynomial of the given representation, by
    calculating the matrix determinant via cypari, a python interface
    to Pari-GP.

    The function only supports evaluating at the variable ``t``.

    The returned object is a cypari query type.

    Parameters
    ----------
    representation : Anything convertible to a
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`).
    quadrant : str
        Determines what principal minor of the Alexander matrix should be
        used in the calculation; all choices *should* give the same answer.
        Must be 'lr', 'ur', 'ul' or 'll' for lower-right, upper-right,
    verbose : bool
        Whether to print information about the procedure. Defaults to False.
    simplify : bool
        If True, tries to simplify the representation before calculating
        the polynomial. Defaults to True.
    '''

    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    if simplify:
        representation.simplify(one=True, two=True, one_extended=True)

    if len(representation._gauss_code) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')

    code = representation._gauss_code[0]

    if len(code) == 0:
        return 1

    mat_mat = _cypari_matrix(code, quadrant=quadrant, verbose=verbose)

    return mat_mat.matdet()


                    
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
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
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
    simplify : bool
        If True, tries to simplify the representation before calculating
        the polynomial. Defaults to True.
    '''
    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    if simplify:
        representation.simplify(one=True, two=True, one_extended=True)

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
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`), or a PlanarDiagram.
    '''
    from pyknotid.representations.gausscode import GaussCode
    from pyknotid.representations.planardiagram import PlanarDiagram
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

q = sym.var('q')
def jones_polynomial(representation, var=q):
    from pyknotid.representations.gausscode import GaussCode
    from pyknotid.representations.planardiagram import PlanarDiagram, Crossing
    if not isinstance(representation, (GaussCode, PlanarDiagram)):
        representation = GaussCode(representation)
    if isinstance(representation, GaussCode):
        representation = PlanarDiagram(representation)

    rep = representation

    if isinstance(q, sym.Symbol):
        A = sym.var('A')
    else:
        A = (var + 0j)**0.25
    
    crossings = crossings_connectedness_order(rep)
    if len(crossings) < 2:
        return 1
    
    web = []
    crossing = crossings[0]
    crossing = PlanarDiagram(crossing)
    diag_1, diag_2 = kauffman_skein(crossing)
    web.append([diag_1, A])
    web.append([diag_2, 1.0/A])
    replacement = -A**2 - A**-2

    for crossing in crossings[1:]:
        crossing = PlanarDiagram(crossing)
        diag_1, diag_2 = kauffman_skein(crossing)
        new_web = []

        for term in web:
            diag_1_product = PlanarDiagram(
                [x for x in diag_1] +
                [x for x in term[0]]).contract_points()
            diag_2_product = PlanarDiagram(
                [x for x in diag_2] +
                [x for x in term[0]]).contract_points()
            contracted_1 = 0
            contracted_2 = 0
            diag_1_new = []
            diag_2_new = []

            for point in diag_1_product:
                if len(point.components()) == 1:
                    contracted_1 += 1
                else:
                    diag_1_new.append(point)
            for point in diag_2_product:
                if len(point.components()) == 1:
                    contracted_2 += 1
                else:
                    diag_2_new.append(point)

            new_term_1 = ([sorted(diag_1_new)] +
                          [term[1] * (replacement ** contracted_1) * A])
            new_term_2 = ([sorted(diag_2_new)] +
                          [term[1] * (replacement ** contracted_2) * 1.0/A])

            new_web.append(new_term_1)
            new_web.append(new_term_2)

        unique_diags = list(set([tuple(x[0]) for x in new_web]))
        contracted_web = [[list(x)] for x in unique_diags]

        for term in new_web:
            for diag in contracted_web:
                if diag[0] == term[0]:
                    if len(diag) == 1:
                        diag.append(term[1])
                    else:
                        diag[1] += term[1]

        web = contracted_web

    # quick and dirty writhe calculation
    crossings = [x for x in rep if isinstance(x, Crossing)]
    total_writhe = 0
    for crossing in crossings:
        total_writhe += crossing.sign()
        
    jones_precursor = web[0][1]

    jones_poly = (-A**3)**total_writhe * jones_precursor / replacement

    if isinstance(var, sym.Symbol):
        return sym.expand(sym.simplify(jones_poly)).subs(A, var**0.25)
    return round(abs(jones_poly))


def crossings_connectedness_order(planar_diagram):
    from pyknotid.representations.planardiagram import Crossing
    crossings = [x for x in planar_diagram if isinstance(x, Crossing)]

    inner_components = crossings[0].components()
    crossings_remaining = crossings[1:]
    ordered_crossings = [crossings[0]]

    for i in range(len(crossings_remaining)):
        connectedness = []
        for crossing in crossings_remaining:
            new_components = (len(list(set(crossing.components() + inner_components))) -
                              len(inner_components))

            connectedness.append(len(crossing.components()) - new_components)
        most_connected_index = connectedness.index(max(connectedness))
        inner_components += crossings_remaining[most_connected_index].components()
        inner_components = list(set(inner_components))
        ordered_crossings.append(crossings_remaining[most_connected_index])
        del crossings_remaining[most_connected_index]

    return ordered_crossings

def contract_points(planar_diagram):
    '''
    For appropriately contracting :class: `Points` in a
    :class: `PlanarDiagram` According to the following rules:

    P_a,b P_b,c -> P_a,c
    P_a,b P_a,b -> P_a,a
    '''
    import copy

    pd_crossings = [x for x in planar_diagram if type(x) == Crossing]
    pd_points = [x for x in planar_diagram if type(x) == Point]

    pd_trimmed = PlanarDiagram()
    for crossing in pd_crossings:
        pd_trimmed.append(crossing)

    substituted_points = []
    for i in range(len(pd_points)):
        if pd_points[i] not in substituted_points:
            pd_trimmed.append(pd_points[i])
            totally_simplified = False
            while not totally_simplified:
                pd_trimmed_old = copy.copy(pd_trimmed)
                for j in range(i+1, len(pd_points)):
                    if (len(pd_trimmed[-1].components()) > 1 and
                            len(pd_points[j].components()) > 1 and
                            pd_points[j] not in substituted_points):
                        # This skips P_a,a situations
                        if pd_trimmed[-1].components() != pd_points[j].components():
                            # This conditions skips (P_a,b P_a,b) situations
                            substituted = False
                            p = 0
                            while substituted is not True and p != 2:
                                q = 0
                                while substituted is not True and q != 2:
                                    if pd_trimmed[-1].components()[p] == pd_points[j].components()[q]:
                                        pd_trimmed.append(Point(pd_trimmed[-1].components()[(p+1) % 2],
                                                                pd_points[j].components()[(q+1) % 2]))
                                        del pd_trimmed[-2]
                                        substituted = True
                                        substituted_points.append(pd_points[j])
                                    q += 1
                                p += 1
                        else:
                            # This deals with (P_a,b P_a,b) situations
                            pd_trimmed.append(Point(pd_trimmed[-1].components()[0],
                                                    pd_trimmed[-1].components()[0]))
                            del pd_trimmed[-2]
                            substituted_points.append(pd_points[j])
                if pd_trimmed_old == pd_trimmed:
                    totally_simplified = True
    return pd_trimmed


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
    num_crossings = int(len(cs)/2)
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


def hyperbolic_volume(representation):
    '''
    The hyperbolic volume, calculated by the SnapPy library for
    studying the topology and geometry of 3-manifolds. This function
    depends on the Spherogram module, distributed with SnapPy or
    available separately.

    Parameters
    ----------
    representation : A PlanarDiagram, or anything convertible to a
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`), or a PlanarDiagram.

    '''
    from pyknotid.representations.gausscode import GaussCode
    from pyknotid.representations.planardiagram import PlanarDiagram
    if not isinstance(representation, (GaussCode, PlanarDiagram)):
        representation = GaussCode(representation)
    if isinstance(representation, GaussCode):
        representation = PlanarDiagram(representation)

    volume = representation.as_spherogram().exterior().volume()
    return volume

def _maxima_matrix(cs, quadrant='lr', verbose=False):
    '''
    Turns the given crossings into a string of maxima code
    representing the Alexander matrix.

    This functions is for internal use only.
    '''
    if len(cs) == 0:
        return ''
    mathmat_entries = {}

    line_num = 0
    num_crossings = int(len(cs)/2)
    crossing_num_counter = 0
    crossing_dict = {}
    crossing_exists = False
    written_indices = []
    for i, crossing in enumerate(cs):
        if verbose and (i+1) % 100 == 0:
            sys.stdout.write('\ri = {0} / {1}'.format(i, len(cs)))
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
            mathmat_entries[
                (crossing_num, line_num % num_crossings)] =  matrix_element
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
            mathmat_entries[(crossing_num, line_num % num_crossings)] =  '-1'
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))
            line_num += 1
            mathmat_entries[
                (crossing_num, line_num % num_crossings)] = new_matrix_element
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))

    if verbose:
        print
    outstrs = ['m: matrix(']
    num_crossings = int(len(cs) / 2)
    for row in range(num_crossings):
        outstrs.append('[')
        for column in range(num_crossings):
            if (row, column) in mathmat_entries:
                value = mathmat_entries[(row, column)]
            else:
                value = '0'
            outstrs.append(value)
            outstrs.append(', ')
        outstrs.pop()
        outstrs.append(']')
        outstrs.append(', ')
    outstrs.pop()
    outstrs.append(');\n')

    outstrs.append(
        {'lr': 'm2: submatrix(1, m, 1)',
         'ur': 'm2: submatrix({nc}, m, 1)'.format(nc=num_crossings),
         'ul': 'm2: submatrix({nc}, m, {nc})'.format(nc=num_crossings),
         'll': 'm2: submatrix(1, m, {nc})'.format(nc=num_crossings)
        }[quadrant])

    outstrs.append(';\n')
    return ''.join(outstrs)


def _cypari_matrix(cs, quadrant='lr', verbose=False):
    '''
    Turns the given crossings into a string of maxima code
    representing the Alexander matrix.

    This functions is for internal use only.
    '''
    if len(cs) == 0:
        return ''
    mathmat_entries = {}

    line_num = 0
    num_crossings = int(len(cs)/2)
    crossing_num_counter = 0
    crossing_dict = {}
    crossing_exists = False
    written_indices = []
    for i, crossing in enumerate(cs):
        if verbose and (i+1) % 100 == 0:
            sys.stdout.write('\ri = {0} / {1}'.format(i, len(cs)))
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
            mathmat_entries[
                (crossing_num, line_num % num_crossings)] =  matrix_element
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
            mathmat_entries[(crossing_num, line_num % num_crossings)] =  '-1'
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))
            line_num += 1
            mathmat_entries[
                (crossing_num, line_num % num_crossings)] = new_matrix_element
            spec = (crossing_num, line_num % num_crossings)
            if spec in written_indices:
                print(spec)
            else:
                written_indices.append((crossing_num,
                                        line_num % num_crossings))

    if verbose:
        print

    print('Warning: quadrant ignored!')
    
    outstrs = ['[']
    num_crossings = int(len(cs) / 2)
    for row in range(num_crossings-1):
        for column in range(num_crossings-1):
            if (row, column) in mathmat_entries:
                value = mathmat_entries[(row, column)]
            else:
                value = '0'
            outstrs.append(value)
            outstrs.append(', ')
        outstrs.pop()
        outstrs.append(';')
    outstrs.pop()
    outstrs.append(']')

    from cypari.gen import pari
    return pari(''.join(outstrs))

def _crossing_arrows_and_signs(gc, crossing_numbers):
    '''Internal function to get a list of crossing arrow indices
    (as per a Gauss diagram) and signs, both in dictionaries.'''

    ## Arrow diagram for v_2 is
    ## appear -> leave -> leave -> appear (with crossed arrows)
    over_crossing_indices = {}
    under_crossing_indices = {}
    signs = {}
    for i, row in enumerate(gc):
        if row[1] == 1:
            over_crossing_indices[row[0]] = i
        else:
            under_crossing_indices[row[0]] = i
        signs[row[0]] = row[2]

    arrows = {index: (over_crossing_indices[index],
                      under_crossing_indices[index])
              for index in crossing_numbers}

    return arrows, signs

def _crossing_arrows_and_signs_numpy(gc, crossing_numbers):
    ## Arrow diagram for v_2 is
    ## appear -> leave -> leave -> appear (with crossed arrows)
    over_crossing_indices = {}
    under_crossing_indices = {}
    signs = {}
    for i, row in enumerate(gc):
        if row[1] == 1:
            over_crossing_indices[row[0]] = i
        else:
            under_crossing_indices[row[0]] = i
        signs[row[0]] = row[2]

    arrows = n.zeros((len(crossing_numbers), 3), dtype=n.long)

    for index, number in enumerate(crossing_numbers):
        row = arrows[index]
        row[0] = over_crossing_indices[number]
        row[1] = under_crossing_indices[number]
        row[2] = signs[number]

    return arrows

def second_order_writhe(representation):
    '''Returns the second order writhe (i1,i3,i2,i4) of the
    representation, as defined in Lin and Wang.
    '''

    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate second order writhe'
                        'for something with more than 1 component')

    gc = gc[0].copy()

    arrows, signs = _crossing_arrows_and_signs(
        gc, representation.crossing_numbers)
    
    crossing_numbers = list(representation.crossing_numbers)
    result = 0
    for index, i1 in enumerate(crossing_numbers):
        arrow1 = arrows[i1]
        a1s, a1e = arrow1
        for i2 in crossing_numbers[index+1:]:
            arrow2 = arrows[i2]
            a2s, a2e = arrow2

            if ((a2s > a1e and a1s > a2s and a2e > a1s) or
                (a2e > a1e and a1s > a2e and a2s > a1s) or
                (a2e > a1s and a1e > a2e and a2s > a1e) or
                (a2s > a1s and a1e > a2s and a2e > a1e)):
            # if ((a2e > a1e and a1s > a2e and a2s > a1s) or
            #     (a2s > a1s and a1e > a2s and a2e > a1e)):
                result += signs[i1] * signs[i2]

    return result

def arnold_2St_2Jplus(representation):
    '''Returns J+ + 2 * St where J+ and St are Arnold's invariants of
    plane curves.

    The calculation is performed by transforming the representation
    into a representation of an unknot by flipping crossings, then
    calculating the second order writhe.

    See 'Invariants of curves and fronts via Gauss diagrams', M
    Polyak, Topology 37, 1998.

    '''

    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate arnold_2St_2Jplus'
                        'for something with more than 1 component')

    gc = gc[0].copy()
    seen_crossings = set()
    changed_crossings = set()
    for row in gc:
        if row[0] not in seen_crossings:
            seen_crossings.add(row[0])
            if row[1] < 0:
                changed_crossings.add(row[0])
                row[1] *= -1
                row[2] *= -1
        elif row[0] in changed_crossings:
            row[1] *= -1
            row[2] *= -1
    arrows, signs = _crossing_arrows_and_signs(
        gc, representation.crossing_numbers)
    
    crossing_numbers = list(representation.crossing_numbers)
    result = 0
    for index, i1 in enumerate(crossing_numbers):
        arrow1 = arrows[i1]
        a1s, a1e = arrow1
        for i2 in crossing_numbers[index+1:]:
            arrow2 = arrows[i2]
            a2s, a2e = arrow2

            if ((a2s > a1e and a1s > a2s and a2e > a1s) or
                (a2e > a1e and a1s > a2e and a2s > a1s) or
                (a2e > a1s and a1e > a2e and a2s > a1e) or
                (a2s > a1s and a1e > a2s and a2e > a1e)):
            # if ((a2e > a1e and a1s > a2e and a2s > a1s) or
            #     (a2s > a1s and a1e > a2s and a2e > a1e)):
                result += signs[i1] * signs[i2]

    return result

def arnold_2St_2Jminus(representation):
    '''Returns J- + 2 * St where J+ and St are Arnold's invariants of
    plane curves.

    See 'Invariants of curves and fronts via Gauss diagrams', M
    Polyak, Topology 37, 1998.

    '''
    return arnold_2St_2Jplus(representation) - len(representation)

def vassiliev_degree_2(representation):
    '''Calculates the Vassiliev invariant of degree 2 of the given
    knot. The representation must have just one knot component,
    this doesn't work for links.

    Parameters
    ==========
    representation : Anything convertible to a
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`).
    '''
    ## See Polyak and Viro
    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate v2'
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
            arrow2 = arrows[i2]
            a2s, a2e = arrow2

            if a2s > a1s and a2e < a1s and a1e > a2s:
                representations_sum += signs[i1] * signs[i2]
            elif a1s > a2s and a1e < a2s and a2e > a1s:
                representations_sum += signs[i1] * signs[i2]

    return representations_sum

def vassiliev_degree_3(representation, try_cython=True):
    '''Calculates the Vassiliev invariant of degree 3 of the given
    knot. The representation must have just one knot component,
    this doesn't work for links.

    Parameters
    ----------
    representation : Anything convertible to a
                     :class:`~pyknotid.representations.gausscode.GaussCode`
        A pyknotid representation class for the knot, or anything that
        can automatically be converted into a GaussCode (i.e. by writing
        :code:`GaussCode(your_object)`).
    try_cython : bool
        Whether to try and use an optimised cython version of the
        routine (takes about 1/3 of the time for complex representations).
        Defaults to True, but the python fallback will be *slower*
        than setting it to False if the cython function is not
        available.
    '''

    if try_cython:
        return _vassiliev_degree_3_numpy(representation)
    return _vassiliev_degree_3_python(representation)

        
def _vassiliev_degree_3_python(representation):
    ## See Polyak and Viro
    from pyknotid.representations.gausscode import GaussCode
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

    print()
    
    return int(round(representations_sum_1 / 2.)) + representations_sum_2

    
def _vassiliev_degree_3_numpy(representation):
    ## See Polyak and Viro
    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    gc = representation._gauss_code
    if len(gc) == 0:
        return 0
    elif len(gc) > 1:
        raise Exception('tried to calculate alexander polynomial'
                        'for something with more than 1 component')

    gc = gc[0]
    arrows = _crossing_arrows_and_signs_numpy(
        gc, representation.crossing_numbers)

    try:
        from pyknotid import cinvariants
    except ImportError:
        print('Failed to import cinvariants. Using *slow* python numpy method.')
    else:
        return int(round(cinvariants.vassiliev_degree_3(arrows)))

    num_crossings = len(arrows) * 2
    
    used_sets = set()
    representations_sum_1 = 0
    representations_sum_2 = 0
    arrow_range = range(len(arrows))
    for i1 in arrow_range:
        if i1 % 10 == 0:
            vprint('\rCurrently comparing index {}   '.format(i1), False)
        arrow1 = arrows[i1]
        a1s, a1e, sign1 = arrow1
        a1e = (a1e - a1s) % num_crossings
        for i2 in arrow_range:
            arrow2 = arrows[i2]
            a2s, a2e, sign2 = arrow2
            a2s = (a2s - a1s) % num_crossings
            a2e = (a2e - a1s) % num_crossings
            for i3 in arrow_range:
                arrow3 = arrows[i3]
                a3s, a3e, sign3 = arrow3
                a3s = (a3s - a1s) % num_crossings
                a3e = (a3e - a1s) % num_crossings

                ordered_indices = tuple(sorted((i1, i2, i3)))
                if ordered_indices in used_sets:
                    continue

                if (a2s < a1e and a3e < a1e and a3e > a2s and
                    a3s > a1e and a2e > a3s):
                    representations_sum_1 += sign1 * sign2 * sign3
                    used_sets.add(ordered_indices)
                if (a2e < a1e and a3s < a1e and a3s > a2e and
                    a2s > a1e and a3e > a2s):
                    representations_sum_2 += sign1 * sign2 * sign3
                    used_sets.add(ordered_indices)

    print
                    
    
    return int(round(representations_sum_1 / 2.)) + representations_sum_2


def self_linking(representation):
    '''Returns the self linking number J(K) of the Gauss code, an
    invariant of virtual knots. See Kauffman 2004 for more
    information.

    Currently only works for knots.
    '''

    from pyknotid.representations.gausscode import GaussCode
    if not isinstance(representation, GaussCode):
        representation = GaussCode(representation)

    if len(representation) == 0:
        return 0
    if len(representation._gauss_code[0]) == 0:
        return 0
    gauss_code = representation._gauss_code
    l = len(gauss_code[0][:,0])
    total_crossings = l/2
    crossing_counter = 1
    slink_counter = 0        

    for crossing_number in representation.crossing_numbers:
        occurences = n.where(gauss_code[0][:, 0] == crossing_number)[0]
        first_occurence = occurences[0]
        second_occurence = occurences[1]
        crossing_difference = second_occurence - first_occurence        

        if(crossing_difference % 2 == 0):
            slink_counter += 2 * gauss_code[0][occurences[0],2]

    return slink_counter   

def virtual_vassiliev_degree_3(representation):
    '''Calculates the virtual Vassiliev invariant of degree 3 (for
    non-long-knots) of the given representation, as described in
    'Finite type invariants of classical and virtual knots' by
    Goussarov, Polyak and Viro.

    Parameters
    ----------
    representation : :class:`~pyknotid.representations.representation.Representation`
        A representation class, or anything convertible to one
        (in principle).
    '''
    ## See Polyak and Viro and Goussarov
    from pyknotid.representations.gausscode import GaussCode
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

    diagrams_found = [[] for _ in range(8)]
    
    crossing_numbers = list(representation.crossing_numbers)
    used_sets = set()
    representations_sum = 0
    for index, i1 in enumerate(crossing_numbers):
        if index % 10 == 0 and len(crossing_numbers) > 10:
            vprint('\rCurrently comparing index {}'.format(index), False)
        arrow1 = arrows[i1]
        a1s, a1e = arrow1
        a1e = (a1e - a1s) % len(gc)
        for i2 in crossing_numbers:
            arrow2 = arrows[i2]
            a2s, a2e = arrow2
            a2s = (a2s - a1s) % len(gc)
            a2e = (a2e - a1s) % len(gc)

            # For the virtual Vassilev of order f, there are still
            # some diagrams with just 2 arrows:
            first_two_ordered_indices = tuple(sorted((i1, i2)))
            if (a2s < a1e and a2e > a1e and
                signs[i1] == signs[i2] and
                first_two_ordered_indices not in used_sets):
                if signs[i1] > 0:
                    representations_sum -= 1
                    # print('7')
                else:
                    representations_sum += 1
                    # print('8')
                used_sets.add(first_two_ordered_indices)
           
            for i3 in crossing_numbers:
                arrow3 = arrows[i3]
                a3s, a3e = arrow3
                a3s = (a3s - a1s) % len(gc)
                a3e = (a3e - a1s) % len(gc)

                ordered_indices = tuple(sorted((i1, i2, i3)))
                if ordered_indices in used_sets:
                    continue

                # These checks are for the diagrams of Polyak, Viro
                # and Goussarov in 'Finite type invariants of
                # classical and virtual knots' (page 13)

                if (a2e < a1e and a3s < a1e and a3s > a2e and
                    a2s > a1e and a3e > a2s):
                    representations_sum += 3*(signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                    # print('1')

                if (a2s < a1e and a3s < a1e and a2s < a3s and
                    a2e < a3e and a2e > a1e):
                    representations_sum -= (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                    # print('2')

                if (a2s < a1e and a3e < a1e and a3e > a2s and
                    a3s > a1e and a2e > a3s):
                    representations_sum += (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                    # print('3')

                if (a2e < a1e and a3s < a1e and a3s > a2e and
                    a3e > a1e and a2s > a3e):
                    representations_sum += (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                    # print('4')

                if (a2e < a1e and a3e < a1e and a3s < a2s and
                    a3s > a1e):
                    representations_sum -= (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                    # print('5')

                if (a2s < a1e and a3s < a1e and a3e < a2e and
                    a3e > a1e):
                    representations_sum -= (signs[i1] * signs[i2] *
                                              signs[i3])
                    used_sets.add(ordered_indices)
                    # print('6')

    print()

    return representations_sum
    return int(round(representations_sum))

