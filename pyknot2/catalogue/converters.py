'''Functions for converting between representations, e.g. from sympy
polynomials to the json lists stored in the database models.  '''

import sympy as sym
import numpy as n    
import json

## 1) Conversion from database to sympy

def sympify_alexander(coeffs):
    '''Takes an Alexander polynomial as a list of coefficients, and
    returns a sympy polynomial representation.'''
    t = sym.var('t')
    index = 0
    result = 0
    for coeff in coeffs:
        result += coeff * t**index
        index += 1
    return result

def sympify_jones(coeffs):
    '''Takes a Jones polynomial as a list of (coefficient, index) tuples,
    and returns a sympy polynomial representation.'''
    q = sym.var('q')
    result = 0
    for coeff, index in coeffs:
        result += coeff*q**index
    return result

def sympify_homfly(coeffs):
    '''Takes a HOMFLY polynomial as a list of (coefficient, a_index,
    z_index) tuples, and returns a sympy polynomial representation.
    '''
    a = sym.var('a')
    z = sym.var('z')
    result = 0
    for coeff, a_index, z_index in coeffs:
        result += coeff * a**a_index * z**z_index
    return result

def json_to_alexander(s):
    '''Takes an Alexander polynomial json serialisation and returns the
    sympy polynomial.'''
    return sympify_alexander(json.loads(s))

def json_to_jones(s):
    '''Takes an Jones polynomial json serialisation and returns the
    sympy polynomial.'''
    return sympify_jones(json.loads(s))

def json_to_homfly(s):
    '''Takes a HOMFLY polynomial json serialisation and returns the
    sympy polynomial.'''
    return sympify_homfly(json.loads(s))
    

## 2) Conversion from sympy to database json

def desympify_alexander(p):
    """Takes a sympy Alexander polynomial p, and returns a tuple of
    coefficients in order.
    """
    if p == 1:
        return (1, )
    p = p.expand()
    terms = p.as_terms()[0]
    coeffs = n.zeros((len(terms),2),dtype=int)
    for i in range(len(terms)):
        entry = terms[i]
        coeffs[i,0] = int(entry[1][0][0])
        coeffs[i,1] = int(entry[1][1][0])

    coeffs = coeffs[n.argsort(coeffs[:,1])]
    if coeffs[0][0] < 0:
        coeffs[:,0] *= -1

    cmin = coeffs[0,1]
    cmax = coeffs[-1,1]

    cs = n.zeros(cmax-cmin+1,dtype=int)
    for entry in coeffs:
        cs[entry[1]-cmin] = int(entry[0])

    return tuple(map(int,cs))

def desympify_jones(p):
    """Takes a sympy Jones polynomial p, and returns a tuple of
    (coefficient, index) tuples.
    """
    if p == 1:
        return (1, )
    p = p.expand()
    terms = p.as_terms()[0]
    coeffs = n.zeros((len(terms), 2), dtype=int)
    for i in range(len(terms)):
        entry = terms[i]
        coeffs[i,0] = int(entry[1][0][0])
        coeffs[i,1] = int(entry[1][1][0])

    coeffs = coeffs[n.argsort(coeffs[:, 1])]
    # if coeffs[0][0] < 0:
    #     coeffs[:, 0] *= -1

    cmin = coeffs[0, 1]
    cmax = coeffs[-1, 1]

    cs = []
    for entry in coeffs:
        cs.append((int(entry[0]), int(entry[1])))
        # Need int to convert from numpy.int64

    return tuple(cs)

def desympify_homfly(p):
    """Takes a sympy HOMFLY polynomial p, and returns a tuple of
    (coefficient, index_a, index_z) tuples.
    """
    if p == 1:
        return (1, )
    p = p.expand()
    terms = p.as_terms()[0]
    coeffs = n.zeros((len(terms), 3), dtype=int)
    for i in range(len(terms)):
        entry = terms[i]
        coeffs[i,0] = int(entry[1][0][0])
        coeffs[i,1] = int(entry[1][1][0])
        coeffs[i,2] = int(entry[1][1][1])

    inds = n.lexsort((coeffs[:,2], coeffs[:,1]))
    coeffs = coeffs[inds]

    cmin = coeffs[0, 1]
    cmax = coeffs[-1, 1]

    cs = []
    for entry in coeffs:
        cs.append((int(entry[0]), int(entry[1]), int(entry[2])))
        # Need int to convert from numpy.int64

    return tuple(cs)

def alexander_to_json(p):
    '''Takes an Alexander polynomial, and returns a json serialised list
    of its coefficients.'''
    return json.dumps(desympify_alexander(p))

def jones_to_json(p):
    '''Takes a Jones polynomial, and returns a json serialised list
    of its coefficients.'''
    return json.dumps(desympify_jones(p))

def homfly_to_json(p):
    '''Takes a HOMFLY polynomial, and returns a json serialised list of
    its coefficients.'''
    return json.dumps(desympify_homfly(p))

## 3) Conversion from rdf to sympy

def rdf_poly_to_sympy(p, vars=None):
    '''Converts polynomials as stored in knot atlas rdf data to sympy
    objects. The process is *very* crude (trim off <math> xml, replace
    characters with python equivalents), but seems to suffice for the
    data currently harvested. 

    vars should be a list of variables in the polynomial
    (e.g. ['t','q']). Appropriate sympy variables are constructed to
    represent these.

    The conversion is done using eval and exec, so you should be *very
    sure* that your rdf doesn't contain malicious code!

    '''
    p = p.strip()
    p = p.replace('<math>', '')
    p = p.replace('</math>', '')
    p = p.strip()
    p = p.replace('{', '(')
    p = p.replace('}', ')')
    p = p.replace('^', '**')
    p = p.replace(' + ','+')
    p = p.replace(' +','+')
    p = p.replace('+ ','+')
    p = p.replace(' - ','-')
    p = p.replace(' -','-')
    p = p.replace('- ','-')
    p = p.replace(' ','*')

    for entry in vars:
        exec(''.join((entry, ' = sym.var(\'', entry, '\')')))

    return eval(p)


## 4) User interface convenience functions (standard names)

def db2py_alexander(s):
    return json_to_alexander(s)

def py2db_alexander(p):
    return alexander_to_json(p)

def db2py_jones(s):
    return json_to_jones(s)

def py2db_jones(p):
    return jones_to_json(p)

def db2py_homfly(s):
    return json_to_homfly(s)

def py2db_homfly(p):
    return homfly_to_json(p)

## 5) Converters between invariants

def homfly_to_jones(p):
    '''Takes a homfly polynomial p, and returns the Jones polynomial
    through variable substitution.'''
    a = sym.var('a')
    z = sym.var('z')
    q = sym.var('q')

    np = p.subs(a, 1/q)
    np = np.subs(z, sym.sqrt(q) - 1/sym.sqrt(q))
    np = np.expand()
    return np

def homfly_other_chirality(p):
    '''Takes a homfly polynomial p, and replaces a with 1/a to give the
    polynomial of the same knot with opposite chirality.'''
    if p == 1:
        return 1
    a = sym.var('a')
    np = p.subs(a, 1/a)
    np = np.expand()
    return np

def jones_other_chirality(p):
    '''Takes a jones polynomial p, and replaces q with 1/q to give the
    polynomial of the same knot with opposite chirality.'''
    if p == 1:
        return 1
    q = sym.var('q')
    np = p.subs(q, 1/q)
    np = np.expand()
    return np
