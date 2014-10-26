'''
Identify knots
==============

Functions for identifying knots based on their polynomials.'''

import database as db
from database import Knot
import converters

db.db.connect()

root_to_attr = {2: Knot.determinant,
                  3: Knot.alexander_imag_3,
                  4: Knot.alexander_imag_4}

def from_invariants(**kwargs):
    '''Takes invariants as kwargs, and does the appropriate conversion to
    return a list of database objects matching all criteria.

    Include the argument return_query=True to get the database query
    (which can have further operations performed on it). By default,
    this is False, and the function just returns a list of Knots.

    Does *not* support all available invariants. Currently, searching
    is supported by:
    - identifier (e.g. '3_1')
    - minimal crossings
    - signature
    - determinant
    - unknotting number
    - Alexander polynomial (provided as sympy)
    - Jones polynomial (provided as sympy)
    - HOMFLY
    - hyperbolic volume
    '''

    return_query = False
    conditions = []
    for invariant, value in kwargs.items():
        invariant = invariant.lower()
        if invariant == 'identifier':
            conditions.append(Knot.identifier == value)
        elif invariant == 'min_crossings':
            conditions.append(Knot.min_crossings == value)
        elif invariant == 'signature':
            conditions.append(Knot.signature == value)
        elif invariant in ['determinant', 'alexander_imag_2',
                           'alex_imag_2', 'weak_2']:
            conditions.append(Knot.determinant == value)
        elif invariant in ['alexander_imag_3', 'alex_imag_3', 'weak_3']:
            conditions.append(Knot.alexander_imag_3 == value)
        elif invariant in ['alexander_imag_4', 'alex_imag_4', 'weak_4']:
            conditions.append(Knot.alexander_imag_4 == value)
        elif invariant in ['roots']:
            for root, result in zip(range(2, len(value)+2), value):
                if result is not None:
                    conditions.append(root_to_attr[root] == result)
        elif invariant == 'unknotting_number':
            conditions.append(Knot.unknotting_number == value)
        elif invariant in ['alexander', 'alex']:
            val = converters.py2db_alexander(value)
            conditions.append(Knot.alexander == val)
        elif invariant == 'jones':
            val = converters.py2db_jones(value)
            chiral_val = converters.py2db_jones(
                converters.jones_other_chirality(value))
            conditions.append(Knot.jones << [val, chiral_val])
        elif invariant in ['homfly', ]:
            val = converters.py2db_homfly(value)
            chiral_val = converters.py2db_homfly(
                converters.homfly_other_chirality(value))
            conditions.append(Knot.homfly << [val, chiral_val])
        elif invariant in ['hypvol', 'hyp_vol', 'hyperbolic_volume']:
            conditions.append(Knot.hyperbolic_volume % '{}*'.format(str(value)))

    selection = Knot.select()
    for condition in conditions:
        selection = selection.where(condition)
    return list(selection)
