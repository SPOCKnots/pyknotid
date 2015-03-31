'''
Identify knots
==============

Functions for identifying knots based on their polynomials.'''

from . import database as db
from pyknot2.catalogue.database import Knot
from pyknot2.catalogue import converters

db.db.connect()

root_to_attr = {2: Knot.determinant,
                  3: Knot.alexander_imag_3,
                  4: Knot.alexander_imag_4}

def from_invariants(return_query=False, **kwargs):
    '''Takes invariants as kwargs, and does the appropriate conversion to
    return a list of database objects matching all the given criteria.

    Does *not* support all available invariants. Currently, searching
    is supported by:

    Parameters
    ----------
    identifier : str
        The name of the knot following knot atlas conventions, e.g. '3_1'
    min_crossings : int
        The minimal crossing number of the knot.
    max_crossings : int
        The maximal known crossing number of the knot. This may be higher
        than its actual crossing number, it serves only to prune the
        results list.
    signature : int
        The signature invariant.
    unknotting_number : int
        The unknotting number of the knot.
    alexander or alex : sympy
        The Alexander polynomial, provided as a sympy expression in a
        single variable (ideally 't').
    determinant or alexander_imag_2: int
        The Alexander polynomial at -1 (== exp(Pi I))
    alexander_imag_3 : int
        The abs of the Alexander polynomial at exp(2 Pi I / 3)
    alexander_imag_4 : int
        The abs of the Alexander polynomial at exp(Pi I / 2)
    roots : iterable
        The abs of the Alexander polnomial at the given roots, assumed
        to start at 2, e.g. passing (3, 2, 1) is the same as identifying
        at determinant=3, alexander_imag_3=2, alexander_imag_4=1. An
        entry of None means the value is ignored in the lookup.
    jones : sympy
        The Jones polynomial, provided as a sympy expression in a single
        variable (ideally 'q').
    homfly : sympy
        The HOMFLY-PT polynomial, provided as a sympy expression in two
        variables.
    hyperbolic_volume or hyp_vol or hypvol : float or str
        The hyperbolic volume of the knot complement. The lookup is a
        string comparison based on the given number of significant digits.
    vassiliev_order_2 or vassiliev_2 or v_2 or v2 : int
        The Vassiliev invariant of order 2.
    vassiliev_order_3 or vassiliev_3 or v_3 or v3 : int
        The Vassiliev invariant of order 3.
    symmetry : string
        The symmetry of the knot, one of 'reversible',
        'positive amphicheiral', 'negative amphicheiral', 'chiral'.
    other : iterable
        A list of other peewee terms that can be chained in ``where()``
        calls, e.g. ``database.Knot.min_crossings < 5``. This provides
        more flexibility than the other options.
    return_query : bool
        If True, returns the database iterator for the objects, otherwise
        returns a list. Defaults to False (i.e. the list). This will
        be much slower if the list is very large, but is convenient
        for most searches.
    '''

    return_query = False
    conditions = []
    for invariant, value in kwargs.items():
        invariant = invariant.lower()
        if invariant == 'identifier':
            conditions.append(Knot.identifier == value)
        elif invariant == 'min_crossings':
            conditions.append(Knot.min_crossings == value)
        elif invariant == 'max_crossings':
            conditions.append(Knot.min_crossings <= value)
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
        elif invariant in ['vassiliev_order_2', 'vassiliev_2', 'v_2', 'v2']:
            conditions.append(Knot.vassiliev_2 == value)
        elif invariant in ['vassiliev_order_3', 'vassiliev_3', 'v_3', 'v3']:
            conditions.append((Knot.vassiliev_3 == value) | (Knot.vassiliev_3 == -1*value))
        elif invariant == 'symmetry':
            conditions.append(Knot.symmetry == value.lower())
        elif invariant == 'other':
            for condition in value:
                conditions.append(condition)

    selection = Knot.select()
    for condition in conditions:
        selection = selection.where(condition)
    selection = selection.order_by(Knot.min_crossings)
    if return_query:
        return selection
    return list(selection)
