
import numpy as np
from functools import wraps
from itertools import combinations, permutations
from collections import defaultdict

from pyknot2.representations.gausscode import GaussCode

def _to_GaussCode(rep):
    if not isinstance(rep, GaussCode):
        rep = GaussCode(rep)
    return rep

def require_GaussCode(func):
    '''Convert the first argument of the function to a GaussCode.'''
    @wraps(func)
    def wrapper(rep, *args, **kwargs):
        rep = _to_GaussCode(rep)
        return func(rep, *args, **kwargs)
    return wrapper

def validate_diagram(d):
    '''Do some basic checks on whether an Arrow diagram (as a string of
    numbered crossings and signs) is valid.'''
    entries = d.split(',')
    counts = defaultdict(lambda: 0)
    signs = defaultdict(lambda: 0)
    for entry in entries:
        number = entry[:-1]
        if entry[-1] not in ('+', '-'):
            raise ValueError('Arrow diagrams must have crossing + or - ',
                             'but this has {}'.format(entry))
        counts[number] += 1
        signs[number] += (1 if entry[-1] == '+' else -1)

    if any([item != 2 for item in counts.values()]):
        raise ValueError('Diagram {} appears invalid: some indices '
                         'appear only once'.format(d))

    if any([item != 0 for item in signs.values()]):
        raise ValueError('Diagram {} appears invalid: sum of signs for '
                         'some crossings do not equal 0')
        
    
    

@require_GaussCode
def writhing_numbers(gc, diagrams, based=False):
    '''Returns the signed sum of representations of the given Arrow
    diagrams in the given representation.

    Parameters
    ----------

    gc : A :class:`~pyknot2.representations.gausscode.GaussCode` or
         equivalent representation.
        The knot for which to find the writhes. 
    diagrams : str or list or tuple
        A list of strings, or single string, representing Arrow
        diagrams, e.g. '1-2+1+2-' for Vassiliev 2.
    based : bool
        Whether the diagrams have basepoints (if True, assumed to be
        just before the first entry).
    '''

    if not isinstance(diagrams, (list, tuple)):
        diagrams = [diagrams]

    for d in diagrams:
        validate_diagram(d)

    level = 0

    code = gc._gauss_code
    code = code[0]
    from pyknot2.invariants import _crossing_arrows_and_signs
    arrows, signs = _crossing_arrows_and_signs(code, gc.crossing_numbers)

    crossing_numbers = list(gc.crossing_numbers)

    # degrees = [len(diagram.split(',')) for diagram in diagrams]

    degrees = defaultdict(lambda: [])
    for diagram in diagrams:
        degrees[len(diagram.split(','))].append(diagram)

    max_degree = max(degrees.keys())

    used_sets = set()

    # representations_sums = [0 for _ in diagrams]
    representations_sums = {d: 0 for d in diagrams}

    cur_arrows = [None for _ in range(max_degree)]

    combs = combinations(crossing_numbers, max_degree)
    for ci, comb in enumerate(combs):
        print('Combination {} of {}'.format(ci, len(combs)))

        cur_arrows = [arrows[i] for i in comb]
        cur_starts = [a[0] for a in cur_arrows]
        cur_ends = [a[1] for a in cur_arrows]

        arrow_indices = {arrow: i+1 for i, arrow in enumerate(cur_arrows)}


        perms = permutations(comb)

        for perm in perms:
            pass


        

        
        

    
    for i, depth in enumerate(range(1, max_degree + 1)):
        for index, arrow_num in enumerate(crossing_numbers):
            arrow = arrows[index]
            arr_s, arr_e = arrow
            arr_e = (arr_e - arr_s) % len(gc)

            cur_arrows = 
        
        
    
        

    return arrows, signs
    



def vassiliev_2(gc):
    return writhing_numbers(gc, '1-,2+,1+,2-', based=True)
