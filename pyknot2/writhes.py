from __future__ import division, print_function
import numpy as np
from math import factorial
from functools import wraps
from itertools import combinations, permutations
from collections import defaultdict

from pyknot2.representations.gausscode import GaussCode
from pyknot2.utils import vprint


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
    gc_len = len(gc)
    code_len = len(code)
    from pyknot2.invariants import _crossing_arrows_and_signs
    arrows, signs = _crossing_arrows_and_signs(code, gc.crossing_numbers)

    crossing_numbers = list(gc.crossing_numbers)

    # degrees = [len(diagram.split(',')) for diagram in diagrams]

    degrees = defaultdict(lambda: [])
    for diagram in diagrams:
        degrees[len(diagram.split(',')) // 2].append(diagram)

    relations = {diagram: [] for diagram in diagrams}
    for diagram in diagrams:
        degree = len(diagram.split(',')) // 2
        num_relations = factorial(degree - 1) * 4

        terms = diagram.split(',')
        numbers = [term[:-1] for term in terms]

        number_strs = list(sorted(set(numbers), key=lambda j: int(j)))
        for i, number in enumerate(number_strs):
            for oi, other_number in enumerate(number_strs[i+1:]):
                oi += i + 1
                if i != 0:
                    if terms.index(number + '-') < terms.index(other_number + '-'):
                        # relations[diagram].append(lambda x, y: x[0] < y[0])
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] < l[oi][0])
                    else:
                        # relations[diagram].append(lambda x, y: x[0] > y[0])
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] > l[oi][0])

                    if terms.index(number + '-') < terms.index(other_number + '+'):
                        # relations[diagram].append(lambda x, y: x[0] < y[1])
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] < l[oi][1])
                    else:
                        # relations[diagram].append(lambda x, y: x[0] > y[1])
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] > l[oi][1])

                if terms.index(number + '+') < terms.index(other_number + '-'):
                    # relations[diagram].append(lambda x, y: x[1] < y[0])
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[oi][0])
                else:
                    # relations[diagram].append(lambda x, y: x[1] > y[0])
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[oi][0])
                    
                # This one is unnecessary?
                if terms.index(number + '+') < terms.index(other_number + '+'):
                    # relations[diagram].append(lambda x, y: x[1] < y[1])
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[oi][1])
                else:
                    # relations[diagram].append(lambda x, y: x[1] > y[1])
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[oi][1])


    max_degree = max(degrees.keys())

    used_sets = set()

    # representations_sums = [0 for _ in diagrams]
    representations_sums = {d: 0 for d in diagrams}
    used_sets = {d: set() for d in diagrams}

    cur_arrows = [None for _ in range(max_degree)]

    combs = combinations(crossing_numbers, max_degree)
    try:
        num_combs = (factorial(len(crossing_numbers)) //
                     factorial(max_degree) //
                     factorial(len(crossing_numbers) - max_degree))
    except ValueError:
        num_combs = 0
    strs = [None for _ in range(max_degree)] * 2
    order = [None for _ in range(max_degree)] * 2
    for ci, comb in enumerate(combs):
        vprint('\rCombination {} of {}    '.format(ci + 1, num_combs),
               newline=False, condition=(ci % 100) == 0)

        if based:
            perms = [comb]
        else:
            perms = permutations(comb)

        for perm in perms:
            cur_arrows = [list(arrows[i]) for i in perm]

            if based and not reduce(lambda x, y: y > x, perm):
                continue

            a1s = cur_arrows[0][0]
            if based:
                a1s = 0

            for i, arrow in enumerate(cur_arrows):
                arrow[0] = (arrow[0] - a1s) % code_len
                arrow[1] = (arrow[1] - a1s) % code_len

            ordered_indices = tuple(sorted(perm))

            for diagram in diagrams:
                if ordered_indices in used_sets[diagram]:
                    continue
                for relation in relations[diagram]:
                    if not relation(cur_arrows):
                        break
                else:
                    representations_sums[diagram] += (
                        reduce(lambda x, y: x*y,
                               [signs[arrow_i] for arrow_i in perm]))
                    used_sets[diagram].add(ordered_indices)
                        
    vprint()

    return representations_sums



def vassiliev_2(gc):
    results = writhing_numbers(gc, '1-,2+,1+,2-', based=True)
    print('results', results)
    return results['1-,2+,1+,2-']


def vassiliev_3(gc):
    results =  writhing_numbers(gc, ['1-,2+,3-,1+,2-,3+',
                                     '1-,2-,3+,1+,3-,2+'], based=False)
    print('results', results)
    return results['1-,2-,3+,1+,3-,2+'] // 2 + results['1-,2+,3-,1+,2-,3+']

def slip_vassiliev_2(gc):
    results = writhing_numbers(gc, ['1+,2-,3-,1-,2+,3+',
                                    '1+,2-,3-,1-,3+,2+',
                                    '1+,2+,3-,1-,2-,3+',
                                    '1+,2+,3-,2-,1-,3+'], based=True)
    print('results', results)
    return results
    return results['1+,2-,3-,1-,2+,3+']
