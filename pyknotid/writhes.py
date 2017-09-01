from __future__ import division, print_function
import numpy as np
from math import factorial
from functools import wraps
from itertools import combinations, permutations
from collections import defaultdict

from pyknotid.representations.gausscode import GaussCode
from pyknotid.utils import vprint


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
                         'some crossings do not equal 0'.format(d))
        

@require_GaussCode
def writhing_numbers(gc, diagrams, based=False):
    '''Returns the signed sum of representations of the given Arrow
    diagrams in the given representation.

    Parameters
    ----------

    gc : A :class:`~pyknotid.representations.gausscode.GaussCode` or
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

    multipliers = defaultdict(lambda : {})
    use_multipliers = False
    if any(['d' in diagram for diagram in diagrams]):
        use_multipliers = True
    if use_multipliers:
        for di, diagram  in enumerate(diagrams):
            terms = diagram.split(',')
            for term in terms:
                if term[-1] == 'd':
                    multiplier = 2
                else:
                    multiplier = 1
                term = int(term[:-2]) if 'd' in term else int(term[:-1])
                multipliers[diagram.replace('d', '')][term] = multiplier
            diagrams[di] = diagram.replace('d', '')

    for d in diagrams:
        validate_diagram(d)

    level = 0

    code = gc._gauss_code
    code = code[0]
    gc_len = len(gc)
    code_len = len(code)
    from pyknotid.invariants import _crossing_arrows_and_signs
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
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] < l[oi][0])
                    else:
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] > l[oi][0])

                if terms.index(number + '-') < terms.index(other_number + '+'):
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] < l[oi][1])
                else:
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] > l[oi][1])

                if terms.index(number + '+') < terms.index(other_number + '-'):
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[oi][0])
                else:
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[oi][0])

                # This one seems to not be necessary for diagrams where all arrows cross?
                if terms.index(number + '+') < terms.index(other_number + '+'):
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[oi][1])
                else:
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[oi][1])

            if i == 0:
                continue
            if terms.index(number + '+') < terms.index(number + '-'):
                relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[i][0])
            else:
                relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[i][0])


    max_degree = max(degrees.keys())

    representations_sums = {d: 0 for d in diagrams}
    used_sets = {d: set() for d in diagrams}

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
        if ci % 10000 == 0:
            vprint('\rCombination {} of {}    '.format(ci + 1, num_combs),
                   newline=False, condition=(ci % 10000) == 0)

        if based:
            perms = [comb]
        else:
            perms = permutations(comb)

        ordered_indices = tuple(sorted(comb))
        for diagram in diagrams:
            if ordered_indices not in used_sets[diagram]:
                break
        else:
            continue

        for perm in perms:
            cur_arrows = [list(arrows[i]) for i in perm]

            a1s = cur_arrows[0][0]
            if based:
                a1s = 0

            for i, arrow in enumerate(cur_arrows):
                arrow[0] = (arrow[0] - a1s) % code_len
                arrow[1] = (arrow[1] - a1s) % code_len


            for diagram in diagrams:
                if ordered_indices in used_sets[diagram]:
                    continue
                for relation in relations[diagram]:
                    if not relation(cur_arrows):
                        break
                else:
                    if use_multipliers:
                        representations_sums[diagram] += (
                            reduce(lambda x, y: x*y,
                                   [signs[arrow_i]**multipliers[diagram][num+1] for num, arrow_i in enumerate(perm)]))
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
                                     '1-,2-,3+,1+,3-,2+'], based=False,
                                forbid_all_symmetry=True)
    print('results', results)
    return results['1-,2-,3+,1+,3-,2+'] // 2 + results['1-,2+,3-,1+,2-,3+']

def vassiliev_4(gc):
    d4 = ['1-,2-,3+,1+,4-,3-,2+,4+',
          '1-,2+,3-,1+,4-,3+,2-,4+',
          '1-,2+,3-,4+,2-,1+,4-,3+',
          '1-,2-,3+,1+,4-,2+,3-,4+',
          '1-,2+,3-,4+,1+,2-,4-,3+',
          '1-,2+,3-,4-,1+,4+,3+,2-',
          '1-,2+,3-,4-,1+,3+,4+,2-',
          '1-,2+,3-,4+,1+,4-,3+,2-',
          '1-,2+,3+,4-,1+,3-,4+,2-',
          '1-,2+,3-,4-,1+,4+,2-,3+',
          '1-,2-,3+,4-,2+,1+,3-,4+',
          '1-,2-,3+,4-,1+,2+,3-,4+',
          '1-,2-,1+,3-,2+,4-,3+,4+',
          '1-,2-,1+,3+,2+,4-,3-,4+']

    d3 = ['1-,2+,3-d,1+,3+d,2-',
          '1-d,2+,3-,1+d,2-,3+']

    results_4 = writhing_numbers(gc, d4)

    results_3 = writhing_numbers(gc, d3)

    total = (results_4[d4[0]] + 6*results_4[d4[1]] + 2*results_4[d4[2]] +
             3*results_4[d4[3]] + results_4[d4[4]] + 2*results_4[d4[5]] +
             2*results_4[d4[6]] - results_4[d4[7]] + results_4[d4[8]] +
             results_4[d4[9]] + 2*results_4[d4[10]] + 2*results_4[d4[11]] +
             results_4[d4[12]] + results_4[d4[13]] +
             results_3[d3[0]] + results_3[d3[1]])

    print('results 4')
    for diagram, value in results_4.items():
        print(diagram, ':', value)
    print('results 3')
    for diagram, value in results_3.items():
        print(diagram, ':', value)
    return total

             
def slip_vassiliev_2(gc):
    codes = ['1-,2+,3+,1+,2-,3-',
             '1-,2+,3+,1+,3-,2-',
             '1-,2-,3+,1+,2+,3-',
             '1-,2-,3+,2+,1+,3-']
    # codes = ['1-,2+,3+,1+,2-,3-']
    results = writhing_numbers(gc, codes, based=True)

    gc = gc.flipped()
    results2 = writhing_numbers(gc, codes, based=True)

    for code in codes:
        print('{}: {}'.format(code, results[code]))
    return np.sum(results.values()) + np.sum(results2.values())
    return np.sum(results.values()), np.sum(results2.values())


def vassiliev_2_long_form(gc):

    results_1 = writhing_numbers(gc, ['1-,2+,1+,2-'], based=False)

    results_2 = writhing_numbers(gc, ['1-,2-d,3-,1+,2+d,3+',
                                      '1-,2-,3-d,1+,3+d,2+',
                                      '1-d,2-,3-,2+,1+d,3+',
                                      '1-,2-,1+,3-d,3+d,2+',
                                      '1-,2-,1+,3+d,3-d,2+'],
                                 based=False)
    
    num_crossings = float(len(gc.crossing_numbers))

    total = np.sum(results_2.values()) + results_1.values()[0]
    if total != 0 and num_crossings == 0:
        raise ValueError('Found no crossings but non-zero total vassiliev')
    if num_crossings == 0:
        return 0

    return total / num_crossings


    


@require_GaussCode
def writhing_numbers_numpy(gc, diagrams, based=False):
    '''Returns the signed sum of representations of the given Arrow
    diagrams in the given representation.

    Parameters
    ----------

    gc : A :class:`~pyknotid.representations.gausscode.GaussCode` or
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
    from pyknotid.invariants import _crossing_arrows_and_signs
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
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] < l[oi][0])
                    else:
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] > l[oi][0])

                    if terms.index(number + '-') < terms.index(other_number + '+'):
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] < l[oi][1])
                    else:
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][0] > l[oi][1])

                if terms.index(number + '+') < terms.index(other_number + '-'):
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[oi][0])
                else:
                    relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[oi][0])

                if i == 0:
                    if terms.index(number + '+') < terms.index(other_number + '+'):
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] < l[oi][1])
                    else:
                        relations[diagram].append(lambda l, i=i, oi=oi: l[i][1] > l[oi][1])


    max_degree = max(degrees.keys())

    used_sets = set()

    # representations_sums = [0 for _ in diagrams]
    representations_sums = {d: 0 for d in diagrams}
    used_sets = {d: set() for d in diagrams}

    print('arrows are', arrows)

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
        if ci % 10000 == 0:
            vprint('\rCombination {} of {}    '.format(ci + 1, num_combs),
                   newline=False)

        if based:
            perms = [comb]
        else:
            perms = permutations(comb)

        ordered_indices = tuple(sorted(comb))
        for diagram in diagrams:
            if ordered_indices not in used_sets[diagram]:
                break
        else:
            continue

        for perm in perms:
            cur_arrows = [list(arrows[i]) for i in perm]

            a1s = cur_arrows[0][0]
            if based:
                a1s = 0

            for i, arrow in enumerate(cur_arrows):
                arrow[0] = (arrow[0] - a1s) % code_len
                arrow[1] = (arrow[1] - a1s) % code_len


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
