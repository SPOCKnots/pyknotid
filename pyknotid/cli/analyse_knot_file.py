#!/usr/bin/env python2

from __future__ import print_function, division

import argparse
import sys
import numpy as np

from os import path
sys.path.append(path.split(path.dirname(path.abspath(__file__)))[0])

from pyknotid.spacecurves.knot import Knot


def knot_from_file(filen):
    with open(filen, 'r') as fileh:
        text = fileh.read()

    return knot_from_text(text)
    

def knot_from_text(text):

    points = []
    for line in text.split('\n'):
        values = line.split(' ')
        if len(values) == 1 and len(values[0]) == 0:
            continue
        if len(values) != 3:
            raise ValueError('Invalid text passed.')
        points.append([float(v) for v in values])

    k = Knot(points, verbose=False)
    return k


def main():
    parser = argparse.ArgumentParser(
        description='Analyse the knot type of a curve in a data file')

    parser.add_argument('filename', help='The file to load')

    parser.add_argument('--identify', dest='identify', action='store_true',
                        help=('Look up the knot in the pyknotid database, '
                              'using its invariants. This may take a few '
                              'seconds on the first run, and can only return '
                              'knots with the same invariants, not necessarily '
                              'perfect matches.'))
    parser.add_argument('--no-identify', dest='identify', action='store_false')
    parser.set_defaults(identify=True)

    parser.add_argument('--rotate', dest='rotate', action='store_true',
                        help=('Rotate the space curve randomly before '
                              'identifying. Usually a good idea, and '
                              'defaults to True.'))
    parser.add_argument('--no-rotate', dest='rotate', action='store_false')
    parser.set_defaults(rotate=True)

    parser.add_argument('--writhe', action='store_true',
                        help=('If passed, will also print an approximation '
                              'for the 3D writhe and average crossing number '
                              'of the knot.'))
    parser.add_argument('--projections', type=int, default=100,
                        help=('The number of projections to average over '
                              'when calculating writhe.'))

    parser.add_argument('--with-vassiliev-3', action='store_true',
                        help=('If passed, also calculates the Vassiliev '
                              'invariant of third order (per Polyak and '
                              'Viro'))

    args = parser.parse_args(sys.argv[1:])

    k = knot_from_file(args.filename)

    print('Loaded line with {} points'.format(len(k.points)))
    
    if args.rotate:
        k.rotate()
    else:
        k.rotate((0.001, 0.0023, 0.0052))

    invariants = ['determinant',
                  'vassiliev_degree_2',
                  'alexander_at_roots']

    print('  determinant:', k.determinant())
    print('  Vassiliev degree 2:', k.vassiliev_degree_2())
    if args.with_vassiliev_3:
        print('  Vassiliev degree 3:', k.vassiliev_degree_3())
    print('  Alexander at roots 2,3,4:', k.alexander_at_root((2, 3, 4)))

    if args.identify:
        ## Identification taken from knot.py (and extended)
        identify_kwargs = {}
        for root in (2, 3, 4):
            identify_kwargs[
                'alex_imag_{}'.format(root)] = k.alexander_at_root(root)

        import sympy as sym
        poly = k.alexander_polynomial(sym.var('t'))
        identify_kwargs['alexander'] = poly

        if len(k.gauss_code()) < 16:
            identify_kwargs['max_crossings'] = len(k.gauss_code())

        if args.with_vassiliev_3:
            identify_kwargs['vassiliev_3'] = k.vassiliev_degree_3()

        from pyknotid.catalogue.identify import from_invariants
        identities = from_invariants(**identify_kwargs)
        if len(identities) < 10:
            print('  Possible identities:', identities)
        else:
            print('  First 10 of {} possible identities:'.format(len(identities)),
                identities[:10])

    import warnings
    warnings.filterwarnings('ignore')
    if args.writhe:
        acn, writhe = k._writhe_and_crossing_numbers(args.projections)
        print('  average crossing number:', acn)
        print('  writhe:', writhe)

if __name__ == "__main__":
    main()
    
