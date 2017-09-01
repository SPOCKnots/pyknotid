'''
Space curve complexity
======================

Functions for evaluating the (mainly geometrical) complexity of space
curves.
'''

from __future__ import print_function

from . import Knot
from . import SpaceCurve
from pyknotid.spacecurves.rotation import (get_rotation_angles,
                                          rotate_to_top)
from pyknotid.utils import vprint
import numpy as n
import numpy as np

import sys


def writhe_and_crossing_number(points, number_of_samples=10,
                               verbose=True, remove_duplicate_angles=False,
                               include_closure=False,
                               **kwargs):
    '''
    Returns the writhe and crossing number of the given curve.

    Parameters
    ----------
    points : array-like
        An Nx3 array of points in the line.
    number_of_samples : int
        The number of projection directions to average over. These will
        be chosen to be roughly uniform on the sphere, as per
        :func:`get_rotation_angles`.
    remove_duplicate_angles : bool
        Whether to remove duplicate angles (i.e. points exactly opposite
        one another on the sphere), adjusting the average appropriately.
        This argument is currently experimental and defaults to False;
        do *not* trust it to remove points correctly.
    **kwargs :
        These are passed to the raw_crossings method of Knot classes
        used internally.
    '''

    angles = get_rotation_angles(number_of_samples)
    if remove_duplicate_angles:
        angles = angles[:int(len(angles) / 2.)]

    crossing_numbers = []
    writhes = []

    for theta, phi in angles:
        k = SpaceCurve(points, verbose=False)
        k._apply_matrix(rotate_to_top(theta, phi))
        crossings = k.raw_crossings(include_closure=include_closure, **kwargs)
        crossing_numbers.append(len(crossings) / 2)
        writhes.append(n.sum(crossings[:, 3]) / 2. if len(crossings) else 0)

    return n.average(crossing_numbers), n.average(writhes)


# def writhe_integral(points, closed=False):

#     ps = points

#     dps = np.roll(points, -1, axis=0) - points

#     writhe = 0.0

#     points2 = points[:-1] if not closed else points

#     for i1, p1 in enumerate(points2):
#         for i2, p2 in enumerate(points2[i1+1:]):
#             i2 += i1 + 1

#             diff = p2 - p1
#             cross = np.cross(dps[i1], dps[i2])
#             writhe += diff.dot(cross) / np.sqrt(np.sum(diff**2))**3

#     return 2 * writhe / (4*np.pi)


def writhe_integral(points, closed=False):

    ps = points

    dps = np.roll(points, -1, axis=0) - points

    writhe = 0.0

    points2 = points[:-1] if not closed else points

    for i1, p1 in enumerate(points[:-3]):
        p2 = points[i1 + 1]
        for i2, p3 in enumerate(points[i1+2:-1]):
            # p2 = p2 + 0.000001
            i2 += i1 + 2

            p4 = points[i2 + 1]

            r12 = p2 - p1
            r13 = p3 - p1
            r14 = p4 - p1
            r23 = p3 - p2
            r24 = p4 - p2
            r34 = p4 - p3

            n1 = np.cross(r13, r14)
            n1 /= np.sqrt(np.sum(n1**2))

            n2 = np.cross(r14, r24)
            n2 /= np.sqrt(np.sum(n2**2))

            n3 = np.cross(r24, r23)
            if np.any(np.abs(n3) > 0.):
                n3 /= np.sqrt(np.sum(n3**2))

            n4 = np.cross(r23, r13)
            if np.any(np.abs(n4) > 0.):
                n4 /= np.sqrt(np.sum(n4**2))

            if np.any(np.isnan(n3)):
                print('!!! nan')
                print(i1, i2)
                print(p1, p2, p3, p4)
                print('nan', r23, r13, np.cross(r23, r13))

            t1, t2, t3, t4 = np.clip([n1.dot(n2),
                                      n2.dot(n3),
                                      n3.dot(n4),
                                      n4.dot(n1)],
                                     -1, 1)

            writhe_contribution = (np.arcsin(t1) +
                                   np.arcsin(t2) +
                                   np.arcsin(t3) +
                                   np.arcsin(t4))

            writhe_contribution *= np.sign(np.cross(r34, r12).dot(r13))

            if np.isnan(writhe_contribution):
                print()
                print('nan')
                print(i1, i2, n1, n2, n3, n4)

            writhe += writhe_contribution


    return 2 * writhe / (4*np.pi)

def higher_order_writhe_integral(points, order=(1, 3, 2, 4), try_cython=True):

    ps = points
    dps = np.roll(points, -1, axis=0) - points

    contributions = np.zeros((len(points[:-1]), len(points[:-1])))
    contributions = np.zeros((len(points), len(points)))

    assert len(set(order)) == len(order)
    assert set(order) == set((1, 2, 3, 4))

    print('Calculating writhe contributions')

    order = [o - 1 for o in order]
    for i1, p1 in enumerate(points[:-3]):
        print('\ri = {} / {}'.format(i1, len(points) - 4), end='')
        sys.stdout.flush()
        p2 = points[i1 + 1]
        for i2, p3 in enumerate(points[i1+2:-1]):
            i2 += i1 + 2

            p4 = points[i2 + 1]

            r12 = p2 - p1
            r13 = p3 - p1
            r14 = p4 - p1
            r23 = p3 - p2
            r24 = p4 - p2
            r34 = p4 - p3

            n1 = np.cross(r13, r14)
            n1 /= np.sqrt(np.sum(n1**2))

            n2 = np.cross(r14, r24)
            n2 /= np.sqrt(np.sum(n2**2))

            n3 = np.cross(r24, r23)
            if np.any(np.abs(n3) > 0.):
                n3 /= np.sqrt(np.sum(n3**2))

            n4 = np.cross(r23, r13)
            if np.any(np.abs(n4) > 0.):
                n4 /= np.sqrt(np.sum(n4**2))

            if np.any(np.isnan(n1)):
                print('!!! nan')
                print(i1, i2)
                print(p1, p2, p3, p4)
                print('nan', r23, r13, np.cross(r23, r13))

            # When the vectors are nearly the same, floating point
            # errors can sometimes make the output a tiny bit higher
            # than 1
            t1, t2, t3, t4 = np.clip([n1.dot(n2),
                                      n2.dot(n3),
                                      n3.dot(n4),
                                      n4.dot(n1)],
                                     -1, 1)

            writhe_contribution = (np.arcsin(t1) +
                                   np.arcsin(t2) +
                                   np.arcsin(t3) +
                                   np.arcsin(t4))

            if np.isnan(writhe_contribution):
                print()
                print('nan!')
                print(i1, i2, n1, n2, n3, n4, writhe_contribution)
                print(n1.dot(n2) > 1, n2.dot(n3) > 1, n3.dot(n4) > 1, n4.dot(n1) > 1)
                print(n1.dot(n2), np.arcsin(n1.dot(n2)))
                print(n2.dot(n3), np.arcsin(n2.dot(n3)))
                print(n3.dot(n4), np.arcsin(n3.dot(n4)))
                print(n4.dot(n1), np.arcsin(n4.dot(n1)))

            writhe_contribution *= np.sign(np.cross(r34, r12).dot(r13))
    
            contributions[i1, i2] = writhe_contribution
            contributions[i2, i1] = writhe_contribution

    print()
    print('Calculating higher order writhe')

    how_func = _higher_order_writhe

    if try_cython:
        try:
            from pyknotid.spacecurves.ccomplexity import cython_higher_order_writhe 
            order = np.array(order)
            how_func = cython_higher_order_writhe
        except ImportError:
            print('Failed to import ccomplexity, using pure python instead')
    
    writhe = how_func(points, contributions, order)

    return writhe / (2*np.pi)**2

def _higher_order_writhe(points, contributions, order):
    writhe = 0.0

    for i1, p1s in enumerate(points[:-3]):
        print('\ri1', i1, len(points) - 4, end='')
        sys.stdout.flush()
        for i2, p2s in enumerate(points[i1+1:-1]):
            i2 += i1 + 1
            for i3, p3s in enumerate(points[i2+1:-1]):
                i3 += i2 + 1
                for i4, p4s in enumerate(points[i3+1:-1]):
                    i4 += i3 + 1

                    indices = (i1, i2, i3, i4)
                    # print('indices', indices)
                    # print('shape', contributions.shape)
                    writhe += (contributions[indices[order[0]],
                                             indices[order[1]]] *
                               contributions[indices[order[2]],
                                             indices[order[3]]])
    print()

    return writhe


def writhing_matrix(points, order=(1, 3, 2, 4)):

    ps = points
    dps = np.roll(points, -1, axis=0) - points

    contributions = np.zeros((len(points[:-1]), len(points[:-1])))
    contributions = np.zeros((len(points), len(points)))

    assert len(set(order)) == len(order)
    assert set(order) == set((1, 2, 3, 4))

    print('Calculating writhe contributions')

    order = [o - 1 for o in order]

    for i1, p1 in enumerate(points[:-3]):
        if i1 % 50 == 0:
            print('\ri = {} / {}'.format(i1, len(points) - 4), end='')
        p2 = points[i1 + 1]
        for i2, p3 in enumerate(points[i1+2:-1]):
            # p2 = p2 + 0.000001
            i2 += i1 + 2

            p4 = points[i2 + 1]

            r12 = p2 - p1
            r13 = p3 - p1
            r14 = p4 - p1
            r23 = p3 - p2
            r24 = p4 - p2
            r34 = p4 - p3

            n1 = np.cross(r13, r14)
            n1 /= np.sqrt(np.sum(n1**2))

            n2 = np.cross(r14, r24)
            n2 /= np.sqrt(np.sum(n2**2))

            n3 = np.cross(r24, r23)
            if np.any(np.abs(n3) > 0.):
                n3 /= np.sqrt(np.sum(n3**2))

            n4 = np.cross(r23, r13)
            if np.any(np.abs(n4) > 0.):
                n4 /= np.sqrt(np.sum(n4**2))

            if np.any(np.isnan(n1)):
                print('!!! nan')
                print(i1, i2)
                print(p1, p2, p3, p4)
                print('nan', r23, r13, np.cross(r23, r13))

            # When the vectors are nearly the same, floating point
            # errors can sometimes make the output a tiny bit higher
            # than 1
            t1, t2, t3, t4 = np.clip([n1.dot(n2),
                                      n2.dot(n3),
                                      n3.dot(n4),
                                      n4.dot(n1)],
                                     -1, 1)

            writhe_contribution = (np.arcsin(t1) +
                                   np.arcsin(t2) +
                                   np.arcsin(t3) +
                                   np.arcsin(t4))

            if np.isnan(writhe_contribution):
                print()
                print('nan!')
                print(i1, i2, n1, n2, n3, n4, writhe_contribution)
                print(n1.dot(n2) > 1, n2.dot(n3) > 1, n3.dot(n4) > 1, n4.dot(n1) > 1)
                print(n1.dot(n2), np.arcsin(n1.dot(n2)))
                print(n2.dot(n3), np.arcsin(n2.dot(n3)))
                print(n3.dot(n4), np.arcsin(n3.dot(n4)))
                print(n4.dot(n1), np.arcsin(n4.dot(n1)))

            writhe_contribution *= np.sign(np.cross(r34, r12).dot(r13))
    
            contributions[i1, i2] = writhe_contribution
            contributions[i2, i1] = writhe_contribution

    print()

    return contributions


def second_order_writhes(points, try_cython=True, basepoint=True):

    contributions = np.zeros((len(points[:-1]), len(points[:-1])))
    contributions = np.zeros((len(points), len(points)))

    print('Calculating writhe contributions')

    for i1, p1 in enumerate(points[:-3]):
        print('\ri = {} / {}'.format(i1, len(points) - 4), end='')
        sys.stdout.flush()
        p2 = points[i1 + 1]
        for i2, p3 in enumerate(points[i1+2:-1]):
            i2 += i1 + 2

            p4 = points[i2 + 1]

            r12 = p2 - p1
            r13 = p3 - p1
            r14 = p4 - p1
            r23 = p3 - p2
            r24 = p4 - p2
            r34 = p4 - p3

            n1 = np.cross(r13, r14)
            n1 /= np.sqrt(np.sum(n1**2))

            n2 = np.cross(r14, r24)
            n2 /= np.sqrt(np.sum(n2**2))

            n3 = np.cross(r24, r23)
            if np.any(np.abs(n3) > 0.):
                n3 /= np.sqrt(np.sum(n3**2))

            n4 = np.cross(r23, r13)
            if np.any(np.abs(n4) > 0.):
                n4 /= np.sqrt(np.sum(n4**2))

            if np.any(np.isnan(n1)):
                print('!!! nan')
                print(i1, i2)
                print(p1, p2, p3, p4)
                print('nan', r23, r13, np.cross(r23, r13))

            # When the vectors are nearly the same, floating point
            # errors can sometimes make the output a tiny bit higher
            # than 1
            t1, t2, t3, t4 = np.clip([n1.dot(n2),
                                      n2.dot(n3),
                                      n3.dot(n4),
                                      n4.dot(n1)],
                                     -1, 1)

            writhe_contribution = (np.arcsin(t1) +
                                   np.arcsin(t2) +
                                   np.arcsin(t3) +
                                   np.arcsin(t4))

            if np.isnan(writhe_contribution):
                print()
                print('nan!')
                print(i1, i2, n1, n2, n3, n4, writhe_contribution)
                print(n1.dot(n2) > 1, n2.dot(n3) > 1, n3.dot(n4) > 1, n4.dot(n1) > 1)
                print(n1.dot(n2), np.arcsin(n1.dot(n2)))
                print(n2.dot(n3), np.arcsin(n2.dot(n3)))
                print(n3.dot(n4), np.arcsin(n3.dot(n4)))
                print(n4.dot(n1), np.arcsin(n4.dot(n1)))

            writhe_contribution *= np.sign(np.cross(r34, r12).dot(r13))
    
            contributions[i1, i2] = writhe_contribution
            contributions[i2, i1] = writhe_contribution

    print()
    print('Calculating higher order writhe')


    if basepoint:
        from pyknotid.spacecurves.ccomplexity import cython_second_order_writhes
    else:
        from pyknotid.spacecurves.ccomplexity import cython_second_order_writhes_no_basepoint as cython_second_order_writhes

    return cython_second_order_writhes(points, contributions)


def second_order_twist(points, z):
    
    contributions = np.zeros((len(points[:-1]), len(points[:-1])))
    contributions = np.zeros((len(points), len(points)))

    contributions = np.zeros(len(points))

    print('Calculating writhe contributions')

    for i1, p1 in enumerate(points[:-3]):
        print('\ri = {} / {}'.format(i1, len(points) - 4), end='')
        sys.stdout.flush()
        p2 = points[i1 + 1]

        p3 = z
        p4 = z

        print('p3 is', p3)
        print('p4 is', p4)

        r12 = p2 - p1
        r13 = p3 - p1
        r14 = p4 - p1
        r23 = p3 - p2
        r24 = p4 - p2
        # r34 = p4 - p3

        # n1 = np.cross(r13, r14)
        # n1 /= np.sqrt(np.sum(n1**2))
        n1 = np.array([0., 0, 0])

        n2 = np.cross(r14, r24)
        n2 /= np.sqrt(np.sum(n2**2))

        # n3 = np.cross(r24, r23)
        # if np.any(np.abs(n3) > 0.):
        #     n3 /= np.sqrt(np.sum(n3**2))
        n3 = np.array([0., 0, 0])

        n4 = np.cross(r23, r13)
        if np.any(np.abs(n4) > 0.):
            n4 /= np.sqrt(np.sum(n4**2))

        if np.any(np.isnan(n1)):
            print('!!! nan')
            # print(i1, i2)
            print(p1, p2, p3, p4)
            print('nan', r23, r13, np.cross(r23, r13))
            return

        # When the vectors are nearly the same, floating point
        # errors can sometimes make the output a tiny bit higher
        # than 1
        t1, t2, t3, t4 = np.clip([n1.dot(n2),
                                  n2.dot(n3),
                                  n3.dot(n4),
                                  n4.dot(n1)],
                                 -1, 1)

        writhe_contribution = (np.arcsin(t1) +
                                np.arcsin(t2) +
                                np.arcsin(t3) +
                                np.arcsin(t4))

        if np.isnan(writhe_contribution):
            print()
            print('nan!')
            print(i1, n1, n2, n3, n4, writhe_contribution)
            print(n1.dot(n2) > 1, n2.dot(n3) > 1, n3.dot(n4) > 1, n4.dot(n1) > 1)
            print(n1.dot(n2), np.arcsin(n1.dot(n2)))
            print(n2.dot(n3), np.arcsin(n2.dot(n3)))
            print(n3.dot(n4), np.arcsin(n3.dot(n4)))
            print(n4.dot(n1), np.arcsin(n4.dot(n1)))

        # writhe_contribution *= np.sign(np.cross(r34, r12).dot(r13))

        contributions[i1] = writhe_contribution

    return contributions
