'''
Space curve complexity
======================

Functions for evaluating the (mainly geometrical) complexity of space
curves.
'''

from . import Knot
from pyknot2.spacecurves.rotation import (get_rotation_angles,
                                          rotate_to_top)
import numpy as n


def writhe_and_crossing_number(points, number_of_samples=10,
                               verbose=True, remove_duplicate_angles=False,
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
        k = Knot(points, verbose=False)
        k._apply_matrix(rotate_to_top(theta, phi))
        crossings = k.raw_crossings(**kwargs)
        crossing_numbers.append(len(crossings) / 2)
        writhes.append(n.sum(crossings[:, 3]) / 2. if len(crossings) else 0)

    return n.average(crossing_numbers), n.average(writhes)
