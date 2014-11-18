'''
Space curve complexity
======================

Functions for evaluating the (mainly geometrical) complexity of space
curves.
'''

from . import Knot
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


def get_rotation_angles(number):
    '''
    Returns a list of theta, phi values, approximately evenly
    distributed on the sphere.

    Uses the generalised spiral points algorithm explained in
    E B Saff and
    A B J Kuijlaars, Distributing many points on a sphere, The
    Mathematical Intelligencer 19(1) 1997.
    '''

    angles = n.zeros((number, 2))
    angles[0] = n.array([n.arccos(-1), 0])

    for k in range(2, number+1):
        h_k = -1. + (2. * (k - 1)) / (number - 1)
        theta = n.arccos(h_k)
        phi = (angles[k-2, 1] + 3.6/n.sqrt(number) *
               1. / n.sqrt(1 - h_k**2)) % (2*n.pi)
        angles[k-1, 0] = theta
        angles[k-1, 1] = phi
    angles[-1, 1] = 0.  # Last phi will be inf otherwise
     
    return angles
        

def rotate_to_top(theta, phi):
    '''
    Returns a rotation matrix that will rotate a sphere such that
    the given positions are at the top.

    Parameters
    ----------
    theta : float
        The latitudinal variable.
    phi : float
        The longitudinal variable.
    '''

    ct = n.cos(theta)
    st = n.sin(theta)
    cp = n.cos(phi)
    sp = n.sin(phi)

    return n.array([[ct, -st, 0], [cp*st, cp*ct, -sp], [sp*st, ct*sp, cp]])
