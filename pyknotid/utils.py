'''
Utility functions
=================
'''

import sys
import numpy as n

def vprint(string='', newline=True, condition=True):
    '''
    Verbose print; prints the given string if the input condition
    is met.

    Pyknotid uses this to make it easy to display changing counters.

    Parameters
    ==========
    string : str
        The string to print.
    newline : bool
        Whether to automatically append a newline.
    condition : bool
        The condition that must be met for the print to occur.
    '''
    if not condition:
        return
    sys.stdout.write(string)
    if newline:
        sys.stdout.write('\n')
    sys.stdout.flush()

def mag(v):
    '''
    Returns the magnitude of the vector v.

    Parameters
    ----------
    v : ndarray
        A vector of any dimension.
    '''
    return n.sqrt(v.dot(v))

def get_rotation_matrix(angles):
    '''
    Returns a rotation matrix based on sequentially rotating
    around the 3 given axes (i.e. by multiplication of the component
    rotation matrices).

    Parameters
    ----------
    angles : iterable
        The psi, theta and phi angles
    '''
    
    phi, theta, psi = angles
    sin = n.sin
    cos = n.cos
    rotmat = n.array([
            [cos(theta)*cos(psi),
             -1*cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi),
             sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi)],
            [cos(theta)*sin(psi),
             cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi),
             -1*sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi)],
            [-1*sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    return rotmat


def ensure_shape_tuple(shape):
    '''If the input is a number, returns a tuple with that number
    repeated three times, otherwise just returns the input.
    '''
    if isinstance(shape, (float, int)):
        return (shape, shape, shape)
    return shape
