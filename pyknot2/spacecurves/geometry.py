'''
Space curve geometry
====================

Functions for evaluating geometrical properties of space curves.
'''

import numpy as n

def arclength(points, include_closure=True):
    '''
    Returns the arclength as the sum of lengths of each segment
    in the piecewise linear line.

    Parameters
    ----------
    points : array-like
        Nx3 array of points in the line
    include_closure : bool
        Whether to include the distance between the final and
        first points. Defaults to True.
    '''

    lengths = n.roll(points, -1, axis=0) - points
    length_mags = n.sqrt(n.sum(lengths*lengths, axis=1))
    arclength = n.sum(length_mags[:-1])
    if include_closure:
        arclength += length_mags[-1]
    return arclength
    
