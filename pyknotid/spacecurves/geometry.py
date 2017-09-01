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

    if len(points) < 2:
        return 0.
        
    lengths = n.roll(points, -1, axis=0) - points
    length_mags = n.sqrt(n.sum(lengths*lengths, axis=1))
    arclength = n.sum(length_mags[:-1])
    if include_closure:
        arclength += length_mags[-1]
    return arclength
    

def radius_of_gyration(points):
    '''
    Returns the radius of gyration of the given vertices, assuming
    each has equal weight (and ignoring the connecting lines).

    Parameters
    ----------
    points : array-like
        Nx3 array of points in the line.
    '''
    av_pos = n.average(points, axis=0)
    diffs = (points - av_pos)**2
    rogs = n.sum(diffs, 1)
    rog = n.average(rogs)
    return n.sqrt(rog)


# def persistences(points, step=None):
#     '''
#     Returns a set of xs and ys for persistence length calculation.
#     '''

#     if step is None:
#         import pyknot.
#         step = n.average
