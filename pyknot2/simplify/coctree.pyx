'''
Cython functions for octree calculations.
'''


import numpy as n
cimport numpy as n

cimport cython

from libc.math cimport abs, pow, sqrt as csqrt, floor, acos

cpdef angle_exceeds(double [:, :] ps, double val=2*n.pi,
                    long include_closure=1):
    '''Returns True if the sum of angles along ps exceeds
    val, else False.

    If include_closure, includes the angles with the line closing
    the end and start points.
    '''
    cdef double angle = 0.
    cdef double [:] nex = ps[0]
    cdef double [:] nex2 = ps[1]
    cdef double [:] dv2 = n.zeros(3, dtype=n.double)
    diff(dv2, nex, nex2)
    divide(dv2, mag(dv2))
    cdef double [:] cur
    cdef double increment
    cdef long lenps = len(ps)
    cdef long [:] checks = n.arange(len(ps)) if include_closure else n.arange(len(ps)-2)
    cdef int i
    for i in checks:
        cur = nex
        nex = nex2
        nex2 = ps[(i+2) % lenps]
        dv = dv2
        diff(dv2, nex, nex2)
        divide(dv2, mag(dv2))
        increment = angle_between(dv, dv2)
        if n.isnan(increment):
            return True
        angle += increment
        if angle > val:
            return True
    assert not n.isnan(angle)
    return False

cdef void diff(double [:] dv2, double [:] nex, double [:] nex2):
    dv2[0] = nex2[0] - nex[0]
    dv2[1] = nex2[1] - nex[1]
    dv2[2] = nex2[2] - nex[2]

cdef double angle_between(double [:] v1, double [:] v2):
    '''Returns angle between v1 and v2, assuming they are normalised to 1.'''
    # clip becaus v1.dot(v2) may exceed 1 due to floating point
    cdef double value = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    if value > 1.:
        value = 1.
    elif value < 0.:
        value = 0.
    return value

cdef void divide(double [:] arr, double val):
    arr[0] = arr[0] / val
    arr[1] = arr[1] / val
    arr[2] = arr[2] / val

cdef double mag(double [:] v):
    return pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2)
