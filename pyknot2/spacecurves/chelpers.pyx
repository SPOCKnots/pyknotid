'''
Cython functions for space curve analysis.
'''

import numpy as n
cimport numpy as n

cimport cython

from libc.math cimport abs, pow, sqrt as csqrt, floor

cpdef find_crossings(double [:] v, double [:] dv,
                     double [:, :] points,
                     double [:] segment_lengths,
                     long current_index,
                     long comparison_index,
                     double max_segment_length,
                     long jump_mode=1
                     ):
    '''
    Searches for crossings between the given vector and any other
    vector in the
    list of points, returning all of them as a list.
                     
    Parameters
    ----------
    v0 : ndarray
        The current point, a 1D vector.
    dv : ndarray
        The vector connecting the current point to the next one
    points : ndarray
        The array or (x, y) values of all the other points
    segment_lengths : ndarray
        The length of each segment joining a point to the
        next one.
    current_index : long
        The index of the point currently being tested.
    comparison_index : long
        The index of the first comparison point
    jump_mode : int
        1 to check every jump distance, 2 to jump based on
        the maximum one, 3 to never jump and check the length
        of every step.
    '''

    cdef list crossings = []
    cdef double vx = v[0]
    cdef double vy = v[1]
    cdef double vz = v[2]
    cdef double dvx = dv[0]
    cdef double dvy = dv[1]
    cdef double dvz = dv[2]

    cdef double twice_max_segment_length = 2*max_segment_length

    cdef int i = 0
    cdef double distance, distance_travelled
    cdef double [:] point
    cdef double [:] next_point
    cdef double jump_x, jump_y, jump_z
    cdef double pz
    cdef double dpz
    cdef long intersect
    cdef double intersect_i, intersect_j
    cdef double crossing_sign
    cdef double crossing_direction
    cdef long jumps
    cdef long num_jumps
    cdef int already_jumped = 0

    while i < len(points) - 1:
        point = points[i]
        distance = csqrt(pow(vx - point[0], 2) + pow(vy - point[1], 2))
        if distance < twice_max_segment_length or already_jumped:
            already_jumped = 0
            next_point = points[i+1]
            jump_x = next_point[0] - point[0]
            jump_y = next_point[1] - point[1]
            jump_z = next_point[2] - point[2]


            intersect, intersect_i, intersect_j = do_vectors_intersect(
                vx, vy, dvx, dvy, point[0], point[1],
                jump_x, jump_y)

            if intersect:
                pz = point[2]
                dpz = jump_z

                crossing_sign = sign((vz + intersect_i * dvz) -
                                     (pz + intersect_j * dpz))

                crossing_direction = sign(cross_product(
                    dvx, dvy, jump_x, jump_y))

                crossings.append([<double>current_index + intersect_i,
                                  (<double>comparison_index + intersect_j +
                                   <double>i),
                                  crossing_sign,
                                  crossing_sign * crossing_direction])
                crossings.append([(<double>comparison_index + intersect_j +
                                   <double>i),
                                  <double>current_index + intersect_i,
                                  -1. * crossing_sign,
                                  crossing_sign * crossing_direction])
            i += 1

        elif jump_mode == 3:
            i += 1  # naive mode - check everything
            already_jumped = 1
        elif jump_mode == 2:
            num_jumps = <long>(floor(distance / max_segment_length)) - 1
            if num_jumps < 1:
                num_jumps = 1
            i += num_jumps
            already_jumped = 1
        else:  # Catch all other jump modes
            distance_travelled = 0.
            jumps = 0
            while (distance_travelled < (distance - max_segment_length) and
                   i < len(points)):
                jumps += 1
                distance_travelled += segment_lengths[i]
                i += 1
            if jumps > 1:
                i -= 2
            already_jumped = 1
            # This keeps jumping until we might be close enough to intersect,
            # without doing vector arithmetic at every step
                                  
    return crossings
                                      



cdef tuple do_vectors_intersect(double px, double py, double dpx, double dpy,
                                double qx, double qy, double dqx, double dqy):
    """Takes four vectors p, dp and q, dq, then tests whether they cross in
    the dp/dq region. Returns this boolean, and the (fractional) point where
    the crossing actually occurs.
    """
    cdef double t, u

    if abs(cross_product(dpx, dpy, dqx, dqy)) < 0.000001:
        return (0, 0., 0.)

    t = cross_product(qx - px, qy - py, dqx, dqy) / cross_product(dpx, dpy,
                                                                  dqx, dqy)
    if t < 1.0 and t > 0.0:
        u = cross_product(qx - px, qy - py, dpx, dpy) / cross_product(dpx, dpy,
                                                                      dqx, dqy)
        if u < 1.0 and u > 0.0:
            return (1, t, u)
    return (0, -1., -1.)
        
cpdef double cross_product(double px, double py, double qx, double qy):
    '''Simple cython cross product for 2D vectors.'''
    return px * qy - py * qx

cpdef double sign(double a):
    return (1. if a > 0. else (-1. if a < 0. else 0.))

cpdef double mag_difference(double [:] a, double [:] b):
    '''The magnitude of the vector joining a and b'''
    return csqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
