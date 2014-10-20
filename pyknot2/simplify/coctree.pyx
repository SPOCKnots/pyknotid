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

cdef void multiply(double [:] arr, double val):
    arr[0] = arr[0] * val
    arr[1] = arr[1] * val
    arr[2] = arr[2] * val

cdef double mag(double [:] v):
    return pow(v[0], 2) + pow(v[1], 2) + pow(v[2], 2)


#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef line_to_segments(line, cuts=None, join_ends=True):
    '''Takes a line (set of points), a list of cut planes in
    x, y, z, and a parameter to decide whether the line
    joining the first and last point should also be cut.

    Returns a list of shorter lines resulting from cutting at
    all these cut planes.'''

    cdef double [:, :] cy_line = line

    cdef double cut_x, cut_y, cut_z
    if cuts is None:
        xmin, ymin, zmin = n.min(line, axis=0) - 1
        xmax, ymax, zmax = n.max(line, axis=0) + 1
        cut_x = (xmax + xmin) / 2.
        cut_y = (ymax + ymin) / 2.
        cut_z = (zmin + zmax) / 2.
    else:
        cut_x, cut_y, cut_z = cuts


    cdef double [:] cy_dv = n.zeros(3, dtype=n.double)
    cdef double [:] cy_nex
    cdef double [:] cy_cur
    cdef double dx, dy, dz
    cdef double x_cut_pos, y_cut_pos, z_cut_pos
        
    # Cut the line wherever it passes through a quad cell boundary
    cdef list segments = []
    cdef long cut_i = 0
    cdef long i
    for i in range(len(line)-1):
        cy_cur = cy_line[i]
        cy_nex = cy_line[i+1]

        diff(cy_dv, cy_cur, cy_nex)
        dx = cy_dv[0]
        dy = cy_dv[1]
        dz = cy_dv[2]

        cross_cut_x = sign(cy_cur[0] - cut_x) != sign(cy_nex[0] - cut_x)
        cross_cut_y = sign(cy_cur[1] - cut_y) != sign(cy_nex[1] - cut_y)
        cross_cut_z = sign(cy_cur[2] - cut_z) != sign(cy_nex[2] - cut_z)

        if (not cross_cut_x and not cross_cut_y and not cross_cut_z):
            continue

        cur = line[i]
        nex = line[i+1]
        dv = cur - nex
        cur = line[i]
        nex = line[i+1]

        if cross_cut_x and cross_cut_y and cross_cut_z:
            x_cut_pos = -1 * (cur[0]-cut_x)/dx
            y_cut_pos = -1 * (cur[1]-cut_y)/dy
            z_cut_pos = -1 * (cur[2]-cut_z)/dz
            order = n.sort((x_cut_pos, y_cut_pos, z_cut_pos))
            # assert 0 < x_cut_pos < 1 and 0 < y_cut_pos < 1 and 0 < z_cut_pos < 1
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            join_point_3 = cur + order[2]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            third_seg = n.vstack((join_point_2, join_point_3))
            line[i] = join_point_3
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
            segments.append(third_seg)
        elif cross_cut_x and cross_cut_y:
            x_cut_pos = -1 * (cur[0]-cut_x)/dx
            y_cut_pos = -1 * (cur[1]-cut_y)/dy
            order = n.sort((x_cut_pos, y_cut_pos))
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            line[i] = join_point_2
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
        elif cross_cut_x and cross_cut_z:
            x_cut_pos = -1 * (cur[0]-cut_x)/dx
            z_cut_pos = -1 * (cur[2]-cut_z)/dz
            order = n.sort((x_cut_pos, z_cut_pos))
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            line[i] = join_point_2
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
        elif cross_cut_y and cross_cut_z:
            y_cut_pos = -1 * (cur[1]-cut_y)/dy
            z_cut_pos = -1 * (cur[2]-cut_z)/dz
            order = n.sort((y_cut_pos, z_cut_pos))
            join_point_1 = cur + order[0]*dv
            join_point_2 = cur + order[1]*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point_1))
            second_seg = n.vstack((join_point_1, join_point_2))
            line[i] = join_point_2
            cut_i = i
            segments.append(first_seg)
            segments.append(second_seg)
        elif cross_cut_x:
            cut_pos = -1 * (cur[0]-cut_x)/dx
            assert 0. <= cut_pos <= 1.
            join_point = cur + cut_pos*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point))
            line[i] = join_point
            cut_i = i
            segments.append(first_seg)
        elif cross_cut_y:
            cut_pos = -1 * (cur[1]-cut_y)/dy
            assert 0. <= cut_pos <= 1.
            join_point = cur + cut_pos*dv
            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point))
            line[i] = join_point
            cut_i = i
            segments.append(first_seg)
        elif cross_cut_z:
            cut_pos = -1 * (cur[2]-cut_z)/dz
            assert 0. <= cut_pos <= 1.
            join_point = cur + cut_pos*dv

            first_seg = n.vstack((line[cut_i:(i+1)].copy(), join_point))
            line[i] = join_point
            # second_seg = n.vstack((join_point, line[(i+1):]))

            cut_i = i
            segments.append(first_seg)

    final_seg = line[cut_i:]
    if cut_i > 0:
        if join_ends:
            first_seg = segments.pop(0)
            segments.append(n.vstack((final_seg, first_seg)))
        else:
            segments.append(final_seg)
    else:
        segments.append(final_seg)

    return segments



cdef double sign(double v):
   if v > 0:
       return 1.0
   elif v < 0:
       return -1.0
   return 0.0
   
