from __future__ import print_function
import sys

import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport abs, pow, sqrt as csqrt, floor

cpdef cython_higher_order_writhe(double [:, :] points,
                        double [:, :] contributions,
                        long [:] order):

    cdef long i1, i2, i3, i4
    cdef long [:] indices = np.zeros(4, dtype=np.int)

    cdef double writhe = 0.0

    for i1 in range(len(points) - 3):
        print('\rcython i1', i1, len(points) - 4, end='')
        sys.stdout.flush()
        indices[0] = i1
        for i2 in range(i1 + 1, len(points) - 1):
            indices[1] = i2
            for i3 in range(i2 + 1, len(points) - 1):
                indices[2] = i3
                for i4 in range(i3 + 1, len(points) - 1):
                    indices[3] = i4

                    writhe += (contributions[indices[order[0]],
                                             indices[order[1]]] *
                               contributions[indices[order[2]],
                                             indices[order[3]]])
    print()

    return writhe 


cpdef cython_second_order_writhes(double [:, :] points,
                                  double [:, :] contributions):

    cdef long i1, i2, i3, i4
    cdef long [:] indices = np.zeros(4, dtype=np.int)

    cdef double writhe_1 = 0.0
    cdef double writhe_2 = 0.0
    cdef double writhe_3 = 0.0

    for i1 in range(len(points) - 3):
        if i1 % 5 == 0:
            print('\rcython i1', i1, len(points) - 4, end='')
        sys.stdout.flush()
        indices[0] = i1
        for i2 in range(i1 + 1, len(points) - 1):
            indices[1] = i2
            for i3 in range(i2 + 1, len(points) - 1):
                indices[2] = i3
                for i4 in range(i3 + 1, len(points) - 1):
                    indices[3] = i4

                    writhe_1 += contributions[i1, i2] * contributions[i3, i4]
                    writhe_2 += contributions[i1, i3] * contributions[i2, i4]
                    writhe_3 += contributions[i1, i4] * contributions[i2, i3]
    print()

    return (writhe_1 / (2*np.pi)**2,
            writhe_2 / (2*np.pi)**2,
            writhe_3 / (2*np.pi)**2)

cpdef cython_second_order_writhes_no_basepoint(double [:, :] points,
                                               double [:, :] contributions):

    cdef long i1, i2, i3, i4
    cdef long [:] indices = np.zeros(4, dtype=np.int)

    cdef double writhe_1 = 0.0
    cdef double writhe_2 = 0.0
    cdef double writhe_3 = 0.0

    for i1 in range(len(points) - 1):
        if i1 % 5 == 0:
            print('\rnbp cython i1', i1, len(points) - 4, end='')
        sys.stdout.flush()
        indices[0] = i1
        possible_i2s = list(range(i1 + 1, len(points) - 1)) + list(range(i1 ))
        for i2 in possible_i2s:
            indices[1] = i2
            if i2 > i1:
                possible_i3s = list(range(i2 + 1, len(points) - 1)) + list(range(i1 ))
            else:
                possible_i3s = list(range(i2 + 1, i1 ))
            for i3 in possible_i3s:
                indices[2] = i3
                if i3 > i1:
                    possible_i4s = list(range(i3 + 1, len(points) - 1)) + list(range(i1 ))
                else:
                    possible_i4s = list(range(i3 + 1, i1 ))
                for i4 in possible_i4s:
                    # print('i1, i2, i3, i4 = {}, {}, {}, {}'.format(i1, i2, i3, i4))
                    indices[3] = i4

                    writhe_1 += contributions[i1, i2] * contributions[i3, i4]
                    writhe_2 += contributions[i1, i3] * contributions[i2, i4]
                    writhe_3 += contributions[i1, i4] * contributions[i2, i3]
    print()

    return (writhe_1 / (2*np.pi)**2,
            writhe_2 / (2*np.pi)**2,
            writhe_3 / (2*np.pi)**2)


# cpdef writhing_matrix(double [:, :] points):
#     for i1 in range(len(points) - 3):
#         print('\ri = {} / {}'.format(i1, len(points) - 4), end='')
#         sys.stdout.flush()
#         p1 = points[i1]
#         for i2 in range(i1 + 2, len(points) - 1):
#             p2 = points[i2]

#             p4 = points[i2 + 1]

#             r12 = p2 - p1
#             r13 = p3 - p1
#             r14 = p4 - p1
#             r23 = p3 - p2
#             r24 = p4 - p2
#             r34 = p4 - p3

#             n1 = np.cross(r13, r14)
#             n1 /= np.sqrt(np.sum(n1**2))

#             n2 = np.cross(r14, r24)
#             n2 /= np.sqrt(np.sum(n2**2))

#             n3 = np.cross(r24, r23)
#             if np.any(np.abs(n3) > 0.):
#                 n3 /= np.sqrt(np.sum(n3**2))

#             n4 = np.cross(r23, r13)
#             if np.any(np.abs(n4) > 0.):
#                 n4 /= np.sqrt(np.sum(n4**2))

#             if np.any(np.isnan(n1)):
#                 print('!!! nan')
#                 print(i1, i2)
#                 print(p1, p2, p3, p4)
#                 print('nan', r23, r13, np.cross(r23, r13))

#             # When the vectors are nearly the same, floating point
#             # errors can sometimes make the output a tiny bit higher
#             # than 1
#             t1, t2, t3, t4 = np.clip([n1.dot(n2),
#                                       n2.dot(n3),
#                                       n3.dot(n4),
#                                       n4.dot(n1)],
#                                      -1, 1)

#             writhe_contribution = (np.arcsin(t1) +
#                                    np.arcsin(t2) +
#                                    np.arcsin(t3) +
#                                    np.arcsin(t4))

#             if np.isnan(writhe_contribution):
#                 print()
#                 print('nan!')
#                 print(i1, i2, n1, n2, n3, n4, writhe_contribution)
#                 print(n1.dot(n2) > 1, n2.dot(n3) > 1, n3.dot(n4) > 1, n4.dot(n1) > 1)
#                 print(n1.dot(n2), np.arcsin(n1.dot(n2)))
#                 print(n2.dot(n3), np.arcsin(n2.dot(n3)))
#                 print(n3.dot(n4), np.arcsin(n3.dot(n4)))
#                 print(n4.dot(n1), np.arcsin(n4.dot(n1)))

#             writhe_contribution *= np.sign(np.cross(r34, r12).dot(r13))
    
#             contributions[i1, i2] = writhe_contribution
#             contributions[i2, i1] = writhe_contribution

#     return contributions
