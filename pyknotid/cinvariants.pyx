
import numpy as n
cimport numpy as n
cimport cython

from pyknotid.utils import vprint

cdef long crude_modulus(long val, long modulo):
    if val < 0:
        return val + modulo
    return val
    

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef vassiliev_degree_3(long [:, :] arrows):
    cdef long num_arrows = len(arrows)
    cdef long num_crossings = len(arrows) * 2

    cdef long a1s, a1e, a2s, a2e, a3s, a3e
    cdef long [:] arrow1, arrow2, arrow3
    cdef long i1, i2, i3
    cdef long sign1, sign2, sign3

    cdef set used_sets = set()
    cdef long representations_sum_1 = 0
    cdef long representations_sum_2 = 0
    cdef tuple ordered_indices
    for i1 in range(num_arrows):
        arrow1 = arrows[i1]
        a1s = arrow1[0]
        a1e = arrow1[1]
        sign1 = arrow1[2]

        a1e = crude_modulus(a1e - a1s, num_crossings)

        for i2 in range(num_arrows):
            arrow2 = arrows[i2]
            a2s = arrow2[0]
            a2e = arrow2[1]
            sign2 = arrow2[2]

            a2s = crude_modulus(a2s - a1s, num_crossings)
            a2e = crude_modulus(a2e - a1s, num_crossings)

            for i3 in range(num_arrows):
                arrow3 = arrows[i3]
                a3s = arrow3[0]
                a3e = arrow3[1]
                sign3 = arrow3[2]

                a3s = crude_modulus(a3s - a1s, num_crossings)
                a3e = crude_modulus(a3e - a1s, num_crossings)

                ordered_indices = tuple(sorted((i1, i2, i3)))
                if ordered_indices in used_sets:
                    continue

                if (a2s < a1e and a3e < a1e and a3e > a2s and
                    a3s > a1e and a2e > a3s):
                    representations_sum_1 += sign1 * sign2 * sign3
                    used_sets.add(ordered_indices)
                if (a2e < a1e and a3s < a1e and a3s > a2e and
                    a2s > a1e and a3e > a2s):
                    representations_sum_2 += sign1 * sign2 * sign3
                    used_sets.add(ordered_indices)

                    
    return representations_sum_1 / 2. + representations_sum_2

    
