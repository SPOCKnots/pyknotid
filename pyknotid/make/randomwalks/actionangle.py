'''Random equilateral polygons
===========================

These functions produce closed random walks as random equilateral
polygons, using the algorithm of J Cantarella et al, "A fast direct
sampling algorithm for equilateral closed polygons". J Phys A 49,
275202 (2016).

This implementation follows the C version available in the `plCurve
<http://www.jasoncantarella.com/wordpress/software/plcurve/>`__
library. The plCurve version is much faster.

'''

import numpy as np
from pyknotid.spacecurves.knot import Knot


def get_closed_loop(length, seed=None, normalisation=10., **kwargs):

    r = np.random.RandomState()
    if seed is not None:
        r.seed(seed)

    distances_okay = False
    while not distances_okay:
        while True:
            ss = r.rand(length - 2) * 2. - 1.
            total = np.sum(ss[:-1])
            if -1. <= total <= 1.:
                ss[-1] = -1. * total
                break

        distances_okay = True

        distances = np.zeros(length - 1)
        distances[0] = 1.
        distances[-1] = 1.  # correct?

        for i in range(1, length - 1):
            distances[i] = distances[i-1] + ss[i-1]

            if (distances[i] + distances[i-1] < 1.0 or
                distances[i] < 1e-12):
                distances_okay = False

    angles = r.rand(length - 3) * 2 * np.pi

    k = Knot(fan_triangulation_action_angle(length, angles, distances),
             **kwargs)
    k.scale(normalisation)
    k.zero_centroid()
    return k



def fan_triangulation_action_angle(length, angles, d):
    normal = np.array([0., 0, 1])

    vt = np.zeros((length, 3))
    vt[0] = np.array([0, 0, 0.])
    vt[1] = np.array([1, 0, 0.])

    for i in range(1, length - 1):
        cos_alpha = (d[i-1]**2 + d[i]**2 - 1.) / (2. * d[i-1] * d[i])
        sin_alpha = np.sqrt(1 - cos_alpha**2)

        f1 = normalised(vt[i])
        f2 = normalised(np.cross(normal, f1))

        vt[i+1] = d[i] * cos_alpha * f1 + d[i] * sin_alpha * f2

        if i < (length - 2):
            f3 = np.cross(normal, normalised(vt[i+1]))
            f3 = normalised(f3)
            normal = np.cos(angles[i-1]) * normal + np.sin(angles[i-1]) * f3
            normal = normalised(normal)

    return vt

def mag(v):
    return np.sqrt(np.sum(v**2))

def normalised(v):
    return v / mag(v)
