'''Named knots and links
=====================

Functions for making certain knots. The knots available are an
arbitrary selection based on known analytic forms.

Each of these functions returns a
:class:`~pyknotid.spacecurves.knot.Knot` or other appropriate pyknotid
space curve class.

API documentation
-----------------

'''

import numpy as n
import numpy as np

from pyknotid.spacecurves.knot import Knot

def unknot(num_points=100):
    '''Returns a simple circle.'''
    data = n.zeros((num_points, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num_points)
    data[:, 0] = 3*n.sin(ts)
    data[:, 1] = 3*n.cos(ts)
    data[:, 2] = n.sin(3*ts)
    return Knot(data)

def k3_1(num_points=100):
    '''Returns a particular trefoil knot conformation.'''
    data = n.zeros((num_points, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num_points)
    data[:, 0] = (2+n.cos(3*ts))*n.cos(2*ts)
    data[:, 1] = (2+n.cos(3*ts))*n.sin(2*ts)
    data[:, 2] = n.sin(3*ts)
    return Knot(data)
trefoil = k3_1


def k4_1(num_points=100):
    '''Returns a particular figure eight knot conformation.'''
    data = n.zeros((num_points, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num_points)
    data[:, 0] = (2+n.cos(2*ts))*n.cos(3*ts)
    data[:, 1] = (2+n.cos(2*ts))*n.sin(3*ts)
    data[:, 2] = n.sin(4*ts)
    return Knot(data)
figure_eight = k4_1

def lissajous(nx=3, ny=2, nz=7, px=0.7, py=0.2, pz=0., num_points=100):
    '''Returns a `Lissajous knot
    <https://en.wikipedia.org/wiki/Lissajous_knot>`__ with the given
    parameters.'''
    data = n.zeros((num_points, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num_points)
    data[:, 0] = n.cos(nx*ts+px)
    data[:, 1] = n.cos(ny*ts+py)
    data[:, 2] = n.cos(nz*ts+pz)
    return Knot(data)


def k5_2(num_points=100):
    '''Returns a Lissajous conformation of the knot 5_2.'''
    return lissajous(3, 2, 7, 0.7, 0.2, 0., num_points)
three_twist = k5_2

def k6_1(num_points=100):
    '''Returns a Lissajous conformation of the knot 6_1.'''
    return lissajous(3, 2, 5, 1.5, 0.2, 0., num_points)
stevedore = k6_1

def k3_1_composite_3_1(num_points=100):
    '''Returns a Lissajous conformation of the composite double trefoil 3_1
    # 3_1.
    '''
    return lissajous(3, 5, 7, 0.7, 1., 0., num_points)
square = k3_1_composite_3_1

def k8_21(num_points=100):
    '''Returns a Lissajous conformation of the knot 8_21.'''
    return lissajous(3, 4, 7, 0.1, 0.7, 0., num_points)
