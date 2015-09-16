'''
Functions for making knots, returning a set of points.
'''

import numpy as n

def unknot(num=100):
    data = n.zeros((num, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num)
    data[:, 0] = 3*n.sin(ts)
    data[:, 1] = 3*n.cos(ts)
    data[:, 2] = n.sin(3*ts)
    return data

def trefoil(num=100):
    data = n.zeros((num, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num)
    data[:, 0] = (2+n.cos(3*ts))*n.cos(2*ts)
    data[:, 1] = (2+n.cos(3*ts))*n.sin(2*ts)
    data[:, 2] = n.sin(3*ts)
    return data


def figure_eight(num=100):
    data = n.zeros((num, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num)
    data[:, 0] = (2+n.cos(2*ts))*n.cos(3*ts)
    data[:, 1] = (2+n.cos(2*ts))*n.sin(3*ts)
    data[:, 2] = n.sin(4*ts)
    return data


def torus_knot(p=3, q=4, num=100):
    '''
    Returns points in the p, q torus knot.

    Note that not all combinations of p and q give knots;
    some curves may pass through themselves.
    '''
    data = n.zeros((num, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num)
    rs = n.cos(q*ts) + 2
    data[:, 0] = rs*n.cos(p*ts)
    data[:, 1] = rs*n.sin(p*ts)
    data[:, 2] = -1*n.sin(q*ts)
    return data


def lissajous(nx=3, ny=2, nz=7, px=0.7, py=0.2, pz=0., num=100):
    data = n.zeros((num, 3), dtype=n.float64)
    ts = n.linspace(0, 2*n.pi, num)
    data[:, 0] = n.cos(nx*ts+px)
    data[:, 1] = n.cos(ny*ts+py)
    data[:, 2] = n.cos(nz*ts+pz)
    return data


def three_twist(num=100):
    return lissajous(3, 2, 7, 0.7, 0.2, 0., num)


def stevedore(num=100):
    return lissajous(3, 2, 5, 1.5, 0.2, 0., num)


def square(num=100):
    return lissajous(3, 5, 7, 0.7, 1., 0., num)


def k8_21(num=100):
    return lissajous(3, 4, 7, 0.1, 0.7, 0., num)
