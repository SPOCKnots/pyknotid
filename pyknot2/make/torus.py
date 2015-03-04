import numpy as n
from fractions import gcd

def components(p, q):
    '''
    Returns the number of components of the p-q torus knot.
    '''
    return gcd(p, q)


def knot(p=3, q=4, num=100):
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
