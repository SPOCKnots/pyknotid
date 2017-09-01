'''Making periodic knots
=====================

Functions for making periodic knots. Each function returns an open
curve that is assumed to repeat infinitely along the axis between its
start and end points. Pass these functions to
pyknotid.spacecurves.periodic for topological analysis.

'''

import numpy as n
import numpy as np
from pyknotid.spacecurves.periodic import CellKnot, PeriodicKnot
from pyknotid.make.knot import trefoil as aperiodic_trefoil

def trefoil(num_points=40):
    points = aperiodic_trefoil(num_points) + 4
    points *= 10
    points[0, -1] = 80.
    points[-1, -1] = 0
    return PeriodicKnot(points, (0, 0, -80.))

def cell_trefoil(num_points=40):
    points = aperiodic_trefoil(num_points) + 4
    points *= 10
    points[0, -1] = 79
    points[-1, -1] = 0
    return CellKnot.folding(points, 80)

def cell_trefoil2(num_points=40):
    points = aperiodic_trefoil(num_points) + 4
    points *= 10

    end = n.zeros((30, 3), dtype=n.float)
    end_start = points[-1]
    end_prev = points[-2]
    end[:, 0] = n.linspace(end_prev[0], end_start[0], 30)
    end[:, 1] = n.linspace(end_prev[1], end_start[1], 30)
    end[:, 2] = n.linspace(end_prev[2], 0, 30.)

    start = n.zeros((30, 3), dtype=n.float)
    start_start = points[0]
    start_prev = points[1]
    start[:, 0] = n.linspace(start_start[0], start_prev[0], 30)
    start[:, 1] = n.linspace(start_start[1], start_prev[1], 30)
    start[:, 2] = n.linspace(79.99, start_prev[2], 30)

    points = n.vstack((start, points[1:-1], end))

    # points[0, -1] = 79
    # points[-1, -1] = 0
    return CellKnot.folding(points, 80)

def simple_link():
    points = n.array([[6, 1, 0],
                      [6, 2, 0],
                      [5, 2.5, 1],
                      [4, 3, 0],
                      [3, 4, 1],
                      [2, 5, 0],
                      [1, 4, -1],
                      [0, 2.5, 0],
                      [1, 1, 1],
                      [2, 0, 0],
                      [3, 1, -1],
                      [4, 1, 0],
                      [5, 2.5, -1],
                      [6, 3, 0],
                      [6, 4, 0]])

    points[:, 0] += 0.05*n.sin(n.linspace(0, 2*n.pi, len(points)))
    points[:, 1] += 0.05*n.cos(n.linspace(0, 2*n.pi, len(points)))
    points[:, 2] += 0.05*(n.sin(n.linspace(0, 2*n.pi, len(points))) * 
                          n.cos(n.linspace(0, 2*n.pi, len(points))))
    return points * 5

def simple_link2():
    points = n.array([[6, 1, 0],
                      [6, 2, 0],
                      [5, 2.5, 1],
                      [4, 3, 0],
                      [3, 4, -1],
                      [2, 5, 0],
                      [1, 4, 1],
                      [0, 2.5, 0],
                      [1, 1, -1],
                      [2, 0, 0],
                      [3, 1, 1],
                      [4, 1, 0],
                      [5, 2.5, -1],
                      [6, 3, 0],
                      [6, 4, 0]])

    points[:, 0] += 0.05*n.sin(n.linspace(0, 2*n.pi, len(points)))
    points[:, 1] += 0.05*n.cos(n.linspace(0, 2*n.pi, len(points)))
    points[:, 2] += 0.05*(n.sin(n.linspace(0, 2*n.pi, len(points))) * 
                          n.cos(n.linspace(0, 2*n.pi, len(points))))
    
    return points * 5

def p3_1():
    points = n.array([[0, 0, 0],
                      [0, 2, 0],
                      [0, 3, 1],
                      [0, 4, 0],
                      [0, 5, -1],
                      [0, 6, 0],
                      [2, 6, 0],
                      [2, 5, 1],
                      [2, 3, 0],
                      [0, 3, -1],
                      [-2, 3, 0],
                      [-2, 5, 1],
                      [0, 5, 1],
                      [2, 5, -1],
                      [3, 5, 0],
                      [3, 7, 0],
                      [0, 7, 0],
                      [0, 8, 0.]])
    return points * 5

def p3_2():
    points = n.array([[6, 1, 0],
                      [6, 2, 0],
                      [5, 2.5, 1],
                      [4, 3, 0],
                      [3, 4, 1],
                      [2, 5, 0],
                      [1, 4, -1],
                      [0, 2.5, 0],
                      [1, 1, 1],
                      [2, 0, 0],
                      [3, 1, -1],
                      [4, 1, 0],
                      [5, 2.5, -1],
                      [6, 3, 0],
                      [6, 4, 0]])

    points[:, 0] += 0.05*n.sin(n.linspace(0, 2*n.pi, len(points)))
    points[:, 1] += 0.05*n.cos(n.linspace(0, 2*n.pi, len(points)))
    points[:, 2] += 0.05*(n.sin(n.linspace(0, 2*n.pi, len(points))) * 
                          n.cos(n.linspace(0, 2*n.pi, len(points))))
    return points * 5

def p3_3():
    points = n.array([[6, 1, 0],
                      [6, 2, 0],
                      [5, 2.5, 1],
                      [4, 3, 0],
                      [3, 4, -1],
                      [2, 5, 0],
                      [1, 4, 1],
                      [0, 2.5, 0],
                      [1, 1, -1],
                      [2, 0, 0],
                      [3, 1, 1],
                      [4, 1, 0],
                      [5, 2.5, -1],
                      [6, 3, 0],
                      [6, 4, 0]])

    points[:, 0] += 0.05*n.sin(n.linspace(0, 2*n.pi, len(points)))
    points[:, 1] += 0.05*n.cos(n.linspace(0, 2*n.pi, len(points)))
    points[:, 2] += 0.05*(n.sin(n.linspace(0, 2*n.pi, len(points))) * 
                          n.cos(n.linspace(0, 2*n.pi, len(points))))
    
    return points * 5

def p4_1():
    points = n.array([[0, 0, 0],
                      [0, 1, 0],
                      [1, 2, 1],
                      [2, 3, 0],
                      [3, 4, -1],
                      [4, 5, 0],
                      [5, 5, 1],
                      [6, 5, 0],
                      [6, 1, 0],
                      [2, 1, 0],
                      [1, 2, -1],
                      [1, 3, 0],
                      [1, 4, 1],
                      [1, 6, 0],
                      [5, 6, 0],
                      [5, 5, -1],
                      [5, 4, -1],
                      [3, 4, 1],
                      [1, 4, -1],
                      [0, 4, 0],
                      [0, 6, 0],
                      [0, 8, 0.]])
    return points * 5
                      
def _p4_something(*crossing_signs):
    s1, s2, s3, s4, s5, s6, s7, s8 = crossing_signs
    points = n.array([[0, 0, 0.],
                      [0, 3, 0],
                      [0, 5, 0],
                      [1, 6, s1],
                      [2, 7, 0],
                      [2, 9, 0],
                      [2, 10, 0],
                      [3, 11, s2],
                      [4, 12, 0],
                      [4, 13, s3],
                      [3, 14, 0],
                      [2, 13, s4],
                      [2, 12, 0],
                      [3, 11, s5],
                      [4, 10, 0],
                      [4, 7, 0],
                      [4, 5, 0],
                      [4, 4, s6],
                      [3, 3, 0],
                      [2, 4, s7],
                      [1, 6, s8],
                      [0, 7, 0],
                      [0, 9, 0]])
    return points * 5

def p4_2():
    return _p4_something(1, 1, -1, 1, -1, 1, -1, -1,)

def p4_3():
    return _p4_something(-1, 1, -1, 1, -1, 1, -1, 1,)



def p4_6():
    print('note: this is a link with a copy of itself')
    points = np.array([[0, 0, 0.],
                       [1, 5, 1],
                       [2, 10, 0],
                       [3, 14, 1],
                       [4, 20, 0],
                       [5, 23, 1],
                       [6, 24, 0],
                       [7, 23, -1],
                       [7, 13, 1],
                       [6, 12, 0],
                       [5, 13, -1],
                       [4, 14, -1],
                       [1, 15, -1],
                       [0, 16, 0],
                       [0, 20, 0]])

    return points * 5


def p5_5():
    points = np.array([[0, 0, 0.],
                       [1, 5, 1],
                       [2, 10, 0],
                       [3, 14, 1],
                       [4, 20, 0],
                       [5, 23, 1],
                       [6, 24, 0],
                       [7, 23, -1],
                       [8, 20, 0],
                       [9, 14, -1],
                       [10, 10, 0],
                       [9, 4, 1],
                       [7, 3, 1],
                       [5, 3, -1],
                       [3, 4, -1],
                       [1, 5, -1],
                       [0, 6, 0],
                       [0, 10, 0]])

    return points * 5


def p5_6():
    points = np.array([[0, 0, 0.],
                       [1, 5, 1],
                       [2, 10, 0],
                       [3, 13, 1],
                       [4, 14, 0],
                       [5, 13, -1],
                       [6, 12, 0],
                       [7, 13, 1],
                       [8, 14, 0],
                       [9, 13, -1],
                       [10, 10, 0],
                       [9, 3, 1],
                       [6, 3, -1],
                       [4, 3, 1],
                       [2, 3, -1],
                       [1, 5, -1],
                       [0, 10, 0]])

    return points * 5

def p5_3():
    points = np.array([[0, 0, 0.],
                       [0, 5, 0],
                       [1, 4, 1],
                       [6, 5, -1],
                       [5, 6, 1],
                       [4, 6, -1],
                       [1, 5, 1],
                       [1, 4, -1],
                       [1, 0, 0],
                       [4, -4, 1],
                       [4.5, -4.5, 0],
                       [5, -4, -1],
                       [5, 0, 0],
                       [6, 5, 1],
                       [1, 5, -1],
                       [0, 6, 0],
                       [0, 10, 0]])

    return points * 5



#######################

def p0_1():
    points = n.array([[0, 0, 0],
                      [1, 5, 0],
                      [0, 10, 0]])
    return points

def p3_1():
    points = n.array([[0, 0, 0],
                      [0, 2, 0],
                      [0, 3, 1],
                      [0, 4, 0],
                      [0, 5, -1],
                      [0, 6, 0],
                      [2, 6, 0],
                      [2, 5, 1],
                      [2, 3, 0],
                      [0, 3, -1],
                      [-2, 3, 0],
                      [-2, 5, 1],
                      [0, 5, 1],
                      [2, 5, -1],
                      [3, 5, 0],
                      [3, 7, 0],
                      [0, 7, 0],
                      [0, 8, 0.]])
    return points * 5

def p3_2__1():
    points = n.array([[6, 1, 0],
                      [6, 2, 0],
                      [5, 2.5, 1],
                      [4, 3, 0],
                      [3, 4, 1],
                      [2, 5, 0],
                      [1, 4, -1],
                      [0, 2.5, 0],
                      [1, 1, 1],
                      [2, 0, 0],
                      [3, 1, -1],
                      [4, 1, 0],
                      [5, 2.5, -1],
                      [6, 3, 0],
                      [6, 4, 0]])

    points[:, 0] += 0.05*n.sin(n.linspace(0, 2*n.pi, len(points)))
    points[:, 1] += 0.05*n.cos(n.linspace(0, 2*n.pi, len(points)))
    points[:, 2] += 0.05*(n.sin(n.linspace(0, 2*n.pi, len(points))) * 
                          n.cos(n.linspace(0, 2*n.pi, len(points))))
    return points * 5

def p3_2__2():
    points = n.array([[6, 1, 0],
                      [6, 2, 0],
                      [5, 2.5, 1],
                      [4, 3, 0],
                      [3, 4, -1],
                      [2, 5, 0],
                      [1, 4, 1],
                      [0, 2.5, 0],
                      [1, 1, -1],
                      [2, 0, 0],
                      [3, 1, 1],
                      [4, 1, 0],
                      [5, 2.5, -1],
                      [6, 3, 0],
                      [6, 4, 0]])

    points[:, 0] += 0.05*n.sin(n.linspace(0, 2*n.pi, len(points)))
    points[:, 1] += 0.05*n.cos(n.linspace(0, 2*n.pi, len(points)))
    points[:, 2] += 0.05*(n.sin(n.linspace(0, 2*n.pi, len(points))) * 
                          n.cos(n.linspace(0, 2*n.pi, len(points))))
    
    return points * 5

def p4_1():
    points = n.array([[0, 0, 0],
                      [0, 1, 0],
                      [1, 2, 1],
                      [2, 3, 0],
                      [3, 4, -1],
                      [4, 5, 0],
                      [5, 5, 1],
                      [6, 5, 0],
                      [6, 1, 0],
                      [2, 1, 0],
                      [1, 2, -1],
                      [1, 3, 0],
                      [1, 4, 1],
                      [1, 6, 0],
                      [5, 6, 0],
                      [5, 5, -1],
                      [5, 4, -1],
                      [3, 4, 1],
                      [1, 4, -1],
                      [0, 4, 0],
                      [0, 6, 0],
                      [0, 8, 0.]])
    return points * 5

def p4_2__1():
    return _p4_something(1, 1, -1, 1, -1, 1, -1, -1,)

def p4_2__2():
    return _p4_something(-1, 1, -1, 1, -1, 1, -1, 1,)

def p4_3__1():
    points = np.array([[0, 0, 0.],
                       [1, 4, -1],
                       [2, 5, 0],
                       [3, 4, 1],
                       [4, 0, 0],
                       [5, -3, -1],
                       [6, -4, 0],
                       [7, -3, 1],
                       [8, 0, 0],
                       [7, 7, -1],
                       [6, 8, 0],
                       [5, 7, 1],
                       [3, 4, -1],
                       [2, 3, 0],
                       [1, 4, 1],
                       [0, 10, 0]])

    return points * 5

def p4_3__2():
    points = np.array([[0, 0, 0.],
                       [1, 4, -1],
                       [2, 5, 0],
                       [3, 4, 1],
                       [4, 0, 0],
                       [5, -3, 1],
                       [6, -4, 0],
                       [7, -3, -1],
                       [8, 0, 0],
                       [7, 7, 1],
                       [6, 8, 0],
                       [5, 7, -1],
                       [3, 4, -1],
                       [2, 3, 0],
                       [1, 4, 1],
                       [0, 10, 0]])

    return points * 5

def p4_4__1():
    # Some of the different crossing representations of this are the
    # same as 4_2
    points = np.array([[0, 0, 0],
                       [4, 5, -1],
                       [5, 7, 1],
                       [6, 10, 0],
                       [5, 17, -1],
                       [3, 18, -1],
                       [2, 18, 0],
                       [2, 16, 1],
                       [4, 15, 1],
                       [4.5, 10, 0],
                       [3, 8, 1],
                       [2, 6, -1],
                       [1, 5, 0],
                       [0, 10, 0]])
    return points * 5

def p4_5__1_false():
    # This is the same as 4_2
    points = np.array([[0, 0, 0.],
                       [1, 4, -1],
                       [5, 6, 1],
                       [8, 10, 0],
                       [5, 16, -1],
                       [4, 18, 1],
                       [3, 19, 0],
                       [2, 18, -1],
                       [1, 14, 1],
                       [3, 10, 0],
                       [4, 8, -1],
                       [3, 7, 0],
                       [2, 8, 1],
                       [0, 10, 0]])
    return points * 5

def p5_4__1():
    points = np.array([[0, 0, 0.],
                       [1, 5, 1],
                       [2, 10, 0],
                       [3, 14, 1],
                       [4, 20, 0],
                       [5, 23, 1],
                       [6, 24, 0],
                       [7, 23, -1],
                       [8, 20, 0],
                       [9, 14, -1],
                       [10, 10, 0],
                       [9, 4, 1],
                       [7, 3, 1],
                       [5, 3, -1],
                       [3, 4, -1],
                       [1, 5, -1],
                       [0, 6, 0],
                       [0, 10, 0]])

    return points * 5

def p5_5__1():
    points = np.array([[0, 0, 0.],
                       [1, 5, 1],
                       [2, 10, 0],
                       [3, 13, 1],
                       [4, 14, 0],
                       [5, 13, -1],
                       [6, 12, 0],
                       [7, 13, 1],
                       [8, 14, 0],
                       [9, 13, -1],
                       [10, 10, 0],
                       [9, 3, 1],
                       [6, 3, -1],
                       [4, 3, 1],
                       [2, 3, -1],
                       [1, 5, -1],
                       [0, 10, 0]])

    return points * 5

def p5_3__1():
    points = np.array([[0, 0, 0.],
                       [0, 5, 0],
                       [1, 4, 1],
                       [6, 5, -1],
                       [5, 6, 1],
                       [4, 6, -1],
                       [1, 5, 1],
                       [1, 4, -1],
                       [1, 0, 0],
                       [4, -4, 1],
                       [4.5, -4.5, 0],
                       [5, -4, -1],
                       [5, 0, 0],
                       [6, 5, 1],
                       [1, 5, -1],
                       [0, 6, 0],
                       [0, 10, 0]])

    return points * 5

