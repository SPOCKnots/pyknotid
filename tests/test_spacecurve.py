import pyknotid.spacecurves.spacecurve as sp
import pyknotid.make as mk

from functools import wraps
import os
from os import path

import numpy as np

import pytest

def pass_trefoil(func):
    def new_func():
        return func(sp.SpaceCurve(mk.trefoil()))
    return new_func


@pass_trefoil
def test_init(k):
    pass

@pass_trefoil
def test_copy(k):
    k2 = k.copy()
    assert np.all(k2.points == k.points)
    assert k2.points is not k.points

@pass_trefoil
def test_points(k):
    assert isinstance(k.points, np.ndarray)

@pass_trefoil
def test_translate(k):
    pos = k.points[0]
    k.translate([10., 20., 30.])
    new_pos = k.points[0]

    assert new_pos[0] - pos[0] == 10.
    assert new_pos[1] - pos[1] == 20.
    assert new_pos[2] - pos[2] == 30.


@pass_trefoil
def test_zero_centroid(k):
    k.points += np.random.random(size=3) * np.random.random() * 10

    k.zero_centroid()

    assert np.all(np.average(k.points, axis=0) < 0.00001)

@pass_trefoil
def test_rotate(k):
    k.rotate()

@pass_trefoil
def test_planar_writhe(k):
    assert np.abs(k.planar_writhe()) == 3

@pass_trefoil
def test_writhe(k):
    k.rotate()
    w = k.writhe(100)
    assert -3.6 < w < -3.4 

@pass_trefoil
def test_gauss_code(k):
    assert str(k.gauss_code(recalculate=True)) == '1+a,2-a,3+a,1-a,2+a,3-a'

@pass_trefoil
def test_reconstructed_space_curve(k):
    k2 = k.reconstructed_space_curve()

    assert k.planar_writhe() == k2.planar_writhe()

@pass_trefoil
def test_write_load(k):
    k.to_json('test_trefoil.json')
    k2 = sp.SpaceCurve.from_json('test_trefoil.json')

    assert k2.planar_writhe() == k2.planar_writhe()

    os.unlink('test_trefoil.json')


@pass_trefoil
def test_octree_simplify(k):
    k.octree_simplify(runs=2)


@pass_trefoil
def test_arclength(k):
    assert np.isclose(k.arclength(), 31.8512, atol=0.01)

@pass_trefoil
def test_rog(k):
    assert np.isclose(k.radius_of_gyration(), 2.244798, atol=0.01)

@pass_trefoil
def test_smooth(k):
    k.smooth()

@pass_trefoil
def test_compiled_vs_python_find_crossings(k):
    try:
        import pyknotid.spacecurves.chelpers
    except ImportError:
        return  # chelpers not installed

    g1 = k.gauss_code(recalculate=True, try_cython=True)
    g2 = k.gauss_code(recalculate=True, try_cython=False)

    assert str(g1) == str(g2)
