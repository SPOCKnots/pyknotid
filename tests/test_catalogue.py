

import pyknot2.spacecurves.spacecurve as sp
import pyknot2.make.knot as mk

from functools import wraps
import os
from os import path

import sys

import numpy as np

import pytest

from pyknot2.catalogue.identify import from_invariants
test_knots = from_invariants(max_crossings=8)


@pytest.mark.parametrize("knot", test_knots)
def test_reconstruction(knot):
    if knot.identifier == '0_1':
        return
    k = knot.space_curve()
    assert k.determinant() == knot.determinant
    assert abs(k.vassiliev_degree_2()) == abs(knot.vassiliev_2)
    assert abs(k.vassiliev_degree_3()) == abs(knot.vassiliev_3)
    assert abs(k.planar_writhe()) == abs(knot.planar_writhe)

    if sys.version_info.major < 3:  # spherogram isn't python3 compatible
        hv = knot.hyperbolic_volume
        if hv == 'Not hyperbolic':
            hv = 0.0
        else:
            try:
                hv = float(hv)
            except ValueError:
                hv = 0.0
        assert np.isclose(hv, k.hyperbolic_volume()[0], atol=0.005,
                          rtol=0.05)
    
