
import pyknot2.spacecurves.knot as spknot
import pyknot2.make.knot as mk

from functools import wraps
import os
from os import path

import numpy as np

import pytest

def pass_trefoil(func):
    def new_func():
        return func(spknot.Knot(mk.trefoil()))
    return new_func

@pass_trefoil
def test_invariants(k):
    assert k.determinant() == 3
    assert k.alexander_at_root((2,3,4)) == [3, 2, 1]
    assert k.vassiliev_degree_2() == 1
    assert k.vassiliev_degree_3() == -1


@pass_trefoil
def test_identify(k):
    try:
        import pyknot2.spacecurves.chelpers
    except ImportError:
        return  # chelpers not installed
