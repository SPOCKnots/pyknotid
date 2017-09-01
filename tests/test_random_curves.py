

import pyknotid.spacecurves.knot as spknot
import pyknotid.make.randomwalks.quaternionic as rw

from functools import wraps
import os
from os import path

import numpy as np

import pytest


def test_random_curves():
    for i in range(10):
        k = spknot.Knot(rw.get_closed_loop(100))
        k.determinant()

    k = spknot.Knot(rw.get_closed_loop(1000))
    k.determinant()
