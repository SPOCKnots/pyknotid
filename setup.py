from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext_modules = [
        Extension("pyknot2.spacecurves.chelpers", ["pyknot2/spacecurves/chelpers.pyx"],
                  libraries=["m"]),
        Extension("pyknot2.simplify.coctree", ["pyknot2/simplify/coctree.pyx"],
                  libraries=["m"]),
        Extension("pyknot2.cinvariants", ["pyknot2/cinvariants.pyx"],
                  libraries=["m"]),
        ]

setup(
  name = 'pyknot2',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [numpy.get_include()],
  ext_modules = ext_modules
)
