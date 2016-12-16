from setuptools import setup, find_packages
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
    name='pyknot2',
    version='1.0',
    description='Tools for analysing knots',
    author='Alexander Taylor',
    author_email='alexander.taylor@bristol.ac.uk',
    install_requires=['numpy', 'networkx', 'planarity',
                      'peewee'],
    cmdclass={'build_ext': build_ext},
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
)
