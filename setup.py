from setuptools import setup, find_packages
from distutils.extension import Extension
# from Cython.Distutils import build_ext
from Cython.Build import cythonize

from os.path import join, sep
from os import walk
import glob

import numpy

package_data = {}

def recursively_include(results, directory, patterns):
    for root, subfolders, files in walk(directory):
        for fn in files:
            if not any([glob.fnmatch.fnmatch(fn, pattern) for pattern in patterns]):
                continue
            filename = join(root, fn)
            directory = 'pyknot2'
            if directory not in results:
                results[directory] = []
            results[directory].append(join(*filename.split(sep)[1:]))

recursively_include(package_data, 'pyknot2',
                    ['*.tmpl', '*.pov', '*.pyx', '*.pxd',
                     ])

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
    install_requires=['numpy', 'networkx', 'planarity'],
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
)
