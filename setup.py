from setuptools import setup, find_packages
from distutils.extension import Extension

from os.path import join, sep, dirname
from os import walk, environ
import glob
import re


package_data = {}

def recursively_include(results, directory, patterns):
    for root, subfolders, files in walk(directory):
        for fn in files:
            if not any([glob.fnmatch.fnmatch(fn, pattern) for pattern in patterns]):
                continue
            filename = join(root, fn)
            directory = 'pyknotid'
            if directory not in results:
                results[directory] = []
            results[directory].append(join(*filename.split(sep)[1:]))

recursively_include(package_data, 'pyknotid',
                    ['*.tmpl', '*.pov', '*.pyx', '*.pxd',
                     '*.py',
                     ])

# Build cython components if possible
try:
    from Cython.Build import cythonize
    import numpy
except ImportError:
    print('Cython or numpy could not be imported, so cythonised calculation '
          'functions will not be built. pyknotid will use Python-only '
          'routines instead. These are slower, but will return the '
          'same result.')
    print('To build the cython components, install cython and numpy and rebuild '
          'pyknotid.')
    ext_modules = []
    include_dirs = []
else:
    ext_modules = [
            Extension("pyknotid.spacecurves.chelpers", ["pyknotid/spacecurves/chelpers.pyx"],
                    libraries=["m"]),
            Extension("pyknotid.spacecurves.ccomplexity", ["pyknotid/spacecurves/ccomplexity.pyx"],
                    libraries=["m"]),
            Extension("pyknotid.simplify.coctree", ["pyknotid/simplify/coctree.pyx"],
                    libraries=["m"]),
            Extension("pyknotid.cinvariants", ["pyknotid/cinvariants.pyx"],
                    libraries=["m"]),
            ]
    ext_modules = cythonize(ext_modules)
    include_dirs = [numpy.get_include()]

pyknotid_init_filen = join(dirname(__file__), 'pyknotid', '__init__.py')
version = None
try:
    with open(pyknotid_init_filen) as fileh:
        lines = fileh.readlines()
except IOError:
    pass
else:
    for line in lines:
        line = line.strip()
        if line.startswith('__version__ = '):
            matches = re.findall(r'["\'].+["\']', line)
            if matches:
                version = matches[0].strip("'").strip('"')
                break
if version is None:
    raise Exception('Error: version could not be loaded from {}'.format(pyknotid_init_filen))

if 'READTHEDOCS' in environ and environ['READTHEDOCS'] == 'True':
    print('Installing for doc only')
    install_requires=['numpy', 'peewee', 'vispy', 'sympy']
else:
    install_requires=['numpy', 'networkx', 'planarity',
                      'peewee', 'vispy', 'sympy', 'appdirs'],


setup(
    name='pyknotid',
    version=version,
    description=('Tools for identifying and analysing knots, in space-curves '
                 'or standard topological representations'),
    author='Alexander Taylor',
    author_email='alexander.taylor@bristol.ac.uk',
    install_requires=install_requires,
    ext_modules=ext_modules,
    include_dirs=include_dirs,
    packages=find_packages(),
    package_data=package_data,
    entry_points={
        'console_scripts': [
            'analyse-knot-file = pyknotid.cli.analyse_knot_file:main',
            'plot-knot = pyknotid.cli.plot_knot:main']
        }
)
