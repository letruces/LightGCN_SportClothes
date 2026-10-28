from setuptools import setup, Extension
import numpy

module = Extension(
    'sampling',
    sources=['sampling.cpp'],
    include_dirs=[numpy.get_include()]
)

setup(
    name='sampling',
    ext_modules=[module]
)