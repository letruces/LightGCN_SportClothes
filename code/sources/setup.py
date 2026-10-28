from pybind11.setup_helpers import Pybind11Extension, build_ext  # pyright: ignore[reportMissingImports]
from setuptools import setup
import numpy

ext_modules = [
    Pybind11Extension(
        'sampling',
        ['sampling.cpp'],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name='sampling',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)