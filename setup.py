import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-O2']

ext_modules = [
    Extension(
    'irls',
        ['irls.cpp'],
        include_dirs=['C:/Users/vergi/OneDrive/Documents/pybind11/include',
					  'C:/Users/vergi/Downloads/Eigen'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='IRLS',
    version='0.0.1',
    description='IRLS algo',
    ext_modules=ext_modules,
)
