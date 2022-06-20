# from distutils.core import setup, Extension
from setuptools import setup, Extension
import numpy
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "src")
module = Extension(
    "tpc_utils_",
    sources=[os.path.join(SRC, "TPC_MODULE_FUNCTIONS.cpp")],
    include_dirs=[numpy.get_include()],
    extra_compile_args=[r"-O2"],
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tpc_utils",
    version="0.0.1",
    author="Guilherme Ferrari Fortino",
    author_email="ferrarifortino@gmail.com",
    maintainer="Guilherme Ferrari Fortino",
    maintainer_email="ferrarifortino@gmail.com",
    description="Package with useful functions for TPC data analysis.",
    long_description=long_description,
    ext_modules=[module],
    install_requires=["numpy>=1.22"],
    python_requires=">=3.8",
    packages=["tpc_utils"],
)
