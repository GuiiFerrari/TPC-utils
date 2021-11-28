# from distutils.core import setup, Extension
from setuptools import setup, Extension
import numpy
import os

# module = Extension("tpc_utils", sources = ["myModule.cpp"], include_dirs=[numpy.get_include()], extra_compile_args=[r"-fopenmp"], extra_link_args=[r"-fopenmp"])
module = Extension("tpc_utils", sources = ["TPC_MODULE_FUNCTIONS.cpp"], include_dirs=[numpy.get_include()])

setup(name = "tpc_utils",
version = "0.1",
author = "Guilherme Ferrari Fortino",
author_email = "ferrarifortino@gmail.com",
maintainer = "Guilherme Ferrari Fortino",
maintainer_email = "ferrarifortino@gmail.com",
description = "Package with useful functions for TPC data analysis.",
ext_modules = [module],
py_modules = ["tpc_utils"])

pasta_atual = os.path.dirname(os.path.realpath(__file__))
build_dir = ""
lib_dir   = ""
# print(pasta_atual)
subpastas = [f.path for f in os.scandir(pasta_atual) if f.is_dir()] 
# print(subpastas)
for dirpath, dirnames, filenames in os.walk(pasta_atual):
    for dirname in dirnames:
        if dirname == "build":
            dirname = os.path.join(dirpath, dirname)
            build_dir = dirname
            break
for dirpath, dirnames, filenames in os.walk(build_dir):
    for dirname in dirnames:
        if "lib" in dirname:
            lib_dir = os.path.join(dirpath, dirname)
file = [x[2] for x in os.walk(lib_dir)][0][0]
os.rename(os.path.join(lib_dir, file), os.path.join(pasta_atual, file))