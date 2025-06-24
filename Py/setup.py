from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("disco_optimized",
        ["disco.cpp"],
        include_dirs=["/usr/include/eigen3"],
        extra_compile_args=["-O3", "-Wall", "-shared", "-fPIC", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="DISCO4HD",
    version="1.0",
    author="Meng Hao",
    email="haombio@gmail.com",
    description="Python package for calculating distance of covariance as a measure of homeostatic dysregulations",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
