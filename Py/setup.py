# setup.py
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("DISCO4HD.disco_optimized",  # 明确模块路径
        ["DISCO4HD/disco.cpp"],  # 注意路径
        include_dirs=["/usr/include/eigen3"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="DISCO4HD",
    version="1.0",
    author="Meng Hao",
    packages=find_packages(),  # 自动查找 DISCO4HD 包
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    package_data={
        "DISCO4HD": ["disco_optimized*.so", "main.py"]  # 包含二进制和脚本
    },
)
