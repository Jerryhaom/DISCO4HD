"""Portable Python package; the optional C++ backend requires Eigen headers."""
from pathlib import Path
import os
import sys
from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}
if os.environ.get("DISCO_BUILD_CPP", "0") == "1":
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    candidates = [os.environ.get("EIGEN3_INCLUDE_DIR"), str(Path(sys.prefix) / "include/eigen3"),
                  "/usr/include/eigen3", "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3"]
    eigen = next((p for p in candidates if p and (Path(p) / "Eigen/Dense").is_file()), None)
    if eigen is None:
        raise RuntimeError("Native build needs Eigen: set EIGEN3_INCLUDE_DIR, or use DISCO_BUILD_CPP=0 for NumPy.")
    compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]
    link_args = []
    if os.environ.get("DISCO_OPENMP", "0") == "1":
        if sys.platform.startswith("linux"):
            compile_args += ["-fopenmp"]; link_args += ["-fopenmp"]
        elif sys.platform == "win32":
            compile_args += ["/openmp"]
        else:
            raise RuntimeError("macOS OpenMP is not configured; use the portable serial native backend.")
    ext_modules = [Pybind11Extension("DISCO4HD.disco_optimized", ["DISCO4HD/disco.cpp"],
                                    include_dirs=[eigen], cxx_std=17,
                                    extra_compile_args=compile_args, extra_link_args=link_args)]
    cmdclass = {"build_ext": build_ext}
setup(name="DISCO4HD", version="1.0.1", author="Meng Hao", packages=find_packages(),
      license="MIT", license_files=["LICENSE"],
      python_requires=">=3.10", install_requires=["numpy>=1.23,<3", "pandas>=1.5,<3"],
      ext_modules=ext_modules, cmdclass=cmdclass,
      description="Correlation-perturbation DISCO with validated NumPy and optional C++ backends")
