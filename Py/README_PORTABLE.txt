Python package validation candidate, version 1.0.1

This subtree is separate from the R package (0.3.1). Python corrections do not
imply R manuscript calculations used the faulty Python extension. The accepted
R centered-sum-of-products implementation is the definition source.

Portable NumPy install: python -m pip install .
Optional serial native build: set DISCO_BUILD_CPP=1 and EIGEN3_INCLUDE_DIR to
the directory containing Eigen/, then python -m build and install that wheel.
Eigen >=3.3 and a C++17 compiler are required; no third-party headers or
machine-specific binaries are redistributed. OpenMP is optional, not required.

The output owns its NumPy memory and C/F-order fields are interpreted correctly.
SS_new = SS + n/(n+1) * delta' delta follows the R source. Finite data,
positive reference variance and matching dimensions are checked.
Zero perturbation returns -Inf, not a floor value. The unfinished Numba branch
is rejected explicitly; no unverified speed claim is made. use_cpp=True falls
back with a warning if native is absent. The native backend is currently serial.

Tests: python -m unittest discover -s tests -v
Prepared CI covers Python 3.10 and 3.12 on Linux/macOS/Windows. Configuration is
not evidence those remote jobs ran. License and author provenance are retained.
