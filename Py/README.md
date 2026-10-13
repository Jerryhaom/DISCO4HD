# DISCO4HD Python

The portable NumPy implementation and optional serial C++ implementation follow
the retained DISCO4HD R source definition. Python 1.0.1 fixes field-layout and
rank-one update defects in the earlier Python extension; these are not evidence
that the R manuscript calculations used the faulty Python code.

Install the NumPy version with `python -m pip install .`. For the optional native
version, install Eigen headers, set `DISCO_BUILD_CPP=1` and
`EIGEN3_INCLUDE_DIR` to the folder containing `Eigen/`, then build and install the
wheel. See `README_PORTABLE.txt` for numerical scope and native build details.

Run `python -m unittest discover -s tests -v`. Native and independent R checks
require the corresponding optional extension and `Rscript`. A skipped optional
test is not native-build evidence. Three-platform CI is prepared in the combined
source tree; preparation is distinct from actual execution.

MIT license; original authorship and copyright are preserved in `LICENSE`.
