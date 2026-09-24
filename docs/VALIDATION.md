# Version 0.3.0 validation

Validated on 2026-09-24 with R 4.5.3 on arm64 macOS.

- `R CMD build` and `R CMD check --no-manual`: **Status: OK**, zero ERROR/WARNING/NOTE.
- **83 test expectations passed**, no failures, warnings or skipped tests; includes the existing two-worker PSOCK backend check.
- New tests reproduce three independent web-calculator fixtures, reconstruct reference samples from sufficient statistics for all five cohorts, verify batching/column-order invariance, preserve subset normalization and reject missing/invalid inputs or incompatible calibration.
- Rebuilding each aggregate reference from controlled inputs agreed with literal individual augmentation to a maximum absolute DISCO difference of 1.03e-10 across the sampled validation rows. See `data-raw/reference-validation.csv`.
- The original local 0.2.0 core R and C++ implementations are preserved. The GitHub 0.1.0-to-0.2.0 migration is documented separately in `NEWS.md`.

The local R installation references a missing gfortran runtime. Validation used a temporary user Makevars file containing `FLIBS =`, linking R's installed BLAS/LAPACK. The released source retains standard `$(FLIBS)` linkage; no system R configuration was modified. OpenMP was unavailable locally; compiled serial and PSOCK parallel paths were tested. Cross-platform GitHub Actions are separate from these local results.

No full rerun of all manuscript models or independent clinical validation is implied by package checks.
