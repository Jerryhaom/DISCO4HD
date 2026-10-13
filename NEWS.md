# DISCO4HD 0.3.1 (staged)

- Add NHANES-calibrated synthetic data, scoring and plotting examples.
- Reject duplicate biomarker column names and invalid one-row age-weight inputs.
- Preserve the caller's random-number state in the simulation helper.
- Exclude the historical participant-level NHANES4 example pending documented
  provenance and redistribution conditions. Scoring functions are unchanged.

# DISCO4HD 0.3.0

- Add `cal_disco10()`, `disco10_reference()` and `disco10_references()` for frozen-reference single-person and batch DISCO / raw DM.
- Bundle aggregate UKB, NHANES, CHARLS, CLHLS and RuLAS references, explicitly identifying 10 / 8 / 9-marker panels, age bands and source sample sizes.
- Match the web calculator's UKB parameters, ln(CRP + 1), unlogged DM and panel-specific comparison calibration.
- Add a units/formula guide, source-data rebuild script and numerical regression tests. Core 0.2.0 methods are unchanged.

# DISCO4HD 0.2.0

- Reject incomplete/non-finite inputs, degenerate references and zero total weights consistently.
- Replace the unimported foreach operator with PSOCK parLapply and guaranteed worker cleanup.
- Remove C++ variance clipping / NaN-to-zero replacement and the artificial log floor.
- Add optional portable OpenMP flags, Fortran linkage and compiled capability reporting.
- Default block decomposition to globally normalized age weights; expose `weighting="unweighted"` for the historical raw definition.
- Correct the factor of two in per-pair means, validate block/reference indices and support a single block.
- Return row mapping and weight matrix. Preserve original rows in cal_disco output.
- Add backend, invalid-input and decomposition regression tests.

Migration: 0.1.0 decomposition raw sums match the unweighted mode on nondegenerate complete input, but its per-pair means were twice the corrected values. The new default decomposition is a scientific definition change. Existing manuscript pipelines and saved outputs are not automatically recomputed. Original 0.1.0 source/tarball should be retained alongside the release archive.
# DISCO4HD 0.3.1

- Add a reproducible synthetic NHANES-calibrated generator, CSV example and
  base-R visualizations with a Markdown walkthrough. No participant records are
  read by the generator; age effects are explicitly artificial.
- Preserve caller RNG state and test synthetic scoring against pure R/C++.
- Give a clear validation error when sample-trained age weights cannot be
  estimated from a single row. Frozen-reference scoring still accepts one row.
