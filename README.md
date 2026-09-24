# DISCO4HD

R implementation of DISCO (Distance of Covariance), accompanying [*Human aging reflects increases in entropy across organ networks*](https://www.medrxiv.org/content/10.1101/2025.05.06.25327108v2). [Web calculator](https://jerryhaom.github.io/DISCO-Calculator/) · [Article analysis code](https://github.com/Jerryhaom/DISCO_Code)

## New in 0.3.0: clinical panel with selectable references

```r
# install.packages("remotes")
remotes::install_github("Jerryhaom/DISCO4HD")
library(DISCO4HD)
disco10_references()
x <- data.frame(CRP=1, GLU=5, CHOL=4.5, HDL=1.3, CREA=80,
                UA=300, WBC=6, UREA=5, RBC=4.8, ALB=42)
cal_disco10(x, reference="UKB")       # DISCO -2.118126; raw DM 2.152269
cal_disco10(x, reference="NHANES")
cal_disco10(x, reference="CHARLS")    # 8 biomarkers
cal_disco10(x, reference="RuLAS")     # 9 biomarkers
cal_disco10(x, reference="UKB", comparison="NHANES")
```

References: UKB / NHANES / CLHLS contain ten biomarkers; CHARLS lacks RBC and ALB, and RuLAS lacks UREA. `reference` selects the actual calculation matrix. Optional `comparison` uses the web calculator's UKB-based distributions for a separate z-score and approximate percentile. DM is unlogged; only CRP is transformed as ln(CRP + 1). No target age or reference records are needed for frozen-reference scoring.

Read the **[10-biomarker guide](docs/10-biomarker-guide.md)** for required units, equations, reference ages and sample sizes, batch examples, provenance, and calibration limits. R help: `?cal_disco10`, `?disco10_reference`, `?disco10_references`. The existing `cal_disco()` and decomposition interfaces are unchanged from the local 0.2.0 release. The historical `Py/` subtree is retained but is not the implementation of these new R functions and is excluded from R package builds.


Distance of Covariance (DISCO), version 0.3.0. Computes age-weighted perturbation of biomarker correlations when an individual is added to a reference population.

## Install

Use R >= 4.1, a C++17 compiler, and the Rcpp / RcppArmadillo headers. Standard R development tools (including the Fortran runtime used by R's BLAS/LAPACK) are required when building from source.

```r
install.packages(c("Rcpp", "RcppArmadillo", "survival"))
install.packages("DISCO4HD_0.3.0.tar.gz", repos = NULL, type = "source")
```

OpenMP is optional: the build uses R's `SHLIB_OPENMP_CXXFLAGS`. `disco_backend_info()$openmp` reports whether the compiled C++ backend supports it. When FALSE, `cpp=TRUE` still accelerates computation but is serial. The `parallel=TRUE` R backend uses PSOCK workers independently of OpenMP.

## Small runnable example (synthetic data only)

```r
library(DISCO4HD)
set.seed(123)
age <- seq(20, 79)
d <- data.frame(age, a=age/10+rnorm(60), b=age/30+rnorm(60),
                c=rnorm(60), e=age/15+rnorm(60))
biomarkers <- c("a", "b", "c", "e")
ref_idx <- which(age <= 34)
r <- cal_disco(d, biomarkers, d[ref_idx, ])
c <- cal_disco(d, biomarkers, d[ref_idx, ], cpp=TRUE)
stopifnot(isTRUE(all.equal(r$DISCO, c$DISCO, tolerance=1e-9)))
z <- disco_modality_decompose(d[biomarkers], list(first=1:2, second=3:4),
                              ref_idx=ref_idx, age=age)
stopifnot(isTRUE(all.equal(unname(z$sub_disco[, "full"]),
                          r$DISCO[z$analysis_idx], tolerance=1e-9)))
```

## Definition and input contract

`w[j,k] = abs(cor(age, X[,j]) * cor(age, X[,k]))`, with diagonal zero and all entries normalized to sum to one. The weights are estimated over all supplied analysis rows (all rows of `d4`, or all rows of `X` in the decomposition), including reference rows when present.

`DISCO[i] = log(n_ref^2 * sum(w * (cor(ref) - cor(rbind(ref, x_i)))^2))`.

The reference-size factor compensates for leading-order perturbation scaling; finite-sample scores need not be exactly invariant to the reference size. Biomarker and age values must be finite and complete, the reference must contain at least three rows with positive variance in each biomarker, and total age weight must be positive. Missing values are rejected consistently by all backends; choose and document imputation before calling the package. Do not include age in the biomarker list unless that is an intentional part of your scientific definition.

`cal_disco(d4, var, ref, parallel=FALSE, cpp=FALSE, ncores=4)` returns the original data frame with a DISCO column. C++ takes precedence when both backend flags are TRUE. Zero perturbation returns `-Inf` without artificial clipping.

## Modality decomposition

`disco_modality_decompose(X, blocks, ref_idx, age=age)` defaults to the same age-weighted definition. It returns only non-reference individuals and exposes `analysis_idx`, `reference_idx`, and `weights`. Raw within/cross components add to `full`; their logarithms do not add. Cross components include both orientations of each feature pair. `mean_per_pair` divides the symmetric sum by twice the number of unordered pairs (the average over directed entries). One-feature blocks have no internal pairs and their per-pair mean is NA.

`weighting="unweighted"` retains the historical unweighted raw perturbation definition. It is not the age-weighted main DISCO score. Survival C-indices and LRT are optional; a single modality has no cross-component LRT. Undefined C-indices are NA and a zero total component prevents the LRT.

## Version changes

See [NEWS.md](NEWS.md). In particular, 0.2.0 changes the decomposition default and rejects missing inputs that older backends handled inconsistently. Do not silently replace historical manuscript scores with newly computed scores.

## Tests

```sh
R CMD build DISCO4HD
R CMD check --no-manual DISCO4HD_0.3.0.tar.gz
```

The tests compare backends with independent correlation recomputation, verify block additivity and invalid inputs, and exercise two PSOCK workers. `DISCO_SKIP_SOCKET_TESTS=1` is an explicit opt-out for restricted sandboxes, not the default.

MIT; see LICENSE.md. Bundled NHANES4 is the pre-existing dataset from 0.1.0; data provenance/documentation is in `?NHANES4`. The synthetic example does not require it.
