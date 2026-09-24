# Clinical DISCO / DM with selectable cohort references

DISCO4HD 0.3.0 adds frozen aggregate reference parameters and batch scoring for the calculator's clinical panel. The original `cal_disco()` remains available for analyses that fit age weights from the supplied population and use a user-defined reference sample.

## Quick start

```r
library(DISCO4HD)
disco10_references()

# Raw biomarker values in the units below, one person per row.
x <- data.frame(CRP=1, GLU=5, CHOL=4.5, HDL=1.3, CREA=80,
                UA=300, WBC=6, UREA=5, RBC=4.8, ALB=42)
cal_disco10(x)                         # UKB: DISCO -2.118126, raw DM 2.152269
cal_disco10(x, reference="NHANES")    # actual NHANES reference and age weights
cal_disco10(x, reference="CHARLS")    # eight biomarkers
cal_disco10(x, reference="RuLAS")     # nine biomarkers

ref <- disco10_reference("CLHLS")
ref$n_ref
ref$age_range
cal_disco10(x, reference=ref)
```

For a CSV, use `x <- read.csv("biomarkers.csv")`, then `write.csv(cal_disco10(x), "scores.csv", row.names=FALSE)`. Age is unnecessary because the age weights are already frozen. Identifier columns are retained. Required biomarker inputs must be numeric, finite, complete and nonnegative; missing markers are not automatically dropped. Extra columns, including markers outside the selected panel, are retained but not used. Input columns that collide with output names must be renamed explicitly. No automatic unit conversion is performed.

## Available references

| Reference | Markers | Reference age rule | Reference N | Sample N for age weights |
|---|---:|---|---:|---:|
| UKB | 10 | age ≤ 40 | 5,282 | 487,517 |
| NHANES | 10 | age < 30 | 6,616 | 37,627 |
| CHARLS | 8; no RBC or ALB | age ≤ 50 | 2,343 | 11,107 |
| CLHLS | 10 | age ≤ 70 | 301 | 2,481 |
| RuLAS | 9; no UREA | age ≤ 74 | 546 | 1,778 |

These references use the youngest available age bands from the cohort analyses. In particular, CLHLS and RuLAS are older-adult reference populations, not universally young or healthy controls. The table describes the analyzed waves, not every measurement available in each cohort. Reference suitability depends on assays, units, population and intended comparison; selecting a cohort is not a guarantee of transportability. The fixed parameters permit scoring small target samples but do not establish clinical validity or a universal reference population.

## Input names and units

| Column | Biomarker | Required raw unit | Internal transform |
|---|---|---|---|
| CRP | C-reactive protein | mg/L | ln(CRP + 1) |
| GLU | Glucose | mmol/L | none |
| CHOL | Total cholesterol | mmol/L | none |
| HDL | HDL cholesterol | mmol/L | none |
| CREA | Creatinine | µmol/L | none |
| UA | Uric acid | µmol/L | none |
| WBC | White blood cell count | 10⁹/L | none |
| UREA | Urea | mmol/L | none |
| RBC | Red blood cell count | 10¹²/L | none |
| ALB | Albumin | g/L | none |

Do not pass BUN in mg/dL as UREA, or already logged CRP as CRP. When converting source datasets, BUN mg/dL × 0.357 gives urea mmol/L; albumin g/dL × 10 gives g/L. The rebuild script documents all source-specific conversions.

## Reference matrix versus comparison population

`reference` determines the matrix, center and age weights used to calculate **DISCO and DM**. `comparison` selects a descriptive distribution for a **UKB-reference DISCO** score. They are distinct operations:

```r
# Full ten-marker primary DISCO / DM plus relative position in NHANES.
cal_disco10(x, reference="UKB", comparison="NHANES")

# Primary score remains ten-marker UKB. Comparison uses the eight-marker
# UKB submatrix because the CHARLS comparison distribution has only eight.
cal_disco10(x, reference="UKB", comparison="CHARLS")
```

With `comparison`, the result adds `comparison_DISCO`, `comparison_marker_count`, `DISCO_z`, `percentile`, and `percentile_band`. Z = (comparison_DISCO − stored cohort mean) / stored cohort SD. Percentiles use linear interpolation between seven rounded empirical z-score knots (5, 10, 25, 50, 75, 90, 95%). Outside these knots, numeric `percentile` is `NA` and the band is `<5` or `>95`. These are approximate ranks, not a Gaussian CDF or mortality-risk estimate.

The frozen comparison tables belong to the UKB-reference scoring definition. Applying them to an NHANES/CHARLS/CLHLS/RuLAS **own-reference** score would be incorrect, so this combination is rejected. For these references, leave `comparison=NULL` and compute DISCO / raw DM only.

## Equations

Let y be raw inputs after transforming only CRP. All bundled matrices use the same UKB scaling constants a and b: z = (y − a) / b. Each selected reference has its own standardized center μ, centered cross-product matrix Q, size n, correlation R and fixed age weights W. The original cohort-specific centers are preserved in μ; UKB scaling does not turn the other matrices into UKB references.

For each target individual:

```text
δ      = z − μ
Q_new  = Q + n/(n+1) δδᵀ
R_new  = diag(Q_new)^(-1/2) Q_new diag(Q_new)^(-1/2)
DISCO  = ln[n² Σ_j Σ_k W_jk (R_jk − R_new,jk)²]
S      = Q/(n−1) + 10^(-8) I
DM     = sqrt(δᵀ S^(-1) δ)
```

Weights are `abs(cor(age, marker_j) * cor(age, marker_k))`, with zero diagonal and normalized to sum to one over **both off-diagonal halves** of the complete reference panel. They are estimated from the finite complete cohort analysis sample, not just the young reference or submitted target batch. DM is **not logged**. DISCO uses the natural log, with no additional +1 or artificial floor; exact zero perturbation produces `-Inf`.

The covariance update is algebraically equivalent to appending the individual to the original reference and recalculating its correlations. It uses sufficient statistics, not a regression approximation or mortality-model prediction.

An explicit subset is supported, for example `cal_disco10(x, panel=c("CRP","GLU","HDL"))`. Subsetting retains the parent weights **without renormalization**, matching the web calculator. Panel-reduced scores have a different scale and cannot use the bundled full-panel calibration; no diagnostic or predictive performance is implied for arbitrary subsets. An own-reference eight-marker CHARLS score also differs from an eight-marker transported UKB score because both the matrix and age weights differ.

## Provenance and reproducibility

The UKB parameters and comparison tables are a full-precision snapshot of [DISCO-Calculator commit 39cce98](https://github.com/Jerryhaom/DISCO-Calculator/commit/39cce9860d0f9f320081fcd01fb4311b30f4cb73). The four other references were rebuilt from the existing harmonized cohort inputs with the same CRP transformation and age bands used in the reference-transport analysis. No additional imputation, outcome selection, or mortality model is used to build them.

`inst/extdata/disco10-references.rds` contains aggregate parameters and input-file hashes only. `data-raw/build-disco10-references.R` records joins, units, transformations and reference construction; its controlled source files are not bundled. From a source checkout with approved inputs:

```sh
DISCO_DATA_ROOT=/path/to/approved-inputs Rscript data-raw/build-disco10-references.R
```

`data-raw/reference-validation.csv` records agreement with literal reference augmentation in all five cohorts. Automated package tests also verify web fixtures, independent covariance reconstruction, batch invariance, input validation, panel handling and separate comparison calibration. These checks verify implementation; they do not establish out-of-sample mortality-model performance.
