#' List bundled clinical biomarker references
#'
#' @return A data.frame with cohort, marker_count, n_ref, age_rule, and markers.
#' @export
#' @examples
#' disco10_references()
disco10_references <- function() {
  refs <- .disco10_registry()$references
  do.call(rbind, lapply(refs, function(r) data.frame(
    cohort = r$cohort, marker_count = length(r$markers), n_ref = r$n_ref,
    age_rule = r$age_rule, markers = paste(r$markers, collapse = ", "),
    row.names = NULL)))
}

#' Select a bundled clinical biomarker reference
#'
#' @param cohort Reference cohort: UKB, NHANES, CHARLS, CLHLS, or RuLAS.
#'   Matching is case insensitive. UKB reproduces the web calculator.
#' @param panel Optional character vector of at least two biomarkers available
#'   in the reference. NULL uses its complete panel. Subset weights are not
#'   renormalized; scores from different panels are not interchangeable.
#' @return A disco10_reference list containing aggregate means, scaling,
#'   centered cross-products, correlations, weights, covariance, units, and
#'   provenance. No participant records are included in this object.
#' @details CHARLS contains eight markers (no RBC or ALB); RuLAS contains nine
#'   (no UREA). The other references contain ten. Reference age ranges differ;
#'   a reference is not automatically suitable just because its cohort matches
#'   a target population. These are frozen research parameters, not clinical
#'   normal ranges. Age weights were fitted in each reference cohort's complete
#'   analysis sample, not estimated again from the submitted individuals.
#' @export
#' @examples
#' ref <- disco10_reference("NHANES")
#' ref$markers
#' ref$n_ref
disco10_reference <- function(cohort = "UKB", panel = NULL) {
  registry <- .disco10_registry()
  cohort <- .disco10_cohort(cohort, names(registry$references))
  .disco10_subset(registry$references[[cohort]], panel)
}

#' Calculate DISCO and raw DM using frozen clinical reference parameters
#'
#' @param data A data.frame with one or more individuals. Biomarker columns
#'   must have the uppercase names and units below. Additional columns are
#'   retained. Age is not required. Missing, nonnumeric, infinite, and negative
#'   biomarker inputs are rejected rather than imputed or silently dropped.
#' @param reference A cohort name (default UKB), or the object returned by
#'   disco10_reference(). This selects the actual calculation matrix and weights.
#' @param panel Optional subset of the reference markers; see disco10_reference().
#' @param comparison Optional comparison cohort for the web calculator's
#'   DISCO z-score and approximate percentile. Only supported with the unmodified
#'   full UKB reference and no panel reduction. CHARLS and RuLAS comparisons
#'   recompute an eight- or nine-marker score using UKB submatrices, without
#'   renormalizing weights. This does not change the primary DISCO or DM.
#' @return The input data.frame with DISCO, DM, reference_cohort, marker_count.
#'   If comparison is supplied, adds comparison_cohort, comparison_marker_count,
#'   comparison_DISCO, DISCO_z, percentile, and percentile_band. Percentile is
#'   linearly interpolated between empirical 5/10/25/50/75/90/95 knots; tails
#'   return NA with band '<5' or '>95', not extrapolated ranks. Existing columns
#'   with output names are rejected. A zero perturbation gives DISCO = -Inf.
#' @details Input units: CRP mg/L; GLU, CHOL, HDL and UREA mmol/L; CREA and UA
#'   micromol/L; WBC 10^9/L; RBC 10^12/L; ALB g/L. CRP is transformed internally
#'   as ln(CRP + 1); do not supply already logged CRP. Other markers are not
#'   logged. DM is the square root of the Mahalanobis quadratic form, without
#'   a logarithm. Its covariance uses a 1e-8 diagonal ridge on the stored
#'   standardized scale, matching the web calculator.
#'
#'   For a standardized individual z and stored reference mean mu, let
#'   delta = z - mu. The centered cross-product after adding that person is
#'   ss_new = ss + n/(n+1) delta delta'. Convert ss_new to correlation R_new.
#'   DISCO = ln(n^2 sum(w * (R_ref - R_new)^2)); the sum includes both off-diagonal
#'   halves and the diagonal weights are zero. This sufficient-statistic update
#'   is algebraically equivalent to adding the person to the reference sample.
#'   Unlike cal_disco(), this function never refits weights in the input batch.
#'   The comparison tables are rounded, descriptive research distributions,
#'   not diagnostic cutoffs or mortality probabilities.
#' @export
#' @examples
#' person <- data.frame(CRP=1, GLU=5, CHOL=4.5, HDL=1.3, CREA=80,
#'                      UA=300, WBC=6, UREA=5, RBC=4.8, ALB=42)
#' cal_disco10(person, reference="UKB", comparison="NHANES")
#' cal_disco10(person, reference="CHARLS")
#' ref <- disco10_reference("RuLAS")
#' cal_disco10(person, reference=ref)
cal_disco10 <- function(data, reference = "UKB", panel = NULL, comparison = NULL) {
  if (is.character(reference)) reference <- disco10_reference(reference)
  if (!inherits(reference, "disco10_reference"))
    stop("reference must be a cohort name or disco10_reference() object", call. = FALSE)
  reference <- .disco10_subset(reference, panel)
  primary_names <- c("DISCO", "DM", "reference_cohort", "marker_count")
  comparison_names <- c("comparison_cohort", "comparison_marker_count", "comparison_DISCO",
                        "DISCO_z", "percentile", "percentile_band")
  output_names <- c(primary_names, if (!is.null(comparison)) comparison_names)
  if (!is.data.frame(data) || anyDuplicated(names(data)))
    stop("data must be a data.frame with unique column names", call. = FALSE)
  if (any(output_names %in% names(data)))
    stop("data already contains output columns; rename them before scoring", call. = FALSE)
  out <- data
  scores <- .disco10_score(data, reference)
  out$DISCO <- scores$DISCO
  out$DM <- scores$DM
  out$reference_cohort <- reference$cohort
  out$marker_count <- length(reference$markers)
  if (!is.null(comparison)) {
    if (!identical(reference, disco10_reference("UKB")))
      stop("comparison requires the unmodified full UKB reference; other calibrations are not bundled", call. = FALSE)
    registry <- .disco10_registry()
    comparison <- .disco10_cohort(comparison, names(registry$comparison))
    cs <- registry$comparison[[comparison]]
    cr <- .disco10_subset(reference, cs$markers)
    ds <- .disco10_score(data, cr)$DISCO
    z <- (ds - cs$mean) / cs$sd
    pct <- stats::approx(cs$z_knots, cs$percentiles, xout=z, rule=1)$y
    band <- ifelse(z < min(cs$z_knots), "<5", ifelse(z > max(cs$z_knots), ">95", "5-95"))
    out$comparison_cohort <- comparison
    out$comparison_marker_count <- length(cs$markers)
    out$comparison_DISCO <- ds
    out$DISCO_z <- z
    out$percentile <- pct
    out$percentile_band <- band
  }
  out
}

.disco10_registry <- function() {
  path <- system.file("extdata", "disco10-references.rds", package="DISCO4HD")
  if (!nzchar(path)) stop("Bundled DISCO reference file is missing", call. = FALSE)
  readRDS(path)
}

.disco10_cohort <- function(cohort, choices) {
  if (!is.character(cohort) || length(cohort) != 1L || is.na(cohort) ||
      !toupper(cohort) %in% toupper(choices))
    stop("cohort must be one of: ", paste(choices, collapse=", "), call. = FALSE)
  choices[match(toupper(cohort), toupper(choices))]
}

.disco10_subset <- function(reference, panel) {
  if (is.null(panel)) return(reference)
  if (!is.character(panel) || length(panel) < 2L || anyNA(panel) ||
      anyDuplicated(panel) || !all(panel %in% reference$markers))
    stop("panel must name at least two distinct available biomarkers", call. = FALSE)
  ii <- match(panel, reference$markers)
  for (field in c("ref_mean", "ref_sd", "mu", "units")) reference[[field]] <- reference[[field]][ii]
  for (field in c("ss", "Rref", "weight", "S")) reference[[field]] <- reference[[field]][ii, ii, drop=FALSE]
  reference$markers <- panel
  reference
}

.disco10_score <- function(data, reference) {
  markers <- reference$markers
  missing <- setdiff(markers, names(data))
  if (length(missing)) stop("Missing required biomarkers: ", paste(missing, collapse=", "), call. = FALSE)
  x <- .disco_matrix(data[, markers, drop=FALSE], "biomarkers")
  if (any(x < 0)) stop("biomarkers must be nonnegative in the documented units", call. = FALSE)
  if ("CRP" %in% markers) x[, "CRP"] <- log1p(x[, "CRP"])
  x <- sweep(sweep(x, 2, reference$ref_mean, "-"), 2, reference$ref_sd, "/")
  delta <- sweep(x, 2, reference$mu, "-")
  n <- reference$n_ref
  ss <- reference$ss
  fac <- n/(n+1)
  dg <- sweep(fac * delta^2, 2, diag(ss), "+")
  acc <- numeric(nrow(delta))
  for (j in seq_len(ncol(delta)-1L)) for (k in seq.int(j+1L, ncol(delta))) {
    r <- (ss[j,k] + fac * delta[,j] * delta[,k]) / sqrt(dg[,j] * dg[,k])
    acc <- acc + 2 * reference$weight[j,k] * (reference$Rref[j,k] - r)^2
  }
  dm2 <- rowSums((delta %*% solve(reference$S)) * delta)
  if (any(!is.finite(acc)) || any(!is.finite(dm2)) || any(dm2 < 0))
    stop("Numerical overflow or invalid reference covariance; check inputs and units", call. = FALSE)
  data.frame(DISCO=log(n^2 * acc), DM=sqrt(dm2))
}
