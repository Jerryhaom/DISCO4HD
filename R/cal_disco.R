#' Calculate Distance of Covariance (DISCO)
#'
#' Quantifies homeostatic dysregulation by comparing biomarker covariance
#' between target population and young reference population.
#'
#' @param d4 Dataframe containing subject-level data with age and biomarkers.
#' @param var Character vector specifying biomarker column names for analysis.
#' @param ref Dataframe containing reference data from young population.
#' @param parallel Logical indicating whether to use parallel computation (default: FALSE).
#' @param cpp Logical indicating whether to use Rcpp implementation (default: FALSE).
#' @param ncores Integer specifying number of CPU cores for parallelization (default: 4).
#'
#' @return A data.frame containing:
#' \itemize{
#'   \item \code{DISCO}: DISCO values
#'   \item Original columns from input \code{d4}
#' }
#'
#' @section Algorithm Details:
#' Computes Mahalanobis-like distance between covariance matrices:
#' 1. Calculate biomarker-age correlations to derive weighting matrix
#' 2. Compare covariance structures:
#'    - Target: Covariance of reference + single subject
#'    - Reference: Covariance of young population
#' 3. Output: log-transformed weighted matrix differences
#'
#' @section Computation Options:
#' - R single-threaded: \code{parallel = FALSE, cpp = FALSE}
#' - R multi-threaded: \code{parallel = TRUE, cpp = FALSE}
#' - C++ optimized: \code{cpp = TRUE} (recommended for large datasets)
#'
#' @examples
#' # Load sample data
#' data(NHANES4)
#' # impute missing data
#' NHANES4=imputeMissings::impute(NHANES4)
#' # Define biomarkers
#' biomarkers <- c("albumin", "alp", "creat", "glucose_mmol", "lymph", "mcv")
#'
#' # Create young reference (age ≤ 30)
#' ref_young <- subset(NHANES4, age <= 30)
#'
#' # Calculate DISCO (single-threaded R)
#' result <- cal_disco(NHANES4, biomarkers, ref_young)
#'
#' \donttest{
#' # Parallel R implementation
#' result_parallel <- cal_disco(NHANES4, biomarkers, ref_young, parallel = TRUE)
#'
#' # C++ implementation
#' result_cpp <- cal_disco(NHANES4, biomarkers, ref_young, cpp = TRUE)
#' }
#'
#' @export
#' @importFrom foreach %dopar%
#' @importFrom parallel makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
cal_disco <- function(d4, var, ref, parallel = FALSE, cpp = FALSE, ncores = 4) {

  if (!all(var %in% names(d4)))  stop("Some variables not found in d4")
  if (!all("age" %in% names(d4)))  stop("Age not found in d4")
  if (!all(var %in% names(ref)))  stop("Some variables not found in ref")

  cc <- apply(d4[, var, drop = FALSE], 2, function(x) cor(d4$age, x, use = "complete.obs"))
  weight <- abs(cc %*% t(cc))
  diag(weight) <- 0
  weight <- weight / sum(weight)

  # 准备数据
  d_ref <- ref[, var, drop = FALSE]
  d_data <- d4[, var, drop = FALSE]
  n_ref <- nrow(d_ref)

  if (cpp) {
    # Rcpp
    mc1 <- disco_optimized_rcpp(
      as.matrix(d_data),
      as.matrix(d_ref),
      as.matrix(weight),
      ncores = ncores
    )
    return(data.frame(d4, DISCO = mc1))
  }
  else if (parallel) {
    #
    if (!requireNamespace("doParallel", quietly = TRUE)) {
      stop("doParallel package required for parallel computation")
    }
    if (!requireNamespace("foreach", quietly = TRUE)) {
      stop("foreach package required for parallel computation")
    }

    `%dopar%` <- foreach::`%dopar%`

    cl <- parallel::makeCluster(ncores)
    doParallel::registerDoParallel(cl)
    ref_cor <- cor(d_ref, use = "pairwise.complete.obs")
    parallel::clusterExport(cl,
                            c("d_ref", "d_data", "weight", "n_ref", "ref_cor"),
                            envir = environment())
    parallel::clusterEvalQ(cl, library(stats))
    results <- foreach::foreach(
      i = 1:nrow(d_data),
      .combine = c,
      .packages = "base"
    ) %dopar% {
      d1 <- rbind(d_ref, d_data[i, ])
      cc1 <- cor(d1, use = "pairwise.complete.obs")
      ds <- sum((ref_cor - cc1)^2 * weight, na.rm = TRUE)
      log(ds * n_ref * n_ref)
    }

    parallel::stopCluster(cl)
    return(data.frame(d4, DISCO = results))
  }
  else {
    # 单线程 R 实现
    mc <- numeric(nrow(d_data))

    ref_cor <- cor(d_ref, use = "pairwise.complete.obs")

    for (i in 1:nrow(d_data)) {
      d1 <- rbind(d_ref, d_data[i, ])
      cc1 <- cor(d1, use = "pairwise.complete.obs")
      ds <- sum((ref_cor - cc1)^2 * weight, na.rm = TRUE)
      mc[i] <- log(ds * n_ref * n_ref)
    }

    return(data.frame(d4, DISCO = mc))
  }
}
