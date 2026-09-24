#' cal_disco
#' @description Calculate distance of covariance for quantifying homeostatic dysregulations
#' @param d4 Dataframe, containing age and biomarker information
#' @param var Enrolled biomarkers variables for analysis
#' @param ref Dataframe, young population information as a reference
#' @param parallel bool, use multiple cores for computation (default: FALSE)
#' @param cpp bool, use Rcpp with multiple cores for computation (default: FALSE)
#' @param ncores numeric, cpu cores for computation (default: 4)
#'
#' @details Biomarkers and age must be finite and complete. The reference needs
#' at least three rows and positive variance in every biomarker. Weights are
#' estimated from all rows of d4. A zero perturbation returns -Inf.
#' @return a data.frame with DISCO values
#' \itemize{
#' \item column DISCO, distance of covariance
#' }
#' @export
#'
#' @examples
#' \dontrun{
#' data(NHANES4)
#' var <- c("albumin", "alp", "creat", "glucose_mmol", "lymph", "mcv", "rdw", "wbc", "ggt")
#' NHANES4y <- NHANES4[which(NHANES4$age <= 30), ]
#' NHANES4m <- cal_disco(NHANES4, var, NHANES4y)
#' }

cal_disco <- function(d4, var, ref, parallel = FALSE, cpp = FALSE, ncores = 4) {
  if (!is.data.frame(d4) || !is.data.frame(ref)) stop("d4 and ref must be data.frames")
  if (!is.character(var) || length(var) < 2L || anyNA(var) || anyDuplicated(var))
    stop("var must name at least two distinct biomarkers")
  if (!all(var %in% names(d4)) || !all(var %in% names(ref))) stop("Some variables not found in d4 or ref")
  if (!is.logical(parallel) || length(parallel) != 1L || is.na(parallel) ||
      !is.logical(cpp) || length(cpp) != 1L || is.na(cpp)) stop("parallel and cpp must be TRUE or FALSE")
  if (!is.numeric(ncores) || length(ncores) != 1L || !is.finite(ncores) ||
      ncores < 1 || ncores > .Machine$integer.max || ncores != floor(ncores)) stop("ncores must be a positive integer")
  d_data <- .disco_matrix(d4[, var, drop = FALSE], "d4", 2L)
  d_ref <- .disco_matrix(ref[, var, drop = FALSE], "ref", 3L)
  .disco_reference(d_ref)
  weight <- .disco_weights(d_data, d4$age)
  n_ref <- nrow(d_ref)
  if (cpp) {
    result <- as.numeric(disco_optimized_rcpp(d_data, d_ref, weight, as.integer(ncores)))
  } else {
    ref_cor <- stats::cor(d_ref)
    score <- function(i) {
      new_cor <- stats::cor(rbind(d_ref, d_data[i, , drop = FALSE]))
      log(sum((ref_cor - new_cor)^2 * weight) * n_ref^2)
    }
    if (parallel) {
      cl <- parallel::makeCluster(ncores)
      on.exit(parallel::stopCluster(cl), add = TRUE)
      # parLapply avoids altering the caller's global foreach backend.
      parallel::clusterExport(cl, c("d_ref", "d_data", "ref_cor", "weight", "n_ref"), envir = environment())
      result <- unlist(parallel::parLapply(cl, seq_len(nrow(d_data)), function(i) {
        new_cor <- stats::cor(rbind(d_ref, d_data[i, , drop = FALSE]))
        log(sum((ref_cor - new_cor)^2 * weight) * n_ref^2)
      }), use.names = FALSE)
    } else result <- vapply(seq_len(nrow(d_data)), score, numeric(1))
  }
  out <- d4
  out$DISCO <- result
  out
}
