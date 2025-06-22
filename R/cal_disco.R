#' cal_disco
#' @description Calculate distance of covariance for quantifying homeostatic dysregulations
#' @param d4 Dataframe, containing age and biomarker information
#' @param var Enrolled biomarkers variables for analysis
#' @param ref Dataframe, young population information as a reference
#' @param parallel bool, use multiple cores for computation (default: FALSE)
#' @param cpp bool, use Rcpp with multiple cores for computation (default: FALSE)
#' @param ncores numeric, cpu cores for computation (default: 4)
#'
#' @return a data.frame with DISCO values
#' \itemize{
#' \item column DISCO, distance of covariance
#' }
#' @export
#'
#' @examples
#' \dontrun{
#' data(NHANES4)
#' var <- c("age", "albumin", "alp", "creat", "glucose_mmol", "lymph", "mcv", "rdw", "wbc", "ggt")
#' NHANES4y <- NHANES4[which(NHANES4$age <= 30), ]
#' NHANES4m <- cal_disco(NHANES4, var, NHANES4y)
#' }

cal_disco <- function(d4, var, ref, parallel = FALSE, cpp = FALSE, ncores = 4) {

  if (!all(var %in% names(d4)))  stop("Some variables not found in d4")
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
    # Rcpp 实现
    if (!exists("disco_optimized_rcpp")) {
      stop("Rcpp function not loaded. Reinstall package with Rcpp support.")
    }
    mc1 <- disco_optimized_rcpp(
      as.matrix(d_data),
      as.matrix(d_ref),
      as.matrix(weight),
      ncores = ncores
    )
    return(data.frame(d4, DISCO = mc1))
  }
  else if (parallel) {
    # 并行 R 实现
    if (!requireNamespace("doParallel", quietly = TRUE)) {
      stop("doParallel package required for parallel computation")
    }

    cl <- parallel::makeCluster(ncores)
    doParallel::registerDoParallel(cl)

    # 导出必要变量
    parallel::clusterExport(cl, c("d_ref", "weight", "n_ref"),
                            envir = environment())

    # 计算 DISCO
    results <- foreach::foreach(
      i = 1:nrow(d_data),
      .combine = c,
      .packages = "base"
    ) %dopar% {
      d1 <- rbind(d_ref, d_data[i, ])
      cc1 <- cor(d1, use = "pairwise.complete.obs")
      ds <- sum((cor(d_ref, use = "pairwise.complete.obs") - cc1)^2 * weight, na.rm = TRUE)
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
