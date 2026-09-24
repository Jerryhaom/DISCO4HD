#' Decompose DISCO into within- and cross-modality components
#'
#' @description
#' Computes the DISCO perturbation for each individual and decomposes it into
#' within-modality and cross-modality contributions defined by a block partition
#' of the features (e.g. clinical vs. proteomics). Optionally evaluates the
#' mortality discrimination of each component and tests whether the cross-modality
#' component adds independent predictive power beyond the within-modality
#' components (likelihood-ratio test).
#'
#' @param X Numeric matrix of biomarkers; rows are individuals, columns are features.
#'   Should be imputed (no missing values) and ideally standardized.
#' @param blocks Named list mapping each modality to its column indices or column
#'   names in `X`, e.g. `list(clinical = 1:58, proteomics = 59:425)`. Every column
#'   of `X` must appear in exactly one block.
#' @param ref_idx Integer or logical vector selecting the reference (young healthy)
#'   individuals. If `NULL`, the reference is defined as `age <= ref_age`.
#' @param time Optional numeric vector of follow-up time (enables C-index / LRT).
#' @param status Optional numeric 0/1 vector of event status (enables C-index / LRT).
#' @param age Optional numeric vector of age (used for the reference cutoff, for
#'   age correlation, and as a covariate in adjusted models).
#' @param sex Optional covariate vector (used in age- and sex-adjusted models).
#' @param weighting Either \code{age} (default, identical weights to cal_disco) or
#'   \code{unweighted} (explicit historical structural perturbation).
#' @param ref_age Age cutoff defining the reference when `ref_idx = NULL` (default 40).
#'
#' @details Age weights use all rows of X, including reference rows, as in
#' cal_disco. Only non-reference rows are returned. Their indices are in
#' analysis_idx. A single block is supported (no cross-component LRT).
#' Zero perturbations give -Inf; undefined correlations/C-indices are NA.
#' @return A list including analysis_idx, reference_idx, weighting, weights and:
#' \itemize{
#'   \item \code{decomposition}: matrix (rows = analysis individuals) with columns
#'     \code{full}, \code{within_<block>}, and \code{cross_<a>_<b>} holding the raw
#'     perturbation sums.
#'   \item \code{per_pair}: data.frame of per-block pair counts and mean per-pair
#'     weighted perturbation per directed pair. Zero-pair blocks return NA.
#'     Unweighted mode uses the same corrected pair counting.
#'   \item \code{sub_disco}: matrix of sub-DISCO values, \code{log(n^2 * S_block)},
#'     with the same columns as \code{decomposition}.
#'   \item \code{cindex}: named vector of mortality C-index per block (if \code{time}
#'     and \code{status} are provided).
#'   \item \code{age_cor}: named vector of Spearman correlation with age per block
#'     (if \code{age} is provided).
#'   \item \code{lrt}: list with the likelihood-ratio test of the cross-modality
#'     component versus the within-modality components (crude and adjusted), when
#'     \code{time}, \code{status}, and \code{age} are provided.
#' }
#' @export
#'
#' @examples
#' \dontrun{
#' # two modalities (e.g. 58 clinical + 367 proteomic features)
#' X <- cbind(clinical, proteomics)
#' blocks <- list(clinical = 1:58, proteomics = 59:425)
#' out <- disco_modality_decompose(
#'   X = X, blocks = blocks,
#'   age = age, ref_age = 40,
#'   time = time, status = status, sex = sex
#' )
#' out$per_pair
#' out$cindex
#' out$lrt
#' }
disco_modality_decompose <- function(X, blocks, ref_idx = NULL,
                                     time = NULL, status = NULL,
                                     age = NULL, sex = NULL, ref_age = 40,
                                     weighting = c("age", "unweighted")) {
  X <- .disco_matrix(X, "X", 3L)
  weighting <- match.arg(weighting)
  if (!is.null(age)) age <- .disco_age(age, nrow(X))
  if (weighting == "age" && is.null(age)) stop("age is required for age-weighted decomposition")
  if (!is.list(blocks) || !length(blocks)) stop("blocks must be a nonempty list")
  block_names <- names(blocks)
  if (is.null(block_names)) block_names <- paste0("modality_", seq_along(blocks))
  if (anyNA(block_names) || any(!nzchar(block_names)) || anyDuplicated(block_names))
    stop("block names must be nonempty and unique")
  names(blocks) <- block_names
  blocks <- lapply(blocks, function(b) {
    if (is.character(b)) {
      if (is.null(colnames(X)) || anyDuplicated(colnames(X))) stop("X needs unique column names for character blocks")
      b <- match(b, colnames(X))
    }
    if (!is.numeric(b) || !length(b) || anyNA(b) || any(!is.finite(b)) ||
        any(b != floor(b)) || any(b < 1L | b > ncol(X))) stop("block indices must be valid column indices")
    as.integer(b)
  })
  all_idx <- unlist(blocks, use.names = FALSE)
  if (anyDuplicated(all_idx) || !identical(sort(all_idx), seq_len(ncol(X))))
    stop("blocks must cover every column of X exactly once")
  if (is.null(ref_idx)) {
    if (is.null(age)) stop("provide ref_idx or age")
    if (!is.numeric(ref_age) || length(ref_age) != 1L || !is.finite(ref_age)) stop("ref_age must be finite")
    ref_idx <- which(age <= ref_age)
  }
  if (is.logical(ref_idx)) {
    if (length(ref_idx) != nrow(X) || anyNA(ref_idx)) stop("logical ref_idx must have one nonmissing value per row")
    ref_idx <- which(ref_idx)
  }
  if (!is.numeric(ref_idx) || anyNA(ref_idx) || any(!is.finite(ref_idx)) ||
      any(ref_idx != floor(ref_idx)) || any(ref_idx < 1 | ref_idx > nrow(X)) || anyDuplicated(ref_idx))
    stop("ref_idx must contain distinct valid row indices")
  ref_idx <- as.integer(ref_idx)
  for (nm in c("time", "status", "sex")) {
    val <- get(nm)
    if (!is.null(val) && (length(val) != nrow(X) || anyNA(val))) stop(nm, " must have one nonmissing value per row")
  }
  if (xor(is.null(time), is.null(status))) stop("provide both time and status")
  if (!is.null(time) && (!is.numeric(time) || any(!is.finite(time)) || any(time < 0))) stop("time must be finite and nonnegative")
  if (!is.null(status) && any(!status %in% c(0, 1))) stop("status must be 0/1")
  Xref <- X[ref_idx, , drop = FALSE]
  .disco_reference(Xref)
  weight <- if (weighting == "age") .disco_weights(X, age) else {
    w <- matrix(1, ncol(X), ncol(X)); diag(w) <- 0; w
  }
  n <- nrow(Xref)
  mu <- colMeans(Xref)
  S <- stats::cov(Xref)
  Rref <- stats::cor(Xref)
  c1 <- (n - 1) / n
  c2 <- 1 / (n + 1)

  analysis <- setdiff(seq_len(nrow(X)), ref_idx)
  if (length(analysis) == 0) stop("no analysis individuals remain (all are reference)")

  k <- length(blocks)
  cross_pairs <- if (k >= 2L) utils::combn(seq_len(k), 2) else matrix(integer(), 2L, 0L)
  col_names <- c(
    "full",
    paste0("within_", block_names),
    if (ncol(cross_pairs)) paste0("cross_", block_names[cross_pairs[1, ]], "_", block_names[cross_pairs[2, ]]) else character()
  )

  # pair counts for per-pair normalization
  sizes <- vapply(blocks, length, integer(1))
  pair_counts <- c(
    full = sum(sizes) * (sum(sizes) - 1) / 2,
    stats::setNames(sizes * (sizes - 1) / 2, paste0("within_", block_names)),
    stats::setNames(sizes[cross_pairs[1, ]] * sizes[cross_pairs[2, ]],
                    if (ncol(cross_pairs)) paste0("cross_", block_names[cross_pairs[1, ]], "_", block_names[cross_pairs[2, ]]) else character())
  )

  decomposition <- matrix(NA_real_, length(analysis), length(col_names),
                          dimnames = list(NULL, col_names))

  for (j in seq_along(analysis)) {
    i <- analysis[j]
    dd <- X[i, ] - mu
    Snew <- c1 * S + c2 * (dd %*% t(dd))
    snew <- sqrt(pmax(diag(Snew), 0))
    Rnew <- Snew / (snew %*% t(snew))
    Diff <- (Rref - Rnew)^2 * weight
    diag(Diff) <- 0

    decomp <- numeric(length(col_names))
    names(decomp) <- col_names
    decomp["full"] <- sum(Diff)
    for (b in seq_along(blocks)) {
      decomp[paste0("within_", block_names[b])] <- sum(Diff[blocks[[b]], blocks[[b]]])
    }
    for (p in seq_len(ncol(cross_pairs))) {
      a <- cross_pairs[1, p]; b <- cross_pairs[2, p]
      decomp[paste0("cross_", block_names[a], "_", block_names[b])] <-
        2 * sum(Diff[blocks[[a]], blocks[[b]]])
    }
    decomposition[j, ] <- decomp
  }

  per_pair <- data.frame(
    block = names(pair_counts),
    n_pairs = as.numeric(pair_counts),
    mean_per_pair = ifelse(pair_counts > 0, colMeans(decomposition) / (2 * as.numeric(pair_counts)), NA_real_),
    stringsAsFactors = FALSE
  )

  sub_disco <- log(n^2 * decomposition)

  out <- list(decomposition = decomposition, per_pair = per_pair, sub_disco = sub_disco,
              analysis_idx = analysis, reference_idx = ref_idx, weighting = weighting, weights = weight)

  if (!is.null(age)) {
    out$age_cor <- vapply(seq_len(ncol(sub_disco)), function(kk) {
      good <- is.finite(sub_disco[, kk])
      if (sum(good) < 3L || stats::sd(sub_disco[good, kk]) == 0) return(NA_real_)
      stats::cor(sub_disco[good, kk], age[analysis][good], method = "spearman")
    }, numeric(1))
    names(out$age_cor) <- col_names
  }

  if (!is.null(time) && !is.null(status)) {
    tm <- time[analysis]; st <- status[analysis]
    dd0 <- data.frame(tm = tm, st = st)
    out$cindex <- vapply(seq_len(ncol(sub_disco)), function(kk) {
      if (any(!is.finite(sub_disco[, kk])) || stats::sd(sub_disco[, kk]) == 0) return(NA_real_)
      f <- survival::coxph(survival::Surv(tm, st) ~ sub_disco[, kk], data = dd0)
      summary(f)$concordance[1]
    }, numeric(1))
    names(out$cindex) <- col_names

    if (!is.null(age) && k >= 2L) {
      # within total and cross total
      within_cols <- grep("^within_", col_names, value = TRUE)
      cross_cols <- grep("^cross_", col_names, value = TRUE)
      within_total <- log(n^2 * rowSums(decomposition[, within_cols, drop = FALSE]))
      cross_total <- log(n^2 * rowSums(decomposition[, cross_cols, drop = FALSE]))
      if (any(!is.finite(within_total)) || any(!is.finite(cross_total))) {
        warning("LRT omitted: a component has zero perturbation (log = -Inf)")
        return(out)
      }
      d_lrt <- data.frame(tm = tm, st = st, within = within_total, cross = cross_total,
                          age = age[analysis])
      if (!is.null(sex)) d_lrt$sex <- sex[analysis]
      base_cov <- if (!is.null(sex)) "+ age + sex" else "+ age"
      A_c <- survival::coxph(survival::Surv(tm, st) ~ within, data = d_lrt)
      B_c <- survival::coxph(survival::Surv(tm, st) ~ within + cross, data = d_lrt)
      A_a <- survival::coxph(stats::as.formula(paste0("survival::Surv(tm, st) ~ within", base_cov)), data = d_lrt)
      B_a <- survival::coxph(stats::as.formula(paste0("survival::Surv(tm, st) ~ within + cross", base_cov)), data = d_lrt)
      out$lrt <- list(
        crude = list(
          cindex_within = summary(A_c)$concordance[1],
          cindex_within_cross = summary(B_c)$concordance[1],
          p_value = stats::anova(A_c, B_c)$`Pr(>|Chi|)`[2]
        ),
        adjusted = list(
          cindex_within = summary(A_a)$concordance[1],
          cindex_within_cross = summary(B_a)$concordance[1],
          p_value = stats::anova(A_a, B_a)$`Pr(>|Chi|)`[2]
        )
      )
    }
  }

  out
}
