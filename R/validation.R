# Shared checks keep all backends on the same mathematical domain.
.disco_matrix <- function(x, label, min_rows = 1L) {
  if (is.data.frame(x) && !all(vapply(x, is.numeric, logical(1))))
    stop(label, " must contain only numeric biomarker columns", call. = FALSE)
  x <- as.matrix(x)
  if (!is.numeric(x) || ncol(x) < 2L || nrow(x) < min_rows)
    stop(label, " must be numeric with at least two columns and ", min_rows, " rows", call. = FALSE)
  if (any(!is.finite(x))) stop(label, " must contain finite complete data; impute or remove missing values explicitly", call. = FALSE)
  x
}
.disco_age <- function(age, n) {
  if (!is.numeric(age) || length(age) != n || any(!is.finite(age)) || stats::sd(age) == 0)
    stop("age must be a finite numeric vector with one value per row and positive variance", call. = FALSE)
  age
}
.disco_reference <- function(x) {
  if (nrow(x) < 3L) stop("reference must contain at least three individuals", call. = FALSE)
  s <- apply(x, 2, stats::sd)
  if (any(!is.finite(s)) || any(s <= 0)) stop("reference biomarkers must have positive finite variance", call. = FALSE)
  invisible(x)
}
.disco_weights <- function(x, age) {
  age <- .disco_age(age, nrow(x))
  if (any(apply(x, 2, stats::sd) <= 0)) stop("analysis biomarkers must have positive variance", call. = FALSE)
  cc <- as.vector(stats::cor(x, age))
  w <- abs(outer(cc, cc)); diag(w) <- 0
  if (any(!is.finite(w)) || sum(w) <= 0) stop("age weights have zero or non-finite total", call. = FALSE)
  w / sum(w)
}
