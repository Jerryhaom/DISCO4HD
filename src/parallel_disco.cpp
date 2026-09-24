#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
bool disco_openmp_enabled() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// [[Rcpp::export]]
arma::vec disco_optimized_rcpp(const arma::mat& d, const arma::mat& d2,
                              const arma::mat& weight, int ncores = 4) {
  const int n = d2.n_rows;
  const int p = d2.n_cols;
  if (n < 3 || p < 2 || d.n_cols != d2.n_cols ||
      weight.n_rows != d2.n_cols || weight.n_cols != d2.n_cols || ncores < 1)
    Rcpp::stop("Invalid reference, weight dimensions or ncores");
  if (!d.is_finite() || !d2.is_finite() || !weight.is_finite() || arma::any(arma::vectorise(weight) < 0))
    Rcpp::stop("Inputs must be finite and weights nonnegative");
  const arma::rowvec mu = arma::mean(d2, 0);
  const arma::mat centered = d2.each_row() - mu;
  const arma::mat ss = centered.t() * centered;
  const arma::vec sd = arma::sqrt(ss.diag());
  if (arma::any(sd <= 0)) Rcpp::stop("Reference biomarkers must have positive variance");
  const arma::mat original = ss / (sd * sd.t());
  arma::vec result(d.n_rows);
  // No R API calls inside an OpenMP worker; report failures after joining.
#ifdef _OPENMP
#pragma omp parallel for num_threads(ncores)
#endif
  for (int i = 0; i < static_cast<int>(d.n_rows); ++i) {
    try {
      const arma::rowvec delta = d.row(i) - mu;
      const arma::mat ss_new = ss + (static_cast<double>(n) / (n + 1.0)) * delta.t() * delta;
      const arma::vec sn = arma::sqrt(ss_new.diag());
      const arma::mat corr = ss_new / (sn * sn.t());
      const arma::mat diff = original - corr;
      const double value = arma::accu(diff % diff % weight);
      result[i] = std::log(value * static_cast<double>(n) * n);
    } catch (...) { result[i] = arma::datum::nan; }
  }
  if (result.has_nan()) Rcpp::stop("Numerical failure in C++ backend; check biomarker scales");
  return result;
}
