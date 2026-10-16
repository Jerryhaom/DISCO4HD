#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::export]]
bool disco_openmp_enabled() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// [[Rcpp::export]]
Rcpp::NumericVector disco_optimized_rcpp(const Rcpp::NumericMatrix& d,
                                       const Rcpp::NumericMatrix& d2,
                                       const Rcpp::NumericMatrix& weight,
                                       int ncores = 4) {
  const int n = d2.nrow(), p = d2.ncol(), queries = d.nrow();
  if (n < 3 || p < 2 || d.ncol() != p || weight.nrow() != p ||
      weight.ncol() != p || ncores < 1)
    Rcpp::stop("Invalid reference, weight dimensions or ncores");
  for (double value : d) if (!std::isfinite(value)) Rcpp::stop("Inputs must be finite");
  for (double value : d2) if (!std::isfinite(value)) Rcpp::stop("Inputs must be finite");
  for (double value : weight)
    if (!std::isfinite(value) || value < 0)
      Rcpp::stop("Inputs must be finite and weights nonnegative");
  std::vector<double> mu(p, 0.0), ss(static_cast<size_t>(p) * p, 0.0), sd(p);
  for (int j = 0; j < p; ++j) {
    for (int k = 0; k < n; ++k) mu[j] += d2(k, j);
    mu[j] /= n;
  }
  for (int j = 0; j < p; ++j) {
    for (int k = 0; k <= j; ++k) {
      double value = 0.0;
      for (int r = 0; r < n; ++r)
        value += (d2(r, j) - mu[j]) * (d2(r, k) - mu[k]);
      ss[static_cast<size_t>(j) * p + k] = value;
      ss[static_cast<size_t>(k) * p + j] = value;
    }
    sd[j] = std::sqrt(ss[static_cast<size_t>(j) * p + j]);
    if (!(sd[j] > 0) || !std::isfinite(sd[j]))
      Rcpp::stop("Reference biomarkers must have positive finite variance");
  }
  Rcpp::NumericVector result(queries);
  const double factor = static_cast<double>(n) / (n + 1.0);
  // Input checks and allocation occur outside workers; no R API calls in workers.
#ifdef _OPENMP
#pragma omp parallel for num_threads(ncores)
#endif
  for (int i = 0; i < queries; ++i) {
    try {
      std::vector<double> delta(p), sn(p);
      for (int j = 0; j < p; ++j) {
        delta[j] = d(i, j) - mu[j];
        sn[j] = std::sqrt(ss[static_cast<size_t>(j) * p + j] + factor * delta[j] * delta[j]);
      }
      double value = 0.0;
      for (int j = 0; j < p; ++j) for (int k = 0; k < p; ++k) {
        const double ref_ss = ss[static_cast<size_t>(j) * p + k];
        const double original = ref_ss / (sd[j] * sd[k]);
        const double augmented = (ref_ss + factor * delta[j] * delta[k]) / (sn[j] * sn[k]);
        const double difference = original - augmented;
        value += difference * difference * weight(j, k);
      }
      result[i] = std::log(value * static_cast<double>(n) * n);
    } catch (...) { result[i] = std::numeric_limits<double>::quiet_NaN(); }
  }
  for (double value : result) if (std::isnan(value))
    Rcpp::stop("Numerical failure in C++ backend; check biomarker scales");
  return result;
}
