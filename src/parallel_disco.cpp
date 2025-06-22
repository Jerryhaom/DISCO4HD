#include <RcppArmadillo.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::export]]

arma::vec disco_optimized_rcpp(const arma::mat& d, 
                               const arma::mat& d2, 
                               const arma::mat& weight, 
                               int ncores = 4) {
  const int n0 = d2.n_rows;
  const int p = d2.n_cols;
  if (d.n_cols != p) {
    stop("d and d2 must have the same number of columns");
  }
  
  const rowvec mu = mean(d2, 0);
  const mat center_d2 = d2.each_row() - mu;
  const mat cov_mat = (center_d2.t() * center_d2) / (n0 - 1.0);
  const vec s = sqrt(cov_mat.diag());
  
  vec results(d.n_rows, fill::zeros);
  
#ifdef _OPENMP
  omp_set_num_threads(ncores);
#pragma omp parallel for
#endif
  for(int k = 0; k < d.n_rows; ++k) {
    try {
      const rowvec x = d.row(k);
      const rowvec delta = x - mu;
      const rowvec delta_mu = delta / (n0 + 1.0);
      
      // Update covariance matrix with mean correction
      const mat delta_outer = delta.t() * delta;
      const mat delta_mu_outer = delta_mu.t() * delta_mu;
      
      const mat sum_old = (n0 - 1.0) * cov_mat;
      const mat sum_new = sum_old + delta_outer - n0 * delta_mu_outer;
      const mat cov_prime = sum_new / n0;
      
      // Calculate new standard deviations
      vec var_prime = cov_prime.diag();
      var_prime = arma::clamp(var_prime, 1e-10, var_prime.max());
      const vec s_prime = sqrt(var_prime);
      
      // Calculate correlation matrices
      const mat s_outer = s_prime * s_prime.t();
      mat cc1 = cov_prime / s_outer;
      cc1.replace(datum::nan, 0.0);
      
      const mat original_corr = cov_mat / (s * s.t());
      const mat diff = original_corr - cc1;
      
      // Calculate weighted sum
      const double sum_val = accu(diff % diff % weight);
      results[k] = (sum_val > 0) ? log(sum_val * n0 * n0) : log(1e-300);
      
    } catch(...) {
      results[k] = NA_REAL;
    }
  }
  return results; // 直接返回 arma::vec
}