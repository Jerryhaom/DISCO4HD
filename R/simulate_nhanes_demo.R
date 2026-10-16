#' Generate a synthetic NHANES-calibrated teaching dataset
#'
#' Generates fictional records using the bundled NHANES reference's aggregate
#' biomarker means and covariance. No participant records are read or sampled.
#' Ages, sex and age trends are illustrative, not estimates of NHANES population
#' distributions. This dataset must not be used for scientific validation.
#'
#' @param n Number of synthetic records (integer, at least 30).
#' @param seed Nonnegative integer random seed. The caller's RNG state is restored.
#' @return A data frame with a synthetic ID, age, sex and the ten biomarkers in
#'   the units required by [cal_disco10()]. Generation details are stored in the
#'   `synthetic_provenance` attribute.
#' @details A multivariate Gaussian is generated on the reference's standardized
#'   analysis scale, then back-transformed (including `expm1` for CRP). Negative
#'   simulated concentrations are clipped to zero. Consequently the returned
#'   moments need not equal the reference moments exactly. Twenty percent of
#'   ages are drawn from 20--29 years and the remainder from 30--85 years; this
#'   artificial design supports a young-reference demonstration. A chosen linear
#'   age shift is added for teaching only, and sex is simulated independently.
#' @export
#' @examples
#' demo <- simulate_nhanes_demo(100, seed = 101)
#' scores <- cal_disco10(demo, reference = "NHANES")
#' head(scores[c("synthetic_id", "DISCO", "DM")])
simulate_nhanes_demo <- function(n = 400L, seed = 20261001L) {
  valid_int <- function(x, lower) is.numeric(x) && length(x)==1L &&
    is.finite(x) && x==floor(x) && x>=lower && x<=.Machine$integer.max
  if(!valid_int(n,30)) stop("n must be an integer of at least 30",call.=FALSE)
  if(!valid_int(seed,0)) stop("seed must be a nonnegative integer",call.=FALSE)
  had_seed <- exists(".Random.seed",envir=.GlobalEnv,inherits=FALSE)
  if(had_seed) old_seed <- get(".Random.seed",envir=.GlobalEnv,inherits=FALSE)
  on.exit({
    if(had_seed) assign(".Random.seed",old_seed,envir=.GlobalEnv)
    else if(exists(".Random.seed",envir=.GlobalEnv,inherits=FALSE))
      rm(list=".Random.seed",envir=.GlobalEnv)
  },add=TRUE)
  set.seed(as.integer(seed))
  r <- disco10_reference("NHANES")
  ny <- max(5L,as.integer(round(n*.2)))
  age <- sample(c(stats::runif(ny,20,29),stats::runif(n-ny,30,85)))
  sex <- sample(c("Female","Male"),n,replace=TRUE)
  z <- matrix(stats::rnorm(n*length(r$markers)),n) %*% chol(r$ss/(r$n_ref-1))
  z <- sweep(z,2,r$mu,"+")
  direction <- c(CRP=.12,GLU=.12,CHOL=.05,HDL=-.05,CREA=.08,UA=.06,
                 WBC=.04,UREA=.09,RBC=-.04,ALB=-.06)
  z <- z+outer((age-25)/10,direction[r$markers])
  x <- sweep(sweep(z,2,r$ref_sd,"*"),2,r$ref_mean,"+")
  colnames(x) <- r$markers
  x[,"CRP"] <- expm1(x[,"CRP"])
  clipped <- colSums(x<0)
  x[x<0] <- 0
  ans <- data.frame(synthetic_id=sprintf("SYN%05d",seq_len(n)),age=age,sex=sex,x,
                    check.names=FALSE,stringsAsFactors=FALSE)
  attr(ans,"synthetic_provenance") <- list(synthetic=TRUE,seed=seed,
    calibration="Bundled NHANES aggregate reference; no individual-level records",
    reference_age_rule=r$age_rule,reference_n=r$n_ref,
    age_design="20% uniform age 20-29; 80% uniform age 30-85 (artificial)",
    age_shift="Chosen illustrative linear shift, not fitted to observed individuals",
    negative_values_clipped=clipped,units=r$units)
  ans
}
