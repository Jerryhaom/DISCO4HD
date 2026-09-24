person10 <- function() data.frame(CRP=c(1,8,0),GLU=c(5,7,3.9),CHOL=c(4.5,6,3.6),HDL=c(1.3,.8,1.6),CREA=c(80,130,60),UA=c(300,480,200),WBC=c(6,12,4),UREA=c(5,9,2.5),RBC=c(4.8,3.4,4),ALB=c(42,30,50))

test_that('UKB scores reproduce independent web fixtures and DM is unlogged', {
 d <- person10(); r <- cal_disco10(d)
 expect_equal(r$DISCO,c(-2.118126394406057,4.921159259470046,1.4551748543547456),tolerance=1e-9)
 expect_equal(r$DM,c(2.152268597777491,10.030433484658058,3.9506639677639126),tolerance=1e-9)
 expect_equal(r[names(d)],d)
 expect_equal(cal_disco10(d[1,,drop=FALSE]),r[1,,drop=FALSE])
 expect_equal(cal_disco10(d[rev(names(d))])$DISCO,r$DISCO)
 expect_equal(cal_disco10(d,disco10_reference('ukb'))$DISCO,r$DISCO)
})

test_that('all bundled references agree with literal reference augmentation', {
 refs <- disco10_references()
 expect_equal(refs$marker_count,c(10L,10L,8L,10L,9L))
 expect_equal(refs$n_ref,c(5282,6616,2343,301,546))
 for (cohort in refs$cohort) {
  r <- disco10_reference(cohort); p <- length(r$markers); n <- r$n_ref
  # Construct a centered reference having exactly the stored sufficient statistics.
  basis <- cbind(1,outer(seq_len(n),seq_len(p),function(i,j) cos(pi*(i-.5)*j/n)))
  q <- qr.Q(qr(basis))[, -1,drop=FALSE]
  xr <- sweep(q %*% chol(r$ss),2,r$mu,'+')
  d <- person10()[r$markers]; z <- as.matrix(d); z[,'CRP'] <- log1p(z[,'CRP'])
  z <- sweep(sweep(z,2,r$ref_mean,'-'),2,r$ref_sd,'/')
  expected <- vapply(seq_len(nrow(z)),function(i) log(n^2*sum(r$weight*(cor(xr)-cor(rbind(xr,z[i,])))^2)),numeric(1))
  expect_equal(cal_disco10(d,cohort)$DISCO,expected,tolerance=1e-8)
  expect_equal(sum(r$weight),1,tolerance=1e-12)
  expect_equal(diag(r$weight),rep(0,p),ignore_attr=TRUE)
  expect_equal(unname(r$S),unname(r$ss/(n-1)+diag(1e-8,p)),tolerance=1e-12)
 }
})

test_that('comparison cohort has separate panel-specific DISCO and calibration', {
 d <- person10()
 ch <- cal_disco10(d,comparison='CHARLS'); ru <- cal_disco10(d,comparison='RuLAS')
 expect_equal(ch$comparison_DISCO,c(-4.7177596820757026,3.2975198908337378,.85220305782945438),tolerance=1e-9)
 expect_equal(ru$comparison_DISCO,c(-2.3242376471311776,4.054018365019223,.30870613426190235),tolerance=1e-9)
 expect_equal(ch$DISCO,cal_disco10(d)$DISCO)
 expect_equal(ch$DISCO_z,(ch$comparison_DISCO+.2586)/1.6108)
 expect_equal(ch$comparison_marker_count,rep(8L,3))
 expect_equal(ch$percentile_band[1:2],c('<5','>95'))
 expect_true(all(is.na(ch$percentile[1:2])))
 uk <- cal_disco10(d,comparison='UKB')
 expect_true(uk$percentile[1]>10 && uk$percentile[1]<25)
 expect_error(cal_disco10(d,reference='NHANES',comparison='NHANES'),'unmodified full UKB')
 expect_error(cal_disco10(d,panel=c('CRP','GLU'),comparison='UKB'),'unmodified full UKB')
})

test_that('subset scoring preserves the original normalization and registry', {
 r <- disco10_reference('UKB'); p <- disco10_reference('UKB',panel=c('CRP','GLU','HDL'))
 expect_equal(p$weight,r$weight[c('CRP','GLU','HDL'),c('CRP','GLU','HDL')])
 expect_lt(sum(p$weight),1)
 expect_equal(disco10_reference('UKB'),r)
 expect_equal(cal_disco10(person10(),p)$DISCO,cal_disco10(person10(),panel=p$markers)$DISCO)
})

test_that('invalid inputs cannot silently change the panel or records', {
 d <- person10()
 expect_error(cal_disco10(d[-1]),'Missing required')
 expect_error(cal_disco10(d[FALSE,]),'rows')
 for (bad in c(NA,NaN,Inf,-1)) {b <- d; b$CRP[1] <- bad; expect_error(cal_disco10(b),'finite|nonnegative')}
 b <- d; b$GLU <- as.character(b$GLU); expect_error(cal_disco10(b),'numeric')
 b <- d; b$DISCO <- 1; expect_error(cal_disco10(b),'output columns')
 expect_error(disco10_reference('invalid'),'cohort must')
 expect_error(disco10_reference(c('UKB','NHANES')),'cohort must')
 expect_error(disco10_reference('CHARLS',panel=c('RBC','ALB')),'available biomarkers')
 expect_error(disco10_reference(panel=c('CRP','CRP')),'distinct')
 expect_error(cal_disco10(d,comparison='invalid'),'cohort must')
})
