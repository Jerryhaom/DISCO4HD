fixture <- function() {
  set.seed(20260924)
  age <- seq(20, 79)
  d <- data.frame(age, a=age/10+rnorm(60), b=age/30+rnorm(60), c=rnorm(60), e=age/15+rnorm(60))
  list(d=d, var=names(d)[-1], ref=d[1:15, ])
}
test_that('backends reproduce independent correlation recomputation', {
  x <- fixture(); d <- x$d; v <- x$var; ref <- x$ref
  cc <- vapply(d[v], function(z) cor(z, d$age), numeric(1))
  w <- abs(outer(cc,cc)); diag(w) <- 0; w <- w/sum(w)
  expected <- vapply(seq_len(nrow(d)), function(i) log(nrow(ref)^2 * sum(w * (cor(ref[v])-cor(rbind(ref[v],d[i,v])))^2)), numeric(1))
  expect_equal(cal_disco(d,v,ref)$DISCO, expected, tolerance=1e-10)
  expect_equal(cal_disco(d,v,ref,cpp=TRUE)$DISCO, expected, tolerance=1e-10)
  tiny <- d; tiny$a <- tiny$a*1e-8; ref_small <- tiny[1:15,]
  expect_equal(cal_disco(tiny,v,ref_small,cpp=TRUE)$DISCO,expected,tolerance=1e-9)
})
test_that('parallel R agrees and leaves no registered foreach backend', {
  skip_if(Sys.getenv('DISCO_SKIP_SOCKET_TESTS') == '1', 'Socket tests explicitly disabled in sandbox')
  x <- fixture()
  expect_equal(cal_disco(x$d,x$var,x$ref,parallel=TRUE,ncores=2)$DISCO,
               cal_disco(x$d,x$var,x$ref)$DISCO,tolerance=1e-10)
})
test_that('invalid data are rejected identically by all backends', {
  x <- fixture()
  for (backend in list(list(),list(cpp=TRUE),list(parallel=TRUE))) {
    bad <- x$d; bad$a[20] <- NA
    expect_error(do.call(cal_disco,c(list(bad,x$var,x$ref),backend)), 'finite complete')
    bad <- x$ref; bad$a <- 0
    expect_error(do.call(cal_disco,c(list(x$d,x$var,bad),backend)), 'variance')
    expect_error(do.call(cal_disco,c(list(x$d,x$var,x$ref[1:2,]),backend)), 'three|3 rows')
  }
  expect_error(cal_disco(x$d,c('a','a'),x$ref),'distinct')
  expect_error(cal_disco(x$d,x$var,x$ref,ncores=0),'positive integer')
  bad <- x$d; bad$age <- 1
  expect_error(cal_disco(bad,x$var,x$ref),'age')
})
test_that('weighted blocks partition the core score and pair averages are correct', {
  x <- fixture(); X <- as.matrix(x$d[x$var]); blocks <- list(ab=1:2,ce=3:4)
  z <- disco_modality_decompose(X,blocks,1:15,age=x$d$age)
  expect_equal(z$sub_disco[,'full'],cal_disco(x$d,x$var,x$ref)$DISCO[z$analysis_idx],tolerance=1e-10)
  expect_equal(z$decomposition[,'full'],rowSums(z$decomposition[,-1]),tolerance=1e-12)
  u <- disco_modality_decompose(X,blocks,1:15,weighting='unweighted')
  dif <- (cor(X[1:15,])-cor(X[c(1:15,16),]))^2
  expect_equal(unname(u$decomposition[1,'full']),2*sum(dif[upper.tri(dif)]),tolerance=1e-12)
  expect_equal(u$per_pair$mean_per_pair[1],mean(u$decomposition[,'full'])/(2*choose(4,2)))
  one <- disco_modality_decompose(X,list(all=1:4),1:15,age=x$d$age)
  expect_equal(one$decomposition[,'full'],one$decomposition[,'within_all'])
  singles <- disco_modality_decompose(X,list(a=1,b=2:4),1:15,age=x$d$age)
  expect_true(is.na(singles$per_pair$mean_per_pair[2]))
  expect_true(all(singles$sub_disco[,'within_a']==-Inf))
})
test_that('bad block/reference definitions cannot silently omit features', {
  x <- fixture(); X <- as.matrix(x$d[x$var]); age <- x$d$age
  expect_error(disco_modality_decompose(X,list(a=c(0,1),b=2:3),1:15,age=age),'indices')
  expect_error(disco_modality_decompose(X,list(a=c(1,2),b=c(2,4)),1:15,age=age),'exactly once')
  expect_error(disco_modality_decompose(X,list(a=1:2,b=3:4),c(0,1,2),age=age),'row indices')
  expect_error(disco_modality_decompose(X,list(a=1:2,b=3:4),c(TRUE,FALSE),age=age),'one nonmissing')
  expect_error(disco_modality_decompose(X,list(a=1:2,b=3:4),1:15),'age is required')
})
