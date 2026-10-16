test_that('synthetic NHANES example is reproducible, complete and usable', {
  d <- simulate_nhanes_demo(100,101)
  expect_identical(d,simulate_nhanes_demo(100,101))
  expect_equal(nrow(d),100L)
  expect_true(all(grepl('^SYN',d$synthetic_id)))
  expect_true(isTRUE(attr(d,'synthetic_provenance')$synthetic))
  markers <- disco10_reference('NHANES')$markers
  expect_true(all(is.finite(as.matrix(d[markers]))))
  expect_true(all(as.matrix(d[markers])>=0))
  f <- cal_disco10(d,reference='NHANES')
  expect_true(all(is.finite(f$DISCO)))
  expect_true(all(is.finite(f$DM)))
  expect_equal(f$DISCO[1],cal_disco10(d[1,,drop=FALSE],reference='NHANES')$DISCO)
  x <- d; x$CRP <- log1p(x$CRP)
  ref <- x[x$age<30,,drop=FALSE]
  r <- cal_disco(x,var=markers,ref=ref,cpp=FALSE,parallel=FALSE)
  c <- cal_disco(x,var=markers,ref=ref,cpp=TRUE,parallel=FALSE)
  expect_equal(r$DISCO,c$DISCO,tolerance=1e-9)
  bad <- x; bad$duplicate <- bad$CRP; names(bad)[ncol(bad)] <- 'CRP'
  expect_error(cal_disco(bad,var=markers,ref=ref),'unique column names')
})

test_that('synthetic generator restores the caller RNG', {
  set.seed(99); before <- .Random.seed
  invisible(simulate_nhanes_demo(30,2))
  expect_identical(.Random.seed,before)
  expect_error(simulate_nhanes_demo(2),'at least 30')
  expect_error(simulate_nhanes_demo(30,NA),'seed')
  expect_error(simulate_nhanes_demo(30,-1),'seed')
  expect_error(simulate_nhanes_demo(30.5),'integer')
})
