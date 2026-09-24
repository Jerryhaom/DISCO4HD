#!/usr/bin/env Rscript
# Rebuild aggregate-only reference parameters from approved controlled inputs.
# Run at the package root. Nothing is written to DISCO_DATA_ROOT.
root <- Sys.getenv('DISCO_DATA_ROOT')
if (!nzchar(root) || !dir.exists(root)) stop('Set DISCO_DATA_ROOT to the approved research input directory')
source_paths <- character()
read_rds <- function(p) {source_paths <<- unique(c(source_paths,p)); readRDS(file.path(root,p))}
read_csv <- function(p) {source_paths <<- unique(c(source_paths,p)); read.csv(file.path(root,p))}
num <- function(x) as.numeric(as.character(x))
join <- function(ids, keys) {
 if(anyDuplicated(keys)) stop('Duplicate source join keys')
 i <- match(as.character(ids),as.character(keys))
 if(anyNA(i)) stop('Unmatched source join keys')
 i
}
# Full-precision snapshot from DISCO-Calculator/index.html at commit 39cce98.
# This preserves web results, including UKB scaling and the DM diagonal ridge.
web <- readRDS('data-raw/web-reference-snapshot.rds')
std <- web$reference; markers <- std$markers
cohorts <- list()
b <- read_rds('Result1/Clean.UKB.Clinical.rds')
a <- read_rds('UKB_Data/extended_covariates_clinical.rds')
b <- b[join(a$ID,b$sampleID),]
cohorts$UKB <- list(age=num(a$age), values=data.frame(CRP=b$CRP,GLU=b$GLU,CHOL=b$TC,HDL=b$HDL.C,CREA=b$CREA,UA=b$GML,WBC=b$wbc,UREA=b$U,RBC=b$RBC,ALB=b$ALB))
source_paths <- c(source_paths,'Package/DISCO4HD/data/NHANES4.rda')
e <- new.env(); load(file.path(root,tail(source_paths,1)),envir=e); a <- e$NHANES4
cohorts$NHANES <- list(age=num(a$age), values=data.frame(CRP=(exp(a$lncrp)-1)*10,GLU=a$glucose_mmol,CHOL=a$totchol*.02586,HDL=a$hdl*.02586,CREA=a$creat*88.4,UA=a$uap*59.5,WBC=a$wbc,UREA=a$bun*.357,RBC=a$rbc,ALB=a$albumin*10))
a <- read_rds('Result4/CHARLS.DISCO.rds')
source_paths <- c(source_paths,'Result4/CHARLS/Blood_Biomarkers.txt')
b <- read.delim(file.path(root,tail(source_paths,1))); b <- b[join(a$id,b$ID),]
cohorts$CHARLS <- list(age=num(a$age),values=data.frame(CRP=b$CRP,GLU=b$Glu/18.0182,CHOL=b$CHOL*.02586,HDL=b$HDL*.02586,CREA=b$Crea*88.4,UA=b$UA*59.5,WBC=b$WBC,UREA=b$Bun*.357))
a <- read_rds('Result4/CLHLS.DISCO.rds')
b <- read_csv('Result4/CLHLS/New/biomarker_dataset_CLHLS_2014-1.tab.csv')
lon <- read_rds('Result4/CLHLS/New/CLHLS.2014_2018.longitudinal.datasets.rds')
lon <- lon[join(b$id,lon$id),]
stopifnot(nrow(a)==nrow(b), isTRUE(all.equal(num(a$age),num(lon$trueage))),
          isTRUE(all.equal(num(a$sex),num(lon$a1))), isTRUE(all.equal(num(a$status),num(lon$dth14_18))))
cohorts$CLHLS <- list(age=num(a$age),values=data.frame(CRP=b$crphs,GLU=b$glu,CHOL=b$cho,HDL=b$hdlc,CREA=b$crea,UA=b$ua,WBC=b$wbc,UREA=b$bun,RBC=b$rbc,ALB=b$alb))
a <- read_rds('Result4/RuLAS.DISCO.rds')
b <- read_csv('Result4/Rugao/2016Rugao.data.x.csv'); b <- b[join(a$ID,b$X2016ID),]
cohorts$RuLAS <- list(age=num(a$age),values=data.frame(CRP=b$CRP2016,GLU=b$GLU2016,CHOL=b$CHOL2016,HDL=b$HDLC2016,CREA=b$CREA2016,UA=b$UA2016,WBC=b$WBC2016,RBC=b$RBC2016,ALB=b$ALB2016))
rm(a,b,lon,e)
units <- c(CRP='mg/L',GLU='mmol/L',CHOL='mmol/L',HDL='mmol/L',CREA='micromol/L',UA='micromol/L',WBC='10^9/L',UREA='mmol/L',RBC='10^12/L',ALB='g/L')
cutoffs <- c(UKB=40,NHANES=30,CHARLS=50,CLHLS=70,RuLAS=74)
refs <- list(); verification <- list()
for (nm in names(cohorts)) {
 c <- cohorts[[nm]]; panel <- names(c$values); ii <- match(panel,markers)
 x <- as.matrix(c$values)
 ok <- rowSums(!is.finite(x)|x<0)==0 & is.finite(c$age)
 x <- x[ok,,drop=FALSE]; age <- c$age[ok]
 x[,'CRP'] <- log1p(x[,'CRP'])
 x <- sweep(sweep(x,2,std$ref_mean[ii],'-'),2,std$ref_sd[ii],'/')
 young <- if (nm=='NHANES') age < cutoffs[nm] else age <= cutoffs[nm]
 xr <- x[young,,drop=FALSE]; mu <- colMeans(xr); ss <- crossprod(sweep(xr,2,mu,'-'))
 w <- abs(tcrossprod(as.vector(cor(x,age)))); diag(w) <- 0; w <- w/sum(w)
 ref <- list(markers=panel,ref_mean=std$ref_mean[ii],ref_sd=std$ref_sd[ii],mu=mu,ss=ss,
             Rref=cov2cor(ss),weight=w,S=ss/(nrow(xr)-1)+diag(1e-8,length(ii)),n_ref=nrow(xr))
 if (nm=='UKB') {
  for (field in c('ref_mean','ref_sd','mu','ss','Rref','weight','S','n_ref'))
   stopifnot(isTRUE(all.equal(unname(ref[[field]]),unname(std[[field]]),tolerance=1e-8)))
  ref <- std
 }
 ref$cohort <- nm; ref$units <- unname(units[panel])
 ref$age_rule <- paste('age',if(nm=='NHANES') '<' else '<=',unname(cutoffs[nm]))
 ref$age_range <- range(age[young]); ref$weight_sample_n <- nrow(x)
 ref$scaling <- 'UKB age <= 40 scaling after ln(CRP_mgL + 1); actual reference center stored in mu'
 ref$weight_source <- 'Absolute outer product of biomarker-age correlations in complete cohort sample; zero diagonal; sum over both off-diagonal halves = 1'
 ref$source <- if(nm=='UKB') 'DISCO-Calculator commit 39cce98; verified against source records' else 'Rebuilt aggregate parameters from cohort sources; finite nonnegative biomarkers and finite age'
 ref$schema_version <- 1L
 # Set names consistently, independent of JSON snapshot simplification.
 for (field in c('ref_mean','ref_sd','mu','units')) names(ref[[field]]) <- panel
 for (field in c('ss','Rref','weight','S')) dimnames(ref[[field]]) <- list(panel,panel)
 class(ref) <- 'disco10_reference'
 # Independent literal correlation recomputation checks the rank-one update.
 err <- vapply(unique(round(seq(1,nrow(x),length.out=7))), function(i) {
  d <- x[i,]-ref$mu
  rn <- cov2cor(ref$ss + ref$n_ref/(ref$n_ref+1)*tcrossprod(d))
  aggregate <- log(ref$n_ref^2*sum(ref$weight*(ref$Rref-rn)^2))
  direct <- log(nrow(xr)^2*sum(ref$weight*(cor(xr)-cor(rbind(xr,x[i,])))^2))
  abs(aggregate-direct)
 },numeric(1))
 stopifnot(max(err)<1e-8,all(is.finite(ref$S)),min(eigen(ref$S,symmetric=TRUE,only.values=TRUE)$values)>0)
 refs[[nm]] <- ref
 verification[[nm]] <- data.frame(cohort=nm,marker_count=length(panel),n_ref=nrow(xr),weight_sample_n=nrow(x),max_direct_error=max(err))
 cat(nm,'markers',length(panel),'reference N',nrow(xr),'weight N',nrow(x),'direct error',max(err),'\n')
}
comparison <- lapply(names(cohorts),function(nm) {
 cs <- web$comparison[[nm]]
 list(markers=names(cohorts[[nm]]$values),mean=cs$mean,sd=cs$sd,z_knots=unlist(cs$pct)[-1],percentiles=c(5,10,25,50,75,90,95))
})
names(comparison) <- names(cohorts)
# jsonlite may preserve JSON null as NA or NULL. Use the seven numeric knots explicitly.
for(nm in names(comparison)) comparison[[nm]]$z_knots <- tail(as.numeric(unlist(web$comparison[[nm]]$pct)),7)
registry <- list(schema_version=1L,version='0.3.0',references=refs,comparison=comparison,
 provenance=list(web_commit='39cce9860d0f9f320081fcd01fb4311b30f4cb73',
 sources=data.frame(path=source_paths,md5=unname(tools::md5sum(file.path(root,source_paths)))),
 reference_policy='Youngest available age bands used in prior cohort analyses; not uniformly young populations',
 comparison_policy='Frozen web tables for transported UKB-reference scores only; not own-reference distributions'))
dir.create('inst/extdata',recursive=TRUE,showWarnings=FALSE)
saveRDS(registry,'inst/extdata/disco10-references.rds',compress='xz',version=2)
write.csv(do.call(rbind,verification),'data-raw/reference-validation.csv',row.names=FALSE)
