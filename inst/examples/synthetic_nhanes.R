# Run from any directory with DISCO4HD >= 0.3.1 installed.
# Output location is explicit; this script never reads participant data.
library(DISCO4HD)
out <- Sys.getenv('DISCO_EXAMPLE_OUTPUT',unset='synthetic_demo_output')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
d <- simulate_nhanes_demo(400,seed=20261001)
scores <- cal_disco10(d,reference='NHANES')
write.csv(d,file.path(out,'synthetic_nhanes_demo.csv'),row.names=FALSE)
write.csv(scores,file.path(out,'synthetic_nhanes_scores.csv'),row.names=FALSE)
writeLines(capture.output(str(attr(d,'synthetic_provenance'))),file.path(out,'synthetic_provenance.txt'))
# Generic, sample-trained DISCO is a separate workflow: explicitly log CRP.
x <- d; x$CRP <- log1p(x$CRP)
markers <- disco10_reference('NHANES')$markers
trained <- cal_disco(x,var=markers,ref=x[x$age<30,],cpp=TRUE,parallel=FALSE)
stopifnot(all(is.finite(trained$DISCO)),all(is.finite(scores$DISCO)),all(is.finite(scores$DM)))
png(file.path(out,'synthetic_nhanes_demo.png'),width=1500,height=500,res=140)
op <- par(mfrow=c(1,3),mar=c(4.5,4.8,3.8,1),las=1,cex=.94)
plot(scores$age,scores$DISCO,pch=16,col=adjustcolor('#E64B35',.45),
     xlab='Simulated age (years)',ylab='DISCO',main='Synthetic example: DISCO')
plot(scores$age,scores$DM,pch=16,col=adjustcolor('#4DBBD5',.5),
     xlab='Simulated age (years)',ylab='DM',main='Synthetic example: DM')
plot(scores$DM,scores$DISCO,pch=16,col=adjustcolor('grey35',.45),
     xlab='DM',ylab='DISCO',main='Synthetic score comparison')
par(op);dev.off()
cat('PASS: synthetic-only demonstration written to',normalizePath(out),'\n')
