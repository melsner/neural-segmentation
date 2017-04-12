library(ggplot2)
library(reshape2)
dat <- read.table("capacity-stats.tsv", head=F)
colnames(dat) <- c("Hc", "Loss (real)", "Real", "Loss (pseud)", "Pseud")
dat2 <- melt(dat, id="Hc")
dat2Vars <- subset(dat2, variable %in% c("Real", "Pseud"))
dat2Vars$Acc <- dat2Vars$variable
plot <- ggplot(dat2Vars,
               aes(x=Hc, y=value, color=Acc)) + geom_line() +
    scale_color_brewer(palette="Dark2") + scale_x_continuous(breaks=dat2$Hc)
