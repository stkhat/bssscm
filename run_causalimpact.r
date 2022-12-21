# ==============================================================================
# Generate the results of CausalImpact
# The output will be used for model comparison in run_bssscm.py
# ==============================================================================

setwd("./")
library("CausalImpact")

set.seed(100)
input_file_path <- "data/empirical/df_long_mapbiomas.csv"
year_start <- 1995
year_prj <- 2011
year_end <- 2019

df <- read.csv(input_file_path)
df <- t(df)
idx_rate <- paste0("df_rate_", year_start:year_end)
data <- df[idx_rate, ]
data <- ts(data, year_start, year_end, frequency = 1)
pre.period <- c(year_start, year_prj - 1)
post.period <- c(year_prj, year_end)

# impact <- CausalImpact(data, pre.period, post.period)

post.period.response <- data[(year_prj - year_start + 1):(year_end - year_start + 1), 1]
y <- data[, 1]
y[(year_prj - year_start + 1):(year_end - year_start + 1)] <- NA
Z <- data[, 2:dim(data)[2]]

# Define the local linear trend model for Z
ss <- AddLocalLevel(list(), Z)
ss <- AddLocalLinearTrend(ss, Z)
# ss <- AddLocalLinearTrend(ss, Z)
bsts.model <- bsts(y ~ Z, ss, niter = 1000)
impact <- CausalImpact(
    bsts.model = bsts.model,
    post.period.response = post.period.response
)

plot(impact)
# plot(impact$model$bsts.model, "coefficients")

dir.create("model_output/causalimpact", showWarnings = FALSE)
path.out <- "model_output/causalimpact/series.csv"
write.csv(impact$series, path.out)

# coeff <- impact$model$bsts.model$coefficients
coeff <- summary(impact$model$bsts.model)$coefficients
write.csv(coeff, "model_output/causalimpact/coeff.csv")
