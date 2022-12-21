# ==============================================================================
# Generate the results of SCM
# The output will be used for model comparison in run_bssscm.py
# ==============================================================================

setwd("./")
library("Synth")
set.seed(1)

# Paths
input_file_path <- "data/empirical/df_tidy_mapbiomas.csv"
input_file_path_long <- "data/empirical/df_long_mapbiomas.csv"
output_file_path <- "model_output/scm/weight.csv"

# Data prep
df <- read.csv(input_file_path)
df$CAR <- as.character(df$NUM_POLY)
ind_controls <- unique(df$NUM_POLY)
ind_controls <- ind_controls[ind_controls != 0]

# Function: estimate the model
run <- function(year_start, year_prj, year_end,
                list_covariates) {
  dataprep.out <-
    dataprep(df,
      predictors = list_covariates,
      dependent = "df_rate",
      unit.variable = "NUM_POLY",
      time.variable = "year",
      unit.names.variable = "CAR",
      treatment.identifier = 0,
      controls.identifier = ind_controls,
      time.predictors.prior = c(year_start:(year_prj - 1)),
      time.optimize.ssr = c(year_start:(year_prj - 1)),
      time.plot = c(year_start:year_end)
    )

  # Run synth
  synth.out <- synth(dataprep.out)

  return(synth.out)
}


covariates_all <- c(
  "DIST_TO_ROAD", "ELEVATION",
  "df_rate_buffer_2003", "df_rate_buffer_2004", "df_rate_buffer_2005"
)
n_covariates <- length(covariates_all)
df_long <- read.csv(input_file_path_long)
cols_df_eval <- paste("df_rate", 2006:2010, sep = "_")
df_rate_treat <- df_long[1, cols_df_eval]
df_rate_control <- df_long[2:nrow(df_long), cols_df_eval]
n_combinations <- 2^n_covariates - 1
errors <- vector(length = n_combinations)

for (i in 1:n_combinations) {
  selected_covariates <- covariates_all[as.logical(intToBits(i)[1:n_covariates])]
  synth.res.i <- run(year_start = 1995, year_prj = 2006, year_end = 2010, selected_covariates)
  df_rate_synth <- as.vector(synth.res.i$solution.w) %*% as.matrix(df_rate_control)
  errors[i] <- sum((df_rate_synth - df_rate_treat)^2)
}
saveRDS(errors, "model_output/scm/errors.rds")
# errors <- readRDS("model_output/scm/errors.rds")

i_opt <- which.min(errors)
list_covariates_opt <- covariates_all[as.logical(intToBits(i_opt)[1:n_covariates])]
print("Selected covariates:")
print(list_covariates_opt)


# Result to be used for model comparison
year_start <- 1995
year_prj <- 2011
year_end <- 2019
list_covariates <- c(
  "DIST_TO_ROAD", "ELEVATION", "df_rate_buffer_2010"
)

synth.out <- run(year_start, year_prj, year_end, list_covariates)

# Export the resulting weight as a csv
write.csv(synth.out$solution.w, output_file_path)
