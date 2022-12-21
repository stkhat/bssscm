"""
Script for reproducing the case study
usage:
    python3 run_bssscm.py
"""

import cmdstanpy

from utils import (compare_baseline, compare_baseline2, prep_data, run_model,
                   summary_results)

# =====================================================
# Settings
# =====================================================
rerun_mcmc = False
file_model = "stan_model/state_space_scm_trend_reg_norm.stan"
path_input_df = "data/empirical/df_long_mapbiomas.csv"
dict_covariates = {
    "DIST_TO_ROAD": "Distance to road [km]",
    "ELEVATION": "Elevation [m]",
    # "df_rate_buffer_2009": "Deforestation rate in buffer area in 2009 [\%]",
    "df_rate_buffer_2010": "Deforestation rate in buffer area in 2010 [\%]",
}
dim_eff = 4
n_mcmc = 2000
n_warmup = 1000
seed = 999


# =====================================================
# Run model for different settings
# =====================================================
for year_obs in [2010, 2015, 2019]:
    for w in [100, 0]:
        data = prep_data(
            path_input_df,
            dict_covariates,
            year_obs,
            w,
            dim_eff=dim_eff,
            year_start=1995,
            year_prj=2011,
            year_end=2020,
        )
        fit = run_model(
            data,
            file_model,
            out_dir=f"model_output/bssscm/mcmc/empirical_norm/obs_w_{w}_by_{year_obs}",
            n_mcmc=n_mcmc,
            n_warmup=n_warmup,
            re_run=rerun_mcmc,
            seed=seed,
        )
        summary_results(
            data,
            fit,
            dict_covariates,
            dir_out=f"report/empirical_norm/obs_by_{year_obs}",
            is_percent=False,
            do_sort=True,
        )

    # Compare baseline between with or without balancing
    fit_summary_b = cmdstanpy.from_csv(
        f"model_output/bssscm/mcmc/empirical_norm/obs_w_100_by_{year_obs}"
    ).summary()
    fit_summary_nb = cmdstanpy.from_csv(
        f"model_output/bssscm/mcmc/empirical_norm/obs_w_0_by_{year_obs}"
    ).summary()
    compare_baseline(
        data,
        fit_summary_b,
        fit_summary_nb,
        dir_out=f"report/empirical_norm/obs_by_{year_obs}",  # f"report/empirical_norm/obs_by_{year_obs}/range_adjusted",
        scale=100,
        # yrange=[-0.2, 3.8],
        # text_y_pos=2.3
    )

    # Compare baseline with SCM and CausalImpact at year 2019
    if year_obs == 2019:
        compare_baseline2(
            data,
            fit_summary_b,
            year_obs,
            dir_out=f"report/empirical_norm",
            scale=100,
        )
