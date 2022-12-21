"""
Collection of functions used in the empirical analysis.
Imported from run_bssscm.py
"""
import os
import shutil
import sys

import cmdstanpy
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

plt.rcParams["font.size"] = 12


def prep_data(
    path_input_df,
    dict_covariates,
    year_obs,
    w,
    dim_eff,
    year_start,
    year_prj,
    year_end,
):
    """Prepare input data for stan model"""
    # Load input data frame
    df = pd.read_csv(path_input_df)
    col_cov = dict_covariates.keys()
    col_df_rate = [f"df_rate_{year}" for year in np.arange(year_start, year_end + 1)]

    y = df.loc[0, col_df_rate]  # Deforestaion rate in PA
    Z = df.loc[1:, col_df_rate]  # Deforestation rate in RRDs
    X = df.loc[:, col_cov]  # Covariates for PA and RRDs
    J = Z.shape[0]

    K = len(col_cov)

    # Normalize covariates
    X_c = X - X.mean(axis=0)
    Cov_Xc_inv = np.linalg.inv(1 / J * X_c.T.dot(X_c))
    L = np.linalg.cholesky(Cov_Xc_inv)
    X_n = X_c.dot(L)

    # Periods
    T_pre = year_prj - year_start
    T_prj = year_obs - year_prj + 1
    T_forth = year_end - year_obs
    T = T_pre + T_prj + T_forth

    # Hyper parameters
    scale_global = dim_eff / ((J - dim_eff) / T_pre**0.5)
    nu_global = 1
    nu_local = 1
    slab_scale = 1
    slab_df = 1

    data = {
        # stan input
        "y": y,
        "Z": Z,
        "K": K,
        "X": X_n,
        "J": J,
        "T_pre": T_pre,
        "T_prj": T_prj,
        "T_forth": T_forth,
        "T": T,
        "w": w,
        "scale_cauchy": 1,
        "scale_global": scale_global,
        "nu_global": nu_global,
        "nu_local": nu_local,
        "slab_scale": slab_scale,
        "slab_df": slab_df,
        # used in summary
        "df": df,
        "year_start": year_start,
        "year_end": year_end,
        "year_prj": year_prj,
        "year_obs": year_obs,
    }

    return data


def run_model(
    data, file_model, out_dir="mcmc", n_mcmc=1000, n_warmup=1000, re_run=True, seed=123
):
    """
    Run the stan model
    """

    if re_run:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=False)
        model = CmdStanModel(stan_file=file_model)
        fit = model.sample(
            chains=1, data=data, iter_sampling=n_mcmc, iter_warmup=n_warmup, seed=seed
        )
        fit.save_csvfiles(out_dir)
    else:
        fit = cmdstanpy.from_csv(out_dir)

    return fit


def plot_convergence(data, beta_show, n_plot, dir_out, plot_shape=(3, 2)):
    """Plot MCMC convergence"""
    plt.figure(figsize=(6, 2.5 * plot_shape[0]))
    for i in range(n_plot):
        plt.subplot(plot_shape[0], plot_shape[1], i + 1)
        plt.plot(beta_show[:, i], color="black", linewidth=0.2)
        plt.title(f"beta_{i+1}")
    # plt.suptitle("MCMC convergence")
    plt.tight_layout()
    plt.savefig(f'{dir_out}/mcmc_convergence_w_{data["w"]}_at_{data["year_obs"]}.png')
    plt.close()


def plot_baseline(data, fit_summary, dir_out, scale):
    """Plot estimation results"""
    # Get BSSSCM baseline
    params = list(fit_summary.index)
    ids_y_bsl = [b for b in params if "y_" in b]
    fit_y_bsl = fit_summary.loc[ids_y_bsl]

    # Plot
    years = np.arange(data["year_start"], data["year_end"] + 1)
    plt.plot(
        years,
        scale * data["y"].values,
        label="Observation",
        color="black",
        linewidth=0.5,
    )
    plt.plot(
        years,
        scale * fit_y_bsl["Mean"].values,
        label="Baseline by BSS-SCM: Posterior mean",
        linestyle="-",
    )
    plt.fill_between(
        years,
        scale * fit_y_bsl["5%"].values,
        scale * fit_y_bsl["95%"].values,
        label="Baseline by BSS-SCM: 90% interval",
        alpha=0.3,
    )
    # plt.plot(years, bsl_scm.values, label="Baseline: SCM", linestyle=':')
    plt.axvline(data["year_prj"], linestyle=":", color="black")
    plt.text(
        data["year_prj"] - 2.2, -0.07, "Project start year", backgroundcolor="white"
    )
    plt.axvspan(
        data["year_obs"] + 1,
        data["year_end"],
        color="gray",
        alpha=0.2,
        label="Prediction periods",
    )
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel("Annual deforestation rate [%]")
    plt.ylim(-0.1, 0.4)
    plt.tight_layout()
    plt.savefig(f'{dir_out}/baseline_w_{data["w"]}_at_{data["year_obs"]}.png')
    plt.close()


def summary_results(data, fit, dict_covariates, dir_out, is_percent, do_sort):
    """Summarize estimation results"""
    os.makedirs(dir_out, exist_ok=True)
    scale = 1 if is_percent else 100

    # Get mcmc array for beta
    beta_arr = fit.stan_variables()["beta"]
    fit_summary = fit.summary()
    fit_summary.to_csv(f'{dir_out}/fit_summary_w_{data["w"]}.csv')

    # sort
    if do_sort:
        ind_rrd_sorted = np.argsort(np.mean(beta_arr, axis=0))[::-1]
        beta_show = beta_arr[:, ind_rrd_sorted]
    else:
        beta_show = beta_arr

    # MCMC convergence
    plot_convergence(data, beta_show, n_plot=6, dir_out=dir_out)

    # Plot Baseline
    plot_baseline(data, fit_summary, dir_out, scale)

    # Get covariate matching table
    get_covaraite_matching_table(data, beta_arr, dict_covariates, dir_out, scale)

    # Plot df rates
    out_path = f"{dir_out}/deforestation_rates.png"
    plot_deforestation_rates(
        data["df"], data["year_start"], data["year_end"], out_path, scale
    )

    # Make kepler result
    make_kepler_for_result(
        data, beta_show, ind_rrd_sorted, dir_out, n_show=5, scale=scale
    )


def plot_deforestation_rates(
    df, year_start, year_end, out_path, scale, lw_rrd=0.3, ymax=2
):
    """"Plot deforestation rates"""
    years = np.arange(year_start + 1, year_end + 1)

    cols = [f"df_rate_{year}" for year in years]
    df_show_pa = df[cols].iloc[0] * scale
    df_show_rrds = df[cols].iloc[1:] * scale
    n_rrds = len(df_show_rrds)

    for i in range(n_rrds):
        label = None if i < n_rrds - 1 else "CARs"
        plt.plot(
            years,
            df_show_rrds.iloc[i].values,
            label=label,
            color="gray",
            linewidth=lw_rrd,
        )
    plt.plot(years, df_show_pa.values, label="PA", color="red", linewidth=2)

    plt.legend(loc="upper left")
    plt.xlabel("Year")
    plt.ylabel("Annual deforestation rate [%]")
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def get_covaraite_matching_table(data, beta_arr, dict_covariates, dir_out, scale):
    """Get covariate matching table as latex format"""
    df_covariate = data["df"][dict_covariates.keys()]

    covariate_pa = df_covariate.iloc[0]
    covariate_rrd = df_covariate.iloc[1:]

    covariate_rrd_mean = covariate_rrd.mean(axis=0)

    beta_mean = beta_arr.mean(axis=0)
    # beta_mean = pd.read_csv("data/weight_scm_mapbiomas.csv", header=0)["w.weight"].values
    covariate_rrd_weighted = covariate_rrd.T.dot(beta_mean)

    out_tab = pd.DataFrame(
        {
            "PA": covariate_pa,
            "RRD: Synthetic": covariate_rrd_weighted,
            "RRD: Average": covariate_rrd_mean,
        }
    ).rename(index=dict_covariates)

    # Convert to percent scale
    ind_rates = [s for s in out_tab.index if "rate" in s.lower()]
    out_tab.loc[ind_rates] = scale * out_tab.loc[ind_rates]
    # out_tab = out_tab.round(3)

    out_tab.style.format(
        {("Numeric", "Integers"): "\${}", ("Numeric", "Floats"): "{:.1f}"}
    )
    out_tab.style.to_latex(
        f'{dir_out}/table_covariate_matching_w_{data["w"]}.tex',
        hrules=True,
        caption="Comparison of covariates (TODO: formatting)",
        label="tab:covariate",
        position="hbtp",
    )


def make_kepler_for_result(data, beta_show, ind_rrd_sorted, dir_out, n_show, scale):
    """Preprocess input for kepler map"""
    # df = pd.read_csv(f'{dir_out}/fit_summary.csv')
    # df.index = df['Unnamed: 0']
    # ind_beta = [idx for idx in list(df.index) if 'beta[' in idx]
    # df_sorted = df.loc[ind_beta].sort_values(by='Mean',ascending=False)
    # df_sorted = df_sorted.iloc[:n_show]
    # ind_rrd_important = [int(idx.replace('beta[','').replace(']','')) for idx in list(df_sorted.index)]

    n_rrd = beta_show.shape[1]
    plt.bar(np.arange(1, n_rrd + 1), beta_show.mean(axis=0))
    plt.xlabel("Index (j) of weight")
    plt.ylabel("weight")
    plt.tight_layout()
    plt.savefig(f'{dir_out}/barplot_weight_w_{data["w"]}.png')
    plt.close()

    ind_rrd_important = list(ind_rrd_sorted[:n_show])
    df = data["df"]
    df_important = df.loc[[0] + ind_rrd_important]
    out_path = f"{dir_out}/deforestation_rates_important.pdf"
    plot_deforestation_rates(
        df_important,
        data["year_start"],
        data["year_end"],
        out_path,
        scale,
        lw_rrd=1,
        ymax=1,
    )

    num_poly_important = list(df_important["NUM_POLY"])
    geojson = "data/polygons/imovels_rrd.geojson"
    gdf = gpd.read_file(geojson)
    gdf_important = gdf.query("NUM_POLY in @num_poly_important").copy()
    gdf_important["weight"] = beta_show[:, :n_show].mean(axis=0)
    gdf_important.to_file(f"{dir_out}/rrd_important.geojson", driver="GeoJSON")


def compare_baseline(
    data,
    fit_summary_b,
    fit_summary_nb,
    dir_out,
    scale,
    yrange=[-0.1, 0.4],
    text_y_pos=-0.07,
):
    """Plot baseline comparison between with or without balancing"""
    os.makedirs(dir_out, exist_ok=True)

    # Get BSSSCM baseline without balancing
    params = list(fit_summary_nb.index)
    ids_y_bsl = [b for b in params if "y_" in b]
    fit_y_bsl_nb = fit_summary_nb.loc[ids_y_bsl]

    # Get BSSSCM baseline with balancing
    params = list(fit_summary_b.index)
    ids_y_bsl = [b for b in params if "y_" in b]
    fit_y_bsl_b = fit_summary_b.loc[ids_y_bsl]

    # Plot
    years = np.arange(data["year_start"], data["year_end"] + 1)
    plt.plot(
        years,
        scale * data["y"].values,
        label="Observation",
        color="black",
        linewidth=0.5,
    )
    plt.plot(
        years,
        scale * fit_y_bsl_b["Mean"].values,
        label="Baseline by BSS-SCM: Posterior mean",
        linestyle="-",
    )
    plt.plot(
        years,
        scale * fit_y_bsl_nb["Mean"].values,
        label="Baseline by BSS-SCM without CB: Posterior mean",
        linestyle=":",
        color="black",
    )
    plt.fill_between(
        years,
        scale * fit_y_bsl_b["5%"].values,
        scale * fit_y_bsl_b["95%"].values,
        label="Baseline by BSS-SCM: 90% interval",
        alpha=0.3,
    )
    # plt.plot(years, bsl_scm.values, label="Baseline: SCM", linestyle=':')
    plt.axvline(data["year_prj"], linestyle=":", color="black")
    plt.text(
        data["year_prj"] - 2.2,
        text_y_pos,
        "Project start year",
        backgroundcolor="white",
    )
    plt.axvspan(
        data["year_obs"] + 1,
        data["year_end"],
        color="gray",
        alpha=0.2,
        label="Prediction periods",
    )
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel("Year")
    plt.ylabel("Annual deforestation rate [%]")
    plt.ylim(*yrange)
    # plt.xlim(*xrange)
    plt.tight_layout()
    plt.savefig(f'{dir_out}/baseline_comp_w_{data["w"]}.pdf')
    plt.close()


def compare_baseline2(data, fit_summary_b, year_obs, dir_out, scale):
    """Compare baseline between different models: BSSSCM, SCM, and CaualImpact"""
    # # Get BSSSCM baseline without balancing
    # params = list(fit_summary_nb.index)
    # ids_y_bsl = [b for b in params if "y_" in b]
    # fit_y_bsl_nb = fit_summary_nb.loc[ids_y_bsl]

    # Get BSSSCM baseline with balancing
    params = list(fit_summary_b.index)
    ids_y_bsl = [b for b in params if "y_" in b]
    fit_y_bsl_b = fit_summary_b.loc[ids_y_bsl]

    # Get SCM baseline
    csv_res_scm = "model_output/scm/weight.csv"
    weight_scm = pd.read_csv(csv_res_scm, header=0)["w.weight"].values
    Z = data["Z"].iloc[:, :-1]
    bsl_scm = Z.T.dot(weight_scm)

    # Get CausalImpact baseline
    csv_res_ci = "model_output/causalimpact/coeff.csv"
    coeff_ci = pd.read_csv(csv_res_ci)
    coeff_ci.index = coeff_ci.iloc[:, 0]
    idx_Z = [f"Z{i}" for i in range(1, Z.shape[0] + 1)]
    weight_ci = coeff_ci.loc[idx_Z, "mean"].values
    bsl_ci = Z.T.dot(weight_ci)

    # # Get PDD baseline
    # pdd_rate = pd.read_csv('data/empirical/df_rate_merged.csv')

    # Plot
    years = np.arange(data["year_start"], data["year_end"])
    # plt.plot(
    #     years,
    #     scale * data["y"].values,
    #     label="Observation",
    #     color="black",
    #     linewidth=0.5,
    # )
    plt.plot(
        years,
        scale * fit_y_bsl_b["Mean"].values[:-1],
        label="Baseline by BSS-SCM: Posterior mean",
        linestyle="-",
    )
    # plt.plot(
    #     years,
    #     scale * fit_y_bsl_nb["Mean"].values,
    #     label="Baseline by BSS-SCM without CB: Posterior mean",
    #     linestyle=":",
    #     color="black",
    # )
    plt.plot(
        years,
        scale * bsl_scm,
        label="Baseline by SCM",
        linestyle="--",
        color="green",
    )
    plt.plot(
        years,
        scale * bsl_ci,
        label="Baseline by CausalImpact: Posterior mean",
        linestyle="-.",
    )
    # plt.plot(
    #     pdd_rate['year'],
    #     pdd_rate['rate_pa_pdd'],
    #     label='Baseline by PDD',
    #     linestyle=':',
    #     color='red'
    # )
    # plt.plot(years, bsl_scm.values, label="Baseline: SCM", linestyle=':')
    plt.axvline(data["year_prj"], linestyle=":", color="black")
    # plt.text(
    #     data["year_prj"] - 2.2, -0.07, "Project start year", backgroundcolor="white"
    # )
    # plt.axvspan(
    #     data["year_obs"] + 1,
    #     data["year_end"],
    #     color="gray",
    #     alpha=0.2,
    #     label="Prediction periods",
    # )
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel("Year")
    plt.ylabel("Annual deforestation rate [%]")
    # plt.ylim(-0.05, 0.45)
    plt.tight_layout()
    plt.savefig(f"{dir_out}/obs_by_{year_obs}/baseline_comp_scm_ci.pdf")
    plt.close()
