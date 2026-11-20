import json
import os
import random as rand
import numpy as np

from simulate.helper_funcs import *
from simulate.rules import *
from simulate.simulate import *
from simulate.working_params import *
import simulate.data as site_data

OUTPUT_DIR = os.environ.get('SIM_OUTPUT_DIR', '.')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'results_and_data')

site_name = "Batwing"

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

def load_best_params(site):
    json_path = os.path.join(
        FIGURES_DIR,
        'all_sites_PSO_parameter_data/data',
        site,
        f'{site}_results.json'
    )
    with open(json_path, 'r') as f:
        payload = json.load(f)

    best = payload["best_params"]

    # Return only the model parameters expected by simulate() (excludes fitting params like disp_r)
    param_keys = [
        "inf_alpha", "inf_beta", "delta", "T_inf", "T_TBD", "T_AD",
        "T_seasonal", "win_length", "win_start", "lambda_win", "lambda_sum",
        "res_gain", "res_max", "k_imm", "theta_imm"
    ]
    return {k: best[k] for k in param_keys}


def load_site_observations(site):
    # Dynamically call the site function in simulate/data.py (e.g., site_data.River())
    site_fn = getattr(site_data, site)
    raw_data = site_fn()

    start_year = raw_data[0]["year"]
    start_day = raw_data[0]["day"]

    obs_times = []
    obs_Hi = []

    for entry in raw_data:
        # Convert (year, day) into relative simulation days starting from t = 0
        t_val = (entry["year"] - start_year) * 365 + (entry["day"] - start_day)
        obs_times.append(t_val)
        obs_Hi.append(entry["Tri_Hi"] + entry["Misc_Hi"])

    sample = [obs_times, obs_Hi]
    initial_row = raw_data[0]
    return sample, initial_row

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

def main():
    parameters = load_best_params(site_name)
    sample, initial_row = load_site_observations(site_name)

    # -------------------------
    # set up initial population
    # -------------------------
    tricolor_num = initial_row["Tri_Hi"]
    tricolor_cluster_sizeMIN = 1
    tricolor_cluster_sizeMAX = 2

    bigbrown_num = initial_row["Misc_Hi"]
    bigbrown_cluster_sizeMIN = 1
    bigbrown_cluster_sizeMAX = 9

    Hi_list = [
        [tricolor_num, tricolor_cluster_sizeMIN, tricolor_cluster_sizeMAX],
        [bigbrown_num, bigbrown_cluster_sizeMIN, bigbrown_cluster_sizeMAX]
    ]

    num_infected = initial_row.get("In", 0)

    # ----------
    # initialize
    # ----------
    final_obs_day = sample[0][-1]
    time = final_obs_day + 365

    history = simulate(make_initial_state(Hi_list, num_infected), steps=time, parameters=parameters)

    print(history.get("SC", 0) / num_infected) if num_infected > 0 else print("none infected")

    plot_wMAPE(
        history=history,
        win_length=parameters["win_length"],
        win_start=parameters["win_start"],
        T_seasonal=parameters["T_seasonal"],
        sample=sample,
        xlim_max=time - 1
    )
    plot_residual_error(
        history=history,
        win_length=parameters["win_length"],
        win_start=parameters["win_start"],
        T_seasonal=parameters["T_seasonal"],
        sample=sample,
        xlim_max=time - 1
    )

if __name__ == "__main__":
    main()