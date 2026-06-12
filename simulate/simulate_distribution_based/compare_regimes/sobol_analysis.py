from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import random as rand
import numpy as np

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.simulate_distribution_based.simulate import *

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# -------------------------
# set up initial population
# -------------------------

# first select number of bats belonging to each species
tricolor_num = 100
tricolor_cluster_sizeMIN = 1
tricolor_cluster_sizeMAX = 2

bigbrown_num = 0
bigbrown_cluster_sizeMIN = 1
bigbrown_cluster_sizeMAX = 9

# hibernating non-infected bats of each species
Hi_list = [[tricolor_num, tricolor_cluster_sizeMIN, tricolor_cluster_sizeMAX], 
           [bigbrown_num, bigbrown_cluster_sizeMIN, bigbrown_cluster_sizeMAX]] 

fraction_infected = 0.01 # in [0, 1]

# NOTICE : the remaining populations (Ot, Im) all start with 0 inhabitants
# NOTICE : resistance starts at 0 for every bat

# ---------------------------
# system-governing parameters
# ---------------------------

# INFECTION PATHWAYS
inf_alpha, inf_beta = 5, 2                  # infected variables for beta distribution
                                            # chance a hibernating bat gets infected (given that PD is on) on any given day
                                            # low: alpha = 1, beta = 10
                                            # moderate: alpha = 2, beta = 5
                                            # high: alpha = 5, beta = 2

delta = 0.05                                # P. destructans decay rate, considered in [0.005, 0.03]

# DEATH OR RECOVERY PATHWAYS
T_inf = 30                                  # approximate time in dayseach bat spends infirm before recovering or dying, 
                                            # considered in [10, 40]

# BOUT and SEASONAL HIBERNATING PATHWAYS
T_TBD = 4.1                                 # length of torpor bout in days, 
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # length of arousal bout in days, 
                                            # considered in [1.74166, 5.63333] for tricolored bats
T_seasonal = 40                             # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
T_win = 210                                 # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal

# BAT IN/OUT FLUX
lambda_win = 0                              # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.001                           # population growth value during summer,
                                            # considered in [0.01, 0.1] 

# -----------------
# types of immunity
# -----------------

# CHECK DISTRIBUTIONS USED IN biology LITERATURE (beta or gamma? exponential?)

immunity_period = 0                         # number of days spent in recovery before re-infection is possible
birth_resistance_max = 0                   # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
recover_resistance_max = 0.02               # resistance after recovery, corresp. w/ rand.normalvariate(0, X)

# ----------
# initialize
# ----------

time = 3650 # total days
init_fractions = [0.01, 0.03, 0.05, 0.10]

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

def main():

    # Define the parameter space w/ ecologically meaningful ranges
    problem = {
        "num_vars": 6,
        "names": ["inf_alpha", "inf_beta", "delta",
                "T_inf", "T_TBD", "T_win"],
        "bounds": [
            [1, 5],         # inf_alpha
            [2, 10],        # inf_beta
            [0.005, 0.05],  # delta
            [10, 40],       # T_inf
            [3.9, 4.3],     # T_TBD
            [150, 210],     # T_win
        ],
    }

    # Generate Saltelli samples: (N * (2*num_vars + 2) total runs)
    # N=128 -> 128 * 18 = 2304 runs; N=64 -> 1152 runs (fast for testing)
    N = 128
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    # Run the model for each sample row
    Y_Pmax = np.zeros(len(param_values))
    Y_Sfinal = np.zeros(len(param_values))
    Y_Mfinal = np.zeros(len(param_values))

    parameters = {
        "inf_alpha": inf_alpha,
        "inf_beta": inf_beta,
        "delta": delta,
        "T_inf": T_inf,
        "T_TBD": T_TBD,
        "T_AD": T_AD,
        "T_seasonal": T_seasonal,
        "T_win": T_win,
        "lambda_win": lambda_win,
        "lambda_sum": lambda_sum,
        "immunity_period": immunity_period,
        "birth_resistance_max": birth_resistance_max,
        "recover_resistance_max": recover_resistance_max,
    }

    for i, row in enumerate(param_values):
        for name, val in zip(problem["names"], row):
            parameters[name] = val
            if name == "inf_alpha":
                parameters[name] = max(1.0, val)    # keep α > 1

        hist = simulate(make_initial_state(Hi_list, fraction_infected, parameters["T_inf"]), time, parameters)
        m = compute_metrics(hist, Hi_list)
        Y_Pmax[i]   = m["P_max"]
        Y_Sfinal[i] = m["S_final"]
        Y_Mfinal[i] = m["M_final"]
        if i % 100 == 0:
            print(f"Sobol run {i}/{len(param_values)}")

    # analyze
    Si_P = sobol.analyze(problem, Y_Pmax,   calc_second_order=False, print_to_console=False)
    Si_S = sobol.analyze(problem, Y_Sfinal, calc_second_order=False, print_to_console=False)
    Si_M = sobol.analyze(problem, Y_Mfinal, calc_second_order=False, print_to_console=False)

    # Plot: grouped bar chart (S1 and ST side by side per parameter)
    def plot_sobol(Si, problem, title, ax):
        names  = problem["names"]
        x      = np.arange(len(names))
        width  = 0.35
        ax.bar(x - width/2, Si["S1"], width, label="S1 (first-order)",
            color="#4393c3", yerr=Si["S1_conf"], capsize=3, error_kw={"lw":0.8})
        ax.bar(x + width/2, Si["ST"], width, label="ST (total)",
            color="#d6604d", yerr=Si["ST_conf"], capsize=3, error_kw={"lw":0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Sobol Index")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_sobol(Si_P, problem, "Sensitivity: Peak Prevalence (P_max)",    axes[0])
    plot_sobol(Si_S, problem, "Sensitivity: Final Persistence (S_final)", axes[1])
    plot_sobol(Si_M, problem, "Sensitivity: Final Mortality (M_final)",   axes[2])
    fig.suptitle("Sobol' Global Sensitivity Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig("figures/sobol_indices.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    