from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import random as rand
import numpy as np

from simulate.simulate_CURRENT.helper_funcs import *
from simulate.simulate_CURRENT.rules import *
from simulate.simulate_CURRENT.simulate import *
from simulate.simulate_CURRENT.working_params import *

import os

OUTPUT_DIR = os.environ.get('results_and_data', '.')
FIGURES_DIR = OUTPUT_DIR #os.path.join(OUTPUT_DIR, 'figures')

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

avg_over = 10

def sample_params():                                                                                                                                     
    return {                                                                                                                                             
        "inf_alpha": inf_alpha,                                                                                               
        "inf_beta": inf_beta,                                               
        "delta": delta,                                                     
        "T_inf": T_inf,                                                     
        "T_TBD": T_TBD,                                                     
        "T_AD": T_AD,                                                       
        "T_seasonal": T_seasonal,                                                                                                               
        "win_length": win_length,                                                                                                              
        "win_start": win_start,                                                                                                              
        "lambda_win": lambda_win,                                           
        "lambda_sum": lambda_sum,                                                                                                    
        "res_gain": res_gain,                                                                                                              
        "res_max": res_max,                                                                                                    
        "k_imm": k_imm,   
        "theta_imm": theta_imm,                                                                                    
    }   

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# -------------------------
# set up initial population
# -------------------------

num_infected = 5

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
T_TBD = 4.1                                 # CONSTANT FOR TRICOLORED # length of torpor bout in days, 
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # CONSTANT FOR TRICOLORED # length of arousal bout in days, 
                                            # considered in [1.74166, 5.63333] for tricolored bats

# -----------------
# types of immunity
# -----------------

res_max = 0.2                               # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
k_imm, theta_imm = 1, 1                     # number of days spent in recovery before re-infection is possible
                                            # corresp. w/ Gamma(k_imm, theta_imm)
res_gain = 0.02                             # resistance AFTER recovery

# ---------------------------------
# latitudinally-averaged parameters
# ---------------------------------

T_seasonal, win_length, win_start, lambda_sum, lambda_win = latitude1_NorthMidwest() # or latitude2_SouthMidwest()

# ----------
# initialize
# ----------

time = 3650 # total days

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================

# initialize accumulators
history_avg_zeros = {
    "Hi": np.zeros(time),
    "Ot": np.zeros(time),
    "In": np.zeros(time),
    "Im": np.zeros(time),
    "De": np.zeros(time),
}


def main():

    # parameter space THAT GETS CHANGED
    # w/ ecologically meaningful ranges
    problem = {
        "num_vars": 8,
        "names": ["inf_alpha", "inf_beta", "delta",
                "T_inf", "res_max", "k_imm", "theta_imm", "res_gain"],
        "bounds": [
            [1, 5],         # inf_alpha
            [2, 10],        # inf_beta
            [0.005, 0.05],  # delta
            [10, 40],       # T_inf
            [0, 0.8],       # res_max
            [0.1, 10],     # k_imm
            [0, 30],     # theta_imm
            [0, 1],     # res_gain
        ],
    }

    # Generate Saltelli samples: (N * (2*num_vars + 2) total runs)
    # N=128 -> 128 * 18 = 2304 runs; N=64 -> 1152 runs (fast for testing)
    N = 512
    param_values = saltelli.sample(problem, N, calc_second_order=False)

    # Run the model for each sample row
    Y_Pmax = np.zeros(len(param_values))
    Y_Sfinal = np.zeros(len(param_values))
    Y_Mfinal = np.zeros(len(param_values))
    Y_R0 = np.zeros(len(param_values))

    parameters = sample_params()

    for i, row in enumerate(param_values):
        for name, val in zip(problem["names"], row):
            parameters[name] = val
            if name == "inf_alpha":
                parameters[name] = max(1.0, val) # keep alpha > 1

        history_avg = history_avg_zeros.copy()

        for j in range(avg_over):

            history = simulate(make_initial_state(Hi_list, num_infected), time, parameters, False)

            for key in history_avg:
                history_avg[key] += np.array(history[key])

        # divide by number of runs for avg
        for key in history_avg:
            history_avg[key] /= avg_over   

        history_avg["SC"] = history["SC"] 

        m = compute_metrics(history_avg, Hi_list, num_infected)
        Y_Pmax[i]   = m["P_max"]
        Y_Sfinal[i] = m["S_final"]
        Y_Mfinal[i] = m["M_final"]
        Y_R0[i]     = m["R0_empirical"]

        if i % 10 == 0:
            print(f"Sobol run {i}/{len(param_values)}")

    # analyze
    Si_P = sobol.analyze(problem, Y_Pmax,   calc_second_order=False, print_to_console=False)
    Si_S = sobol.analyze(problem, Y_Sfinal, calc_second_order=False, print_to_console=False)
    Si_M = sobol.analyze(problem, Y_Mfinal, calc_second_order=False, print_to_console=False)
    Si_R0 = sobol.analyze(problem, Y_R0,    calc_second_order=False, print_to_console=False)

    # save data to csv
    sobol_data = {
        "Parameter": problem["names"],
        
        # R0 indices
        "R0_S1": Si_R0["S1"],
        "R0_S1_conf": Si_R0["S1_conf"],
        "R0_ST": Si_R0["ST"],
        "R0_ST_conf": Si_R0["ST_conf"],
        
        # P_max (Peak Prevalence) indices
        "Pmax_S1": Si_P["S1"],
        "Pmax_S1_conf": Si_P["S1_conf"],
        "Pmax_ST": Si_P["ST"],
        "Pmax_ST_conf": Si_P["ST_conf"],
        
        # S_final (Final Persistence) indices
        "Sfinal_S1": Si_S["S1"],
        "Sfinal_S1_conf": Si_S["S1_conf"],
        "Sfinal_ST": Si_S["ST"],
        "Sfinal_ST_conf": Si_S["ST_conf"],
        
        # M_final (Final Mortality) indices
        "Mfinal_S1": Si_M["S1"],
        "Mfinal_S1_conf": Si_M["S1_conf"],
        "Mfinal_ST": Si_M["ST"],
        "Mfinal_ST_conf": Si_M["ST_conf"],
    }
    
    df_sobol = pd.DataFrame(sobol_data)
    csv_path = os.path.join(OUTPUT_DIR, "sobol_analysis_data.csv")
    df_sobol.to_csv(csv_path, index=False)
    print(f"\nSaved Sobol numerical data to: {csv_path}\n")

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

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten() # flattens the 2x2 array so we can index it 0-3

    # R0 drives Peak Prevalence, which drives Mortality and Persistence
    plot_sobol(Si_R0, problem, "Sensitivity: Empirical R0", axes[0])
    plot_sobol(Si_P, problem, "Sensitivity: Peak Prevalence (P_max)", axes[1])
    plot_sobol(Si_S, problem, "Sensitivity: Final Persistence (S_final)", axes[2])
    plot_sobol(Si_M, problem, "Sensitivity: Final Mortality (M_final)", axes[3])
    
    fig.suptitle("Sobol' Global Sensitivity Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,"sobol_analysis_plot.pdf"), bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    
