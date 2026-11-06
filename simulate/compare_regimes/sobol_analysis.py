from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol
import pandas as pd
import random as rand
import numpy as np

from simulate.helper_funcs import *
from simulate.rules import *
from simulate.simulate import *
from simulate.working_params import *

import os

OUTPUT_DIR = os.environ.get('SIM_OUTPUT_DIR', '.')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'results_and_data')

# publication-grade theme style
from simulate.style_config import *
apply_publication_style()

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

time = 365*20 # total days

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
    problem = {
        "num_vars": 8,
        # Formatted cleanly to match LaTeX parameter standards
        "names": [
            r"$\alpha_{\mathrm{inf}}$", r"$\beta_{\mathrm{inf}}$", r"$\delta$",
            r"$T_{\mathrm{inf}}$", r"$r_{\mathrm{max}}$", r"$r_{\mathrm{gain}}$", r"$k_{\mathrm{imm}}$",
            r"$\theta_{\mathrm{imm}}$"
        ],
        "bounds": [
            [1, 5],
            [2, 10],
            [0.005, 0.05],
            [10, 40],
            [0, 0.8],
            [0, 1],
            [0.1, 10],
            [0, 30],
        ],
    }

    N = 2**18
    param_values = sobol_sample.sample(problem, N, calc_second_order=False)

    Y_Pmax = np.zeros(len(param_values))
    Y_Sfinal = np.zeros(len(param_values))
    Y_Mfinal = np.zeros(len(param_values))
    Y_R0 = np.zeros(len(param_values))

    parameters = sample_params()

    for i, row in enumerate(param_values):
        for name, val in zip(problem["names"], row):
            # map back parameter dictionary keys correctly
            raw_keys = ["inf_alpha", "inf_beta", "delta", "T_inf", "res_max", "res_gain", "k_imm", "theta_imm"]
            for r_key, p_name in zip(raw_keys, problem["names"]):
                if p_name == name:
                    parameters[r_key] = max(1.0, val) if r_key == "inf_alpha" else val

        history_avg = history_avg_zeros.copy()
        for j in range(avg_over):
            history = simulate(make_initial_state(Hi_list, num_infected), time, parameters, False)
            for key in history_avg:
                history_avg[key] += np.array(history[key])

        for key in history_avg:
            history_avg[key] /= avg_over   

        history_avg["SC"] = history["SC"] 

        m = compute_metrics(history_avg, Hi_list, num_infected)
        Y_Pmax[i]   = m["P_max"]
        Y_Sfinal[i] = m["S_final"]
        Y_Mfinal[i] = m["M_final"]
        Y_R0[i]     = m["R0_empirical"]

        if i % 1000 == 0:
            print(f"Sobol run {i}/{len(param_values)}")

    Si_P = sobol.analyze(problem, Y_Pmax, calc_second_order=False, print_to_console=False)
    Si_S = sobol.analyze(problem, Y_Sfinal, calc_second_order=False, print_to_console=False)
    Si_M = sobol.analyze(problem, Y_Mfinal, calc_second_order=False, print_to_console=False)
    Si_R0 = sobol.analyze(problem, Y_R0, calc_second_order=False, print_to_console=False)

    sobol_data = {
        "Parameter": [n.replace('$', '').replace('\\', '') for n in problem["names"]],
        "R0_S1": Si_R0["S1"], "R0_S1_conf": Si_R0["S1_conf"],
        "R0_ST": Si_R0["ST"], "R0_ST_conf": Si_R0["ST_conf"],
        "Pmax_S1": Si_P["S1"], "Pmax_S1_conf": Si_P["S1_conf"],
        "Pmax_ST": Si_P["ST"], "Pmax_ST_conf": Si_P["ST_conf"],
        "Sfinal_S1": Si_S["S1"], "Sfinal_S1_conf": Si_S["S1_conf"],
        "Sfinal_ST": Si_S["ST"], "Sfinal_ST_conf": Si_S["ST_conf"],
        "Mfinal_S1": Si_M["S1"], "Mfinal_S1_conf": Si_M["S1_conf"],
        "Mfinal_ST": Si_M["ST"], "Mfinal_ST_conf": Si_M["ST_conf"],
    }
    
    df_sobol = pd.DataFrame(sobol_data)
    csv_path = os.path.join(OUTPUT_DIR, "sobol_analysis_data.csv")
    df_sobol.to_csv(csv_path, index=False)

    def plot_sobol(Si, problem, title, ax):
        names = problem["names"]
        x = np.arange(len(names))
        width = 0.38
        
        ax.bar(x - width/2, Si["S1"], width, label=r"First-order ($S_1$)",
               color=COLORS["Sobol_S1"], yerr=Si["S1_conf"], capsize=3, error_kw={"lw":0.8, "color":"#333333"})
        ax.bar(x + width/2, Si["ST"], width, label=r"Total-order ($S_T$)",
               color=COLORS["Sobol_ST"], yerr=Si["ST_conf"], capsize=3, error_kw={"lw":0.8, "color":"#333333"})
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Sobol Sensitivity Index")
        ax.set_title(title, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    plot_sobol(Si_R0, problem, r"Sensitivity: Empirical $R_0$", axes[0])
    plot_sobol(Si_P, problem, r"Sensitivity: Peak Prevalence ($P_{\max}$)", axes[1])
    plot_sobol(Si_S, problem, r"Sensitivity: Final Persistence ($S_{\text{final}}$)", axes[2])
    plot_sobol(Si_M, problem, r"Sensitivity: Final Mortality ($M_{\text{final}}$)", axes[3])
    
    plt.savefig(os.path.join(FIGURES_DIR, "sobol_analysis_plot.pdf"), bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()