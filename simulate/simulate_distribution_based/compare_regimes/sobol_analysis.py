from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import random as rand
import numpy as np

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.simulate_distribution_based.simulate import *

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

num_infected = 0 # DO NOT CHANGE
for i in range(len(Hi_list)):
    num_infected += int(Hi_list[i][0]*fraction_infected) # DO NOT CHANGE

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
T_TBD = 4.1                                 # CONFIDENT # length of torpor bout in days, 
                                            # considered in [3.9, 4.3] for tricolored bats
T_AD = 88.5/1440                            # CONFIDENT # length of arousal bout in days, 
                                            # considered in [1.74166, 5.63333] for tricolored bats
T_seasonal = 40                             # CONFIDENT # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
win_length = 95                             # CONFIDENT # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal
win_start = 297                             # CONFIDENT # approximate day in calendar year that Te : 1 -> 0

# BAT IN/OUT FLUX
lambda_win = 0                              # CONFIDENT # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.00013942579094               # CONFIDENT # population growth value during summer,
                                            # considered in [0.01, 0.1] 

# -----------------
# types of immunity
# -----------------

res_max = 0.2                               # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
k_imm, theta_imm = 1, 1                     # number of days spent in recovery before re-infection is possible
                                            # corresp. w/ Gamma(k_imm, theta_imm)
res_gain = 0.02                             # resistance AFTER recovery

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
        "num_vars": 6,
        "names": ["inf_alpha", "inf_beta", "delta",
                "T_inf", "T_TBD", "win_length"],
        "bounds": [
            [1, 5],         # inf_alpha
            [2, 10],        # inf_beta
            [0.005, 0.05],  # delta
            [10, 40],       # T_inf
            [3.9, 4.3],     # T_TBD
            [150, 210],     # win_length
        ],
    }

    # Generate Saltelli samples: (N * (2*num_vars + 2) total runs)
    # N=128 -> 128 * 18 = 2304 runs; N=64 -> 1152 runs (fast for testing)
    N = 64
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
    plt.savefig("figures/sobol_analysis_plot.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    