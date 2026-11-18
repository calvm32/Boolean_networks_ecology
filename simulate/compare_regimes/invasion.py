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

avg_over = 20

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

T_seasonal, win_length, win_start, lambda_sum, lambda_win = latitude2_SouthMidwest() # or latitude2_SouthMidwest()

# ----------
# initialize
# ----------

delta = 0.01                                # P. destructans decay rate, considered in [0.005, 0.03]

time = 365*20 # total days
init_fractions = [0.01, 0.01, 0.10, 0.25]
NAME_APPEND = "delta0.01_lat2SMW"

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
    parameters = sample_params()
    cmap = plt.colormaps["OrRd"].resampled(len(init_fractions) + 2)
    
    # Pre-run simulation data gathering across fractions
    prevalence_runs = []
    persistence_runs = []
    t = np.arange(time)

    for idx, frac in enumerate(init_fractions):
        num_infected = int(tricolor_num * frac)
        history_avg = {key: np.zeros(time, dtype=float) for key in ("Hi", "Ot", "In", "Im", "De")}

        for i in range(avg_over):
            history = simulate(make_initial_state(Hi_list, num_infected), steps=time, parameters=parameters, Print=False)
            for key in history_avg:
                history_avg[key] += np.array(history[key])

        for key in history_avg:
            history_avg[key] /= avg_over   
        
        m = compute_metrics(history_avg, Hi_list, num_infected)
        prevalence_runs.append((frac, m["_P"]))
        persistence_runs.append((frac, m["_S"]))

    # 1. Disease Prevalence Figure
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.2))
    for idx, (frac, p_data) in enumerate(prevalence_runs):
        color = cmap(idx + 2)
        ax1.plot(t, p_data, color=color, linewidth=1.6, label=f"{int(frac*100)}% Initial Infection")
    ax1.set_title(r"Disease Prevalence $P(t)$", fontweight='bold')
    ax1.set_xlabel("Time Step (Days)")
    ax1.set_ylabel("Prevalence Ratio")
    ax1.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.5)
    fig1.savefig(os.path.join(FIGURES_DIR, f'invasion_prevalence_{NAME_APPEND}.pdf'), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig1)

    # 2. Population Persistence Figure
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.2))
    for idx, (frac, s_data) in enumerate(persistence_runs):
        color = cmap(idx + 2)
        ax2.plot(t, s_data, color=color, linewidth=1.6, label=f"{int(frac*100)}% Initial Infection")
    ax2.set_title(r"Population Persistence $S(t)$", fontweight='bold')
    ax2.set_xlabel("Time Step (Days)")
    ax2.set_ylabel("Persistence Ratio")
    ax2.legend(frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.5)
    fig2.savefig(os.path.join(FIGURES_DIR, f'invasion_persistence_{NAME_APPEND}.pdf'), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig2)

if __name__ == "__main__":
    main()