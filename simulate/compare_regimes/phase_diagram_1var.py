import random as rand
import numpy as np
import matplotlib.pyplot as plt

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

num_infected = 1

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

# ----------------------------------
# phase diagram comparison variables
# ----------------------------------

times_list = [180, 365, 3*365, 10*365, 20*365]
num_params = 10 #100

# outbreak severity: inf_alpha vs. T_inf = acute phase
param_change = "inf_alpha"
parameters_list = np.linspace(1,8,num_params)

title = "Phase Diagram"

totals_list = []

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================


def main():
    parameters = sample_params()
    totals_list = []

    for val in parameters_array:
        parameters[param_change] = float(val)
        history = simulate(make_initial_state(Hi_list, num_infected), steps=times_list[-1], parameters=parameters, Print=False)
        total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
        totals_list.append(total)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 8), constrained_layout=True)
    axes = axes.ravel()

    for i, time in enumerate(times_list):
        totals_at_time = [total[time-1] for total in totals_list]
        
        axes[i].plot(parameters_array, totals_at_time, color=COLORS["Hi"], marker='o', markersize=3, linewidth=1.8)
        axes[i].set_xlabel(r"Parameter: $\alpha_{\mathrm{inf}}$")
        axes[i].set_ylabel(f"Population at Day {time}")
        axes[i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[i].set_title(f"Time Horizon: Day {time}", fontweight='bold')
        axes[i].grid(True)

    plt.savefig(os.path.join(FIGURES_DIR, 'outbreakseverity_1var_sweep.pdf'), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()