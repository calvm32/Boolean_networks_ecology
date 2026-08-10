import random as rand
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.simulate_distribution_based.simulate import *

import os

OUTPUT_DIR = os.environ.get('SIM_OUTPUT_DIR', '.')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

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

# ----------------------------------
# phase diagram comparison variables
# ----------------------------------

times_list = [180, 365, 3*365, 10*365, 20*365, 40*365]
num_params = 10

# dead vs inf
param_change = ["inf_alpha", "inf_beta"]
parameters_list = [np.linspace(0.1,10,num_params), np.linspace(0.1,10,num_params)]
title = "inf_alphavbeta"

# --------------
# actual testing
# --------------

totals_list = np.empty((num_params,num_params), dtype=object)

for i in range(num_params):
    for j in range(num_params):
        totals_list[i, j] = []


# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================


def main():
    parameters = sample_params()

    for i in range(len(parameters_list[0])):
        parameters[param_change[0]] = parameters_list[0][i]

        for j in range(len(parameters_list[1])):
            parameters[param_change[1]] = parameters_list[1][j]

            history = simulate(make_initial_state(Hi_list, num_infected), steps=times_list[-1], parameters=parameters, Print=False)
            total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])

            totals_list[i][j] = total
            if (i % 10 == 0) and (j % 10 == 0): # save some time
                print(f"done w/ parameter ({i},{j})/({len(parameters_list[0])}x{len(parameters_list[1])})")

    rows = 2
    cols = len(times_list)// rows
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10), subplot_kw={'projection': '3d'}, constrained_layout=True)
    axes = axes.ravel()

    # for 3d plotting
    X, Y = np.meshgrid(np.ravel(parameters_list[0]), np.ravel(parameters_list[1])) # create meshgrid for X and Y
    Z = np.zeros(X.shape) # initialize Z values (total population) for the 3D plot

    for i, time in enumerate(times_list):
        Z.fill(0) # clear Z before updating with new values

        for j in range(totals_list.shape[0]):
            for k in range(totals_list.shape[1]):
                Z[j, k] = totals_list[j][k][time - 1]

        # plot the surface for the current time slice
        ax = axes[i]
        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_xlabel(f"{param_change[0]}")
        ax.set_ylabel(f"{param_change[1]}")
        ax.set_zlabel(f"Population at day {time}")
        ax.set_title(f"Population at day {time}", pad=15)

    for ax in axes:
        ax.zaxis.labelpad = 10
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
    
    fig.set_constrained_layout_pads(
        w_pad=0.05,   # width padding
        h_pad=0.1,    # height padding
        hspace=0.2,
        wspace=0.2
    )

    plt.savefig(os.path.join(FIGURES_DIR,'phase_diagram2var.pdf'), format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()