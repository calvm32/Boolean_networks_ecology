import random as rand
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter

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

num_infected = 10

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

T_seasonal, win_length, win_start, lambda_sum, lambda_win = latitude2_SouthMidwest() # or latitude2_SouthMidwest()

# ----------------------------------
# phase diagram comparison variables
# ----------------------------------

times_list = [365, 3*365, 10*365, 20*365] # must be even and in ascending order
num_params = 12 #32 # how many samples in the given range
num_seeds = 5   # number of stochastic realizations per cell

# longterm survival: delta vs. k_imm = final persistance heatmap
# param_change = ["delta", "k_imm"]
# parameters_list = [np.linspace(0.001,0.04,num_params), np.linspace(0.1,10,num_params)]

# outbreak severity: inf_alpha vs. T_inf = acute phase
param_change = ["inf_alpha", "T_inf"]
parameters_list = [np.linspace(1,8,num_params), np.linspace(10,45,num_params)]

title = "Phase Diagram"

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


# store lists of arrays to hold multiple seeds per cell
totals_list = np.empty((num_params, num_params), dtype=object)
for i in range(num_params):
    for j in range(num_params):
        totals_list[i, j] = []

def main():
    parameters = sample_params()
    totals_list = np.zeros((len(parameters_list[0]), len(parameters_list[1]), times_list[-1]))

    for i in range(len(parameters_list[0])):
        parameters[param_change[0]] = parameters_list[0][i]

        for j in range(len(parameters_list[1])):
            parameters[param_change[1]] = parameters_list[1][j]

            seed_runs = []
            for seed in range(num_seeds):
                history = simulate(make_initial_state(Hi_list, num_infected), steps=times_list[-1], parameters=parameters, Print=False)
                total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])
                seed_runs.append(total)

            totals_list[i, j] = np.mean(seed_runs, axis=0)

        if i % 4 == 0:
            print(f"Completed parameter row {i}/{len(parameters_list[0])}")

    X, Y = np.meshgrid(parameters_list[0], parameters_list[1], indexing='ij')
    initial_total = totals_list[0, 0, 0]
    rows = 2
    cols = len(times_list) // rows

    # --- 1. 3D Total Population Surface Plots ---
    fig1, axes1 = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 10), subplot_kw={'projection': '3d'}, constrained_layout=True)
    axes1 = axes1.ravel()
    Z_pop = np.zeros(X.shape)

    for idx, time in enumerate(times_list):
        Z_pop.fill(0)
        for i in range(len(parameters_list[0])):
            for j in range(len(parameters_list[1])):
                start_idx = max(0, time - 365)
                Z_pop[i, j] = np.mean(totals_list[i, j, start_idx:time])

        ax = axes1[idx]
        Z_smooth = gaussian_filter(Z_pop, sigma=0.3)
        ax.plot_surface(X, Y, Z_smooth, cmap='viridis', edgecolor='none', alpha=0.92, antialiased=True)

        ax.set_xlabel(r"Infection Parameter ($\alpha_{\mathrm{inf}}$)", labelpad=8)
        ax.set_ylabel(r"Infectious Period ($T_{\mathrm{inf}}$)", labelpad=8)
        ax.set_zlabel("Population Count", labelpad=8)
        ax.set_title(f"Total Population at Year {time // 365} (Day {time})", pad=10, fontweight='bold')

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    plt.savefig(os.path.join(FIGURES_DIR, 'outbreak_total_population_3d.pdf'), bbox_inches='tight')
    plt.show()

    # --- 2. 2D Likelihood of Survival Heatmap ---
    fig2, axes2 = plt.subplots(nrows=rows, ncols=cols, figsize=(14, 8), constrained_layout=True)
    axes2 = axes2.ravel()
    Z_like = np.zeros(X.shape)

    for idx, time in enumerate(times_list):
        ax = axes2[idx]
        Z_like.fill(0)
        for i in range(len(parameters_list[0])):
            for j in range(len(parameters_list[1])):
                start_idx = max(0, time - 365)
                mean_pop = np.mean(totals_list[i, j, start_idx:time])
                Z_like[i, j] = mean_pop / initial_total

        Z_smooth_like = gaussian_filter(Z_like, sigma=0.4)
        mesh = ax.pcolormesh(X, Y, Z_smooth_like, shading='auto', cmap='YlGnBu', vmin=0, vmax=1)

        ax.set_xlabel(r"Infection Parameter ($\alpha_{\mathrm{inf}}$)")
        ax.set_ylabel(r"Infectious Period ($T_{\mathrm{inf}}$)")
        ax.set_title(f"Likelihood of Survival at Year {time // 365} (Day {time})", fontweight='bold')
        ax.grid(False)

    cbar = fig2.colorbar(mesh, ax=axes2.tolist(), orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label(r"Likelihood of Survival Ratio ($E[N(t)] / N_0$)", fontsize=10)

    plt.savefig(os.path.join(FIGURES_DIR, 'outbreak_survival_likelihood_heatmap.pdf'), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()