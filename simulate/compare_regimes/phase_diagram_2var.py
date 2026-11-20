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

T_seasonal, win_length, win_start, lambda_sum, lambda_win = latitude1_NorthMidwest() # or latitude2_SouthMidwest()

# ----------------------------------
# phase diagram comparison variables
# ----------------------------------

times_list = [365, 3*365, 5*365, 10*365] # must be even and in ascending order
num_params = 20 # how many samples in the given range
num_seeds = 4   # number of stochastic realizations per cell (avgd later)

# longterm survival: delta vs. res_gain
param_change = ["delta", "res_gain"]
parameters_list = [np.linspace(0.005,0.05,num_params), np.linspace(0,1,num_params)]
NAME_APPEND = "final_persistence:delta_vs_resgain_lat1NMW"
param1_label = r"Fungal decay rate ($\delta$)"
param2_label = r"Acquired resistance ($r_{\mathrm{gain}}$)"

# # immune paradox 1: inf_alpha vs. res_gain
# param_change = ["inf_alpha", "res_gain"]
# parameters_list = [np.linspace(1,10,num_params), np.linspace(0,1,num_params)]
# NAME_APPEND = "immune_paradox:infalpha_vs_resgain_lat1NMW"
# param1_label = r"Infection parameter ($\alpha_{\mathrm{inf}}$)"
# param2_label = r"Acquired resistance ($r_{\mathrm{gain}}$)"

# # longevity: T_inf vs. delta
# param_change = ["T_inf", "delta"]
# parameters_list = [np.linspace(5,45,num_params), np.linspace(0.005,0.05,num_params)]
# NAME_APPEND = "longevity:Tinf_vs_delta_lat2SMW"
# param1_label = r"Infected period ($T_{\mathrm{inf}}$)"
# param2_label = r"Fungal decay rate ($\delta$)"

# # peak vs. longterm: T_inf vs. res_gain
# param_change = ["T_inf", "res_gain"]
# parameters_list = [np.linspace(5,45,num_params), np.linspace(0,1,num_params)]
# NAME_APPEND = "peakvslongterm:Tinf_vs_resgain_lat2SMW"
# param1_label = r"Infected period ($T_{\mathrm{inf}}$)"
# param2_label = r"Acquired resistance ($r_{\mathrm{gain}}$)"

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

    # 1. 3D Total Population Surface Plots (4 Separate PDFs)
    Z_pop = np.zeros(X.shape)

    for idx, time in enumerate(times_list):
        year = time // 365
        Z_pop.fill(0)
        for i in range(len(parameters_list[0])):
            for j in range(len(parameters_list[1])):
                start_idx = max(0, time - 365)
                Z_pop[i, j] = np.mean(totals_list[i, j, start_idx:time])

        fig1 = plt.figure(figsize=(6.5, 5.0))
        ax = fig1.add_subplot(111, projection='3d')

        Z_smooth = gaussian_filter(Z_pop, sigma=0.3)
        ax.plot_surface(X, Y, Z_smooth, cmap='viridis', edgecolor='none', alpha=0.92, antialiased=True)

        ax.set_xlabel(param1_label, labelpad=10)
        ax.set_ylabel(param2_label, labelpad=10)
        ax.set_zlabel("Population Count", labelpad=6)
        ax.set_title(f"Total Population at Year {year}", pad=-12, fontweight='bold')

        ax.tick_params(axis='x', pad=3)
        ax.tick_params(axis='y', pad=3)
        ax.tick_params(axis='z', pad=3)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

        # Fixed margins instead of bbox_inches='tight', which ignores 3D axis labels
        fig1.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=0.90)

        # Rotate slightly and pull camera back to prevent right-side z-axis clipping
        ax.view_init(elev=28, azim=-50)
        ax.set_box_aspect([1.0, 1.0, 0.75], zoom=0.88)

        out_path = os.path.join(FIGURES_DIR, f'phase_diagram:{NAME_APPEND}_yr{year}.pdf')
        fig1.savefig(out_path)  # no bbox_inches='tight'
        plt.close(fig1)

    # 2. 2D Likelihood of Survival Heatmap
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

        ax.set_xlabel(param1_label)
        ax.set_ylabel(param2_label)
        ax.set_title(f"Likelihood of Survival at Year {time // 365}", fontweight='bold')
        ax.grid(False)

    cbar = fig2.colorbar(mesh, ax=axes2.tolist(), orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label(r"Likelihood of Survival Ratio ($E[N(t)] / N_0$)", fontsize=10)

    plt.savefig(os.path.join(FIGURES_DIR, 'phase_survival_heatmap:' + NAME_APPEND + '.pdf'), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()