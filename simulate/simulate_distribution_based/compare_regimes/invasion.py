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
win_length = 210                                 # length of winter season in days in Nebraska mines
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

T_im = 0                         # number of days spent in recovery before re-infection is possible
res_max = 0                   # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
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
    parameters = {
        "inf_alpha": inf_alpha,
        "inf_beta": inf_beta,
        "delta": delta,
        "T_inf": T_inf,
        "T_TBD": T_TBD,
        "T_AD": T_AD,
        "T_seasonal": T_seasonal,
        "win_length": win_length,
        "lambda_win": lambda_win,
        "lambda_sum": lambda_sum,
        "T_im": T_im,
        "res_max": res_max,
        "recover_resistance_max": recover_resistance_max,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.cm.get_cmap("YlOrRd", len(init_fractions))
    
    for idx, frac in enumerate(init_fractions):

        hist = simulate(make_initial_state(Hi_list, frac), time, parameters)
        m = compute_metrics(hist, Hi_list)
        t = np.arange(time)
        color = cmap(idx)

        axes[0].plot(t, m["_P"], color=color,
                     label=f"{int(frac*100)}% initially infected")
        axes[1].plot(t, m["_S"], color=color,
                     label=f"{int(frac*100)}% initially infected")

    for ax, ylabel in zip(axes, ["Prevalence P(t)", "Persistence S(t)"]):
        ax.set_xlabel("Day"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Invasion Scenarios: Effect of Initial Infection Size")
    fig.tight_layout()
    plt.savefig("figures/invasion_scenarios.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    history = simulate(make_initial_state(Hi_list, frac), time, parameters=parameters)
    plot_history_highlights(history, win_length)
    

if __name__ == "__main__":
    main()
    