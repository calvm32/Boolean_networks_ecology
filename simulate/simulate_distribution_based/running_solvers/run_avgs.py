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

fraction_infected = 0   # choose in [0, 1]

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

time = 3650             # total days

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================


# initialize accumulators
history_avg = {
    "Hi": np.zeros(time),
    "Ot": np.zeros(time),
    "In": np.zeros(time),
    "Im": np.zeros(time),
    "De": np.zeros(time),
}

def simulate(initial_state, steps, parameters):
    state = initial_state
    T_win = parameters["T_win"]

    history = {
        "Hi": np.empty(steps,dtype=np.int32),
        "Ot": np.empty(steps,dtype=np.int32),
        "In": np.empty(steps,dtype=np.int32),
        "Im": np.empty(steps,dtype=np.int32),
        "De": np.empty(steps,dtype=np.int32),
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= T_win: # T_win
            state["Te"] = 0   
        else:
            state["Te"] = 1 # summer
        counts = count(state)

        history["Hi"][t] = (counts["Hi"])
        history["Ot"][t] = (counts["Ot"])
        history["In"][t] = (counts["In"])
        history["Im"][t] = (counts["Im"])
        history["De"][t] = (counts["De"])

        state = step(state, parameters)

        if t % 50 == 0:
            print(f"done w/ simulation at step {t}")

    return history


def main():
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

    for i in range(avg_over):

        history = simulate(
            make_initial_state(Hi_list, fraction_infected, T_inf),
            steps=time,
            parameters=parameters
        )

        for key in history_avg:
            history_avg[key] += np.array(history[key])

    # divide by number of runs for avg
    for key in history_avg:
        history_avg[key] /= avg_over    

    plot_history_highlights(history_avg, T_win)

if __name__ == "__main__":
    main()