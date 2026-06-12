import random as rand
import numpy as np

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_rough_original.rules_vectorized import *

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

# NOTICE : the remaining populations (Ot, Im, In) all start with 0 inhabitants
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

# WINTER AWAKENING PATHWAY
awake_a = -3                                # represents a baseline arousal tendency, 
                                            # considered in a [-4,-2]
awake_b = 0.7                               # represents a social coupling strength, 
                                            # considered in [0.3, 1]

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
lambda_win = 0.0                         # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.05                           # population growth value during summer,
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

time = 3650 # total days
empty_pop = np.empty((0,5), dtype=float)

def make_initial_state():
    # NOTICE : each inhabitant node contains the following information:
    # [ ON/OFF, 
    #   resistance number AKA res_num, 
    #   clustering number AKA mu_i, 
    #   infection number MINUS days spent infected (i.e. days left infirm), 
    #   0 for just entered hibernation OR 1 for exited hibernation at least once (to track arousal periods)
    # ]
    return {
        "Hi": np.array([
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0]
                for i in range(len(Hi_list))
                for _ in range(Hi_list[i][0])
              ], dtype=float),
        "Ot": empty_pop.copy(),
        "In": empty_pop.copy(),
        "Im": empty_pop.copy(),
        "De": 0, # only need total numbers of dead
        "Re": 1,
        "Te": 0,
        "Hu": 0,
        "PD": 0,
    }

def simulate(initial_state, steps, parameters):
    state = initial_state
    T_win = parameters["T_win"]

    history = {
        "Hi": np.empty(steps,dtype=np.int32),
        "Ot": np.empty(steps,dtype=np.int32),
        "In": np.empty(steps,dtype=np.int32),
        "Im": np.empty(steps,dtype=np.int32),
        "De": 0,
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
        history["De"] = (counts["De"])

        state = step(state, parameters)

        print(f"done{t}")

    return history


def main():
    parameters = {
        "inf_alpha": inf_alpha,
        "inf_beta": inf_beta,
        "delta": delta,
        "awake_a": awake_a,
        "awake_b": awake_b,
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

    history = simulate(make_initial_state(), steps=time, parameters=parameters)
    plot_history_highlights(history, T_win)

if __name__ == "__main__":
    main()