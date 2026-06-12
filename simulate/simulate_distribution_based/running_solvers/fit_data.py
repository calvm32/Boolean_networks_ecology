import random as rand
import numpy as np
import copy

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.data import *

# --------------------
# set up control group
# --------------------

data = happy_jack_data()

START_YEAR = data[0]["year"]
SAMPLE_DAY = 140

obs_times = []
obs_Ot = []
obs_Ot = []
obs_In = []

for d in data:
    t = SAMPLE_DAY + 365 * (d["year"] - START_YEAR)
    
    obs_times.append(t)
    obs_Ot.append(d["Ot"])
    obs_Ot.append(d["Ot"])
    obs_In.append(d["In"])

def sample_params():
    return {
        "inf_alpha": inf_alpha,
        "inf_beta": inf_beta,
        "delta": delta,
        "awake_a": awake_a,
        "awake_b": awake_b,
        "T_inf": T_inf,
        "T_TBD": T_inf,
        "T_AD": T_inf,
        "T_seasonal": T_inf,
        "T_win": T_win,
        "lambda_win": lambda_win,
        "lambda_sum": lambda_sum,
        "immunity_period": immunity_period,
        "birth_resistance_max": birth_resistance_max,
        "recover_resistance_max": recover_resistance_max,
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
    
def loss(parameters, runs=2):
    losses = []

    for _ in range(runs):
        sim = simulate(make_initial_state(), steps=max(obs_times)+1, parameters=parameters)

        error = 0.0
        for i, t in enumerate(obs_times):
            pred_N = sim["Ot"][t]
            pred_O = sim["Ot"][t]
            pred_I = sim["In"][t]

            error += (pred_N - obs_Ot[i])**2
            error += (pred_O - obs_Ot[i])**2
            error += (pred_I - obs_In[i])**2

        losses.append(error / len(obs_times))

    return np.mean(losses)

# ----------
# run things
# ----------

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

        # if t % 50 == 0:
        #     print(f"done w/ simulation at step {t}")

    return history


def main():
    parameters = sample_params()
    best = None
    best_loss = float("inf")
    n_iter = 20

    for i in range(n_iter):
        params = sample_params()
        L = loss(params)
        
        if L < best_loss:
            best_loss = L
            best = params
            print("New best:", best_loss, best)
        
        print(f"checked {i}")

    best_sim = simulate(make_initial_state(Hi_list, fraction_infected), steps = 4500, parameters=best)
    plot_history_highlights(best_sim, T_win, sample=[obs_times, obs_Ot])


if __name__ == "__main__":
    main()


"""
New best: 2183.6363636363635 {'p_infected': 0.007998399524002234, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.21995833929973632, 'p_netchange': 0.0005224715374178368, 'water': 941.0402072309407, 'food': 5000, 'T_win': 120}
New best: 3743.8863636363635 {'p_infected': 0.041521217775843514, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.3986495789950476, 'p_netchange': 0.000515467966854555, 'water': 990.266884779426, 'food': 5000, 'T_win': 120}
New best: 1235.3636363636365 {'p_infected': 0.01524217865079747, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.2697086101327978, 'p_netchange': 0.0004065503241910787, 'water': 735.3878583939961, 'food': 5000, 'T_win': 120}
New best: 362.0909090909091 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5629538071675663, 'p_netchange': 0.0002439501436381033, 'water': 1829.7787389194987, 'food': 5000, 'T_win': 120}
New best: 358.22727272727275 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.48303984638268127, 'p_netchange': 0.00020677835763457532, 'water': 5000, 'food': 5000, 'T_win': 120}
New best: 349.79545454545456 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5573477611867865, 'p_netchange': 0.0002470489971729022, 'water': 5000, 'food': 5000, 'T_win': 120}
New best: 340.4318181818182 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5375541529135586, 'p_netchange': 0.0002226736397319496, 'water': 5000, 'food': 5000, 'T_win': 120}
New best: 353.59090909090907 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0, 'p_recover': 0, 'p_hibernate': 0.5217376853281397, 'p_netchange': 0.0002371693157187258, 'water': 5000, 'food': 5000, 'T_win': 127.79382882757909}
New best: 351.3636363636364 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.14737923993374222, 'p_recover': 0, 'p_hibernate': 0.5398474209465732, 'p_netchange': 0.00024587321732499006, 'water': 5000, 'food': 5000, 'T_win': 120}
GLOBAL BEST: 324.8863636363636 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08177036053584033, 'p_recover': 0, 'p_hibernate': 0.506130033345668, 'p_netchange': 0.00022945884090207235, 'water': 5000, 'food': 5000, 'T_win': 120}
GLOBAL BEST: 302.3636363636364 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.26956807970377467, 'p_recover': 0, 'p_hibernate': 0.5448954517680774, 'p_netchange': 0.00021523603923560332, 'water': 927.0382284770973, 'food': 414.8102016331979, 'T_win': 120}
GLOBAL BEST: 324.70454545454544 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08629881107887209, 'p_recover': 0, 'p_hibernate': 0.4521499873392658, 'p_netchange': 0.00021204118330130886, 'water': 191.12299861491675, 'food': 58.85014554319323, 'T_win': 120}
GLOBAL BEST: 355.5909090909091 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.053821586907585324, 'p_recover': 0, 'p_hibernate': 0.5757344511791069, 'p_netchange': 0.00021559487804763611, 'water': 616.2438584507863, 'food': 60.8566143812073, 'T_win': 120}
GLOBAL BEST: 344.77272727272725 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.14891365897212455, 'p_recover': 0, 'p_hibernate': 0.4777174885328155, 'p_netchange': 0.0002050202186951778, 'water': 412.01349345275577, 'food': 898.6018997126824, 'T_win': 120}
New best: 317.9545454545455 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.04345960635454034, 'p_recover': 0, 'p_hibernate': 0.5332425535631224, 'p_netchange': 0.00024123902192027023, 'water': 869.4131416026416, 'food': 920.9508227333656, 'T_win': 120}
New best: 321.1363636363636 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.11996019118621024, 'p_recover': 0, 'p_hibernate': 0.5358468255532175, 'p_netchange': 0.00022540489216466345, 'water': 246.9063201140136, 'food': 386.8256630780159, 'T_win': 120}
New best: 308.72727272727275 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08, 'p_recover': 0, 'p_hibernate': 0.5, 'p_netchange': 0.0002, 'water': 350.34867097205273, 'food': 324.50109337947646, 'T_win': 120}
New best: 308.04545454545456 {'p_infected': 0, 'p_dead': 0, 'p_awake': 0.08, 'p_recover': 0, 'p_hibernate': 0.5, 'p_netchange': 0.0002, 'water': 371.46665433373187, 'food': 164.4683879543965, 'T_win': 120}

RESULTS:
    - p_awake = 0.08
    - p_netchange = 0.00021
    - p_hibernate = 0.5
    - food and water genuinely seem to have no effect here, 
      but that makes sense given the data we're testing against

"""