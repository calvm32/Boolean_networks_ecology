import random as rand
import numpy as np
import copy

from simulate.simulate_distribution_based.helper_funcs import *
from simulate.simulate_distribution_based.rules import *
from simulate.simulate_distribution_based.simulate import *
from simulate.data import *

# --------------------
# set up control group
# --------------------

data = happy_jack_data()

START_YEAR = data[0]["year"]
SAMPLE_DAY = data[0]["day"]

obs_times = []
obs_Hi = []
obs_Ot = []
obs_In = []

for d in data:
    t = d["day"] + 365*(d["year"]-START_YEAR)
    
    obs_times.append(t)
    obs_Ot.append(d["Tri_Ot"] + d["Misc_Ot"])
    obs_In.append(d["In"])

def sample_params():                                                                                                                                     
    return {                                                                                                                                             
        "inf_alpha": inf_alpha,                                                                                               
        "inf_beta": inf_beta,                                               
        "delta": delta,                                                     
        "T_inf": T_inf,                                                     
        "T_TBD": T_TBD,                                                     
        "T_AD": T_AD,                                                       
        "T_seasonal": rand.uniform(20,60),                                                                                                               
        "win_length": rand.uniform(80,180),                                                                                                              
        "win_start": rand.uniform(230,280),                                                                                                              
        "lambda_win": lambda_win,                                           
        "lambda_sum": rand.uniform(0.00015, 0.00025),                                                                                                    
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
T_seasonal = 35                             # approx. transition time in days between hibernating and not
                                            # considered in 10-40 maybe?
win_length = 95                             # length of winter season in days in Nebraska mines
                                            # considered in 5-7 months, depending on transition period T_seasonal
win_start = 264                             # approximate day in calendar year that Te : 1 -> 0

# BAT IN/OUT FLUX
lambda_win = 0                              # population growth value during winter, 
                                            # considered in [0, 0.01] 
lambda_sum = 0.00015317467856               # population growth value during summer,
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

time = 3650             # total days

# ==========================================================================================================================
# ==========================================================================================================================
# ==========================================================================================================================
    
def loss(parameters, runs=2):
    losses = []

    for _ in range(runs):
        sim = simulate(make_initial_state(Hi_list, fraction_infected), steps=max(obs_times)+1, parameters=parameters)

        error = 0.0
        for i, t in enumerate(obs_times):
            pred_Ot = sim["Ot"][t]
            pred_In = sim["In"][t]

            error += (pred_Ot - obs_Ot[i])**2
            error += (pred_In - obs_In[i])**2

        losses.append(error / len(obs_times))

    return np.mean(losses)

# ----------
# run things
# ----------

def simulate(initial_state, steps, parameters):
    state = initial_state
    win_length = parameters["win_length"]

    history = {
        "Hi": np.empty(steps,dtype=np.int32),
        "Ot": np.empty(steps,dtype=np.int32),
        "In": np.empty(steps,dtype=np.int32),
        "Im": np.empty(steps,dtype=np.int32),
        "De": np.empty(steps,dtype=np.int32),
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= win_length: # win_length
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
        L = abs(loss(params))
        
        if L < best_loss:
            best_loss = L
            best = params
            print("New best:", best_loss, best)
        
        print(f"checked {i}")

    best_sim = simulate(make_initial_state(Hi_list, fraction_infected), steps = 4500, parameters=best)
    plot_history_highlights(best_sim, win_length, win_start, sample=[obs_times, obs_Ot])


if __name__ == "__main__":
    main()
