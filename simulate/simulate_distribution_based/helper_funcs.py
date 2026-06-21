import matplotlib.pyplot as plt
import numpy as np
from simulate.simulate_distribution_based.rules import *

# ---------------------
# setup before each run
# ---------------------

def make_initial_state(Hi_list, fraction_infected, T_inf):
    # NOTICE : each inhabitant node contains the following information:
    # [ ON/OFF, 
    #   resistance number AKA res_num, 
    #   clustering number AKA mu_i, 
    #   infection number MINUS days spent infected (i.e. days left infirm), 
    #   0 for just entered hibernation OR 1 for exited hibernation at least once (to track arousal periods)
    # ]
    
    empty_pop = np.empty((0,5), dtype=float)

    return {
        "Hi": [
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0]
                for i in range(len(Hi_list))
                for _ in range(Hi_list[i][0] - int(Hi_list[i][0]*fraction_infected))
              ],
        "Ot": empty_pop.copy(),
        "In": [
                [1, 0, rand.uniform(Hi_list[i][1], Hi_list[i][2]), 0, 0]
                for i in range(len(Hi_list))
                for _ in range(int(Hi_list[i][0]*fraction_infected))
              ],
        "Im": empty_pop.copy(),
        "De": 0, # only need total numbers of dead
        "Re": 1,
        "Te": 0,
        "Hu": 0,
        "PD": 0,
    }

# ----------------------------------------
# computing individaul & population values
# ----------------------------------------

def step(state, parameters):
    agg = aggregate(state)

    env_next = update_environment(state, agg, parameters)
    pop_next = update_individuals(state, {**state, **env_next}, parameters)

    return {**state, **env_next, **pop_next}
    
def aggregate(state):
    return {
        "Hi_any": len(state["Hi"]) > 0,
        "Ot_any": len(state["Ot"]) > 0,
        "In_any": len(state["In"]) > 0,
        "Im_any": len(state["Im"]) > 0,
        "Hi_sum": len(state["Hi"]),
        "Ot_sum": len(state["Ot"]),
        "In_sum": len(state["In"]),
        "Im_sum": len(state["Im"]),
    }

def count(state):
    return {
        "Hi": len(state["Hi"]),
        "Ot": len(state["Ot"]),
        "In": len(state["In"]),
        "Im": len(state["Im"]),
        "De": state["De"],
    }

def perturb(params, keys, scale=0.15):
    new = params.copy()
    for k in keys:
        val = params[k]
        if val > 0:
            new[k] = max(0, val + np.random.normal(0, scale * val))
    return new

def compute_metrics(history, Hi_list):
    
    N0 = 0 # total pop
    for i in range(len(Hi_list)):
        N0 += Hi_list[i][0] 

    T = len(history["Hi"])
    alive = (np.array(history["Hi"])
           + np.array(history["Ot"])
           + np.array(history["In"])
           + np.array(history["Im"]))   # := N(t)

    # population metrics
    N_min = alive.min() # min population ever reached
    N_max = alive.max() # max population ever reached
    N_final = alive[-1] # final pop

    # prevalence: P(t) = In(t) / N(t)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(alive > 0, np.array(history["In"]) / alive, 0.0)

    P_max = P.max()            # peak prevalence
    T_Pmax = int(np.argmax(P)) # day of peak prevalence
    P_avg = P.mean()           # time-averaged prevalence

    # persistence: S(t) = N(t) / N(0)
    S = alive / N0
    S_final = S[-1]                         # final persistence

    # mortality burded: M(t) = De(t) / N(0)
    M = np.array(history["De"]) / N0
    M_final = M[-1]
    M_max = M.max()
    death_days = np.where(np.array(history["De"]) > 0)[0]
    T_De = int(death_days[0]) if len(death_days) > 0 else np.nan    # first death

    # disease invasion rate: I_new(t) = In(t+1) - In(t)
    In_arr = np.array(history["In"])
    I_new = np.diff(In_arr, prepend=In_arr[0])                      # new infections per day
    with np.errstate(invalid='ignore', divide='ignore'):
        I_rate = np.where(alive > 0, I_new / alive, 0.0)            # per-capita rate

    return {
        "N_min":    N_min,
        "N_final":  N_final,
        "P_max":    P_max,
        "T_Pmax":   T_Pmax,
        "P_avg":    P_avg,
        "S_final":  S_final,
        "M_final":  M_final,
        "M_max":    M_max,
        "T_De":     T_De,

        # full time-series (needed for Monte Carlo bands)
        "_P":       P,
        "_S":       S,
        "_M":       M,
        "_I_rate":  I_rate,
        "_alive":   alive,
    }

# ------------------------------
# plotting at the end of the run
# ------------------------------

def plot_history(history, sample=[]):
    t = range(len(history["Hi"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize = (14,7)
    )

    # -----------------
    # individual counts
    # -----------------

    ax1.plot(t, history["Hi"], label="Hibernating (Hi)")
    ax1.plot(t, history["Ot"], label="Non-hibernating, non-infected, non-immune (Ot)")
    ax1.plot(t, history["In"], label="Infected (In)")
    ax1.plot(t, history["Im"], label="Immune (Im)")
    ax1.plot(t, history["De"], label="Deceased (De)")

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics (Boolean Network)")
    ax1.legend()
    ax1.grid()

    # ------------
    # total counts
    # ------------

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])

    ax2.plot(t, total, label="Total tricolored bats")

    # if there's sample data, compare:
    if len(sample) != 0:
        obs_times = sample[0]; obs_Ot = sample[1]
        ax2.scatter(obs_times, obs_Ot, label="Observed total tricolored bats")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Population count")
    ax2.set_title("Bat Population Dynamics (Boolean Network)")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

def plot_history_highlights(history, win_length, win_start, sample=[]):
    t = range(len(history["Hi"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        constrained_layout=True,
        figsize = (14,7)
    )

    # -----------------
    # individual counts
    # -----------------

    ax1.plot(t, history["Hi"], label="Hibernating (Hi)")
    ax1.plot(t, history["Ot"], label="Non-hibernating, non-infected, non-immune (Ot)")
    ax1.plot(t, history["In"], label="Infected (In)")
    ax1.plot(t, history["Im"], label="Immune (Im)")
    ax1.plot(t, history["De"], label="Deceased (De)")

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics (Boolean Network)")
    ax1.legend()
    ax1.grid()

    # ------------
    # total counts
    # ------------

    total = np.array(history["Hi"]) + np.array(history["Ot"]) + np.array(history["In"]) + np.array(history["Im"])

    ax2.plot(t, total, label="Total tricolored bats")

    # if there's sample data, compare:
    if len(sample) != 0:
        obs_times = sample[0]; obs_Ot = sample[1]
        ax2.scatter(obs_times, obs_Ot, label="Observed total tricolored bats")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Population count")
    ax2.set_title("Bat Population Dynamics (Boolean Network)")
    ax2.legend()
    ax2.grid()

    # highlight the win_length data
    for year in range(-1, len(t)//365 + 1):
        if year == -1:
            start = 0
        else:
            start = win_start + year*365
        end = start + win_length

        ax1.axvspan(start, end, alpha=0.1)
        ax2.axvspan(start, end, alpha=0.1)

    plt.tight_layout()
    plt.show()

