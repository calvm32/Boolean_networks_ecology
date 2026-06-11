import matplotlib.pyplot as plt
import numpy as np
from simulate.simulate_distribution_based.rules import *

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
        "De": sum(bat[0] for bat in state["De"]),
        "Im": len(state["Im"])
    }

def perturb(params, keys, scale=0.15):
    new = params.copy()
    for k in keys:
        val = params[k]
        if val > 0:
            new[k] = max(0, val + np.random.normal(0, scale * val))
    return new

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
    ax1.plot(t, history["De"], label="Deceased (De)")
    ax1.plot(t, history["Im"], label="Imcovered (Im)")

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

def plot_history_highlights(history, T_win, sample=[]):
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
    ax1.plot(t, history["De"], label="Deceased (De)")
    ax1.plot(t, history["Im"], label="Imcovered (Im)")

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

    # highlight the T_win data
    for time in t:
        
        if time % 365 == T_win and time <= 365:
            ax1.axvspan(time - T_win, time, facecolor='blue', alpha=0.1)
            ax2.axvspan(time - T_win, time, facecolor='blue', alpha=0.1)
        if time % 365 == T_win:
            ax1.axvspan(time - T_win + 365, time + 365, facecolor='blue', alpha=0.1)
            ax2.axvspan(time - T_win + 365, time + 365, facecolor='blue', alpha=0.1)

    plt.tight_layout()
    plt.show()