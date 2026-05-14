import matplotlib.pyplot as plt
import numpy as np
from simulate.rules import *

def step(state, parameters):
    agg = aggregate(state)

    env_next = update_environment(state, agg, parameters)
    pop_next = update_individuals(state, {**state, **env_next}, parameters)

    return {**state, **env_next, **pop_next}
    
def aggregate(state):
    return {
        "Hi_any": len(state["Hi"]) > 0,
        "NHi_NIn_any": len(state["NHi_NIn"]) > 0,
        "Ot_any": len(state["Ot"]) > 0,
        "In_any": len(state["In"]) > 0,
        "Hi_sum": len(state["Hi"]),
        "NHi_NIn_sum": len(state["NHi_NIn"]),
        "Ot_sum": len(state["Ot"]),
        "In_sum": len(state["In"]),
    }

def count(state):
    return {
        "Hi": len(state["Hi"]),
        "NHi_NIn": len(state["NHi_NIn"]),
        "Ot": len(state["Ot"]),
        "In": len(state["In"]),
        "De": len(state["De"]),
        "Re": len(state["Re"])
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
    ax1.plot(t, history["NHi_NIn"], label="Non-hibernating, non-infected (NHi_NIN)")
    ax1.plot(t, history["In"], label="Infected (In)")
    ax1.plot(t, history["De"], label="Deceased (De)")
    ax1.plot(t, history["Re"], label="Recovered (Re)")
    ax1.plot(t, history["Ot"], label="Other species (Ot)") if np.any(history["Ot"]) else None

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics (Boolean Network)")
    ax1.legend()
    ax1.grid()

    # ------------
    # total counts
    # ------------

    total = np.array(history["Hi"]) + np.array(history["NHi_NIn"]) + np.array(history["In"]) + np.array(history["Re"])

    ax2.plot(t, total, label="Total tricolored bats")
    ax2.plot(t, history["Ot"], label="Other species (Ot)") if np.any(history["Ot"]) else None

    # if there's sample data, compare:
    if len(sample) != 0:
        obs_times = sample[0]; obs_NHi_NIn = sample[1]
        ax2.scatter(obs_times, obs_NHi_NIn, label="Observed total tricolored bats")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Population count")
    ax2.set_title("Bat Population Dynamics (Boolean Network)")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

def plot_history_highlights(history, winter, sample=[]):
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
    ax1.plot(t, history["NHi_NIn"], label="Non-hibernating, non-infected (NHi_NIN)")
    ax1.plot(t, history["In"], label="Infected (In)")
    ax1.plot(t, history["De"], label="Deceased (De)")
    ax1.plot(t, history["Re"], label="Recovered (Re)")
    ax1.plot(t, history["Ot"], label="Other species (Ot)") if np.any(history["Ot"]) else None

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics (Boolean Network)")
    ax1.legend()
    ax1.grid()

    # ------------
    # total counts
    # ------------

    total = np.array(history["Hi"]) + np.array(history["NHi_NIn"]) + np.array(history["In"]) + np.array(history["Re"])

    ax2.plot(t, total, label="Total tricolored bats")
    ax2.plot(t, history["Ot"], label="Other species (Ot)") if np.any(history["Ot"]) else None

    # if there's sample data, compare:
    if len(sample) != 0:
        obs_times = sample[0]; obs_NHi_NIn = sample[1]
        ax2.scatter(obs_times, obs_NHi_NIn, label="Observed total tricolored bats")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Population count")
    ax2.set_title("Bat Population Dynamics (Boolean Network)")
    ax2.legend()
    ax2.grid()

    # highlight the winter data
    for time in t:
        
        if time % 365 == winter and time <= 365:
            ax1.axvspan(time - winter, time, facecolor='blue', alpha=0.1)
            ax2.axvspan(time - winter, time, facecolor='blue', alpha=0.1)
        if time % 365 == winter:
            ax1.axvspan(time - winter + 365, time + 365, facecolor='blue', alpha=0.1)
            ax2.axvspan(time - winter + 365, time + 365, facecolor='blue', alpha=0.1)

    plt.tight_layout()
    plt.show()