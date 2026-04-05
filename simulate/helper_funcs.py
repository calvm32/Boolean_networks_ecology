import matplotlib.pyplot as plt
import numpy as np
from rules import *

def step(state, parameters):
    agg = aggregate(state)

    env_next = update_environment(state, agg, parameters)
    pop_next = update_individuals(state, {**state, **env_next}, parameters)

    return {**state, **env_next, **pop_next}
    
def aggregate(state):
    return {
        "Hi_any": any(state["Hi"]),
        "NHi_NIn_any": any(state["NHi_NIn"]),
        "Ot_any": any(state["Ot"]),
        "In_any": any(state["In"]),
    }

def count(state):
    return {
        "Hi": sum(state["Hi"]),
        "NHi_NIn": sum(state["NHi_NIn"]),
        "Ot": sum(state["Ot"]),
        "In": sum(state["In"]),
        "De": sum(state["De"]),
    }


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
    ax1.plot(t, history["Ot"], label="Other species (Ot)") if np.any(history["Ot"]) else None

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Population count")
    ax1.set_title("Bat Population Dynamics (Boolean Network)")
    ax1.legend()
    ax1.grid()

    # ------------
    # total counts
    # ------------

    total = np.array(history["Hi"]) + np.array(history["NHi_NIn"]) + np.array(history["In"])

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