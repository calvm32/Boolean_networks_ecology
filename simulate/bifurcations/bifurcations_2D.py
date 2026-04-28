import random as rand
import numpy as np
import matplotlib.pyplot as plt

from simulate.helper_funcs import *
from simulate.rules import *

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.0         # chance a hibernating bat gets infected (given that WNS is on) on any given day
p_dead = 0.005            # chance that an infected bat dies on any given day
p_recover = 0.01          # chance of recovering and going back into hibernation on any given day
p_awake = 0.08            # OKAY # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_hibernate = 0.5         # CONFIDENT # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_netchange = 0.000215    # CONFIDENT # chance of new bat due to immigration/birth per day

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

# population counts
Hi_num = 500            # hibernating bats
NHi_NIn_num = 0         # non-hibernating non-infected bats
In_num = 1              # non-hibernating infected bats
Ot_num = 0              # other bats
Re_num = 0              # recovered bats

# resource limits
water = 290             # OKAY # number of bats it would take to deplete water completely
food = 170              # OKAY # number of bats it would take to deplete food completely

time = 3650             # total days
winter = 120            # CONFIDENT # length of winter season in Nebraska mines
immunity_period = 0    # number of days spent in recovery before re-infection is possible

# ------------------
# bifurcation values
# ------------------

times_list = [180, 365, 3*365, 10*365, 20*365, 40*365]
parameters_list = np.linspace(0.001,0.5,100)
totals_list = []

param_change = "p_recover"

# ----------
# initialize
# ----------

def make_initial_state():
    return {
        "Hi": [1]*(Hi_num),
        "NHi_NIn": [1]*NHi_NIn_num,
        "Ot": [1]*Ot_num,
        "In": [1]*In_num,
        "De": [0]*(Hi_num + NHi_NIn_num + In_num),
        "Re": [0]*Re_num,
        "Wa": 1,
        "Fo": 1,
        "Te": 0,
        "Hu": 0,
        "WNS": 0,
    }

def simulate(initial_state, steps, parameters):
    state = initial_state
    winter = parameters["winter"]

    history = {
        "Hi":[],
        "NHi_NIn":[],
        "Ot":[],
        "In":[],
        "De":[],
        "Re":[],
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= winter: # winter
            state["Te"] = 0   
        else:
            state["Te"] = 1 # summer
        counts = count(state)

        history["Hi"].append(counts["Hi"])
        history["NHi_NIn"].append(counts["NHi_NIn"])
        history["Ot"].append(counts["Ot"])
        history["In"].append(counts["In"])
        history["De"].append(counts["De"])
        history["Re"].append(counts["Re"])

        state = step(state, parameters)

    return history


def main():

    inhabitant_nodes = ["Hi", "NHi_NIn", "In", "Ot", "De", "Re"]
    resource_nodes = ["Wa", "Fo"]
    environment_nodes = ["Te", "Hu", "El", "Po", "Su", "Ba", "WNS"]

    nodes = inhabitant_nodes + resource_nodes + environment_nodes

    parameters = {
        "p_infected": p_infected,
        "p_dead": p_dead,
        "p_awake": p_awake,
        "p_recover": p_recover,
        "p_hibernate": p_hibernate,
        "p_netchange": p_netchange,
        "water": water,
        "food": food,
        "winter": winter,
        "immunity_period": immunity_period
    }

    for i in parameters_list:
        parameters[param_change] = i

        history = simulate(make_initial_state(), steps=times_list[-1], parameters=parameters)
        total = np.array(history["Hi"]) + np.array(history["NHi_NIn"]) + np.array(history["In"]) + np.array(history["Re"])
        totals_list.append(total)
        print(i)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 10))
    axes = axes.ravel() # for iteration

    for i, time in enumerate(times_list):
        totals_at_time = []
        for total in totals_list:
            totals_at_time.append(total[time-1])
        
        axes[i].plot(parameters_list, totals_at_time)
        axes[i].set_xlabel(f"{param_change}")
        axes[i].set_ylabel(f"Population at day {time}")

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()