import random as rand
import numpy as np
from helper_funcs import *
from rules import *

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.01         # chance a hibernating bat gets infected (given that WNS is on) on any given day
p_dead = 0.005            # chance that an infected bat dies on any given day
p_awake = 0.02            # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_recover = 0.01          # chance of recovering and going back into hibernation on any given day
p_hibernate = 0.5         # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_influx = 0.001          # chance of new bat due to immigration/birth per day

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

# population counts
Hi_num = 200            # hibernating bats
NHi_NIn_num = 0         # non-hibernating non-infected bats
In_num = 1              # non-hibernating infected bats
Ot_num = 0              # other bats

# resource limits
water = 5000            # number of bats it would take to deplete water completely
food = 5000             # number of bats it would take to deplete food completely

time = 1000              # total days
winter = 120            # length of winter season

# ----------
# initialize
# ----------

state0 = {
    "Hi": [1]*Hi_num,
    "NHi_NIn": [1]*NHi_NIn_num,
    "Ot": [1]*Ot_num,
    "In": [1]*In_num,
    "De": [0]*(Hi_num + NHi_NIn_num + In_num),

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

        state = step(state, parameters)

    return history


def main():

    inhabitant_nodes = ["Hi", "NHi_NIn", "In", "Ot", "De"]
    resource_nodes = ["Wa", "Fo"]
    environment_nodes = ["Te", "Hu", "El", "Po", "Su", "Ba", "WNS"]

    nodes = inhabitant_nodes + resource_nodes + environment_nodes

    parameters = {
        "p_infected": p_infected,
        "p_dead": p_dead,
        "p_awake": p_awake,
        "p_recover": p_recover,
        "p_hibernate": p_hibernate,
        "p_influx": p_influx,
        "water": water,
        "food": food,
        "winter": winter
    }

    if In_num != 0:
        state0["WNS"] = 1

    history = simulate(state0, steps=time, parameters=parameters)
    plot_history(history)

if __name__ == "__main__":
    main()