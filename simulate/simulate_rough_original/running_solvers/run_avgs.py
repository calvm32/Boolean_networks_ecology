import random as rand
import numpy as np

from simulate.simulate_rough_original.helper_funcs import *
from simulate.simulate_rough_original.rules import *

avg_over = 20 # avg of X runs

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.005                          # chance a hibernating bat gets infected (given that PD is on) on any given day
p_dead = 0.005                              # chance that an infected bat dies on any given day
p_recover = 1-(1-(1-p_dead)**30)**(1/30)    # CONFIDENT # chance of recovering and going back into hibernation on any given day
p_awake = 0.08                              # OKAY # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_hibernate = 0.5                           # CONFIDENT # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_netchange = 0.000215                      # CONFIDENT # chance of new bat due to immigration/birth per day
contact_rate = 10                           # population-dependent rate of contact btwn health bat and WNS infected bat or surface

# -----------------
# types of immunity
# -----------------

immunity_period = 0                         # number of days spent in recovery before re-infection is possible
birth_resistance_max = 0.02                 # hereditary resistance of newborn, corresp. w/ rand.normalvariate(0, X)
recover_resistance_max = 0.3                # resistance after recovery, corresp. w/ rand.normalvariate(0, X)

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

# population counts
Hi_num = 100            # hibernating bats
NHO_num = 0         # non-hibernating non-infected bats
In_num = 1              # non-hibernating infected bats
Ot_num = 0              # other bats
Im_num = 0              # recovered bats

# resource limits
water = 1000            # OKAY # number of bats it would take to deplete water completely
food = 1000             # OKAY # number of bats it would take to deplete food completely

time = 2*3650           # total days
win_length = 120            # CONFIDENT # length of winter season in Nebraska mines

# ----------
# initialize
# ----------

res_num = 0             # starting resistance for bats in the hibernaculum

def make_initial_state():
    return {
        "Hi": [[1, res_num] for _ in range(Hi_num)],
        "NHO": [[1, res_num] for _ in range(NHO_num)],
        "Ot": [[1, res_num] for _ in range(Ot_num)],
        "In": [[1, res_num] for _ in range(In_num)],
        "De": [[0, res_num] for _ in range(Hi_num + NHO_num + In_num)],
        "Im": [[0, res_num] for _ in range(Im_num)],
        "Wa": 1,
        "Fo": 1,
        "Te": 0,
        "Hu": 0,
        "PD": 0,
    }


# initialize accumulators
history_avg = {
    "Hi": np.zeros(time),
    "NHO": np.zeros(time),
    "Ot": np.zeros(time),
    "In": np.zeros(time),
    "De": np.zeros(time),
    "Im": np.zeros(time),
}


def simulate(initial_state, steps, parameters):
    state = initial_state
    win_length = parameters["win_length"]

    history = {
        "Hi":[],
        "NHO":[],
        "Ot":[],
        "In":[],
        "De":[],
        "Im":[],
    }

    for t in range(steps):

        # Seasonal tempcycle
        if (t % 365) <= win_length: # win_length
            state["Te"] = 0   
        else:
            state["Te"] = 1 # summer
        counts = count(state)

        history["Hi"].append(counts["Hi"])
        history["NHO"].append(counts["NHO"])
        history["Ot"].append(counts["Ot"])
        history["In"].append(counts["In"])
        history["De"].append(counts["De"])
        history["Im"].append(counts["Im"])

        state = step(state, parameters)

    return history


def main():

    inhabitant_nodes = ["Hi", "NHO", "In", "Ot", "De", "Im"]
    resource_nodes = ["Wa", "Fo"]
    environment_nodes = ["Te", "Hu", "El", "Po", "Su", "Ba", "PD"]

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
        "water0": water,
        "food0": food,
        "win_length": win_length,
        "immunity_period": immunity_period,
        "contact_rate": contact_rate,
        "birth_resistance_max": birth_resistance_max,
        "recover_resistance_max": recover_resistance_max
    }

    for i in range(avg_over):

        history = simulate(
            make_initial_state(),
            steps=time,
            parameters=parameters
        )

        for key in history_avg:
            history_avg[key] += np.array(history[key])

    # divide by number of runs for avg
    for key in history_avg:
        history_avg[key] /= avg_over    

    plot_history_highlights(history_avg, win_length)

if __name__ == "__main__":
    main()