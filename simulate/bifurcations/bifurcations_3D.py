import random as rand
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulate.helper_funcs import *
from simulate.rules import *

# ------------------------------------------
# hibernacula-INDEPENDENT initial conditions
# ------------------------------------------

# probabilities
p_infected = 0.01         # chance a hibernating bat gets infected (given that WNS is on) on any given day
p_dead = 0.005            # chance that an infected bat dies on any given day
p_recover = 0.01          # chance of recovering and going back into hibernation on any given day
p_awake = 0.08            # OKAY # chance of a waking bat arousing a hibernating bat from torpor on any given day
p_hibernate = 0.5         # CONFIDENT # chance of a bat switching between hibernating and not (given that Te switches) on any given day
p_netchange = 0.000215    # CONFIDENT # chance of new bat due to immigration/birth per day

# ----------------------------------------
# hibernacula-DEPENDENT initial conditions
# ----------------------------------------

# population counts
Hi_num = 500             # hibernating bats
NHi_NIn_num = 0         # non-hibernating non-infected bats
In_num = 1              # non-hibernating infected bats
Ot_num = 0              # other bats
Re_num = 0              # recovered bats

# resource limits
water = 290             # OKAY # number of bats it would take to deplete water completely
food = 170              # OKAY # number of bats it would take to deplete food completely

time = 3650             # total days
winter = 120            # CONFIDENT # length of winter season in Nebraska mines
recovery_period = 130    # number of days spent in recovery before re-infection is possible

# last tested 60
# ------------------
# bifurcation values
# ------------------

#times_list = [180, 365, 3*365]
times_list = [180, 365, 3*365, 10*365, 20*365, 40*365]
#times_list = [3*365, 10*365, 40*365]

param_change = ["p_infected", "p_recover"]
num_params = 30
parameters_list = [np.linspace(0.001,0.1,num_params), np.linspace(0.0001,0.5,num_params)]
totals_list = np.empty((num_params,num_params), dtype=object)

for i in range(num_params):
    for j in range(num_params):
        totals_list[i, j] = []

# ----------
# initialize
# ----------

def make_initial_state():
    return {
        "Hi": [1]*(Hi_num + Re_num),
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
        "p_netchange": p_netchange,
        "water": water,
        "food": food,
        "winter": winter,
        "recovery_period": recovery_period
    }

    for i in range(len(parameters_list[0])):
        parameters[param_change[0]] = parameters_list[0][i]

        for j in range(len(parameters_list[1])):
            parameters[param_change[1]] = parameters_list[1][j]

            history = simulate(make_initial_state(), steps=times_list[-1], parameters=parameters)
            total = np.array(history["Hi"]) + np.array(history["NHi_NIn"]) + np.array(history["In"])

            totals_list[i][j] = total
            if (i % 10 == 0) and (j % 10 == 0): # save some time
                print(f"list ({i},{j})")

    rows = 2
    cols = len(times_list)// rows
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12*rows//cols), subplot_kw={'projection': '3d'}, constrained_layout=True)
    axes = axes.ravel()

    # for 3d plotting
    X, Y = np.meshgrid(np.ravel(parameters_list[0]), np.ravel(parameters_list[1])) # create meshgrid for X and Y
    Z = np.zeros(X.shape) # initialize Z values (total population) for the 3D plot

    for i, time in enumerate(times_list):
        Z.fill(0) # clear Z before updating with new values

        for j in range(totals_list.shape[0]):
            for k in range(totals_list.shape[1]):
                Z[j, k] = totals_list[j][k][time - 1]

        # plot the surface for the current time slice
        ax = axes[i]
        ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_xlabel(f"{param_change[0]}")
        ax.set_ylabel(f"{param_change[1]}")
        ax.set_zlabel(f"Population at day {time}")
        ax.set_title(f"Population at day {time}")

    plt.savefig("3D_bifurcations.png", dpi=200, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()